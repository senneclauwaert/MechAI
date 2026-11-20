"""The job queue: a table instead of a message broker.

The api calls enqueue(). The worker calls claim_next() in a loop. Two workers never take
the same job, thanks to `FOR UPDATE SKIP LOCKED`.
"""

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import case, or_, select, update
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.models import Job

LOCK_TIME = timedelta(minutes=10)  # a crashed job is picked up again after this
RETRY_DELAY = timedelta(seconds=30)


def enqueue(
    db: Session, job_type: str, payload: dict[str, Any] | None = None, *, commit: bool = True
) -> Job:
    job = Job(type=job_type, payload=payload or {})
    db.add(job)
    if commit:
        db.commit()
    else:
        db.flush()
    return job


def get_job(db: Session, job_id: uuid.UUID) -> Job | None:
    return db.get(Job, job_id)


def claim_next(db: Session) -> Job | None:
    """Take the next job that is due, and lock it so no other worker takes it."""
    now = datetime.now(UTC)
    db.execute(
        update(Job)
        .where(
            Job.status == "running",
            Job.locked_until < now,
            Job.attempts >= get_settings().job_max_attempts,
        )
        .values(status="failed", locked_until=None)
    )
    job = db.scalars(
        select(Job)
        .where(
            Job.run_after <= now,
            Job.attempts < get_settings().job_max_attempts,
            or_(
                Job.status == "queued",
                # running, but its lock ran out: the worker probably crashed
                (Job.status == "running") & (Job.locked_until < now),
            ),
        )
        .order_by(Job.run_after)
        .limit(1)
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    ).first()
    if job is None:
        db.commit()
        return None
    job.status = "running"
    job.attempts += 1
    job.locked_until = now + LOCK_TIME
    db.commit()
    return job


def _owned(job: Job, claim_token: int):
    return (
        (Job.id == job.id)
        & (Job.status == "running")
        & (Job.attempts == claim_token)
        & (Job.locked_until > datetime.now(UTC))
    )


def renew_lease(db: Session, job: Job, claim_token: int) -> bool:
    renewed = db.scalar(
        update(Job)
        .where(_owned(job, claim_token))
        .values(locked_until=datetime.now(UTC) + LOCK_TIME)
        .returning(Job.id)
    )
    db.commit()
    return renewed is not None


def mark_done(db: Session, job: Job, claim_token: int) -> None:
    db.execute(update(Job).where(_owned(job, claim_token)).values(status="done", locked_until=None))
    db.commit()


def mark_failed(db: Session, job: Job, error: str, claim_token: int) -> None:
    """Retry later, or give up, only while this worker still owns the claim."""
    exhausted = Job.attempts >= get_settings().job_max_attempts
    db.execute(
        update(Job)
        .where(_owned(job, claim_token))
        .values(
            last_error=error,
            locked_until=None,
            status=case((exhausted, "failed"), else_="queued"),
            run_after=case(
                (exhausted, Job.run_after),
                else_=datetime.now(UTC) + RETRY_DELAY * claim_token,
            ),
        )
    )
    db.commit()
