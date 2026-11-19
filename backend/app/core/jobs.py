"""The job queue: a table instead of a message broker.

The api calls enqueue(). The worker calls claim_next() in a loop. Two workers never take
the same job, thanks to `FOR UPDATE SKIP LOCKED`.
"""

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.models import Job

LOCK_TIME = timedelta(minutes=10)  # a crashed job is picked up again after this
RETRY_DELAY = timedelta(seconds=30)


def enqueue(db: Session, job_type: str, payload: dict[str, Any] | None = None) -> Job:
    job = Job(type=job_type, payload=payload or {})
    db.add(job)
    db.commit()
    return job


def claim_next(db: Session) -> Job | None:
    """Take the next job that is due, and lock it so no other worker takes it."""
    now = datetime.now(UTC)
    job = db.scalars(
        select(Job)
        .where(
            Job.run_after <= now,
            or_(
                Job.status == "queued",
                # running, but its lock ran out: the worker probably crashed
                (Job.status == "running") & (Job.locked_until < now),
            ),
        )
        .order_by(Job.run_after)
        .limit(1)
        .with_for_update(skip_locked=True)
    ).first()
    if job is None:
        return None
    job.status = "running"
    job.attempts += 1
    job.locked_until = now + LOCK_TIME
    db.commit()
    return job


def mark_done(db: Session, job: Job) -> None:
    job.status = "done"
    job.locked_until = None
    db.commit()


def mark_failed(db: Session, job: Job, error: str) -> None:
    """Retry later, or give up after the maximum number of attempts."""
    job.last_error = error
    job.locked_until = None
    if job.attempts >= get_settings().job_max_attempts:
        job.status = "failed"
    else:
        job.status = "queued"
        job.run_after = datetime.now(UTC) + RETRY_DELAY * job.attempts
    db.commit()
