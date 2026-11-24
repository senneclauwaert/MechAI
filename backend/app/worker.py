"""The background worker: takes jobs from core.jobs and runs them.

Start it with: python -m app.worker
Add a new job type by adding a function to HANDLERS.
"""

import logging
import signal
import time
from collections.abc import Callable
from threading import Event, Thread
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import jobs
from app.core.database import SessionLocal
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.modules.knowledge import service as knowledge

log = logging.getLogger("mechai.worker")


# Test jobs, to try the queue by hand through the dev endpoints.
def echo(db: Session, payload: dict[str, Any]) -> None:
    """A test job: it only writes its payload to the log."""
    log.info("echo job: %s", payload)


def fail(db: Session, payload: dict[str, Any]) -> None:
    """A test job that always fails, to see the retries."""
    raise AppError("CORE_TEST_FAILURE", "This test job always fails.")


# Job type (the `type` column of core.jobs) -> function that runs it.
HANDLERS: dict[str, Callable[[Session, dict[str, Any]], None]] = {
    "echo": echo,
    "fail": fail,
    "embed_text": knowledge.embed_text_job,
    "ingest_document": knowledge.ingest_document_job,
}

# Shutdown: SIGTERM/SIGINT let the current job finish, then the loop stops.
running = True


# Signal handler: finish the current job, then leave the loop.
def stop(*_: object) -> None:
    global running
    running = False
    log.info("stopping after the current job")


# Runs in a thread next to a long job and keeps extending its lock, so another
# worker does not think it crashed. Stops when the job ends or the claim is lost.
def _heartbeat(job_id: object, claim_token: int, finished: Event) -> None:
    while not finished.wait(jobs.LOCK_TIME.total_seconds() / 3):
        try:
            with SessionLocal() as db:
                job = db.get(jobs.Job, job_id)
                if job is None or not jobs.renew_lease(db, job, claim_token):
                    return
        except SQLAlchemyError:
            log.exception("could not renew lease for job %s", job_id)


def run_one(db: Session) -> bool:
    """Run one job. Returns False when the queue is empty."""
    job = jobs.claim_next(db)
    if job is None:
        return False
    log.info("job %s (%s) started, attempt %s", job.id, job.type, job.attempts)
    claim_token = job.attempts  # immutable ownership token for this attempt
    finished = Event()
    heartbeat = Thread(target=_heartbeat, args=(job.id, claim_token, finished), daemon=True)
    heartbeat.start()
    handler = HANDLERS.get(job.type)
    try:
        if handler is None:
            raise AppError("CORE_UNKNOWN_JOB", f"No handler for job type '{job.type}'.")
        handler(db, job.payload)
    except AppError as exc:
        db.rollback()
        log.warning("job %s failed: %s %s", job.id, exc.code, exc.message)
        jobs.mark_failed(db, job, f"{exc.code}: {exc.message}", claim_token)
    except Exception:
        db.rollback()
        log.exception("job %s crashed", job.id)
        jobs.mark_failed(
            db, job, "CORE_INTERNAL: unexpected error, see the worker log", claim_token
        )
    else:
        jobs.mark_done(db, job, claim_token)
        log.info("job %s done", job.id)
    finally:
        finished.set()
        heartbeat.join(timeout=1)
    return True


# Main loop: run jobs until the queue is empty, then sleep for worker_poll_seconds.
def main() -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    poll = get_settings().worker_poll_seconds
    log.info("worker started, polling every %ss", poll)
    while running:
        with SessionLocal() as db:
            try:
                had_job = run_one(db)
            except SQLAlchemyError:
                db.rollback()
                log.exception("could not poll the job queue")
                had_job = False
        if not had_job:
            time.sleep(poll)


if __name__ == "__main__":
    main()
