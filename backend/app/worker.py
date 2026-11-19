"""The background worker: takes jobs from core.jobs and runs them.

Start it with: python -m app.worker
Add a new job type by adding a function to HANDLERS.
"""

import logging
import signal
import time
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import jobs
from app.core.database import SessionLocal
from app.core.errors import AppError
from app.core.logging import setup_logging
from app.modules.knowledge import service as knowledge

log = logging.getLogger("mechai.worker")


def echo(db: Session, payload: dict[str, Any]) -> None:
    """A test job: it only writes its payload to the log."""
    log.info("echo job: %s", payload)


def fail(db: Session, payload: dict[str, Any]) -> None:
    """A test job that always fails, to see the retries."""
    raise AppError("CORE_TEST_FAILURE", "This test job always fails.")


HANDLERS: dict[str, Callable[[Session, dict[str, Any]], None]] = {
    "echo": echo,
    "fail": fail,
    "embed_text": knowledge.embed_text_job,
}

running = True


def stop(*_: object) -> None:
    global running
    running = False
    log.info("stopping after the current job")


def run_one(db: Session) -> bool:
    """Run one job. Returns False when the queue is empty."""
    job = jobs.claim_next(db)
    if job is None:
        return False
    log.info("job %s (%s) started, attempt %s", job.id, job.type, job.attempts)
    handler = HANDLERS.get(job.type)
    try:
        if handler is None:
            raise AppError("CORE_UNKNOWN_JOB", f"No handler for job type '{job.type}'.")
        handler(db, job.payload)
    except AppError as exc:
        db.rollback()
        log.warning("job %s failed: %s %s", job.id, exc.code, exc.message)
        jobs.mark_failed(db, job, f"{exc.code}: {exc.message}")
    except Exception:
        db.rollback()
        log.exception("job %s crashed", job.id)
        jobs.mark_failed(db, job, "CORE_INTERNAL: unexpected error, see the worker log")
    else:
        jobs.mark_done(db, job)
        log.info("job %s done", job.id)
    return True


def main() -> None:
    setup_logging()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    poll = get_settings().worker_poll_seconds
    log.info("worker started, polling every %ss", poll)
    while running:
        with SessionLocal() as db:
            had_job = run_one(db)
        if not had_job:
            time.sleep(poll)


if __name__ == "__main__":
    main()
