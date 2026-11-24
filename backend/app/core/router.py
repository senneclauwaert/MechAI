"""Core endpoints: health check and (in dev) a way to put a test job in the queue."""

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import text

from app.config import get_settings
from app.core import jobs
from app.core.database import DB
from app.core.errors import AppError, not_found
from app.core.models import Job

router = APIRouter(tags=["core"])


# Health check: also checks that the database answers.
@router.get("/health")
def health(db: DB) -> dict[str, str]:
    try:
        db.execute(text("SELECT 1"))
    except Exception as exc:
        raise AppError("CORE_DATABASE_DOWN", "The database cannot be reached.", 503) from exc
    return {"status": "ok", "database": "ok"}


# Dev-only job endpoints: put a job in the queue and follow its status.
class JobIn(BaseModel):
    type: str
    payload: dict[str, Any] = {}


# What the dev job endpoints return about a job.
class JobOut(BaseModel):
    id: str
    type: str
    status: str
    attempts: int
    last_error: str | None

    @classmethod
    def of(cls, job: Job) -> "JobOut":
        return cls(
            id=str(job.id),
            type=job.type,
            status=job.status,
            attempts=job.attempts,
            last_error=job.last_error,
        )


# Dependency that hides an endpoint outside dev (answers 404 as if it does not exist).
def _dev_only() -> None:
    if get_settings().environment != "dev":
        raise not_found("Page")


@router.post("/dev/jobs", dependencies=[Depends(_dev_only)])
def add_test_job(body: JobIn, db: DB) -> JobOut:
    """Dev only: put a job in the queue, to play with the worker."""
    return JobOut.of(jobs.enqueue(db, body.type, body.payload))


# Dev only: read the status, attempts and last error of a job.
@router.get("/dev/jobs/{job_id}", dependencies=[Depends(_dev_only)])
def get_test_job(job_id: str, db: DB) -> JobOut:
    job = db.get(Job, job_id)
    if job is None:
        raise not_found("Job")
    return JobOut.of(job)
