"""Core: health check, error format and the job queue with the worker."""

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core import jobs
from app.worker import run_one


# API: health and the shared error format
def test_health(client: TestClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_error_shape(client: TestClient) -> None:
    response = client.get("/api/v1/dev/jobs/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "CORE_NOT_FOUND"
    assert error["reference"].startswith("req_")


def test_validation_error_shape(client: TestClient) -> None:
    response = client.post("/api/v1/dev/jobs", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CORE_VALIDATION_FAILED"


# Job queue and worker
def test_worker_runs_a_job(db: Session) -> None:
    job = jobs.enqueue(db, "echo", {"msg": "hi"})
    assert run_one(db)
    db.refresh(job)
    assert job.status == "done"


def test_failed_job_is_retried_then_failed(db: Session) -> None:
    job = jobs.enqueue(db, "fail")
    run_one(db)
    db.refresh(job)
    assert job.status == "queued"  # will be retried later
    assert job.last_error is not None and job.last_error.startswith("CORE_TEST_FAILURE")
    job.attempts = 3
    job.status = "running"
    from datetime import UTC, datetime

    job.locked_until = datetime.now(UTC) + jobs.LOCK_TIME
    db.flush()
    jobs.mark_failed(db, job, "CORE_TEST_FAILURE: again", job.attempts)
    assert job.status == "failed"
