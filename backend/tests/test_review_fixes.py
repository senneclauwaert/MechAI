from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import Settings
from app.core import jobs
from app.core.errors import AppError
from app.core.models import Job
from app.llm import FakeLLM, MistralLLM
from app.modules.knowledge.schemas import PlaygroundTextIn


def test_settings_and_empty_title() -> None:
    assert Settings(_env_file=None).embed_dimensions == 1024
    for dimensions in (512, 1025):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, embed_dimensions=dimensions)
    with pytest.raises(ValidationError):
        PlaygroundTextIn(title="", text="text")


@pytest.mark.parametrize("vectors", [[], [None], [[1.0], [2.0]]])
def test_invalid_embeddings(vectors) -> None:
    llm = object.__new__(MistralLLM)
    llm.settings = Settings(_env_file=None)
    llm.client = Mock()
    llm.client.embeddings.create.return_value = SimpleNamespace(
        data=[SimpleNamespace(embedding=v) for v in vectors]
    )
    with pytest.raises(AppError) as error:
        llm.embed(["one"])
    assert error.value.code == "KNOWLEDGE_EMBEDDING_FAILED"
    assert error.value.status == 503
    assert FakeLLM().embedding_model != llm.embedding_model


def test_health_alias_and_http_errors(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}
    for path, method, status in [("/missing", "get", 404), ("/health", "post", 405)]:
        response = getattr(client, method)(path)
        assert response.status_code == status
        assert response.json()["error"]["reference"].startswith("req_")
        if status == 405:
            assert "GET" in response.headers["allow"]


def test_expired_claim_cap_and_ownership(db: Session) -> None:
    db.execute(delete(Job))
    old = datetime.now(UTC) - timedelta(minutes=11)
    capped = Job(type="echo", status="running", attempts=3, locked_until=old)
    retry = Job(type="echo", status="running", attempts=1, locked_until=old)
    db.add_all([capped, retry])
    db.commit()
    claimed = jobs.claim_next(db)
    assert claimed.id == retry.id
    assert claimed.attempts == 2
    db.refresh(capped)
    assert capped.status == "failed"
    assert capped.locked_until is None
    lease = claimed.locked_until
    jobs.mark_done(db, claimed, 1)
    jobs.mark_failed(db, claimed, "stale", 1)
    assert not jobs.renew_lease(db, claimed, 1)
    db.refresh(claimed)
    assert claimed.status == "running"
    assert claimed.last_error is None
    assert jobs.renew_lease(db, claimed, 2)
    db.refresh(claimed)
    assert claimed.locked_until >= lease
    jobs.mark_done(db, claimed, 2)
    db.refresh(claimed)
    assert claimed.status == "done"


def test_poll_recovers_database_failure(monkeypatch) -> None:
    from sqlalchemy.exc import SQLAlchemyError

    from app import worker

    db = Mock()
    context = Mock()
    context.__enter__ = Mock(return_value=db)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(worker, "SessionLocal", Mock(return_value=context))
    monkeypatch.setattr(worker, "run_one", Mock(side_effect=SQLAlchemyError("offline")))
    monkeypatch.setattr(worker.signal, "signal", Mock())
    monkeypatch.setattr(worker, "running", True)
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        worker.running = False

    monkeypatch.setattr(worker.time, "sleep", sleep)
    worker.main()
    db.rollback.assert_called_once()
    assert sleeps == [worker.get_settings().worker_poll_seconds]


def test_heartbeat_renews_while_work_continues(monkeypatch) -> None:
    from app import worker

    job = Mock()
    db = Mock()
    db.get.return_value = job
    context = Mock()
    context.__enter__ = Mock(return_value=db)
    context.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(worker, "SessionLocal", Mock(return_value=context))
    renew = Mock(return_value=True)
    monkeypatch.setattr(jobs, "renew_lease", renew)
    finished = Mock()
    finished.wait.side_effect = [False, True]
    worker._heartbeat("job-id", 2, finished)
    renew.assert_called_once_with(db, job, 2)
    assert finished.wait.call_args.args == (200.0,)


def test_embedding_reuse_and_search_isolate_models(db: Session) -> None:
    from app.modules.knowledge import service
    from app.modules.knowledge.models import Chunk

    db.execute(delete(Chunk))
    original, _ = service.add_playground_text(db, "Axle", "replace axle bearing")
    original.embedding = [1.0] * 1024
    original.embedding_model = "mistral-embed"
    duplicate, _ = service.add_playground_text(db, "Axle", "replace axle bearing")
    service.embed_text_job(db, {"chunk_id": str(duplicate.id)})
    db.refresh(duplicate)
    assert duplicate.embedding_model == FakeLLM().embedding_model
    assert list(duplicate.embedding) != list(original.embedding)
    hits = service.search(db, "axle bearing")
    by_id = {hit["chunk_id"]: hit for hit in hits}
    assert by_id[str(duplicate.id)]["vector_rank"] == 1
    assert by_id[str(original.id)]["vector_rank"] is None
