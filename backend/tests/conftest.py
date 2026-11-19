"""Tests run against the database from .env (start it first: docker compose ... up -d db).

Each test runs inside a transaction that is rolled back, so the database stays clean.
Later (CI issue) this can move to a throwaway database with testcontainers.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.database import engine, get_db
from app.main import app


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never call Mistral: without a key, get_llm() returns the fake client."""
    monkeypatch.setattr(get_settings(), "mistral_api_key", "")


@pytest.fixture
def db() -> Iterator[Session]:
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture
def client(db: Session) -> Iterator[TestClient]:
    app.dependency_overrides[get_db] = lambda: db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()
