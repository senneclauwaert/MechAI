"""Tests for POST /api/v1/assistant/ask: request limits, outcomes and the error shape.

The pipeline itself is tested in test_assistant_service.py; here only the HTTP layer is checked.
Fake clients only, no network. The handbook rows are written for these tests.
"""

import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.modules.assistant import service
from app.modules.assistant.models import Trace
from app.modules.knowledge import service as knowledge_service
from app.modules.knowledge.models import Chunk, Document
from tests.test_assistant_service import (
    ANSWER,
    QUESTION,
    TRANSLATION,
    ScriptedLLM,
    add_section,
    ok,
)

URL = "/api/v1/assistant/ask"


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dev environment, AI on, text may go out, a known threshold and cap."""
    s = get_settings()
    monkeypatch.setattr(s, "environment", "dev")
    monkeypatch.setattr(s, "ai_enabled", True)
    monkeypatch.setattr(s, "allow_external_embeddings", True)
    monkeypatch.setattr(s, "refusal_threshold", 0.7)
    monkeypatch.setattr(s, "daily_question_cap", 300)


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> ScriptedLLM:
    """The scripted client, used by both services."""
    ai = ScriptedLLM()
    monkeypatch.setattr(service, "get_llm", lambda: ai)
    monkeypatch.setattr(knowledge_service, "get_llm", lambda: ai)
    return ai


@pytest.fixture
def handbook(db: Session) -> Session:
    """One handbook with one rear axle section, and no other chunks."""
    db.execute(delete(Chunk))
    doc = Document(title="Test kart manual", source_type="handbook", revision="A")
    db.add(doc)
    db.flush()
    add_section(db, doc, "4.5")
    return db


def assert_error(response: Any, status: int, code: str) -> None:
    """The response is an error with the standard shape."""
    assert response.status_code == status
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]
    assert error["reference"]


# A normal question gives 200 with a cited answer and a stored trace (the question is stripped).
def test_ask_returns_an_answer_and_stores_a_trace(
    client: TestClient, handbook: Session, llm: ScriptedLLM
) -> None:
    llm.replies = [ok(TRANSLATION), ok(ANSWER)]

    response = client.post(URL, json={"question": f"  {QUESTION}  ", "kart_number": "07"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "answer"
    assert body["answer"]["steps"][0]["sources"] == ["S1"]
    assert body["sources"][0]["id"] == "S1"
    assert "text" not in body["sources"][0]
    trace = handbook.get(Trace, uuid.UUID(body["trace_id"]))
    assert trace is not None
    assert trace.question == QUESTION  # stripped
    assert trace.kart_number == "07"
    assert trace.input == "typed"


# An emergency is a normal 200 response: no answer, no sources, no AI call.
def test_ask_returns_200_for_an_emergency(
    client: TestClient, handbook: Session, llm: ScriptedLLM
) -> None:
    response = client.post(URL, json={"question": "There is smoke coming from the kart"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "emergency"
    assert body["answer"] is None and body["sources"] == []
    assert llm.chat_calls == []


@pytest.mark.parametrize(
    "body",
    [
        {"question": ""},
        {"question": "     "},
        {"question": " a "},
        {"question": "x" * 501},
        {},
        {"question": QUESTION, "kart_number": ""},
        {"question": QUESTION, "kart_number": "12345678901"},
        {"question": QUESTION, "kart_number": "07; drop"},
        {"question": QUESTION, "input": "video"},
    ],
)
# A bad body (empty, too short, too long, wrong kart number or input) gives the 422 error shape.
def test_ask_rejects_a_bad_request(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post(URL, json=body)

    assert_error(response, 422, "CORE_VALIDATION_FAILED")


# Outside dev the endpoint answers 403 and calls nothing.
def test_ask_is_not_allowed_in_prod(client: TestClient, llm: ScriptedLLM) -> None:
    get_settings().environment = "prod"

    response = client.post(URL, json={"question": QUESTION})

    assert_error(response, 403, "ASSISTANT_NOT_ALLOWED")
    assert llm.chat_calls == []


# At the daily cap a normal question gets 429 and no AI call.
def test_ask_stops_at_the_daily_cap(
    client: TestClient, handbook: Session, llm: ScriptedLLM
) -> None:
    get_settings().daily_question_cap = 1
    handbook.add(
        Trace(question="old", input="typed", safety_level="normal", outcome="answer", model="m")
    )
    handbook.flush()

    response = client.post(URL, json={"question": QUESTION})

    assert_error(response, 429, "ASSISTANT_DAILY_LIMIT")
    assert llm.chat_calls == []


# No embedded handbook text and a normal question gives 409 in the error shape.
def test_ask_gives_409_when_nothing_is_embedded(
    client: TestClient, db: Session, llm: ScriptedLLM
) -> None:
    db.execute(delete(Chunk))
    llm.replies = [ok(TRANSLATION)]

    response = client.post(URL, json={"question": f"{QUESTION} {uuid.uuid4().hex}"})

    assert_error(response, 409, "KNOWLEDGE_NOTHING_EMBEDDED")


# The route is part of the OpenAPI schema.
def test_the_endpoint_is_in_the_openapi_schema(client: TestClient) -> None:
    paths = client.get("/api/openapi.json").json()["paths"]

    assert "post" in paths[URL]


# A high-voltage question at the daily cap is not an error: 200 with pages and the cap as reason.
def test_ask_shows_pages_for_a_high_voltage_question_at_the_cap(
    client: TestClient, handbook: Session, llm: ScriptedLLM
) -> None:
    get_settings().daily_question_cap = 1
    handbook.add(
        Trace(question="old", input="typed", safety_level="normal", outcome="answer", model="m")
    )
    handbook.flush()

    response = client.post(URL, json={"question": "Replace the battery near the axle"})

    assert response.status_code == 200
    body = response.json()
    assert body["outcome"] == "high_voltage"
    assert body["reason_code"] == "ASSISTANT_DAILY_LIMIT"
    assert len(body["sources"]) == 1
    assert llm.chat_calls == [] and llm.embed_calls == 0
