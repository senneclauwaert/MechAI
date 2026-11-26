"""Tests for the assistant.traces table: a full and a minimal Trace round-trip, with defaults."""

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.modules.assistant.models import Trace


# Every column of a trace survives a save and a read.
def test_full_trace_is_saved_and_read_back(db: Session) -> None:
    trace = Trace(
        question="How do I check the tyre pressure?",
        kart_number="07",
        input="text",
        question_language="en",
        safety_level="normal",
        outcome="answered",
        search_terms={"en": "tyre pressure"},
        retrieved=[{"chunk_id": "c1", "vector_rank": 1, "text_rank": 2, "similarity": 0.8}],
        sections_sent=["s1"],
        best_similarity=0.8,
        prompt_versions={"answer": "answer_v1"},
        model="test-model",
        document_revisions=[{"document_id": "d1", "revision": "A"}],
        tokens_in=120,
        tokens_out=40,
        latency_ms={"search": 12, "answer": 300, "total": 330},
        answer={"steps": ["Check it."]},
        flags=["number_dropped"],
        error_code="ASSISTANT_TEST",
        request_id="req123",
    )
    db.add(trace)
    db.flush()
    db.expire_all()

    saved = db.get(Trace, trace.id)
    assert saved is not None
    assert saved.question == "How do I check the tyre pressure?"
    assert saved.retrieved[0]["chunk_id"] == "c1"
    assert saved.sections_sent == ["s1"]
    assert saved.answer == {"steps": ["Check it."]}
    assert saved.tokens_in == 120
    assert saved.flags == ["number_dropped"]
    assert saved.created_at is not None


# A trace with only the required columns gets defaults, and None is SQL NULL.
def test_minimal_trace_gets_defaults_and_sql_null(db: Session) -> None:
    trace = Trace(
        question="Where is the fuse box?", input="voice", safety_level="normal", outcome="refused"
    )
    db.add(trace)
    db.flush()
    db.expire_all()

    saved = db.get(Trace, trace.id)
    assert saved is not None
    assert saved.retrieved == []
    assert saved.sections_sent == []
    assert saved.document_revisions == []
    assert saved.flags == []
    assert saved.prompt_versions == {}
    assert saved.latency_ms == {}
    assert saved.tokens_in == 0
    assert saved.tokens_out == 0
    assert saved.model is None
    assert saved.answer is None
    assert saved.search_terms is None

    # None must be SQL NULL, not the JSON value null
    row = db.execute(
        text("SELECT answer IS NULL, search_terms IS NULL FROM assistant.traces WHERE id = :id"),
        {"id": trace.id},
    ).one()
    assert tuple(row) == (True, True)
