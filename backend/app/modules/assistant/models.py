"""Tables of the assistant schema (vault: Data model). For now only the trace of each question."""

from typing import Any

from sqlalchemy import Float, Index, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


# One row per question asked: what was searched, what was sent to the model, what came back
# and why (outcome, flags). No handbook text is copied here, only ids and scores.
# A null JSONB column is stored as SQL NULL (none_as_null), never as the JSON value null.
class Trace(Base):
    __tablename__ = "traces"
    __table_args__ = (
        Index("traces_created_at", "created_at"),
        {"schema": "assistant"},
    )

    question: Mapped[str] = mapped_column(String(500))
    kart_number: Mapped[str | None] = mapped_column(String(10))
    input: Mapped[str] = mapped_column(String(10))
    question_language: Mapped[str | None] = mapped_column(String(5))
    safety_level: Mapped[str] = mapped_column(String(20))
    outcome: Mapped[str] = mapped_column(String(20))
    search_terms: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))
    # list of {chunk_id, vector_rank, text_rank, similarity, score}
    retrieved: Mapped[list[Any]] = mapped_column(JSONB(none_as_null=True), default=list)
    sections_sent: Mapped[list[Any]] = mapped_column(JSONB(none_as_null=True), default=list)
    best_similarity: Mapped[float | None] = mapped_column(Float)
    prompt_versions: Mapped[dict[str, Any]] = mapped_column(JSONB(none_as_null=True), default=dict)
    # null model means no AI call was made
    model: Mapped[str | None] = mapped_column(String(100))
    document_revisions: Mapped[list[Any]] = mapped_column(JSONB(none_as_null=True), default=list)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    # milliseconds per step: translate, search, answer, total
    latency_ms: Mapped[dict[str, Any]] = mapped_column(JSONB(none_as_null=True), default=dict)
    # the parsed model answer, also when the checks dropped it
    answer: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))
    # what the model wrote, only when it could not be parsed or had no steps: list of {call, text}
    raw_replies: Mapped[list[Any]] = mapped_column(
        JSONB(none_as_null=True), default=list, server_default="[]"
    )
    flags: Mapped[list[Any]] = mapped_column(JSONB(none_as_null=True), default=list)
    error_code: Mapped[str | None] = mapped_column(String(100))
    request_id: Mapped[str | None] = mapped_column(String(20))
