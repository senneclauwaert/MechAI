"""Tables of the knowledge schema (vault: Data model)."""

import uuid
from datetime import date

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, Date, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

EMBED_DIMENSIONS = 1024  # mistral-embed


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = {"schema": "knowledge"}

    title: Mapped[str] = mapped_column(String(300))
    kart_model: Mapped[str | None] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(5), default="en")
    source_type: Mapped[str] = mapped_column(String(20), default="handbook")
    revision: Mapped[str | None] = mapped_column(String(50))
    published_on: Mapped[date | None] = mapped_column(Date)
    superseded_by: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("knowledge.documents.id"))
    file_path: Mapped[str | None] = mapped_column(String(500))
    file_hash: Mapped[str | None] = mapped_column(String(64), unique=True)
    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("core.jobs.id", ondelete="SET NULL")
    )


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="pages_document_page"),
        {"schema": "knowledge"},
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge.documents.id", ondelete="CASCADE"), index=True
    )
    page_number: Mapped[int] = mapped_column(Integer)
    image_path: Mapped[str | None] = mapped_column(String(500))
    text: Mapped[str] = mapped_column(Text, default="")
    kind: Mapped[str] = mapped_column(String(20), default="text")


class Section(Base):
    __tablename__ = "sections"
    __table_args__ = {"schema": "knowledge"}

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge.documents.id", ondelete="CASCADE"), index=True
    )
    section_number: Mapped[str | None] = mapped_column(String(20))
    heading_path: Mapped[str] = mapped_column(String(500), default="")
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, default="")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    safety_level: Mapped[str] = mapped_column(String(20), default="normal")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        Index(
            "chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("chunks_tsv", "tsv", postgresql_using="gin"),
        Index("chunks_hash", "content_hash", "embedding_model"),
        {"schema": "knowledge"},
    )

    section_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("knowledge.sections.id", ondelete="CASCADE"), index=True
    )
    text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBED_DIMENSIONS))
    embedding_model: Mapped[str | None] = mapped_column(String(100))
    # English full-text search column, filled by PostgreSQL itself
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )
