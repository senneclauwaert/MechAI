"""The logic of the knowledge module. Other modules call these functions, never the tables.

For now this is a playground to try RAG: add a text, the worker embeds it, then search.
The real ingestion (PDF -> pages -> sections -> chunks) comes in its own issue.
"""

import hashlib
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core import jobs
from app.llm import get_llm
from app.modules.knowledge import errors
from app.modules.knowledge.models import Chunk, Document, Section

PLAYGROUND_TITLE = "Playground"
RRF_K = 60  # standard value for Reciprocal Rank Fusion


def _playground(db: Session) -> Document:
    doc = db.scalars(select(Document).where(Document.title == PLAYGROUND_TITLE)).first()
    if doc is None:
        doc = Document(title=PLAYGROUND_TITLE, source_type="procedure")
        db.add(doc)
        db.flush()
    return doc


def add_playground_text(db: Session, title: str, text: str) -> tuple[Chunk, uuid.UUID]:
    """Save a text as one section with one chunk, and ask the worker to embed it."""
    doc = _playground(db)
    section = Section(document_id=doc.id, heading_path=title, text=text)
    db.add(section)
    db.flush()
    chunk_text = f"{PLAYGROUND_TITLE} > {title}\n{text}"  # context header + text
    chunk = Chunk(
        section_id=section.id,
        text=chunk_text,
        content_hash=hashlib.sha256(chunk_text.encode()).hexdigest(),
        token_count=len(chunk_text.split()),
    )
    db.add(chunk)
    db.commit()
    job = jobs.enqueue(db, "embed_text", {"chunk_id": str(chunk.id)})
    return chunk, job.id


def embed_text_job(db: Session, payload: dict[str, Any]) -> None:
    """Worker job: embed one chunk (reuses an earlier embedding of the same text)."""
    chunk = db.get(Chunk, uuid.UUID(payload["chunk_id"]))
    if chunk is None:
        raise errors.chunk_not_found()
    llm = get_llm()
    model = llm.embedding_model
    same = db.scalars(
        select(Chunk).where(
            Chunk.content_hash == chunk.content_hash,
            Chunk.embedding_model == model,
            Chunk.embedding.is_not(None),
        )
    ).first()
    chunk.embedding = same.embedding if same else llm.embed([chunk.text])[0]
    chunk.embedding_model = model
    db.commit()


def search(db: Session, question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Hybrid search: vector top 20 + full-text top 20, merged with RRF (vault: RAG pipeline)."""
    llm = get_llm()
    active = (Chunk.embedding_model == llm.embedding_model) & Chunk.embedding.is_not(None)
    has_embeddings = db.scalar(select(func.count()).where(active))
    if not has_embeddings:
        raise errors.nothing_embedded()

    query_vector = llm.embed([question])[0]
    by_vector = db.scalars(
        select(Chunk)
        .where(active)
        .order_by(Chunk.embedding.cosine_distance(query_vector))
        .limit(20)
    ).all()

    ts_query = func.websearch_to_tsquery("english", question)
    by_text = db.scalars(
        select(Chunk)
        .where(Chunk.tsv.op("@@")(ts_query))
        .order_by(func.ts_rank(Chunk.tsv, ts_query).desc())
        .limit(20)
    ).all()

    vector_rank = {c.id: i + 1 for i, c in enumerate(by_vector)}
    text_rank = {c.id: i + 1 for i, c in enumerate(by_text)}
    chunks = {c.id: c for c in [*by_vector, *by_text]}

    def rrf(chunk_id: uuid.UUID) -> float:
        score = 0.0
        for ranks in (vector_rank, text_rank):
            if chunk_id in ranks:
                score += 1 / (RRF_K + ranks[chunk_id])
        return score

    best = sorted(chunks, key=rrf, reverse=True)[:limit]
    return [
        {
            "chunk_id": str(cid),
            "heading": chunks[cid].text.split("\n", 1)[0],
            "text": chunks[cid].text,
            "vector_rank": vector_rank.get(cid),
            "text_rank": text_rank.get(cid),
            "score": round(rrf(cid), 5),
        }
        for cid in best
    ]
