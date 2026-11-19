"""Endpoints under /api/v1/knowledge. Thin: they only call service.py."""

from typing import Annotated

from fastapi import APIRouter, Query

from app.core.database import DB
from app.modules.knowledge import service
from app.modules.knowledge.schemas import PlaygroundTextIn, PlaygroundTextOut, SearchHit, SearchOut

router = APIRouter(tags=["knowledge"])


@router.post("/playground", status_code=201)
def add_text(body: PlaygroundTextIn, db: DB) -> PlaygroundTextOut:
    """Add a test text. The worker embeds it in the background."""
    chunk, job_id = service.add_playground_text(db, body.title, body.text)
    return PlaygroundTextOut(chunk_id=str(chunk.id), job_id=str(job_id))


@router.get("/search")
def search(db: DB, q: Annotated[str, Query(min_length=2, max_length=500)]) -> SearchOut:
    hits = service.search(db, q)
    return SearchOut(question=q, hits=[SearchHit(**h) for h in hits])
