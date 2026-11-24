"""Endpoints under /api/v1/knowledge. Thin: they only call service.py."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Form, Query, Response, UploadFile

from app.config import get_settings
from app.core.database import DB
from app.modules.knowledge import service
from app.modules.knowledge.schemas import (
    DocumentOut,
    DocumentStatusOut,
    PlaygroundTextIn,
    PlaygroundTextOut,
    SearchHit,
    SearchOut,
)

router = APIRouter(tags=["knowledge"])


# Playground: try embeddings and search with your own text.
@router.post("/playground", status_code=201)
def add_text(body: PlaygroundTextIn, db: DB) -> PlaygroundTextOut:
    """Add a test text. The worker embeds it in the background."""
    chunk, job_id = service.add_playground_text(db, body.title, body.text)
    return PlaygroundTextOut(chunk_id=str(chunk.id), job_id=str(job_id))


# Hybrid search over all embedded chunks, best matches first.
@router.get("/search")
def search(db: DB, q: Annotated[str, Query(min_length=2, max_length=500)]) -> SearchOut:
    hits = service.search(db, q)
    return SearchOut(question=q, hits=[SearchHit(**h) for h in hits])


# Handbook upload. 201 for a new document, 200 when the same file was uploaded before.
@router.post("/documents", status_code=201)
def upload_document(
    file: UploadFile,
    title: Annotated[str, Form(min_length=1, max_length=300)],
    db: DB,
    response: Response,
    revision: Annotated[str | None, Form(max_length=50)] = None,
    kart_model: Annotated[str | None, Form(max_length=100)] = None,
) -> DocumentOut:
    """Upload a handbook PDF. The worker reads it in the background (dev only until #26)."""
    # One byte more than the limit is enough for the service to see the file is too large
    data = file.file.read(get_settings().max_pdf_mb * 1024 * 1024 + 1)
    document, job, already_ingested = service.upload_document(db, data, title, revision, kart_model)
    if already_ingested:
        response.status_code = 200
    return DocumentOut(
        document_id=str(document.id),
        job_id=str(job.id),
        title=document.title,
        revision=document.revision,
        already_ingested=already_ingested,
    )


# Ingestion progress of one document, for polling after an upload.
@router.get("/documents/{document_id}/status")
def document_status(document_id: uuid.UUID, db: DB) -> DocumentStatusOut:
    return DocumentStatusOut(**service.document_status(db, document_id))
