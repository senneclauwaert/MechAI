"""Endpoints under /api/v1/assistant. Thin: they only call service.py."""

from fastapi import APIRouter

from app.core.database import DB
from app.modules.assistant import service
from app.modules.assistant.schemas import AskIn, AskOut

router = APIRouter(tags=["assistant"])


# Ask one question. 200 for every outcome; the body says what happened (answer, pages only, ...).
@router.post("/ask")
def ask(body: AskIn, db: DB) -> AskOut:
    """Answer a question from the handbook, or explain why not (dev only until #26)."""
    return service.ask(db, body.question, body.kart_number, body.input)
