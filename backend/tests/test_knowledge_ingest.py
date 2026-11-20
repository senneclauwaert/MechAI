import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.jobs import enqueue, get_job
from app.modules.knowledge.models import Document, Page


def _document(file_hash: str | None = None) -> Document:
    return Document(title="Test manual", file_hash=file_hash)


def test_document_with_file_hash_saves(db: Session) -> None:
    db.add(_document("a" * 64))
    db.flush()


def test_same_file_hash_is_rejected(db: Session) -> None:
    db.add(_document("a" * 64))
    db.flush()
    db.add(_document("a" * 64))
    with pytest.raises(IntegrityError):
        db.flush()


def test_same_page_number_twice_is_rejected(db: Session) -> None:
    document = _document()
    db.add(document)
    db.flush()
    db.add(Page(document_id=document.id, page_number=1))
    db.add(Page(document_id=document.id, page_number=1))
    with pytest.raises(IntegrityError):
        db.flush()


def test_get_job_returns_job_or_none(db: Session) -> None:
    job = enqueue(db, "test")
    assert get_job(db, job.id) is not None
    assert get_job(db, uuid.uuid4()) is None
