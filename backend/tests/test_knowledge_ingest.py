import hashlib
import uuid
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.errors import AppError
from app.core.jobs import enqueue, get_job
from app.llm import FakeLLM
from app.modules.knowledge import errors, service
from app.modules.knowledge.models import Chunk, Document, Page, Section
from app.modules.knowledge.service import ParsedItem, SectionDraft
from app.modules.storage import service as storage
from app.worker import run_one


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


def _items() -> list[ParsedItem]:
    return [
        ParsedItem("text", "Intro text.", 1),
        ParsedItem("heading", "5 Rear axle", 38),
        ParsedItem("text", "Axle text.", 38),
        ParsedItem("heading", "5.3 Rear mechanical section", 40),
        ParsedItem("text", "Mechanical text.", 40),
        ParsedItem("heading", "Warning", 41),
        ParsedItem("heading", "5.3.1 Bearings", 42),
        ParsedItem("table", "| a | b |\n| 1 | 2 |", 42),
    ]


def test_heading_paths() -> None:
    sections = service.build_sections(_items(), "Manual")
    assert [s.heading_path for s in sections] == [
        "Manual",
        "5 Rear axle",
        "5 Rear axle > 5.3 Rear mechanical section",
        "5 Rear axle > 5.3 Rear mechanical section > 5.3.1 Bearings",
    ]


def test_text_before_first_heading_goes_to_title_section() -> None:
    first = service.build_sections(_items(), "Manual")[0]
    assert first.number is None
    assert first.text == "Intro text."


def test_page_range_and_unnumbered_heading_stays_text() -> None:
    mechanical = service.build_sections(_items(), "Manual")[2]
    assert mechanical.text == "Mechanical text.\n\nWarning"
    axle_to_bearings = service.build_sections(_items()[1:], "Manual")
    assert axle_to_bearings[0].page_start == 38
    assert axle_to_bearings[-1].page_end == 42


def test_section_without_text_is_dropped() -> None:
    items = [ParsedItem("heading", "1 Empty", 1), ParsedItem("heading", "2 Full", 2)]
    items.append(ParsedItem("text", "Some text", 2))
    assert [s.heading_path for s in service.build_sections(items, "Manual")] == ["2 Full"]


def test_long_section_is_split_into_parts() -> None:
    section = SectionDraft("1", "1 Big", 3, 4, "\n\n".join(["word " * 30] * 4))
    parts = service.split_long_sections([section], max_tokens=100)
    assert [p.heading_path for p in parts] == ["1 Big", "1 Big (part 2)"]
    assert all(p.page_start == 3 and p.page_end == 4 for p in parts)


def test_table_is_never_split() -> None:
    table = "| a | b |\n" + "| word | word |\n" * 100
    section = SectionDraft("1", "1 Table", 1, 1, table)
    assert len(service.split_long_sections([section], max_tokens=50)) == 1
    assert service.make_chunks(table, chunk_tokens=50, overlap_tokens=10) == [table.strip()]


def test_chunks_stay_within_limit_and_overlap() -> None:
    paragraphs = [" ".join(f"p{i}w{n}" for n in range(60)) for i in range(4)]  # 80 tokens each
    chunks = service.make_chunks("\n\n".join(paragraphs), chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) == 4
    assert all(service.approx_tokens(c) <= 100 for c in chunks)
    for before, after in zip(chunks, chunks[1:], strict=False):
        assert before.split()[-15:] == after.split()[:15]  # the boundary words are shared


def test_short_last_paragraph_is_repeated_whole() -> None:
    text = "\n\n".join(["a " * 70, "short end", "b " * 70])
    chunks = service.make_chunks(text, chunk_tokens=100, overlap_tokens=20)
    assert chunks[1].startswith("short end")


def test_very_long_paragraph_is_cut_by_words_with_overlap() -> None:
    paragraph = " ".join(f"w{n}" for n in range(300))
    chunks = service.make_chunks(paragraph, chunk_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(service.approx_tokens(c) <= 100 for c in chunks)
    assert chunks[0].split()[-1] in chunks[1].split()


def test_safety_level_uses_whole_words() -> None:
    assert service.safety_level_for("Disconnect the battery") == "high_voltage"
    assert service.safety_level_for("Check the High-Voltage cable") == "high_voltage"
    assert service.safety_level_for("Rear axle bearing") == "normal"
    assert service.safety_level_for("batteryless") == "normal"


def test_context_header() -> None:
    one = SectionDraft("1", "1 Axle", 5, 5, "x")
    many = SectionDraft("1", "1 Axle", 5, 7, "x")
    assert service.context_header("Manual", one) == "Manual > 1 Axle (p. 5)"
    assert service.context_header("Manual", many) == "Manual > 1 Axle (p. 5-7)"


def _chunk(text: str, section_id: uuid.UUID | None = None) -> Chunk:
    return Chunk(
        section_id=section_id,
        text=text,
        content_hash=service.content_hash(text),
        token_count=1,
    )


def _spy_embed(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    sizes: list[int] = []
    original = FakeLLM.embed

    def spy(self: FakeLLM, texts: list[str]) -> list[list[float]]:
        sizes.append(len(texts))
        return original(self, texts)

    monkeypatch.setattr(FakeLLM, "embed", spy)
    return sizes


def test_embed_chunks_sends_same_text_once(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _spy_embed(monkeypatch)
    chunks = [_chunk("same text"), _chunk("same text")]
    assert service.embed_chunks(db, chunks) == 1
    assert sizes == [1]
    assert chunks[0].embedding is not None and chunks[1].embedding is not None
    assert chunks[0].embedding_model == FakeLLM.embedding_model


def test_embed_chunks_reuses_existing_embedding(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    document = _document()
    db.add(document)
    db.flush()
    section = Section(document_id=document.id, heading_path="x", text="x")
    db.add(section)
    db.flush()
    old = _chunk("known text", section.id)
    db.add(old)
    db.flush()
    service.embed_chunks(db, [old])
    sizes = _spy_embed(monkeypatch)
    assert service.embed_chunks(db, [_chunk("known text")]) == 0
    assert sizes == []


def test_embed_chunks_works_in_batches_of_64(db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    sizes = _spy_embed(monkeypatch)
    chunks = [_chunk(f"text number {n}") for n in range(130)]
    assert service.embed_chunks(db, chunks) == 130
    assert sizes == [64, 64, 2]


# --- the ingest job (Docling and pypdfium2 are replaced by fakes) ---


def _fake_parse(data: bytes) -> tuple[list[ParsedItem], dict[int, str]]:
    items = [
        ParsedItem("heading", "5 Rear axle", 1),
        ParsedItem("text", "Replace the axle bearing.", 1),
        ParsedItem("heading", "5.3 Battery", 2),
        ParsedItem("text", "Disconnect the battery first.", 2),
        ParsedItem("table", "| part | torque |\n| bolt | 25 Nm |", 3),
    ]
    return items, {1: "Replace the axle bearing.", 2: "Disconnect the battery first.", 3: ""}


@pytest.fixture
def ingest_document(db: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Document:
    monkeypatch.setattr(get_settings(), "files_dir", str(tmp_path))
    monkeypatch.setattr(service, "parse_pdf", _fake_parse)
    monkeypatch.setattr(service, "render_page_png", lambda data, number, dpi: b"png%d" % number)
    path = storage.save_file("knowledge/source", "manual.bin", b"fake file")
    document = Document(title="Test manual", file_path=path)
    db.add(document)
    db.flush()
    return document


def _counts(db: Session, document: Document) -> tuple[int, int, int]:
    pages = db.scalar(select(func.count()).where(Page.document_id == document.id))
    sections = db.scalar(select(func.count()).where(Section.document_id == document.id))
    chunks = db.scalar(
        select(func.count())
        .select_from(Chunk)
        .join(Section, Chunk.section_id == Section.id)
        .where(Section.document_id == document.id)
    )
    return pages or 0, sections or 0, chunks or 0


def test_ingest_job_saves_pages_sections_and_chunks(
    db: Session, ingest_document: Document, tmp_path: Path
) -> None:
    service.ingest_document_job(db, {"document_id": str(ingest_document.id)})

    assert _counts(db, ingest_document) == (3, 2, 2)
    for number in (1, 2, 3):
        png = tmp_path / "knowledge" / str(ingest_document.id) / f"page-{number:03}.png"
        assert png.read_bytes() == b"png%d" % number
    page = db.scalars(select(Page).where(Page.page_number == 1)).one()
    assert page.image_path == f"knowledge/{ingest_document.id}/page-001.png"
    assert page.text == "Replace the axle bearing."

    sections = db.scalars(select(Section).where(Section.document_id == ingest_document.id)).all()
    levels = {section.section_number: section.safety_level for section in sections}
    assert levels == {"5": "normal", "5.3": "high_voltage"}

    chunks = db.scalars(
        select(Chunk).join(Section).where(Section.document_id == ingest_document.id)
    )
    for chunk in chunks:
        assert chunk.text.startswith("Test manual > 5 Rear axle")
        assert chunk.embedding is not None


def test_ingest_job_retry_is_clean_and_embeds_nothing(
    db: Session, ingest_document: Document, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = {"document_id": str(ingest_document.id)}
    service.ingest_document_job(db, payload)
    first = _counts(db, ingest_document)

    sizes = _spy_embed(monkeypatch)
    service.ingest_document_job(db, payload)
    assert _counts(db, ingest_document) == first
    assert sum(sizes) == 0


def test_ingest_job_parse_failure_is_stored_in_last_error(
    db: Session, ingest_document: Document, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(data: bytes) -> tuple[list[ParsedItem], dict[int, str]]:
        raise errors.parse_failed()

    monkeypatch.setattr(service, "parse_pdf", broken)
    job = enqueue(db, "ingest_document", {"document_id": str(ingest_document.id)})
    assert run_one(db)
    db.refresh(job)
    assert job.last_error is not None
    assert job.last_error.startswith("KNOWLEDGE_PARSE_FAILED")


def test_ingest_job_missing_document(db: Session) -> None:
    with pytest.raises(AppError) as caught:
        service.ingest_document_job(db, {"document_id": str(uuid.uuid4())})
    assert caught.value.code == "KNOWLEDGE_DOCUMENT_NOT_FOUND"


# --- upload and status endpoints ---

FAKE_PDF = b"%PDF-1.4 fake"


@pytest.fixture
def upload_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(get_settings(), "files_dir", str(tmp_path))
    monkeypatch.setattr(service, "parse_pdf", _fake_parse)
    monkeypatch.setattr(service, "render_page_png", lambda data, number, dpi: b"png%d" % number)
    return tmp_path


def _upload(client: TestClient, data: bytes = FAKE_PDF, **fields: str) -> Response:
    form = {"title": "Test manual", **fields}
    return client.post(
        "/api/v1/knowledge/documents",
        data=form,
        files={"file": ("whatever name.pdf", data, "application/pdf")},
    )


def _status(client: TestClient, document_id: str) -> dict[str, Any]:
    return client.get(f"/api/v1/knowledge/documents/{document_id}/status").json()


def test_upload_creates_document_file_and_queued_job(
    client: TestClient, db: Session, upload_env: Path
) -> None:
    response = _upload(client, revision="rev 2")
    assert response.status_code == 201
    body = response.json()
    assert body["already_ingested"] is False
    assert body["revision"] == "rev 2"

    saved = upload_env / "knowledge" / body["document_id"] / "source.pdf"
    assert saved.read_bytes() == FAKE_PDF
    document = db.get(Document, uuid.UUID(body["document_id"]))
    assert document is not None
    assert document.file_path == f"knowledge/{body['document_id']}/source.pdf"
    assert document.source_type == "handbook"
    assert document.language == "en"
    job = get_job(db, uuid.UUID(body["job_id"]))
    assert job is not None
    assert (job.type, job.status) == ("ingest_document", "queued")
    assert job.payload == {"document_id": body["document_id"]}
    assert document.job_id == job.id


def test_same_file_again_returns_existing_document_without_new_job(
    client: TestClient, upload_env: Path
) -> None:
    first = _upload(client).json()
    second = _upload(client)
    assert second.status_code == 200
    assert second.json()["already_ingested"] is True
    assert second.json()["document_id"] == first["document_id"]
    assert second.json()["job_id"] == first["job_id"]


def test_same_file_after_failed_job_queues_a_new_job(
    client: TestClient, db: Session, upload_env: Path
) -> None:
    first = _upload(client).json()
    job = get_job(db, uuid.UUID(first["job_id"]))
    assert job is not None
    job.status = "failed"
    db.commit()

    second = _upload(client)
    assert second.status_code == 200
    assert second.json()["document_id"] == first["document_id"]
    assert second.json()["job_id"] != first["job_id"]
    new_job = get_job(db, uuid.UUID(second.json()["job_id"]))
    assert new_job is not None and new_job.status == "queued"


def test_upload_that_loses_a_race_returns_the_winner(
    client: TestClient, db: Session, upload_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_hash = hashlib.sha256(FAKE_PDF).hexdigest()
    winner = Document(title="Winner", file_hash=file_hash)
    db.add(winner)
    db.commit()

    # The first lookup finds nothing, as if the other upload committed just after it
    real_lookup = service._find_by_hash
    lookups = iter([None])
    monkeypatch.setattr(
        service, "_find_by_hash", lambda db, h: next(lookups, None) or real_lookup(db, h)
    )

    response = _upload(client)
    assert response.status_code == 200
    assert response.json()["already_ingested"] is True
    assert response.json()["document_id"] == str(winner.id)
    assert not (upload_env / "knowledge").exists()  # the lost upload saved no file


def test_png_renamed_to_pdf_is_refused(client: TestClient, upload_env: Path) -> None:
    response = _upload(client, data=b"\x89PNG\r\n\x1a\n not a pdf")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "KNOWLEDGE_NOT_A_PDF"


def test_too_large_file_is_refused(
    client: TestClient, upload_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "max_pdf_mb", 1)
    response = _upload(client, data=b"%PDF-" + b"0" * (1024 * 1024 - 4))  # 1 MB + 1 byte
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "KNOWLEDGE_FILE_TOO_LARGE"


def test_upload_is_refused_in_prod(
    client: TestClient, upload_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "environment", "prod")
    response = _upload(client)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "KNOWLEDGE_UPLOAD_NOT_ALLOWED"


def test_upload_without_title_is_refused(client: TestClient, upload_env: Path) -> None:
    response = client.post(
        "/api/v1/knowledge/documents", files={"file": ("a.pdf", FAKE_PDF, "application/pdf")}
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "CORE_VALIDATION_FAILED"


def test_status_is_queued_then_done_with_counts(
    client: TestClient, db: Session, upload_env: Path
) -> None:
    document_id = _upload(client).json()["document_id"]
    queued = _status(client, document_id)
    assert queued["status"] == "queued"
    assert (queued["page_count"], queued["section_count"], queued["chunk_count"]) == (0, 0, 0)

    assert run_one(db)
    done = _status(client, document_id)
    assert done["status"] == "done"
    assert done["error_code"] is None
    assert (done["page_count"], done["section_count"], done["chunk_count"]) == (3, 2, 2)


def test_status_is_failed_with_error_code_when_the_parser_fails(
    client: TestClient, db: Session, upload_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(data: bytes) -> tuple[list[ParsedItem], dict[int, str]]:
        raise errors.parse_failed()

    monkeypatch.setattr(service, "parse_pdf", broken)
    monkeypatch.setattr(get_settings(), "job_max_attempts", 1)
    document_id = _upload(client).json()["document_id"]
    assert run_one(db)
    failed = _status(client, document_id)
    assert failed["status"] == "failed"
    assert failed["error_code"] == "KNOWLEDGE_PARSE_FAILED"


def test_status_of_unknown_document_is_404(client: TestClient) -> None:
    response = client.get(f"/api/v1/knowledge/documents/{uuid.uuid4()}/status")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "CORE_NOT_FOUND"
