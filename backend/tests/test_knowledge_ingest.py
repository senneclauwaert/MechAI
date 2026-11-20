import uuid

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.jobs import enqueue, get_job
from app.llm import FakeLLM
from app.modules.knowledge import service
from app.modules.knowledge.models import Chunk, Document, Page, Section
from app.modules.knowledge.service import ParsedItem, SectionDraft


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
