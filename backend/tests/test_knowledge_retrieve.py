"""Tests for retrieve_sections and external_text_allowed (knowledge module).

Rows are built by hand with the fake embedding client, inside the rolled-back transaction of the
`db` fixture. The texts are short kart texts written for these tests.
"""

import uuid
from typing import NoReturn

import pytest
from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.errors import AppError
from app.llm import ChatResult, FakeLLM, LLMClient
from app.modules.knowledge import service
from app.modules.knowledge.models import Chunk, Document, Section

FAKE = FakeLLM()


# A client that is not the fake one, so external_text_allowed looks at the settings.
class OtherLLM:
    embedding_model = "other-embed"
    chat_model = "other-chat"

    def chat(
        self,
        messages: list[dict[str, str]],
        json_output: bool = False,
        max_tokens: int | None = None,
    ) -> ChatResult:
        return ChatResult("", 0, 0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def transcribe(self, audio: bytes, file_name: str, language: str | None = None) -> str:
        return ""


# A fake client whose embedding call is down.
class DownEmbedLLM(FakeLLM):
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise AppError("ASSISTANT_AI_UNAVAILABLE", "down", 503)


def make_document(
    db: Session,
    source_type: str = "handbook",
    superseded: bool = False,
    revision: str | None = "A",
) -> Document:
    """Save a document; `superseded` points it to a newer document."""
    doc = Document(title=f"Kart manual {uuid.uuid4().hex[:6]}", source_type=source_type)
    doc.revision = revision
    db.add(doc)
    db.flush()
    if superseded:
        newer = Document(title="Newer manual", source_type=source_type)
        db.add(newer)
        db.flush()
        doc.superseded_by = newer.id
        db.flush()
    return doc


def make_section(db: Session, doc: Document, tokens: int = 50, number: str = "1.1") -> Section:
    """Save a section of a document."""
    section = Section(
        document_id=doc.id,
        section_number=number,
        heading_path=f"Chapter > {number} Part",
        page_start=3,
        page_end=4,
        text=f"Body of section {number}",
        token_count=tokens,
    )
    db.add(section)
    db.flush()
    return section


def make_chunk(
    db: Session, section: Section, text: str, model: str | None = FAKE.embedding_model
) -> Chunk:
    """Save a chunk with a fake embedding (model None: no embedding at all)."""
    chunk = Chunk(
        section_id=section.id,
        text=text,
        content_hash=uuid.uuid4().hex,
        token_count=10,
        embedding=FAKE.embed([text])[0] if model else None,
        embedding_model=model,
    )
    db.add(chunk)
    db.flush()
    return chunk


@pytest.fixture
def empty_db(db: Session) -> Session:
    """The db fixture without any chunk (undone after the test)."""
    db.execute(delete(Chunk))
    return db


def test_finds_the_right_section(empty_db: Session) -> None:
    doc = make_document(empty_db)
    axle = make_section(empty_db, doc, number="4.5")
    seat = make_section(empty_db, doc, number="7.1")
    make_chunk(empty_db, axle, "Replace the rear axle bearing and tighten the nut.")
    make_chunk(empty_db, seat, "Adjust the seat position with the two side levers.")

    result = service.retrieve_sections(empty_db, "replace the rear axle bearing", ["axle"], [])

    assert result.sections[0].section_id == axle.id
    assert result.sections[0].document_title == doc.title
    assert result.sections[0].revision == "A"
    assert result.best_similarity is not None and result.best_similarity > 0
    assert result.embedding_error is None
    assert result.chunks[0].vector_rank == 1


def test_two_chunks_of_one_section_give_one_section(empty_db: Session) -> None:
    doc = make_document(empty_db)
    section = make_section(empty_db, doc)
    make_chunk(empty_db, section, "Rear axle bearing removal steps.")
    make_chunk(empty_db, section, "Rear axle bearing fitting steps.")

    result = service.retrieve_sections(empty_db, "rear axle bearing", ["axle"], [])

    assert len(result.chunks) == 2
    assert [s.section_id for s in result.sections] == [section.id]


def test_playground_and_superseded_documents_are_left_out(empty_db: Session) -> None:
    playground = make_document(empty_db, source_type="procedure")
    old = make_document(empty_db, superseded=True)
    current = make_document(empty_db)
    for doc in (playground, old):
        make_chunk(empty_db, make_section(empty_db, doc), "Rear axle bearing from the old text.")
    wanted = make_section(empty_db, current)
    make_chunk(empty_db, wanted, "Rear axle bearing from the current handbook.")

    result = service.retrieve_sections(empty_db, "rear axle bearing", ["axle"], [])

    assert [s.section_id for s in result.sections] == [wanted.id]


def test_only_playground_text_means_nothing_embedded(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db, source_type="procedure"))
    make_chunk(empty_db, section, "Rear axle bearing.")

    with pytest.raises(AppError) as error:
        service.retrieve_sections(empty_db, "axle", ["axle"], [], use_embeddings=False)
    assert error.value.code == "KNOWLEDGE_NOTHING_EMBEDDED"


def test_wrong_embedding_model_means_nothing_embedded(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    make_chunk(empty_db, section, "Rear axle bearing.", model="some-other-model")

    with pytest.raises(AppError) as error:
        service.retrieve_sections(empty_db, "rear axle", ["axle"], [])
    assert error.value.code == "KNOWLEDGE_NOTHING_EMBEDDED"


def test_one_matching_keyword_is_enough(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Replace the rear axle bearing.")

    result = service.retrieve_sections(
        empty_db, "x", ["axle", "unknownwordzzz"], [], use_embeddings=False
    )

    assert result.chunks[0].chunk_id == chunk.id
    assert result.chunks[0].text_rank == 1
    assert result.best_similarity is None


def test_part_number_ranks_first_in_the_text_list(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    make_chunk(empty_db, section, "axle axle axle bearing axle axle")
    with_part = make_chunk(empty_db, section, "Order bearing AB-1234 for the hub.")

    result = service.retrieve_sections(
        empty_db, "x", ["axle"], ["AB-1234", "x" * 51, " "], use_embeddings=False
    )

    ranks = {c.chunk_id: c.text_rank for c in result.chunks}
    assert ranks[with_part.id] == 1


def test_odd_characters_in_keywords_do_not_break_the_query(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Replace the rear axle bearing.")

    result = service.retrieve_sections(
        empty_db, "x", ['"axle', "rear-axle'", "or", "-bearing", "OR"], [], use_embeddings=False
    )

    assert [c.chunk_id for c in result.chunks] == [chunk.id]


def test_text_search_input_is_cleaned() -> None:
    cleaned = service._text_search_input(['"Axle', "axle", "a-b", "OR", "or"], "ignored")
    assert cleaned == "Axle or a or b"
    assert service._text_search_input([], "Is the 'seat' loose?") == "Is or the or seat or loose"
    many = service._text_search_input([f"w{i}" for i in range(40)], "")
    assert len(many.split(" or ")) == 30


def make_ranked_sections(db: Session, tokens: list[int]) -> list[Section]:
    """Sections whose chunks mention "axle" 3, 2, 1 ... times, so they rank in this order."""
    doc = make_document(db)
    sections = []
    for index, size in enumerate(tokens):
        section = make_section(db, doc, tokens=size, number=f"{index + 1}.1")
        make_chunk(db, section, " ".join(["axle"] * (len(tokens) - index)))
        sections.append(section)
    return sections


def test_section_count_limit(empty_db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "answer_sections", 2)
    sections = make_ranked_sections(empty_db, [10, 10, 10])

    result = service.retrieve_sections(empty_db, "x", ["axle"], [], use_embeddings=False)

    assert [s.section_id for s in result.sections] == [sections[0].id, sections[1].id]


def test_token_limit_skips_a_section_that_does_not_fit(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "answer_source_tokens", 500)
    sections = make_ranked_sections(empty_db, [400, 400, 50])

    result = service.retrieve_sections(empty_db, "x", ["axle"], [], use_embeddings=False)

    assert [s.section_id for s in result.sections] == [sections[0].id, sections[2].id]


def test_first_section_is_kept_when_it_alone_is_too_long(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "answer_source_tokens", 500)
    sections = make_ranked_sections(empty_db, [900, 10])

    result = service.retrieve_sections(empty_db, "x", ["axle"], [], use_embeddings=False)

    assert [s.section_id for s in result.sections] == [sections[0].id]
    assert result.sections[0].text == sections[0].text  # never cut


def test_embedding_error_keeps_the_text_hits(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Replace the rear axle bearing.")
    monkeypatch.setattr(service, "get_llm", lambda: DownEmbedLLM())

    result = service.retrieve_sections(empty_db, "secret question words", ["axle"], [])

    assert result.embedding_error == "ASSISTANT_AI_UNAVAILABLE"
    assert result.best_similarity is None
    assert result.chunks[0].chunk_id == chunk.id
    assert result.chunks[0].similarity is None
    assert "secret question words" not in caplog.text


def test_without_embeddings_the_client_is_never_created(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = make_section(empty_db, make_document(empty_db))
    make_chunk(empty_db, section, "Replace the rear axle bearing.")

    def no_client() -> NoReturn:
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr(service, "get_llm", no_client)

    result = service.retrieve_sections(empty_db, "x", ["axle"], [], use_embeddings=False)

    assert len(result.sections) == 1


@pytest.mark.parametrize(
    ("allow", "ai_enabled", "expected"),
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_external_text_allowed(
    monkeypatch: pytest.MonkeyPatch, allow: bool, ai_enabled: bool, expected: bool
) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "allow_external_embeddings", allow)
    monkeypatch.setattr(settings, "ai_enabled", ai_enabled)

    other: LLMClient = OtherLLM()
    assert service.external_text_allowed(other) is expected
    assert service.external_text_allowed(FakeLLM()) is True  # the fake client is always allowed
