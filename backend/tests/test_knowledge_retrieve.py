"""Tests for retrieve_sections and external_text_allowed (knowledge module).

Rows are built by hand with the fake embedding client, inside the rolled-back transaction of the
`db` fixture. The texts are short kart texts written for these tests.
"""

import unicodedata
import uuid
from typing import Any, NoReturn

import pytest
from sqlalchemy import delete
from sqlalchemy import text as sql_text
from sqlalchemy.exc import DBAPIError
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


# A fake client that embeds every query as one fixed vector, so similarities are known up front.
class FixedQueryLLM(FakeLLM):
    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _ in texts]


def unit(first: float, second: float = 0.0) -> list[float]:
    """A 1024-dim vector with the first two values set, the rest 0 (cosine is easy to compute)."""
    dims = get_settings().embed_dimensions
    return [first, second] + [0.0] * (dims - 2)


def use_query_vector(monkeypatch: pytest.MonkeyPatch, vector: list[float]) -> None:
    """Make retrieve_sections embed every query as `vector`."""
    monkeypatch.setattr(service, "get_llm", lambda: FixedQueryLLM(vector))


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
    db: Session,
    section: Section,
    text: str,
    model: str | None = FAKE.embedding_model,
    vector: list[float] | None = None,
) -> Chunk:
    """Save a chunk with a fake embedding, or with `vector` (model None: no embedding at all)."""
    chunk = Chunk(
        section_id=section.id,
        text=text,
        content_hash=uuid.uuid4().hex,
        token_count=10,
        embedding=(vector or FAKE.embed([text])[0]) if model else None,
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


def test_similarity_values_are_exact(empty_db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    doc = make_document(empty_db)
    high = make_chunk(empty_db, make_section(empty_db, doc, number="1.1"), "a", vector=unit(1.0))
    mid = make_chunk(
        empty_db, make_section(empty_db, doc, number="2.1"), "b", vector=unit(0.6, 0.8)
    )
    low = make_chunk(
        empty_db, make_section(empty_db, doc, number="3.1"), "c", vector=unit(0.0, 1.0)
    )

    result = service.retrieve_sections(empty_db, "q", ["zzzunknown"], [])

    by_id = {c.chunk_id: c for c in result.chunks}
    assert by_id[high.id].similarity == pytest.approx(1.0)
    assert by_id[mid.id].similarity == pytest.approx(0.6)
    assert by_id[low.id].similarity == pytest.approx(0.0)
    assert [by_id[c.id].vector_rank for c in (high, mid, low)] == [1, 2, 3]
    assert result.best_similarity == pytest.approx(1.0)


def test_best_similarity_of_low_chunks_is_low(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    section = make_section(empty_db, make_document(empty_db))
    make_chunk(empty_db, section, "a", vector=unit(0.6, 0.8))  # similarity 0.6
    make_chunk(empty_db, section, "b", vector=unit(0.0, 1.0))  # similarity 0.0

    result = service.retrieve_sections(empty_db, "q", ["zzzunknown"], [])

    assert result.best_similarity == pytest.approx(0.6)  # the maximum, not the minimum or 0.4


def test_rrf_scores_and_order(empty_db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    doc = make_document(empty_db)
    both_section = make_section(empty_db, doc, number="1.1")
    vector_section = make_section(empty_db, doc, number="2.1")
    text_section = make_section(empty_db, doc, number="3.1")
    # vector list: both (rank 1), vector_only (rank 2); text list: text_only (1), both (2)
    both = make_chunk(empty_db, both_section, "axle", vector=unit(1.0))
    vector_only = make_chunk(empty_db, vector_section, "seat", vector=unit(0.6, 0.8))
    text_only = make_chunk(empty_db, text_section, "axle axle axle", model=None)

    result = service.retrieve_sections(empty_db, "q", ["axle"], [])

    by_id = {c.chunk_id: c for c in result.chunks}
    assert (by_id[both.id].vector_rank, by_id[both.id].text_rank) == (1, 2)
    assert (by_id[vector_only.id].vector_rank, by_id[vector_only.id].text_rank) == (2, None)
    assert (by_id[text_only.id].vector_rank, by_id[text_only.id].text_rank) == (None, 1)
    assert by_id[both.id].score == pytest.approx(1 / 61 + 1 / 62)
    assert by_id[vector_only.id].score == pytest.approx(1 / 62)
    assert by_id[text_only.id].score == pytest.approx(1 / 61)
    assert by_id[text_only.id].similarity is None
    assert [c.chunk_id for c in result.chunks] == [both.id, text_only.id, vector_only.id]


def test_equal_distances_are_ordered_by_chunk_id(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    section = make_section(empty_db, make_document(empty_db))
    # 25 identical vectors, more than the 20 that are kept, saved in descending id order
    ids = sorted(uuid.uuid4() for _ in range(25))
    for chunk_id in reversed(ids):
        chunk = Chunk(
            id=chunk_id,
            section_id=section.id,
            text="seat",
            content_hash=uuid.uuid4().hex,
            embedding=unit(1.0),
            embedding_model=FAKE.embedding_model,
        )
        empty_db.add(chunk)
        empty_db.flush()

    result = service.retrieve_sections(empty_db, "q", ["axle"], [])

    assert [c.chunk_id for c in result.chunks] == ids[:20]
    assert [c.vector_rank for c in result.chunks] == list(range(1, 21))


def test_vector_hits_sort_equal_distances_again_in_python() -> None:
    first, second = sorted(uuid.uuid4() for _ in range(2))

    # A session whose rows come back in the wrong id order (a relaxed HNSW scan may do this)
    class WrongOrderSession:
        def execute(self, stmt: Any) -> "WrongOrderSession":
            return self

        def all(self) -> list[tuple[Chunk, float]]:
            return [(Chunk(id=second), 0.5), (Chunk(id=first), 0.5)]

    session: Any = WrongOrderSession()
    hits = service._vector_hits(session, service.select(Chunk), unit(1.0))
    assert [chunk.id for chunk, _ in hits] == [first, second]


def test_search_breaks_score_ties_by_chunk_id(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    section = make_section(empty_db, make_document(empty_db))
    low, high = sorted(uuid.uuid4() for _ in range(2))
    # Both are rank 1 in one list only, so both score 1/61; the text-only chunk has the lower id
    vector_only = Chunk(id=high, text="seat", embedding=unit(1.0))
    vector_only.embedding_model = FAKE.embedding_model
    text_only = Chunk(id=low, text="axle")
    for chunk in (vector_only, text_only):
        chunk.section_id = section.id
        chunk.content_hash = uuid.uuid4().hex
        empty_db.add(chunk)
        empty_db.flush()

    hits = service.search(empty_db, "axle")

    assert [hit["chunk_id"] for hit in hits] == [str(low), str(high)]
    assert hits[0]["score"] == hits[1]["score"]


def test_ai_off_never_creates_a_client(empty_db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Replace the rear axle bearing.")
    monkeypatch.setattr(get_settings(), "ai_enabled", False)

    def no_client() -> NoReturn:
        raise AssertionError("get_llm must not be called")

    monkeypatch.setattr(service, "get_llm", no_client)

    result = service.retrieve_sections(empty_db, "x", ["axle"], [], use_embeddings=True)

    assert [c.chunk_id for c in result.chunks] == [chunk.id]
    assert result.best_similarity is None
    assert result.embedding_error is None


def test_best_similarity_comes_only_from_returned_sections(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_query_vector(monkeypatch, unit(1.0))
    monkeypatch.setattr(get_settings(), "answer_sections", 1)
    doc = make_document(empty_db)
    close = make_section(empty_db, doc, number="1.1")
    other = make_section(empty_db, doc, number="2.1")
    make_chunk(empty_db, close, "seat", vector=unit(1.0))  # similarity 1, no text match
    make_chunk(empty_db, other, "axle axle", vector=unit(0.6, 0.8))  # similarity 0.6, text match

    result = service.retrieve_sections(empty_db, "q", ["axle"], [])

    assert [s.section_id for s in result.sections] == [other.id]  # the close section is dropped
    assert result.best_similarity == pytest.approx(0.6)


def test_zero_vectors_give_no_score(empty_db: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    section = make_section(empty_db, make_document(empty_db))
    zero_chunk = make_chunk(empty_db, section, "axle", vector=unit(0.0))
    # a zero vector in the database: the chunk is not a vector hit
    use_query_vector(monkeypatch, unit(1.0))
    result = service.retrieve_sections(empty_db, "q", ["axle"], [])
    assert result.best_similarity is None
    assert result.chunks[0].chunk_id == zero_chunk.id
    assert result.chunks[0].similarity is None
    assert result.chunks[0].vector_rank is None

    # a zero query vector: nothing has a score
    make_chunk(empty_db, section, "axle bearing", vector=unit(1.0))
    use_query_vector(monkeypatch, unit(0.0))
    result = service.retrieve_sections(empty_db, "q", ["axle"], [])
    assert result.best_similarity is None
    assert all(c.similarity is None for c in result.chunks)


def test_control_characters_are_removed_from_all_inputs(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Order bearing AB-1234 for the rear axle.")

    result = service.retrieve_sections(
        empty_db, "rear\x00 axle\x01 bearing\x1f", ["ax\x00le", "\x07"], ["AB-\x001234", "\x00"]
    )

    assert result.chunks[0].chunk_id == chunk.id
    # keywords with no usable word: the words of the query are used
    only_control = service.retrieve_sections(
        empty_db, "rear axle", ["\x00", "\x1b"], [], use_embeddings=False
    )
    assert [c.chunk_id for c in only_control.chunks] == [chunk.id]


# A fake client that remembers the texts it was asked to embed.
class RecordingLLM(FakeLLM):
    def __init__(self) -> None:
        self.embedded: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed(texts)


def test_inputs_are_cleaned_before_embedding_and_text_search(
    empty_db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    section = make_section(empty_db, make_document(empty_db))
    make_chunk(empty_db, section, "Order bearing AB-1234 for the rear axle.")
    llm = RecordingLLM()
    monkeypatch.setattr(service, "get_llm", lambda: llm)
    seen: dict[str, Any] = {}
    real_input, real_hits = service._text_search_input, service._text_hits

    # Spies: record what the private helpers receive, then call the real ones.
    def spy_input(keywords: list[str], query: str) -> str:
        seen["keywords"], seen["query"] = keywords, query
        return real_input(keywords, query)

    def spy_hits(
        db: Session, base: Any, ts_input: str, part_numbers: list[str] | None = None
    ) -> list[Chunk]:
        seen["parts"] = part_numbers
        return real_hits(db, base, ts_input, part_numbers)

    monkeypatch.setattr(service, "_text_search_input", spy_input)
    monkeypatch.setattr(service, "_text_hits", spy_hits)

    service.retrieve_sections(
        empty_db, "rear\x00 axle\x01 bearing", ["ax\x00le", "\x07"], ["AB-\x001234", "\x00"]
    )

    assert llm.embedded == ["rear  axle  bearing"]
    assert not any(unicodedata.category(c) == "Cc" for c in llm.embedded[0])
    assert seen["query"] == "rear  axle  bearing"
    assert seen["keywords"] == ["ax le", " "]
    assert seen["parts"] == ["AB- 1234"]


def test_lone_surrogates_do_not_cause_an_error(empty_db: Session) -> None:
    section = make_section(empty_db, make_document(empty_db))
    chunk = make_chunk(empty_db, section, "Order bearing AB-1234 for the rear axle.")

    result = service.retrieve_sections(
        empty_db, "rear \ud800 axle", ["ax\udc00le", "bearing"], ["AB-\ud8001234"]
    )

    assert result.chunks[0].chunk_id == chunk.id


def test_clean_replaces_control_characters_by_a_space() -> None:
    assert service._clean("a\x00b\tc\nd\x7fe") == "a b c d e"
    assert service._clean("a\ud800b\udfffc") == "a b c"
    assert service._clean("Axle-é 12") == "Axle-é 12"


def test_database_errors_do_not_show_bound_values(db: Session) -> None:
    with pytest.raises(DBAPIError) as error:
        # a division by zero: the database message itself does not repeat the value
        db.execute(sql_text("SELECT length(CAST(:value AS text)) / 0"), {"value": "secret-term"})
    assert "secret-term" not in str(error.value)
