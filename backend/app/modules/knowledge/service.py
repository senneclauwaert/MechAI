"""The logic of the knowledge module. Other modules call these functions, never the tables.

For now this is a playground to try RAG: add a text, the worker embeds it, then search.
The real ingestion (PDF -> pages -> sections -> chunks) comes in its own issue.
"""

import hashlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core import jobs
from app.llm import get_llm
from app.modules.knowledge import errors
from app.modules.knowledge.models import Chunk, Document, Section

PLAYGROUND_TITLE = "Playground"
RRF_K = 60  # standard value for Reciprocal Rank Fusion
EMBED_BATCH = 64  # texts per embedding call

# "5.3.1 Bearings" -> number "5.3.1", title "Bearings"
SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

# Whole words or phrases that mark a chunk as high voltage. The handbook is English.
# The query-side safety levels (#34) are separate and stricter; this tag is checked by hand once.
HIGH_VOLTAGE_WORDS = [
    "battery",
    "batteries",
    "bms",
    "charger",
    "charging",
    "controller",
    "high voltage",
    "high-voltage",
    "inverter",
    "contactor",
    "fuse",
    "motor cable",
    "insulation",
]
HIGH_VOLTAGE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in HIGH_VOLTAGE_WORDS) + r")\b", re.IGNORECASE
)


@dataclass(frozen=True)
class ParsedItem:
    """One piece of the handbook as the parser found it."""

    kind: Literal["heading", "text", "table"]
    text: str
    page: int


@dataclass
class SectionDraft:
    """A numbered section of the handbook, not saved yet."""

    number: str | None
    heading_path: str
    page_start: int
    page_end: int
    text: str


def approx_tokens(text: str) -> int:
    """A cheap token estimate: about 4 tokens for every 3 words."""
    return round(len(text.split()) * 4 / 3)


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _parent_numbers(number: str) -> list[str]:
    """Parents from the top: "5.3.1" gives ["5", "5.3"]."""
    parts = number.split(".")
    return [".".join(parts[:i]) for i in range(1, len(parts))]


def build_sections(items: list[ParsedItem], document_title: str) -> list[SectionDraft]:
    """Group parsed items into numbered sections, each with its full heading path."""
    # (number, title, items) per section; the first one is the title section
    groups: list[tuple[str | None, str, list[ParsedItem]]] = [(None, document_title, [])]
    for item in items:
        match = SECTION_HEADING.match(item.text) if item.kind == "heading" else None
        if match:
            groups.append((match.group(1), item.text, []))
        else:
            groups[-1][2].append(item)  # also a heading without a number: it stays text

    titles = {number: title for number, title, _ in groups if number}
    sections = []
    for number, title, group_items in groups:
        text = "\n\n".join(item.text for item in group_items)
        if not text.strip():
            continue
        if number is None:
            path = title
        else:
            parents = [titles[n] for n in _parent_numbers(number) if n in titles]
            path = " > ".join([*parents, title])
        pages = [item.page for item in group_items]
        sections.append(SectionDraft(number, path, min(pages), max(pages), text))
    return sections


def split_long_sections(sections: list[SectionDraft], max_tokens: int) -> list[SectionDraft]:
    """Split sections above max_tokens at paragraph boundaries."""
    result = []
    for section in sections:
        if approx_tokens(section.text) <= max_tokens:
            result.append(section)
            continue
        parts: list[list[str]] = [[]]
        for paragraph in section.text.split("\n\n"):
            candidate = "\n\n".join([*parts[-1], paragraph])
            if parts[-1] and approx_tokens(candidate) > max_tokens:
                parts.append([paragraph])
            else:
                parts[-1].append(paragraph)
        for number, part in enumerate(parts, start=1):
            suffix = f" (part {number})" if number > 1 else ""
            result.append(
                SectionDraft(
                    section.number,
                    section.heading_path + suffix,
                    section.page_start,
                    section.page_end,
                    "\n\n".join(part),
                )
            )
    return result


def safety_level_for(text: str) -> str:
    return "high_voltage" if HIGH_VOLTAGE_PATTERN.search(text) else "normal"


def context_header(document_title: str, section: SectionDraft) -> str:
    if section.page_start == section.page_end:
        pages = f"p. {section.page_start}"
    else:
        pages = f"p. {section.page_start}-{section.page_end}"
    return f"{document_title} > {section.heading_path} ({pages})"


def _is_table(paragraph: str) -> bool:
    return any(line.startswith("|") for line in paragraph.split("\n"))


def _cut_by_words(paragraph: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Cut one long paragraph into pieces; each piece starts with the end of the one before."""
    words = paragraph.split()
    size = max(1, chunk_tokens * 3 // 4)  # words per piece
    step = max(1, size - overlap_tokens * 3 // 4)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return pieces


def _overlap_text(paragraph: str, overlap_tokens: int) -> str:
    """The end of the previous chunk that is repeated at the start of the next one."""
    if overlap_tokens == 0 or _is_table(paragraph):
        return ""
    if approx_tokens(paragraph) <= overlap_tokens:
        return paragraph
    return " ".join(paragraph.split()[-(overlap_tokens * 3 // 4) :])


def make_chunks(text: str, chunk_tokens: int, overlap_tokens: int) -> list[str]:
    """Pack paragraphs into chunks of at most chunk_tokens (a whole table may be longer)."""
    chunks: list[str] = []
    current: list[str] = []  # paragraphs of the chunk being built
    last_paragraph = ""  # the last real paragraph of the previous chunk

    def flush() -> None:
        if current:
            chunks.append("\n\n".join(current))
            current.clear()

    for paragraph in (p.strip() for p in text.split("\n\n")):
        if not paragraph:
            continue
        if approx_tokens(paragraph) > chunk_tokens:
            flush()
            if _is_table(paragraph):
                chunks.append(paragraph)  # a table is never cut
            else:
                chunks.extend(_cut_by_words(paragraph, chunk_tokens, overlap_tokens))
            last_paragraph = ""
            continue
        if current and approx_tokens("\n\n".join([*current, paragraph])) > chunk_tokens:
            flush()
            overlap = _overlap_text(last_paragraph, overlap_tokens)
            # the overlap only helps when the new paragraph still fits next to it
            if overlap and approx_tokens(f"{overlap}\n\n{paragraph}") <= chunk_tokens:
                current.append(overlap)
        current.append(paragraph)
        last_paragraph = paragraph
    flush()
    return chunks


def embed_chunks(db: Session, chunks: list[Chunk]) -> int:
    """Set the embedding of every chunk. Returns how many texts were sent to the AI.

    Texts that were embedded before (same hash and model), or that appear twice in the list, are
    not sent again. Does not commit.
    """
    llm = get_llm()
    model = llm.embedding_model
    hashes = {chunk.content_hash for chunk in chunks}
    rows = db.execute(
        select(Chunk.content_hash, Chunk.embedding).where(
            Chunk.content_hash.in_(hashes),
            Chunk.embedding_model == model,
            Chunk.embedding.is_not(None),
        )
    ).all()
    known = {content: vector for content, vector in rows}

    to_send: dict[str, str] = {}  # hash -> text, each text once
    for chunk in chunks:
        if chunk.content_hash not in known:
            to_send.setdefault(chunk.content_hash, chunk.text)

    hashes_to_send = list(to_send)
    for start in range(0, len(hashes_to_send), EMBED_BATCH):
        batch = hashes_to_send[start : start + EMBED_BATCH]
        vectors = llm.embed([to_send[h] for h in batch])
        known.update(zip(batch, vectors, strict=True))

    for chunk in chunks:
        chunk.embedding = known[chunk.content_hash]
        chunk.embedding_model = model
    return len(to_send)


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
        content_hash=content_hash(chunk_text),
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
    embed_chunks(db, [chunk])
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
