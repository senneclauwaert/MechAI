"""The logic of the knowledge module. Other modules call these functions, never the tables.

It covers:
- handbook ingestion: PDF -> pages (text + PNG) -> sections -> chunks -> embeddings,
  run by the worker after an upload;
- the playground: add a text, the worker embeds it;
- hybrid search (vector + full-text, merged with RRF) over the chunks;
- retrieval of whole parent sections for the assistant (`retrieve_sections`).
"""

import hashlib
import io
import logging
import math
import re
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from sqlalchemy import Select, case, delete, func, or_, select
from sqlalchemy import text as sql_text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import jobs
from app.core.errors import AppError, not_found
from app.core.models import Job
from app.llm import FakeLLM, LLMClient, get_llm
from app.modules.knowledge import errors
from app.modules.knowledge.models import Chunk, Document, Page, Section
from app.modules.storage import service as storage

log = logging.getLogger("mechai.knowledge")

# Tuning constants
PLAYGROUND_TITLE = "Playground"
RRF_K = 60  # standard value for Reciprocal Rank Fusion
EMBED_BATCH = 64  # texts per embedding call
MAX_PAGE_PIXELS = 6000  # longest side of a page image

# Patterns used while building sections
# "5.3.1 Bearings" -> number "5.3.1", title "Bearings"
SECTION_HEADING = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(\S.*)$")

# Whole words or phrases that mark a chunk as high voltage. The handbook is English.
# The query-side safety levels (#34) are separate and stricter; this tag is checked by hand once.
HIGH_VOLTAGE_WORDS = [
    "battery",
    "batteries",
    "bms",
    "hv",
    "charger",
    "chargers",
    "charging",
    "controller",
    "controllers",
    "inverter",
    "inverters",
    "contactor",
    "contactors",
    "fuse",
    "fuses",
    "motor cable",
    "motor cables",
    "insulation",
]
# Between the words of a phrase ("motor cable") any whitespace is allowed, also a line break.
# Between "high" and "voltage": whitespace, a hyphen, a Unicode hyphen or dash (U+2010 to U+2014),
# the minus sign (U+2212) or a soft hyphen (U+00AD), in any mix.
HIGH_VOLTAGE_PATTERN = re.compile(
    r"\b("
    + "|".join(
        [
            *(re.escape(word).replace(r"\ ", r"\s+") for word in HIGH_VOLTAGE_WORDS),
            r"high[\s\-‐-—−­]+voltage",
        ]
    )
    + r")\b",
    re.IGNORECASE,
)


# In-memory shapes used between parsing and saving
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
    safety_level: str = "normal"  # of the whole section, so every part of a split gets the same


# Small text helpers
def approx_tokens(text: str) -> int:
    """A cheap token estimate: about 4 tokens for every 3 words."""
    return round(len(text.split()) * 4 / 3)


# SHA-256 of a chunk text; the same text gets the same hash, so its embedding can be reused.
def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# Sections: group the parsed items under their numbered headings, then split
# sections that are too long.
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
            groups.append((match.group(1), item.text, [item]))
        else:
            groups[-1][2].append(item)  # also a heading without a number: it stays text

    titles = {number: title for number, title, _ in groups if number}
    sections = []
    for number, title, group_items in groups:
        body_items = group_items[1:] if number is not None else group_items
        text = "\n\n".join(item.text for item in body_items)
        if not text.strip():
            continue
        if number is None:
            path = title
        else:
            parents = [titles[n] for n in _parent_numbers(number) if n in titles]
            path = " > ".join([*parents, title])
        pages = [item.page for item in group_items]
        # The heading counts too: "6 Battery > 6.2 Removal" is high voltage whatever the body says
        level = safety_level_for(path + "\n" + text)
        sections.append(SectionDraft(number, path, min(pages), max(pages), text, level))
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
                    section.safety_level,
                )
            )
    return result


# "high_voltage" when the text mentions any high-voltage word, else "normal".
def safety_level_for(text: str) -> str:
    return "high_voltage" if HIGH_VOLTAGE_PATTERN.search(text) else "normal"


# First line of every chunk: document > heading path (pages). Gives the model and
# the search the context of a piece that is read on its own.
def context_header(document_title: str, section: SectionDraft) -> str:
    if section.page_start == section.page_end:
        pages = f"p. {section.page_start}"
    else:
        pages = f"p. {section.page_start}-{section.page_end}"
    return f"{document_title} > {section.heading_path} ({pages})"


# Chunking: pack paragraphs into chunks of about chunk_tokens, with a small overlap
# between chunks. Tables are never cut; a paragraph that is too long is cut by words.
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

    # Close the chunk being built and start a new one.
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


# Embeddings
def external_text_allowed(llm: LLMClient) -> bool:
    """True when handbook text may go to this client: the fake one always, else both settings."""
    settings = get_settings()
    if isinstance(llm, FakeLLM):
        return True
    return settings.allow_external_embeddings and settings.ai_enabled


def _check_external_embeddings(llm: LLMClient) -> None:
    """Raise when this client may not get text (see external_text_allowed)."""
    if not external_text_allowed(llm):
        raise errors.external_embeddings_off()


def embed_chunks(db: Session, chunks: list[Chunk]) -> int:
    """Set the embedding of every chunk. Returns how many texts were sent to the AI.

    Texts that were embedded before (same hash and model), or that appear twice in the list, are
    not sent again. Does not commit.
    """
    llm = get_llm()
    _check_external_embeddings(llm)
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


# Playground: all test texts live in one "Playground" document.
def _playground(db: Session) -> Document:
    doc = db.scalars(select(Document).where(Document.title == PLAYGROUND_TITLE)).first()
    if doc is None:
        doc = Document(title=PLAYGROUND_TITLE, source_type="procedure")
        db.add(doc)
        db.flush()
    return doc


def add_playground_text(db: Session, title: str, text: str) -> tuple[Chunk, uuid.UUID]:
    """Save a text as one section with one chunk, and ask the worker to embed it (dev only)."""
    _require_dev_until_auth(errors.playground_not_allowed())
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


# PDF reading. Only the worker calls these: the api never opens a PDF.
def parse_pdf(data: bytes) -> tuple[list[ParsedItem], dict[int, str]]:
    """Read a PDF with Docling. Returns the items in reading order and the text of every page."""
    # Imported here so the api and the tests start fast (the same pattern as llm.py)
    from docling.datamodel.base_models import DocumentStream, InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import (
        DocItem,
        SectionHeaderItem,
        TableItem,
        TextItem,
        TitleItem,
    )

    settings = get_settings()
    try:
        options = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
        if settings.docling_models_dir:
            options.artifacts_path = settings.docling_models_dir
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        stream = DocumentStream(name="upload.pdf", stream=io.BytesIO(data))
        doc = converter.convert(stream).document

        items: list[ParsedItem] = []
        page_texts = {number: "" for number in range(1, doc.num_pages() + 1)}
        for item, _level in doc.iterate_items():  # body only: page headers and footers are skipped
            if not isinstance(item, DocItem) or not item.prov:
                continue
            kind: Literal["heading", "text", "table"]
            if isinstance(item, TableItem):
                kind, text = "table", item.export_to_markdown(doc=doc)
            elif isinstance(item, SectionHeaderItem | TitleItem):
                kind, text = "heading", item.text
            elif isinstance(item, TextItem):
                kind, text = "text", item.text
            else:
                continue  # pictures
            text = text.strip()
            if not text:
                continue
            page = item.prov[0].page_no
            items.append(ParsedItem(kind, text, page))
            page_texts[page] = (page_texts.get(page, "") + "\n\n" + text).strip()
        return items, page_texts
    except Exception as exc:  # Docling raises many kinds; the worker logs this one with its trace
        raise errors.parse_failed() from exc


def render_scale(width_pt: float, height_pt: float, dpi: int) -> float:
    """Pixels per point for a page image; a huge page is scaled down to MAX_PAGE_PIXELS."""
    return min(dpi / 72, MAX_PAGE_PIXELS / max(width_pt, height_pt))


def render_page_png(data: bytes, page_number: int, dpi: int) -> bytes:
    """Draw one page (numbered from 1) as a PNG."""
    import pypdfium2

    try:
        pdf = pypdfium2.PdfDocument(data)
        try:
            page = pdf[page_number - 1]
            width, height = page.get_size()
            image = page.render(scale=render_scale(width, height, dpi)).to_pil()
        finally:
            pdf.close()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception as exc:  # pypdfium2 raises PdfiumError, but also plain errors for odd pages
        raise errors.parse_failed() from exc


def count_pdf_pages(data: bytes) -> int:
    """Count the pages without parsing the PDF (much cheaper than Docling)."""
    import pypdfium2

    try:
        pdf = pypdfium2.PdfDocument(data)
    except pypdfium2.PdfiumError as exc:
        raise errors.parse_failed() from exc
    try:
        return len(pdf)
    finally:
        pdf.close()


# Ingestion job: everything for one document happens in one transaction, so a
# failed run leaves the previous data in place and a retry starts clean.
def ingest_document_job(db: Session, payload: dict[str, Any]) -> None:
    """Worker job: PDF -> pages (text + PNG) -> sections -> chunks -> embeddings. One commit."""
    settings = get_settings()
    document = db.get(Document, uuid.UUID(payload["document_id"]))
    if document is None:
        raise errors.document_not_found()

    # Checks first, so nothing is read, parsed or rendered for a job that cannot succeed
    _check_external_embeddings(get_llm())
    data = storage.open_file(document.file_path or "")
    if count_pdf_pages(data) > settings.max_pdf_pages:
        raise errors.too_many_pages(settings.max_pdf_pages)
    items, page_texts = parse_pdf(data)

    # Build the sections and their chunks in memory; nothing is saved yet.
    drafts = split_long_sections(build_sections(items, document.title), settings.section_max_tokens)
    sections: list[Section] = []
    chunks: list[Chunk] = []
    for draft in drafts:
        section = Section(
            id=uuid.uuid4(),  # set now, so the chunks can point to it before anything is saved
            document_id=document.id,
            section_number=draft.number,
            heading_path=draft.heading_path,
            page_start=draft.page_start,
            page_end=draft.page_end,
            text=draft.text,
            token_count=approx_tokens(draft.text),
            safety_level=draft.safety_level,
        )
        sections.append(section)
        header = context_header(document.title, draft)
        for piece in make_chunks(draft.text, settings.chunk_tokens, settings.chunk_overlap_tokens):
            text = f"{header}\n{piece}"
            chunks.append(
                Chunk(
                    section_id=section.id,
                    text=text,
                    content_hash=content_hash(text),
                    token_count=approx_tokens(text),
                )
            )
    # Order: render first (it can fail, and embeddings cost money), then embed while the old
    # chunks still exist (a retry reuses their embeddings), then replace the old rows.
    pages: list[Page] = []
    for number, text in page_texts.items():
        png = render_page_png(data, number, settings.page_image_dpi)
        image_path = storage.save_file(f"knowledge/{document.id}", f"page-{number:03}.png", png)
        pages.append(
            Page(document_id=document.id, page_number=number, image_path=image_path, text=text)
        )
    embedded = embed_chunks(db, chunks)

    # a retry starts clean; the chunks go with their sections (ON DELETE CASCADE)
    db.execute(delete(Section).where(Section.document_id == document.id))
    db.execute(delete(Page).where(Page.document_id == document.id))
    db.add_all(pages)
    db.add_all(sections)
    db.flush()  # sections first: the chunks point to them
    db.add_all(chunks)
    db.commit()
    log.info(
        "ingested document %s: %d pages, %d sections, %d chunks, %d embedded",
        document.id,
        len(page_texts),
        len(drafts),
        len(chunks),
        embedded,
    )


# Upload helpers
def _find_by_hash(db: Session, file_hash: str) -> Document | None:
    return db.scalars(select(Document).where(Document.file_hash == file_hash)).first()


# Queue an ingestion job and link it to the document, in one commit.
def _queue_ingest(db: Session, document: Document) -> Job:
    job = jobs.enqueue(db, "ingest_document", {"document_id": str(document.id)}, commit=False)
    document.job_id = job.id
    db.commit()
    return job


def _known_document(db: Session, document: Document) -> tuple[Document, Job, bool]:
    """The file was uploaded before: queue a new job only when the last one failed (or is gone)."""
    db.refresh(document, with_for_update=True)
    job = jobs.get_job(db, document.job_id) if document.job_id else None
    if job is None or job.status == "failed":
        job = _queue_ingest(db, document)
    return document, job, True


def _require_dev_until_auth(error: AppError) -> None:
    """Until #26 adds the admin check, the handbook endpoints only work in the dev environment."""
    if get_settings().environment != "dev":
        raise error


# Upload: check the file, skip duplicates, save the PDF and queue the ingestion.
def upload_document(
    db: Session, data: bytes, title: str, revision: str | None, kart_model: str | None
) -> tuple[Document, Job, bool]:
    """Save an uploaded handbook and queue its ingestion. The last value is `already_ingested`."""
    settings = get_settings()
    _require_dev_until_auth(errors.upload_not_allowed())
    if len(data) > settings.max_pdf_mb * 1024 * 1024:
        raise errors.file_too_large(settings.max_pdf_mb)
    # The client's file name and content type are not trusted: look at the first bytes
    if not data.startswith(b"%PDF-"):
        raise errors.not_a_pdf()
    # The api never opens the PDF (PDFium is not thread-safe and the file is untrusted):
    # the worker checks the page count before Docling runs.

    file_hash = hashlib.sha256(data).hexdigest()
    existing = _find_by_hash(db, file_hash)
    if existing is not None:
        return _known_document(db, existing)

    document = Document(
        id=uuid.uuid4(),
        title=title,
        revision=revision,
        kart_model=kart_model,
        language="en",
        source_type="handbook",
        file_hash=file_hash,
    )
    db.add(document)
    try:
        db.flush()
    except IntegrityError:
        # Two uploads of the same file at the same moment: the other one won, return its document.
        # No file was saved yet, and every document has its own folder, so nothing is overwritten.
        db.rollback()
        winner = _find_by_hash(db, file_hash)
        if winner is None:
            raise
        return _known_document(db, winner)

    # Our own file name, never the client's
    document.file_path = storage.save_file(f"knowledge/{document.id}", "source.pdf", data)
    return document, _queue_ingest(db, document), False


# Status: the state of the last ingestion job and how many rows it produced.
def document_status(db: Session, document_id: uuid.UUID) -> dict[str, Any]:
    """Where the ingestion of a document stands, with the counts of what was saved."""
    _require_dev_until_auth(errors.status_not_allowed())
    document = db.get(Document, document_id)
    if document is None:
        raise not_found("Document")
    job = jobs.get_job(db, document.job_id) if document.job_id else None
    status = job.status if job else "queued"
    error_code = None
    if status == "failed" and job and job.last_error:
        error_code = job.last_error.split(":", 1)[0]  # the worker stores "CODE: message"

    # Counts of what was saved for this document
    pages = db.scalar(select(func.count()).where(Page.document_id == document_id))
    sections = db.scalar(select(func.count()).where(Section.document_id == document_id))
    chunks = db.scalar(
        select(func.count())
        .select_from(Chunk)
        .join(Section, Chunk.section_id == Section.id)
        .where(Section.document_id == document_id)
    )
    return {
        "document_id": str(document.id),
        "title": document.title,
        "revision": document.revision,
        "status": status,
        "error_code": error_code,
        "page_count": pages or 0,
        "section_count": sections or 0,
        "chunk_count": chunks or 0,
    }


# Search helpers, used by search() and retrieve_sections()
def _vector_hits(
    db: Session, base: Select[Any], query_vector: list[float]
) -> list[tuple[Chunk, float]]:
    """The 20 chunks closest in meaning, as (chunk, cosine distance), closest first.

    `base` is a select of Chunk that already holds the filters. The rows are sorted again here
    because an HNSW scan in relaxed order may return them slightly out of order. A distance that
    is not a number (a zero vector has none) means "no score": that chunk is not a vector hit.
    Equal distances are ordered by chunk id, so the ranks are the same on every call.
    """
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = db.execute(base.add_columns(distance).order_by(distance, Chunk.id).limit(20)).all()
    hits = [(chunk, float(dist)) for chunk, dist in rows if dist is not None]
    return sorted(
        ((chunk, dist) for chunk, dist in hits if math.isfinite(dist)),
        key=lambda row: (row[1], row[0].id),
    )


def _text_hits(
    db: Session,
    base: Select[Any],
    ts_input: str,
    part_numbers: list[str] | None = None,
) -> list[Chunk]:
    """The 20 chunks that best match the words, best first.

    Chunks that contain one of the part numbers (exact words, no stemming) come before the rest.
    """
    ts_query = func.websearch_to_tsquery("english", ts_input)
    word_match = Chunk.tsv.op("@@")(ts_query)
    part_matches = [
        func.to_tsvector("simple", Chunk.text).op("@@")(func.plainto_tsquery("simple", part))
        for part in part_numbers or []
    ]
    stmt = base.where(or_(word_match, *part_matches))
    if part_matches:
        stmt = stmt.order_by(case((or_(*part_matches), 0), else_=1))
    stmt = stmt.order_by(func.ts_rank(Chunk.tsv, ts_query).desc(), Chunk.id).limit(20)
    return list(db.scalars(stmt).all())


def _rrf_score(vector_rank: int | None, text_rank: int | None) -> float:
    """Reciprocal Rank Fusion: the sum of 1 / (60 + rank) over the lists the chunk is in."""
    score = 0.0
    for rank in (vector_rank, text_rank):
        if rank is not None:
            score += 1 / (RRF_K + rank)
    return score


# Search
def search(db: Session, question: str, limit: int = 5) -> list[dict[str, Any]]:
    """Hybrid search: vector top 20 + full-text top 20, merged with RRF (vault: RAG pipeline)."""
    llm = get_llm()
    active = (Chunk.embedding_model == llm.embedding_model) & Chunk.embedding.is_not(None)
    has_embeddings = db.scalar(select(func.count()).where(active))
    if not has_embeddings:
        raise errors.nothing_embedded()

    # 1. Vector search: the 20 chunks closest in meaning (cosine distance).
    query_vector = llm.embed([question])[0]
    by_vector = [chunk for chunk, _ in _vector_hits(db, select(Chunk).where(active), query_vector)]

    # 2. Full-text search: the 20 chunks that best match the words of the question.
    by_text = _text_hits(db, select(Chunk), question)

    # 3. Merge both lists with Reciprocal Rank Fusion: a chunk scores higher the better it
    # ranks in either list, and highest when it is in both.
    vector_rank = {c.id: i + 1 for i, c in enumerate(by_vector)}
    text_rank = {c.id: i + 1 for i, c in enumerate(by_text)}
    chunks = {c.id: c for c in [*by_vector, *by_text]}

    # RRF score of one chunk
    def rrf(chunk_id: uuid.UUID) -> float:
        return _rrf_score(vector_rank.get(chunk_id), text_rank.get(chunk_id))

    # Highest score first, ties broken by chunk id
    best = sorted(chunks, key=lambda cid: (-rrf(cid), cid))[:limit]
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


# Retrieval of parent sections for the assistant
@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk that the search found, with its ranks and scores (kept for the trace)."""

    chunk_id: uuid.UUID
    section_id: uuid.UUID
    vector_rank: int | None
    text_rank: int | None
    similarity: float | None  # 1 - cosine distance; None when the vector search did not find it
    score: float  # RRF score: only for ordering, never for the "not in the handbook" decision


@dataclass(frozen=True)
class RetrievedSection:
    """A whole handbook section (never cut) with the data needed to cite it."""

    section_id: uuid.UUID
    section_number: str | None
    heading_path: str
    page_start: int | None
    page_end: int | None
    text: str
    token_count: int
    safety_level: str
    document_title: str
    revision: str | None


@dataclass(frozen=True)
class Retrieval:
    """The result of retrieve_sections."""

    chunks: list[RetrievedChunk]  # the best 20 chunks, best first
    sections: list[RetrievedSection]  # the sections for the answer step, best first
    best_similarity: float | None  # highest similarity among the returned sections, or None
    embedding_error: str | None  # error code when the query could not be embedded


MAX_PART_NUMBERS = 10
MAX_PART_NUMBER_CHARS = 50
MAX_TEXT_WORDS = 30


# Chunks of current handbook text only: no Playground text, no superseded documents.
def _handbook_chunks() -> Select[Any]:
    return (
        select(Chunk)
        .join(Section, Chunk.section_id == Section.id)
        .join(Document, Section.document_id == Document.id)
        .where(Document.source_type == "handbook", Document.superseded_by.is_(None))
    )


def _clean(text: str) -> str:
    """Replace control characters (category Cc) and lone surrogates (Cs) by a space.

    NUL and surrogates cannot be stored or sent to the database as UTF-8.
    """
    return "".join(" " if unicodedata.category(char) in ("Cc", "Cs") else char for char in text)


def _search_words(source: str) -> list[str]:
    """The distinct words of a text (letters, digits, underscores), without the word "or"."""
    words: list[str] = []
    seen: set[str] = set()
    for word in re.findall(r"\w+", source):
        key = word.lower()
        if key != "or" and key not in seen:
            seen.add(key)
            words.append(word)
    return words


def _text_search_input(keywords: list[str], query: str) -> str:
    """Words for websearch_to_tsquery, joined with "or" because that function ANDs its words.

    Only letters, digits and underscores survive, so quotes and hyphens cannot change the query.
    When the keywords hold no usable word, the words of the query are used instead.
    """
    words = _search_words(" ".join(keywords)) or _search_words(query)
    return " or ".join(words[:MAX_TEXT_WORDS])


def _pick_sections(db: Session, ranked_section_ids: list[uuid.UUID]) -> list[RetrievedSection]:
    """Load the sections (best first, from the ranked chunks) that fit the limits.

    At most `answer_sections` sections, and no section that would pass `answer_source_tokens`;
    the first one is always kept, however long. A section is never cut.
    """
    settings = get_settings()
    order = list(dict.fromkeys(ranked_section_ids))  # each section once, best chunk first
    rows = db.execute(
        select(Section, Document.title, Document.revision)
        .join(Document, Section.document_id == Document.id)
        .where(Section.id.in_(order))
    ).all()
    found = {section.id: (section, title, revision) for section, title, revision in rows}

    picked: list[RetrievedSection] = []
    tokens = 0
    for section_id in order:
        if len(picked) >= settings.answer_sections:
            break
        section, title, revision = found[section_id]
        if picked and tokens + section.token_count > settings.answer_source_tokens:
            continue
        tokens += section.token_count
        picked.append(
            RetrievedSection(
                section_id=section.id,
                section_number=section.section_number,
                heading_path=section.heading_path,
                page_start=section.page_start,
                page_end=section.page_end,
                text=section.text,
                token_count=section.token_count,
                safety_level=section.safety_level,
                document_title=title,
                revision=revision,
            )
        )
    return picked


def retrieve_sections(
    db: Session,
    query: str,
    keywords: list[str],
    part_numbers: list[str],
    *,
    use_embeddings: bool = True,
) -> Retrieval:
    """Hybrid search over current handbook text; returns the parent sections of the best chunks.

    `query` is a full English sentence: the vector search and the similarity score use it.
    `keywords` and `part_numbers` feed the full-text search. When the query cannot be embedded,
    the text hits are still returned and `embedding_error` holds the error code. With AI off
    (`AI_ENABLED=false`) there is no vector search and no client is created. NUL and other
    control characters are removed from all inputs first, and search terms are never logged.
    """
    query = _clean(query)
    keywords = [_clean(keyword) for keyword in keywords]
    base = _handbook_chunks()
    if not db.scalar(select(func.count()).select_from(base.subquery())):
        raise errors.nothing_embedded()

    vector_hits: list[tuple[Chunk, float]] = []
    embedding_error: str | None = None
    if use_embeddings and get_settings().ai_enabled:
        llm = get_llm()  # only here: without embeddings no client is created
        active = base.where(
            Chunk.embedding_model == llm.embedding_model, Chunk.embedding.is_not(None)
        )
        if not db.scalar(select(func.count()).select_from(active.subquery())):
            raise errors.nothing_embedded()
        try:
            query_vector = llm.embed([query])[0]
        except AppError as exc:
            log.warning("query embedding failed: %s", exc.code)  # the code only, never the text
            embedding_error = exc.code
        else:
            db.execute(sql_text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            vector_hits = _vector_hits(db, active, query_vector)

    parts = [_clean(part).strip() for part in part_numbers]
    parts = [p for p in parts if p and len(p) <= MAX_PART_NUMBER_CHARS][:MAX_PART_NUMBERS]
    text_hits = _text_hits(db, base, _text_search_input(keywords, query), parts)

    # Merge both lists with RRF; the best 20 chunks, ties broken by chunk id
    vector_rank = {c.id: i + 1 for i, (c, _) in enumerate(vector_hits)}
    similarity = {c.id: 1 - dist for c, dist in vector_hits}
    text_rank = {c.id: i + 1 for i, c in enumerate(text_hits)}
    found = {c.id: c for c in [*(c for c, _ in vector_hits), *text_hits]}
    scores = {cid: _rrf_score(vector_rank.get(cid), text_rank.get(cid)) for cid in found}
    best = sorted(found, key=lambda cid: (-scores[cid], cid))[:20]
    sections = _pick_sections(db, [found[cid].section_id for cid in best])

    # The refusal score counts only chunks of the sections that are really returned
    returned = {section.section_id for section in sections}
    section_similarities = [similarity[c.id] for c, _ in vector_hits if c.section_id in returned]

    return Retrieval(
        chunks=[
            RetrievedChunk(
                chunk_id=cid,
                section_id=found[cid].section_id,
                vector_rank=vector_rank.get(cid),
                text_rank=text_rank.get(cid),
                similarity=similarity.get(cid),
                score=scores[cid],
            )
            for cid in best
        ],
        sections=sections,
        best_similarity=max(section_similarities, default=None),
        embedding_error=embedding_error,
    )
