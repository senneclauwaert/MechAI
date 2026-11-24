"""The question pipeline of the assistant: from a mechanic's question to a checked answer.

`ask()` follows these steps: code safety check on the question as typed, daily cap, translate
(one AI call), search (knowledge module), decide, write the answer, verify it, save the trace.
At the daily cap a high-voltage question skips the AI and gets pages from a keyword search.
Code decides first and the AI second; the stricter safety level always wins. The AI never writes
steps for high-voltage or emergency questions, and any failed check drops all generated steps.
Every question that gets past the access and cap checks leaves a trace row.

Logs never carry the question, the answer, a prompt or handbook text.
"""

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.errors import AppError
from app.core.logging import request_id_var
from app.llm import LLMClient, get_llm
from app.modules.assistant import checks, errors
from app.modules.assistant.models import Trace
from app.modules.assistant.schemas import (
    AnswerDraft,
    AnswerOut,
    AskOut,
    PartOut,
    SourceOut,
    StepOut,
    Translation,
)
from app.modules.knowledge import service as knowledge
from app.modules.knowledge.service import Retrieval, RetrievedSection

log = logging.getLogger("mechai.assistant")

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"
TRANSLATE_VERSION = "translate_v1"
ANSWER_VERSION = "answer_v1"

# Reason codes that are defined outside this module (llm.py and the knowledge module).
AI_UNAVAILABLE = "ASSISTANT_AI_UNAVAILABLE"
AI_DISABLED = "ASSISTANT_AI_DISABLED"
NOTHING_EMBEDDED = "KNOWLEDGE_NOTHING_EMBEDDED"
EXTERNAL_TEXT_OFF = "KNOWLEDGE_EXTERNAL_EMBEDDINGS_OFF"

# The translate answer is a small JSON object; 600 tokens leave room for 20 keywords and 10 part
# numbers (400 was too tight for long questions).
TRANSLATE_MAX_TOKENS = 600
TRACE_QUESTION_CHARS = 500
CLOSEST_SOURCES = 3  # sections shown when the answer is "not in the handbook"


def _read_prompt(version: str) -> str:
    """Read one prompt file (once, when this module is loaded)."""
    return (PROMPTS_DIR / f"{version}.md").read_text(encoding="utf-8")


TRANSLATE_PROMPT = _read_prompt(TRANSLATE_VERSION)
ANSWER_PROMPT = _read_prompt(ANSWER_VERSION)


# What we know about one question while it moves through the pipeline; it becomes the trace row.
@dataclass
class _Run:
    question: str  # cleaned (no NUL, no lone surrogates): for the model, the search and the trace
    raw_question: str  # as typed: only the code safety gate reads it, so nothing is hidden
    kart_number: str | None
    input_kind: str
    level: str = "normal"
    outcome: str = "retrieval_only"
    language: str | None = None
    reason: str | None = None
    search_terms: dict[str, Any] | None = None
    retrieval: Retrieval | None = None
    sections_sent: list[str] = field(default_factory=list)
    prompt_versions: dict[str, str] = field(default_factory=dict)
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency: dict[str, int] = field(default_factory=dict)
    draft: dict[str, Any] | None = None
    answer: AnswerOut | None = None
    flags: list[str] = field(default_factory=list)
    raise_after: AppError | None = None  # raised after the trace is saved


# ---- Cleaning text ----------------------------------------------------------------------------


def _safe(text: str) -> str:
    """Remove NUL characters and lone surrogates (Postgres and UTF-8 cannot hold them)."""
    return text.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")


def _safe_json(value: Any) -> Any:
    """Clean every string inside a JSON-like value; NaN and infinity become null (JSONB cannot)."""
    if isinstance(value, str):
        return _safe(value)
    if isinstance(value, list):
        return [_safe_json(item) for item in value]
    if isinstance(value, dict):
        return {_safe(str(key)): _safe_json(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_draft(draft: AnswerDraft) -> AnswerDraft | None:
    """The draft with every text cleaned, so the checks see exactly what the mechanic would see.

    None when cleaning makes it invalid (for example a step that was only NUL characters).
    """
    try:
        return AnswerDraft.model_validate(_safe_json(draft.model_dump()))
    except ValidationError:
        return None


def _ms(since: float) -> int:
    """Milliseconds since a time.monotonic() value."""
    return int((time.monotonic() - since) * 1000)


# ---- Messages for the model -------------------------------------------------------------------


def data_block(tag: str, text: str) -> str:
    """Wrap text in a tag; "<" is escaped so the text cannot close its own block."""
    return f"<{tag}>\n{text.replace('<', '&lt;')}\n</{tag}>"


def _source_block(source_id: str, section: RetrievedSection) -> str:
    """One handbook section as a data block with its id, heading and pages (all escaped)."""
    heading = section.heading_path.replace("<", "&lt;").replace('"', "&quot;")
    pages = ""
    if section.page_start is not None:
        pages = f"{section.page_start}-{section.page_end or section.page_start}"
    body = section.text.replace("<", "&lt;")
    return f'<source id="{source_id}" heading="{heading}" pages="{pages}">\n{body}\n</source>'


def _source_id(index: int) -> str:
    """The id of the n-th section sent to the model: S1, S2, ..."""
    return f"S{index + 1}"


def _translate_messages(question: str) -> list[dict[str, str]]:
    """System prompt plus the question in a data block."""
    return [
        {"role": "system", "content": TRANSLATE_PROMPT},
        {"role": "user", "content": data_block("question", question)},
    ]


def _answer_messages(question: str, sections: list[RetrievedSection]) -> list[dict[str, str]]:
    """System prompt plus the question and the numbered sections, each in a data block."""
    blocks = [data_block("question", question)]
    blocks += [_source_block(_source_id(i), s) for i, s in enumerate(sections)]
    return [
        {"role": "system", "content": ANSWER_PROMPT},
        {"role": "user", "content": "\n".join(blocks)},
    ]


def _check_sources(sections: list[RetrievedSection]) -> dict[str, checks.SourceText]:
    """What the answer check compares with: the original text, plus the section number, heading
    path and page range as a separate label (a number with a unit must be in the text)."""
    sources: dict[str, checks.SourceText] = {}
    for i, s in enumerate(sections):
        label = f"{s.section_number or ''} {s.heading_path} pages {s.page_start}-{s.page_end}"
        sources[_source_id(i)] = checks.SourceText(body=s.text, label=label)
    return sources


# ---- Calling the model ------------------------------------------------------------------------


def _parse[T: BaseModel](model: type[T], text: str) -> T | None:
    """Read the model's JSON into a schema; None when it is invalid JSON or has the wrong shape."""
    try:
        return model.model_validate_json(text)
    except ValidationError:
        return None


def _translate(run: _Run, llm: LLMClient | None) -> Translation | None:
    """One translate call. None means it failed or gave unusable output (no retry)."""
    started = time.monotonic()
    run.prompt_versions["translate"] = TRANSLATE_VERSION
    result: Translation | None = None
    if llm is not None:
        run.model = llm.chat_model  # set before the call: a failed call still counts for the cap
        try:
            reply = llm.chat(
                _translate_messages(run.question), json_output=True, max_tokens=TRANSLATE_MAX_TOKENS
            )
        except AppError as exc:
            log.warning("translate call failed: %s", exc.code)
        else:
            run.tokens_in += reply.tokens_in
            run.tokens_out += reply.tokens_out
            if not reply.truncated:
                result = _parse(Translation, reply.text)
    run.latency["translate"] = _ms(started)
    return result


def _write_answer(
    run: _Run, llm: LLMClient, sections: list[RetrievedSection]
) -> AnswerDraft | None:
    """The answer call, with one retry when the output is invalid or cut off.

    The draft is cleaned once here (see `_clean_draft`); the checks and the response use that
    version. Returns None and sets the reason when the call fails or the output stays invalid.
    """
    started = time.monotonic()
    run.prompt_versions["answer"] = ANSWER_VERSION
    run.model = llm.chat_model
    messages = _answer_messages(run.question, sections)
    draft: AnswerDraft | None = None
    run.reason = errors.answer_invalid().code
    for _attempt in range(2):
        try:
            reply = llm.chat(
                messages, json_output=True, max_tokens=get_settings().answer_max_tokens
            )
        except AppError as exc:
            log.warning("answer call failed: %s", exc.code)
            run.reason = AI_UNAVAILABLE
            break
        run.tokens_in += reply.tokens_in
        run.tokens_out += reply.tokens_out
        draft = None if reply.truncated else _parse(AnswerDraft, reply.text)
        if draft is not None:
            draft = _clean_draft(draft)
        if draft is not None:
            run.reason = None
            break
    run.latency["answer"] = _ms(started)
    return draft


# ---- Pipeline steps ---------------------------------------------------------------------------


def _check_daily_cap(db: Session) -> None:
    """Raise when today's questions that used the AI (since 00:00 UTC) reached the cap."""
    day_start = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    used = db.scalar(
        select(func.count())
        .select_from(Trace)
        .where(Trace.model.is_not(None), Trace.created_at >= day_start)
    )
    if (used or 0) >= get_settings().daily_question_cap:
        raise errors.daily_limit()


def _code_gate(run: _Run) -> bool:
    """Step 1: the word lists decide on the question as typed. True when the question is stopped."""
    run.level = checks.question_level(run.raw_question)
    if run.level == "emergency":
        run.outcome = "emergency"
        return True
    if checks.is_bypass(run.raw_question):
        run.outcome = "refused"
        return True
    return False


def _english_text(tr: Translation) -> str:
    """All English text the model produced, for the word-list checks."""
    parts = [tr.question_en, tr.search_terms_en, *tr.keywords_en, *tr.part_numbers]
    return " ".join(parts)


def _apply_translation(run: _Run, tr: Translation | None) -> bool:
    """Step 3: combine the levels (stricter wins). True when the question is stopped."""
    if tr is None:
        run.level = checks.strictest(run.level, "high_voltage")
        run.flags.append("translate_failed")
        run.reason = AI_UNAVAILABLE
        return False
    english = _english_text(tr)
    run.language = tr.question_language
    run.search_terms = {
        "question_en": tr.question_en,
        "search_terms_en": tr.search_terms_en,
        "keywords_en": tr.keywords_en,
        "part_numbers": tr.part_numbers,
        "safety_flag": tr.safety_flag,
        "is_bypass": tr.is_bypass,
        "is_wiring": tr.is_wiring,
    }
    run.level = checks.strictest(run.level, tr.safety_flag, checks.question_level(english))
    if run.level == "emergency":  # code emergency is checked before bypass
        run.outcome = "emergency"
        return True
    if tr.is_bypass or checks.is_bypass(english):  # the AI flag counts like the word list
        run.outcome = "refused"
        return True
    return False


def _search(
    db: Session, run: _Run, tr: Translation | None, *, use_embeddings: bool
) -> Retrieval | None:
    """Step 4: hybrid search. None when nothing is embedded and the question is already decided."""
    query = run.question
    keywords: list[str] = []
    part_numbers: list[str] = []
    if tr is not None:
        query = tr.question_en  # never blank: a blank translation is invalid (see Translation)
        keywords = [k for k in [*tr.keywords_en, tr.search_terms_en] if k.strip()]
        part_numbers = tr.part_numbers
    started = time.monotonic()
    try:
        retrieval = knowledge.retrieve_sections(
            db, query, keywords, part_numbers, use_embeddings=use_embeddings
        )
    except AppError as exc:
        if exc.code != NOTHING_EMBEDDED:
            raise
        run.reason = NOTHING_EMBEDDED
        if run.level == "high_voltage":  # the level found so far is not lost
            run.outcome = "high_voltage"
        else:
            run.outcome = "retrieval_only"
            run.raise_after = exc
        return None
    finally:
        run.latency["search"] = _ms(started)
    if use_embeddings and run.model is None:
        run.model = get_settings().chat_model  # the query embedding is an AI call for the cap
    run.retrieval = retrieval
    return retrieval


def _is_wiring(run: _Run, tr: Translation | None) -> bool:
    """True when the question as typed, its English terms or the AI flag say it is about wiring."""
    if checks.is_wiring(run.raw_question):
        return True
    return tr is not None and (tr.is_wiring or checks.is_wiring(_english_text(tr)))


def _decide(run: _Run, retrieval: Retrieval, wiring: bool, llm: LLMClient | None) -> bool:
    """Step 5: set the outcome when answering is not allowed. True when answering may go on."""
    if run.level == "high_voltage":
        run.outcome = "high_voltage"
        return False
    if wiring:
        run.outcome = "retrieval_only"
        return False
    if retrieval.embedding_error is not None:
        run.outcome = "retrieval_only"
        run.reason = retrieval.embedding_error
        return False
    best = retrieval.best_similarity
    # Written so that NaN also counts as "below the threshold".
    if not retrieval.sections or best is None or not (best >= get_settings().refusal_threshold):
        run.outcome = "not_in_handbook"
        return False
    if llm is None:
        run.outcome = "retrieval_only"
        run.reason = AI_UNAVAILABLE
        return False
    if not knowledge.external_text_allowed(llm):
        run.outcome = "retrieval_only"
        run.reason = EXTERNAL_TEXT_OFF
        return False
    return True


def _answer_step(run: _Run, llm: LLMClient, sections: list[RetrievedSection]) -> None:
    """Steps 6 and 7: write the answer, then verify it. Sets the outcome."""
    run.sections_sent = [str(s.section_id) for s in sections]
    if any(s.safety_level == "high_voltage" for s in sections):
        run.flags.append("hv_section_sent")  # a hint for the trace only
    draft = _write_answer(run, llm, sections)
    if draft is None:
        run.outcome = "retrieval_only"
        return
    run.draft = draft.model_dump()
    if not draft.steps:
        run.outcome = "not_in_handbook"
        return
    flags = checks.check_answer(draft, _check_sources(sections))
    if flags:  # any failed check drops ALL generated steps
        run.flags += flags
        if "unsafe_words" in flags:
            run.level = checks.strictest(run.level, "high_voltage")
            run.outcome = "high_voltage"
        else:
            run.outcome = "retrieval_only"
        return
    run.outcome = "answer"
    run.answer = AnswerOut(
        steps=[StepOut(**s.model_dump()) for s in draft.steps],
        parts=[PartOut(**p.model_dump()) for p in draft.parts],
        warnings=[StepOut(**w.model_dump()) for w in draft.warnings],
        confidence=draft.confidence,
        missing_information=draft.missing_information,
    )


def _run_without_ai(db: Session, run: _Run, reason: str) -> None:
    """Keyword search on the question only, no AI call of any kind (no chat, no embedding).

    Used when the AI is off and for a high-voltage question at the daily cap; `reason` tells which.
    """
    run.reason = reason
    retrieval = _search(db, run, None, use_embeddings=False)
    if retrieval is not None:
        run.outcome = "high_voltage" if run.level == "high_voltage" else "retrieval_only"


def _run_with_ai(db: Session, run: _Run) -> None:
    """Steps 2 to 7 with the AI on.

    At the daily cap a normal question gets the 429 at once (no trace). A high-voltage question
    is a safety question: it gets the pages of a keyword search (no AI call, not even the client
    is created), outcome `high_voltage` and the cap as reason code, and it is logged.
    """
    try:
        _check_daily_cap(db)  # before any AI call
    except AppError as exc:
        if run.level != "high_voltage":
            raise
        _run_without_ai(db, run, exc.code)
        return
    try:
        llm: LLMClient | None = get_llm()
    except AppError as exc:
        log.warning("no AI client: %s", exc.code)
        llm = None
    tr = _translate(run, llm)
    if _apply_translation(run, tr):
        return
    retrieval = _search(db, run, tr, use_embeddings=True)
    if retrieval is None:
        return
    if _decide(run, retrieval, _is_wiring(run, tr), llm) and llm is not None:
        _answer_step(run, llm, retrieval.sections)


# ---- Result and trace -------------------------------------------------------------------------


def _shown_sections(run: _Run) -> list[RetrievedSection]:
    """Sections to show as sources: none for emergency or refused, 3 when not in the handbook."""
    if run.retrieval is None or run.outcome in ("emergency", "refused"):
        return []
    sections = run.retrieval.sections
    return sections[:CLOSEST_SOURCES] if run.outcome == "not_in_handbook" else sections


def _result(run: _Run, trace_id: str | None) -> AskOut:
    """Build the response; the answer is only set for the outcome "answer"."""
    sources = [
        SourceOut(
            id=_source_id(i),
            section_id=s.section_id,
            section_number=s.section_number,
            heading_path=s.heading_path,
            page_start=s.page_start,
            page_end=s.page_end,
            document_title=s.document_title,
            revision=s.revision,
        )
        for i, s in enumerate(_shown_sections(run))
    ]
    return AskOut.model_validate(
        {
            "trace_id": trace_id,
            "outcome": run.outcome,
            "safety_level": run.level,
            "question_language": run.language,
            "answer": run.answer if run.outcome == "answer" else None,
            "sources": sources,
            "reason_code": run.reason,
        }
    )


def _new_trace(run: _Run) -> Trace:
    """The trace row of a question: ids and scores only, never section text. All text is cleaned."""
    retrieval = run.retrieval
    revisions = sorted(
        {(s.document_title, s.revision) for s in retrieval.sections} if retrieval else set(),
        key=lambda pair: (pair[0], pair[1] or ""),
    )
    retrieved = (
        [
            {
                "chunk_id": str(c.chunk_id),
                "vector_rank": c.vector_rank,
                "text_rank": c.text_rank,
                "similarity": c.similarity,
                "score": c.score,
            }
            for c in retrieval.chunks
        ]
        if retrieval
        else []
    )
    best = retrieval.best_similarity if retrieval else None
    return Trace(
        id=uuid.uuid4(),
        question=_safe(run.question)[:TRACE_QUESTION_CHARS],
        kart_number=_safe(run.kart_number) if run.kart_number else None,
        input=run.input_kind,
        question_language=run.language,
        safety_level=run.level,
        outcome=run.outcome,
        search_terms=_safe_json(run.search_terms),
        retrieved=_safe_json(retrieved),
        sections_sent=run.sections_sent,
        best_similarity=best if best is not None and math.isfinite(best) else None,
        prompt_versions=run.prompt_versions,
        model=run.model,
        document_revisions=_safe_json([{"title": t, "revision": r} for t, r in revisions]),
        tokens_in=run.tokens_in,
        tokens_out=run.tokens_out,
        latency_ms=run.latency,
        answer=_safe_json(run.draft),
        flags=sorted(set(run.flags)),
        error_code=run.reason if run.outcome != "answer" else None,
        request_id=(request_id_var.get() or "")[:20] or None,
    )


def _save_trace(db: Session, run: _Run) -> str | None:
    """Save and commit the trace; returns its id.

    For emergency and refused a database error is logged (no question text) and the answer is
    still returned without a trace id. For every other outcome the error goes up.
    """
    trace = _new_trace(run)
    try:
        db.add(trace)
        db.commit()
    except SQLAlchemyError as exc:
        db.rollback()
        if run.outcome in ("emergency", "refused"):
            # Only the class name: the text of a database error can hold the question.
            log.error("could not save the %s trace: %s", run.outcome, type(exc).__name__)
            return None
        raise
    return str(trace.id)


def _log_result(run: _Run, trace_id: str | None, started: float) -> None:
    """One log line per question: trace id, outcome, level, total ms. Never any text."""
    warn = run.outcome in ("emergency", "high_voltage", "refused")
    (log.warning if warn else log.info)(
        "question done: trace=%s outcome=%s level=%s total_ms=%d",
        trace_id,
        run.outcome,
        run.level,
        _ms(started),
    )


def ask(db: Session, question: str, kart_number: str | None, input_kind: str) -> AskOut:
    """Answer one question from the handbook, or explain why not, and save a trace.

    Raises ASSISTANT_NOT_ALLOWED in prod (until login exists, no trace) and
    ASSISTANT_DAILY_LIMIT when the cap is reached (no trace). A high-voltage question at the cap
    is not an error: it returns pages with the cap as reason code. KNOWLEDGE_NOTHING_EMBEDDED is
    raised after saving the trace, for a normal question when no handbook text is embedded.
    """
    if get_settings().environment != "dev":
        raise errors.not_allowed()
    started = time.monotonic()
    run = _Run(
        question=_safe(question),
        raw_question=question,
        kart_number=kart_number,
        input_kind=input_kind,
    )
    if not _code_gate(run):
        if get_settings().ai_enabled:
            _run_with_ai(db, run)
        else:
            _run_without_ai(db, run, AI_DISABLED)
    run.latency["total"] = _ms(started)
    trace_id = _save_trace(db, run)
    _log_result(run, trace_id, started)
    if run.raise_after is not None:
        raise run.raise_after
    return _result(run, trace_id)
