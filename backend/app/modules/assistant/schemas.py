"""Shapes of the assistant: the model output (translation, draft) and the result of `ask()`.

Unknown keys in the model output are ignored (the Pydantic default). The checks in `checks.py`
and the pipeline decide what happens with a draft; these shapes only validate its form.
"""

import uuid
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)


class Translation(BaseModel):
    """Result of the translate call: language, English search texts and the model's flags."""

    question_language: Literal["nl", "fr", "en"]
    # The whole question as one natural English sentence (vector search and similarity score).
    question_en: str = Field(max_length=500)
    search_terms_en: str = Field(max_length=500)
    keywords_en: list[Annotated[str, Field(max_length=100)]] = Field(max_length=20)
    part_numbers: list[Annotated[str, Field(max_length=50)]] = Field(max_length=10)
    # Required on purpose: a missing flag makes the translation invalid (stricter level wins).
    safety_flag: Literal["normal", "high_voltage", "emergency"]
    # Also required, and a real true or false (not "false" or 0): the model says whether the
    # question asks to bypass a protection or open a battery pack, and whether it is about wiring.
    is_bypass: StrictBool
    is_wiring: StrictBool

    @model_validator(mode="after")
    def _needs_text(self) -> "Translation":
        """An empty translation is a failed one: the English question and some search text must
        hold real text (spaces only do not count)."""
        if not self.question_en.strip():
            raise ValueError("question_en is empty")
        if not self.search_terms_en.strip() and not any(k.strip() for k in self.keywords_en):
            raise ValueError("no search terms")
        return self


class StepDraft(BaseModel):
    """One repair step, with the sections it is based on (S1, S2, ...)."""

    text: str = Field(min_length=1, max_length=1000)
    sources: list[str] = Field(min_length=1)


class WarningDraft(BaseModel):
    """One warning, with the sections it is based on."""

    text: str = Field(min_length=1, max_length=1000)
    sources: list[str] = Field(min_length=1)


class PartDraft(BaseModel):
    """A part the repair needs, with the section that names it."""

    name: str = Field(min_length=1, max_length=200)
    part_number: str | None = Field(default=None, max_length=50)
    source: str

    @field_validator("part_number", mode="before")
    @classmethod
    def _no_number_is_none(cls, value: object) -> object:
        """The model sometimes writes "null" or "" for a part without a number: that is None."""
        if isinstance(value, str) and value.strip().casefold() in ("", "null"):
            return None
        return value


class AnswerDraft(BaseModel):
    """The model's answer before the checks: steps, parts, warnings and what is missing."""

    # What the model found in the sources. Only used to notice an answer without steps; it is
    # handbook text, so it is left out of every dump (trace and response).
    evidence: list[str] = Field(default_factory=list, max_length=20, exclude=True)
    steps: list[StepDraft] = Field(max_length=20)
    parts: list[PartDraft] = Field(max_length=20)
    warnings: list[WarningDraft] = Field(max_length=10)
    confidence: Literal["low", "medium", "high"]
    missing_information: list[Annotated[str, Field(max_length=300)]] = Field(max_length=10)


class AskIn(BaseModel):
    """The request of POST /ask. The question is stripped before the length check."""

    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=2, max_length=500)]
    kart_number: str | None = Field(
        default=None, min_length=1, max_length=10, pattern=r"^[0-9A-Za-z-]+$"
    )
    input: Literal["typed", "voice"] = "typed"


# What `ask()` returns to the caller (the endpoint of B6 sends it as it is).
class StepOut(BaseModel):
    """A step or a warning of a checked answer, with the sources (S1, ...) it is based on."""

    text: str
    sources: list[str]


class PartOut(BaseModel):
    """A part of a checked answer."""

    name: str
    part_number: str | None
    source: str


class AnswerOut(BaseModel):
    """A checked answer; only present when every check passed."""

    steps: list[StepOut]
    parts: list[PartOut]
    warnings: list[StepOut]
    confidence: Literal["low", "medium", "high"]
    missing_information: list[str]


class SourceOut(BaseModel):
    """A handbook section the answer step looked at: where to find it, never its text."""

    id: str  # S1 .. S5
    section_id: uuid.UUID
    section_number: str | None
    heading_path: str
    page_start: int | None
    page_end: int | None
    document_title: str
    revision: str | None


class AskOut(BaseModel):
    """The result of one question. `answer` is filled only when `outcome` is "answer"."""

    trace_id: str | None  # null only when saving an emergency or refused trace failed
    outcome: Literal[
        "answer", "not_in_handbook", "retrieval_only", "high_voltage", "emergency", "refused"
    ]
    safety_level: Literal["normal", "high_voltage", "emergency"]
    question_language: Literal["nl", "fr", "en"] | None
    answer: AnswerOut | None
    sources: list[SourceOut]
    reason_code: str | None
