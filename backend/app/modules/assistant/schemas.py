"""Shapes of what the model returns for the assistant: the translation and the answer draft.

Unknown keys in the model output are ignored (the Pydantic default). The checks in `checks.py`
and the pipeline decide what happens with a draft; these shapes only validate its form.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class Translation(BaseModel):
    """Result of the translate call: language, English search texts and the model's safety flag."""

    question_language: Literal["nl", "fr", "en"]
    # The whole question as one natural English sentence (vector search and similarity score).
    question_en: str = Field(max_length=500)
    search_terms_en: str = Field(max_length=500)
    keywords_en: list[Annotated[str, Field(max_length=100)]] = Field(max_length=20)
    part_numbers: list[Annotated[str, Field(max_length=50)]] = Field(max_length=10)
    # Required on purpose: a missing flag makes the translation invalid (stricter level wins).
    safety_flag: Literal["normal", "high_voltage", "emergency"]


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


class AnswerDraft(BaseModel):
    """The model's answer before the checks: steps, parts, warnings and what is missing."""

    steps: list[StepDraft] = Field(max_length=20)
    parts: list[PartDraft] = Field(max_length=20)
    warnings: list[WarningDraft] = Field(max_length=10)
    confidence: Literal["low", "medium", "high"]
    missing_information: list[Annotated[str, Field(max_length=300)]] = Field(max_length=10)
