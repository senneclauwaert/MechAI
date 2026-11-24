"""Request and response shapes of the knowledge module."""

from typing import Literal

from pydantic import BaseModel, Field


# Playground
class PlaygroundTextIn(BaseModel):
    title: str = Field(min_length=1, max_length=300, examples=["Rear axle bearing"])
    text: str = Field(min_length=1, max_length=20_000)


class PlaygroundTextOut(BaseModel):
    chunk_id: str
    job_id: str


# Search: one hit with its rank in each search list and the merged RRF score.
class SearchHit(BaseModel):
    chunk_id: str
    heading: str
    text: str
    vector_rank: int | None
    text_rank: int | None
    score: float


class SearchOut(BaseModel):
    question: str
    hits: list[SearchHit]


# Handbook upload and ingestion status
class DocumentOut(BaseModel):
    document_id: str
    job_id: str
    title: str
    revision: str | None
    already_ingested: bool


class DocumentStatusOut(BaseModel):
    document_id: str
    title: str
    revision: str | None
    status: Literal["queued", "running", "done", "failed"]
    error_code: str | None
    page_count: int
    section_count: int
    chunk_count: int
