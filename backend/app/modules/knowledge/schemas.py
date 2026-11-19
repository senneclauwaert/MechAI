"""Request and response shapes of the knowledge module."""

from pydantic import BaseModel, Field


class PlaygroundTextIn(BaseModel):
    title: str = Field(min_length=1, max_length=300, examples=["Rear axle bearing"])
    text: str = Field(min_length=1, max_length=20_000)


class PlaygroundTextOut(BaseModel):
    chunk_id: str
    job_id: str


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
