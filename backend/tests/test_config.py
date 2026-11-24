"""Settings: validation of the values read from the environment."""

import pytest
from pydantic import ValidationError

from app.config import Settings


def test_password_with_special_characters_is_escaped() -> None:
    settings = Settings(_env_file=None, postgres_password="p@ss/w:rd")  # type: ignore[call-arg,arg-type]
    assert "p%40ss%2Fw%3Ard" in settings.db_url


def test_chunk_tokens_must_be_in_range() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", chunk_tokens=0)  # type: ignore[call-arg,arg-type]


def test_external_embeddings_are_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_EXTERNAL_EMBEDDINGS", raising=False)
    settings = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg,arg-type]
    assert settings.allow_external_embeddings is False


def test_max_pdf_pages_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", max_pdf_pages=0)  # type: ignore[call-arg,arg-type]


@pytest.mark.parametrize(("tokens", "overlap"), [(50, 50), (300, 301)])
def test_overlap_must_be_less_than_chunk_size(tokens: int, overlap: int) -> None:
    with pytest.raises(ValidationError, match="strictly less"):
        Settings(
            _env_file=None, postgres_password="x", chunk_tokens=tokens, chunk_overlap_tokens=overlap
        )  # type: ignore[call-arg,arg-type]


def test_default_chunk_sizes_are_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHUNK_TOKENS", raising=False)
    monkeypatch.delenv("CHUNK_OVERLAP_TOKENS", raising=False)
    settings = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg,arg-type]
    assert (settings.chunk_tokens, settings.chunk_overlap_tokens) == (300, 50)


def test_assistant_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "REFUSAL_THRESHOLD",
        "ANSWER_SECTIONS",
        "ANSWER_SOURCE_TOKENS",
        "ANSWER_MAX_TOKENS",
        "AI_TIMEOUT_SECONDS",
        "DAILY_QUESTION_CAP",
    ]:
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg,arg-type]
    assert (s.refusal_threshold, s.answer_sections, s.answer_source_tokens) == (0.70, 5, 6000)
    assert (s.answer_max_tokens, s.ai_timeout_seconds, s.daily_question_cap) == (800, 20, 300)


@pytest.mark.parametrize(
    "field",
    [
        {"refusal_threshold": -0.1},
        {"refusal_threshold": 1.1},
        {"answer_sections": 0},
        {"answer_sections": 11},
        {"answer_source_tokens": 499},
        {"answer_source_tokens": 20001},
        {"answer_max_tokens": 99},
        {"answer_max_tokens": 4001},
        {"ai_timeout_seconds": 0},
        {"ai_timeout_seconds": 121},
        {"daily_question_cap": 0},
        {"daily_question_cap": 100001},
    ],
)
def test_assistant_settings_out_of_range(field: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", **field)  # type: ignore[call-arg,arg-type]


@pytest.mark.parametrize("value", ["", "/custom/models"])
def test_models_directory(value: str) -> None:
    settings = Settings(_env_file=None, postgres_password="x", docling_models_dir=value)  # type: ignore[call-arg,arg-type]
    assert settings.docling_models_dir == (value or "/opt/docling-models")
