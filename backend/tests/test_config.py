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


@pytest.mark.parametrize("value", ["", "/custom/models"])
def test_models_directory(value: str) -> None:
    settings = Settings(_env_file=None, postgres_password="x", docling_models_dir=value)  # type: ignore[call-arg,arg-type]
    assert settings.docling_models_dir == (value or "/opt/docling-models")
