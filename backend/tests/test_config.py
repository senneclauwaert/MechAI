import pytest
from pydantic import ValidationError

from app.config import Settings


def test_password_with_special_characters_is_escaped() -> None:
    settings = Settings(_env_file=None, postgres_password="p@ss/w:rd")  # type: ignore[call-arg,arg-type]
    assert "p%40ss%2Fw%3Ard" in settings.db_url


def test_chunk_tokens_must_be_in_range() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", chunk_tokens=0)  # type: ignore[call-arg,arg-type]


def test_external_embeddings_are_off_by_default() -> None:
    settings = Settings(_env_file=None, postgres_password="x")  # type: ignore[call-arg,arg-type]
    assert settings.allow_external_embeddings is False


def test_max_pdf_pages_must_be_at_least_one() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", max_pdf_pages=0)  # type: ignore[call-arg,arg-type]
