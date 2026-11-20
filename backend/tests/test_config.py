import pytest
from pydantic import ValidationError

from app.config import Settings


def test_password_with_special_characters_is_escaped() -> None:
    settings = Settings(_env_file=None, postgres_password="p@ss/w:rd")  # type: ignore[call-arg,arg-type]
    assert "p%40ss%2Fw%3Ard" in settings.db_url


def test_chunk_tokens_must_be_in_range() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, postgres_password="x", chunk_tokens=0)  # type: ignore[call-arg,arg-type]
