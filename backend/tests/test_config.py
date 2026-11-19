from app.config import Settings


def test_password_with_special_characters_is_escaped() -> None:
    settings = Settings(_env_file=None, postgres_password="p@ss/w:rd")  # type: ignore[call-arg,arg-type]
    assert "p%40ss%2Fw%3Ard" in settings.db_url
