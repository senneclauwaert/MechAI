"""All settings come from environment variables (or the .env file in the repo root)."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo root, so `uvicorn` started from backend/ still finds ../.env
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(ROOT_ENV, ".env"), extra="ignore")

    environment: str = "dev"  # dev or prod

    # Database. Inside Docker the host is "db"; on your laptop it is "localhost".
    postgres_user: str = "mechai"
    postgres_password: str = "mechai"
    postgres_db: str = "mechai"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None  # set this to override the parts above

    # Mistral
    mistral_api_key: str = ""
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    embed_dimensions: int = 1024
    stt_model: str = "voxtral-mini-latest"
    ai_enabled: bool = True

    # Files (the Docker volume)
    files_dir: str = "/data/files"

    # Worker
    worker_poll_seconds: float = 2.0
    job_max_attempts: int = 3

    auth_mode: str = "dev"  # dev or pin
    supercube_client: str = "fake"  # fake or real

    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
