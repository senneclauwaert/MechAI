"""App settings, read from environment variables (or the .env file in the repo root).

Every other file gets its settings through get_settings(), never from os.environ.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL

# The repo root, so `uvicorn` started from backend/ still finds ../.env
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


# One field per environment variable (the name in upper case, e.g. CHUNK_TOKENS).
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(ROOT_ENV, ".env"), extra="ignore")

    # dev turns on the dev-only endpoints; prod hides them.
    environment: Literal["dev", "prod"] = "dev"

    # Database. Inside Docker the host is "db"; on your laptop it is "localhost".
    postgres_user: str = "mechai"
    postgres_password: SecretStr  # no default: must be set in .env
    postgres_db: str = "mechai"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None  # set this to override the parts above

    # Mistral
    mistral_api_key: SecretStr | None = None
    chat_model: str = "mistral-small-latest"
    embed_model: str = "mistral-embed"
    embed_dimensions: int = Field(default=1024, ge=1024, le=1024)
    stt_model: str = "voxtral-mini-latest"
    ai_enabled: bool = True

    # Files (the Docker volume)
    files_dir: str = "/data/files"

    # Handbook ingestion
    chunk_tokens: int = Field(default=300, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)
    section_max_tokens: int = Field(default=1500, ge=300, le=8000)
    max_pdf_mb: int = Field(default=50, ge=1, le=50)
    page_image_dpi: int = Field(default=150, ge=72, le=300)
    max_pdf_pages: int = Field(default=500, ge=1, le=5000)
    # Handbook text goes to the AI provider only after Supercube's written OK; it also needs
    # ai_enabled. Ingestion refuses a real client when either one is off, and the retrieval of
    # sections skips the query embedding when ai_enabled is off (text search only).
    # It covers ALL handbook text sent to Mistral (embeddings and chat); the name stays for now.
    allow_external_embeddings: bool = False
    docling_models_dir: str = "/opt/docling-models"

    # Assistant
    # Below this best similarity score the answer is "not in the handbook".
    # 0.70 is a placeholder until it is measured on real questions.
    refusal_threshold: float = Field(default=0.70, ge=0, le=1)
    answer_sections: int = Field(default=5, ge=1, le=10)  # handbook sections given to the AI
    answer_source_tokens: int = Field(default=6000, ge=500, le=20000)  # max source text per answer
    answer_max_tokens: int = Field(default=800, ge=100, le=4000)  # max tokens the AI may write
    ai_timeout_seconds: int = Field(default=20, ge=1, le=120)  # per chat call
    daily_question_cap: int = Field(default=300, ge=1, le=100000)  # questions per day, all users

    # Worker
    worker_poll_seconds: float = 5.0
    job_max_attempts: int = 3

    # Switches for later issues: login mode and the Supercube API client.
    auth_mode: str = "dev"  # dev or pin
    supercube_client: str = "fake"  # fake or real

    # Validators: fix up or reject values after they are read.
    @field_validator("docling_models_dir")
    @classmethod
    def empty_models_dir_uses_baked_models(cls, value: str) -> str:
        # Compose can override the image environment with an empty value.
        return value or "/opt/docling-models"

    @model_validator(mode="after")
    # The overlap between two chunks must be smaller than a chunk, or chunking never moves on.
    def validate_chunk_overlap(self) -> "Settings":
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError("chunk_overlap_tokens must be strictly less than chunk_tokens")
        return self

    # The SQLAlchemy connection string: DATABASE_URL if set, else built from the parts.
    @property
    def db_url(self) -> str:
        if self.database_url:
            return self.database_url
        # URL.create escapes special characters (an "@" or "/" in the password)
        url = URL.create(
            "postgresql+psycopg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )
        return url.render_as_string(hide_password=False)


# Read the settings once and reuse the same object; tests patch fields on it.
@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # postgres_password comes from .env
