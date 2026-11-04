"""Typed application settings.

Values come from environment variables first, then from the `.env` file at the
repository root. A missing required value stops the program at startup with a
clear error, instead of failing later in the middle of a request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/profile_rag/settings.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",  # .env may hold settings other tools read
    )

    database_url: PostgresDsn
    test_database_url: PostgresDsn | None = None

    # Seconds to wait for a pooled connection before reporting the database
    # as unavailable.
    db_connect_timeout: float = Field(default=2.0, gt=0)

    # The local Ollama server that computes embeddings.
    ollama_url: str = "http://127.0.0.1:11434"


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    return Settings()  # values come from the environment
