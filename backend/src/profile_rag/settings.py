"""Typed application settings.

Values come from environment variables first, then from the `.env` file at the
repository root. A missing required value stops the program at startup with a
clear error, instead of failing later in the middle of a command.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/src/profile_rag/settings.py -> parents[3] is the repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]

CONFIG = SettingsConfigDict(
    env_file=REPO_ROOT / ".env",
    env_file_encoding="utf-8",
    extra="ignore",  # .env may hold settings other tools read
)


class Settings(BaseSettings):
    model_config = CONFIG

    database_url: PostgresDsn
    test_database_url: PostgresDsn | None = None

    # The local Ollama server that computes embeddings.
    ollama_url: str = "http://127.0.0.1:11434"

    # Pi, which runs the answer model (generation/pi_rpc.py). `pi_bin` is a
    # command name on PATH or a path to the executable. The variables have a
    # PROFILE_RAG_ prefix because Pi itself exports PI_MODEL, PI_PROVIDER, and
    # others to every command it runs. Unprefixed, a `profile-rag ask` started
    # from a Pi shell would take the parent session's model over .env.
    pi_bin: str = Field(default="pi", validation_alias="PROFILE_RAG_PI_BIN")
    pi_model: str = Field(
        default="anthropic/claude-opus-5-5", validation_alias="PROFILE_RAG_PI_MODEL"
    )
    pi_timeout_seconds: float = Field(
        default=120.0, gt=0, validation_alias="PROFILE_RAG_PI_TIMEOUT_SECONDS"
    )

    # Cloudflare Workers AI, which embeds for the models that have a
    # `workers_ai_name` (vectorstore/workers_ai_embeddings.py). The token needs
    # the "Workers AI: Read" permission. Ollama models do not need either
    # value.
    cf_account_id: str | None = Field(default=None, validation_alias="PROFILE_RAG_CF_ACCOUNT_ID")
    cf_api_token: SecretStr | None = Field(
        default=None, validation_alias="PROFILE_RAG_CF_API_TOKEN"
    )


@lru_cache
def get_settings() -> Settings:
    """Return the settings, loaded once per process."""
    # Values come from the environment; Pyright does not know pydantic-settings.
    return Settings()  # pyright: ignore[reportCallIssue]


class TracingSettings(BaseSettings):
    """The LangSmith settings that tracing.py reads.

    Separate from `Settings` so commands that need no database still start
    when DATABASE_URL is missing. The names are the standard ones the
    langsmith library reads, so they have no PROFILE_RAG_ prefix. Only
    LANGSMITH_TRACING turns tracing on; see tracing.py.
    """

    model_config = CONFIG

    langsmith_tracing: bool = Field(default=False, validation_alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
