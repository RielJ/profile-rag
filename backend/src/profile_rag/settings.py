"""Typed application settings.

Values come from environment variables first, then from the `.env` file at the
repository root. A missing required value stops the program at startup with a
clear error, instead of failing later in the middle of a request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # Seconds to wait for a pooled connection before reporting the database
    # as unavailable.
    db_connect_timeout: float = Field(default=2.0, gt=0)

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

    # Which backend runs the answer model (generation/chat_models.py): Pi, as
    # above, or an OpenAI-compatible HTTP API such as Groq or OpenRouter, for
    # a deployed server where Pi cannot run. The PROFILE_RAG_LLM_* settings
    # apply to the `openai` backend only.
    llm: Literal["pi", "openai"] = Field(default="pi", validation_alias="PROFILE_RAG_LLM")
    llm_base_url: str = Field(
        default="https://api.groq.com/openai/v1", validation_alias="PROFILE_RAG_LLM_BASE_URL"
    )
    llm_model: str = Field(default="openai/gpt-oss-120b", validation_alias="PROFILE_RAG_LLM_MODEL")
    llm_api_key: SecretStr | None = Field(default=None, validation_alias="PROFILE_RAG_LLM_API_KEY")
    llm_timeout_seconds: float = Field(
        default=60.0, gt=0, validation_alias="PROFILE_RAG_LLM_TIMEOUT_SECONDS"
    )

    # The search behind `POST /api/ask` (api/search.py): the embedding model
    # key (vectorstore/models.py), and the retriever mode and `k` used when a
    # request does not set them. The modes are retrievers/factory.py's.
    api_model: str = Field(default="nomic-v15", validation_alias="PROFILE_RAG_API_MODEL")
    api_mode: Literal["dense", "hybrid", "mmr"] = Field(
        default="hybrid", validation_alias="PROFILE_RAG_API_MODE"
    )
    api_k: int = Field(default=8, ge=1, le=20, validation_alias="PROFILE_RAG_API_K")
    # Who may ask the API, and how often (api/access.py). Unset means no
    # check, which suits a local server.
    access_key: SecretStr | None = Field(default=None, validation_alias="PROFILE_RAG_ACCESS_KEY")
    client_questions_per_hour: int | None = Field(
        default=None, ge=1, validation_alias="PROFILE_RAG_CLIENT_QUESTIONS_PER_HOUR"
    )
    questions_per_day: int | None = Field(
        default=None, ge=1, validation_alias="PROFILE_RAG_QUESTIONS_PER_DAY"
    )

    # What computes the API's question embeddings: the local Ollama server, or
    # fastembed inside the process (vectorstore/fastembed_embeddings.py), for
    # a deployed server without Ollama. Indexing uses Ollama. A model with a
    # `workers_ai_name` ignores this setting and indexing's Ollama: only
    # Workers AI has its weights (vectorstore/embedders.py).
    embedder: Literal["ollama", "fastembed"] = Field(
        default="ollama", validation_alias="PROFILE_RAG_EMBEDDER"
    )

    # Cloudflare Workers AI, which embeds for the models that have a
    # `workers_ai_name` (vectorstore/workers_ai_embeddings.py), in the CLI and
    # the API. The token needs the "Workers AI: Read" permission. Ollama and
    # fastembed models do not need either value.
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
