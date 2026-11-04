"""Tests for the Pi settings in settings.py.

Pi exports PI_MODEL, PI_PROVIDER, and others to every command it runs. The
settings read PROFILE_RAG_PI_* only, so a `profile-rag ask` started from a Pi
shell keeps the configured model.
"""

import pytest
from pydantic import PostgresDsn
from pydantic_settings import SettingsConfigDict

from profile_rag.settings import Settings

DATABASE_URL = "postgresql://user:password@127.0.0.1:5443/db"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("PROFILE_RAG_PI_BIN", "PROFILE_RAG_PI_MODEL", "PROFILE_RAG_PI_TIMEOUT_SECONDS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", DATABASE_URL)


class EnvironmentOnly(Settings):
    """Settings that ignore the repository's .env file."""

    # Pydantic merges this with the parent's config, replacing env_file only.
    model_config = SettingsConfigDict(env_file=None)


def load() -> Settings:
    return EnvironmentOnly(database_url=PostgresDsn(DATABASE_URL))


def test_a_parent_pi_sessions_variables_do_not_change_the_pi_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("PI_BIN", "/elsewhere/pi")
    monkeypatch.setenv("PI_TIMEOUT_SECONDS", "1")
    settings = load()
    assert settings.pi_model == "anthropic/claude-opus-5-5"
    assert settings.pi_bin == "pi"
    assert settings.pi_timeout_seconds == 120.0


def test_the_prefixed_variables_set_the_pi_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PI_MODEL", "parent-session-model")
    monkeypatch.setenv("PROFILE_RAG_PI_MODEL", "anthropic/claude-sonnet-5-5")
    monkeypatch.setenv("PROFILE_RAG_PI_BIN", "/opt/pi/bin/pi")
    monkeypatch.setenv("PROFILE_RAG_PI_TIMEOUT_SECONDS", "30")
    settings = load()
    assert settings.pi_model == "anthropic/claude-sonnet-5-5"
    assert settings.pi_bin == "/opt/pi/bin/pi"
    assert settings.pi_timeout_seconds == 30.0
