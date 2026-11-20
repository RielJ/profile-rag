"""Tests for vectorstore/embedders.py, and the demo check that uses the same split.

Every service is faked: the tests record which module's function was
called, so no Ollama or Cloudflare is needed.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from pydantic import SecretStr

from profile_rag import demo_check
from profile_rag.settings import Settings
from profile_rag.vectorstore import embedders, ollama_embeddings, workers_ai_embeddings
from profile_rag.vectorstore.models import BGE_M3, NOMIC_V15, EmbeddingModel

SECRET = "cf-token-in-test"


@pytest.fixture(autouse=True)
def no_cloudflare_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Settings read the environment; a shell with real credentials must not leak in."""
    monkeypatch.delenv("PROFILE_RAG_CF_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("PROFILE_RAG_CF_API_TOKEN", raising=False)


def settings(**values: str) -> Settings:
    return Settings.model_validate(
        {
            "database_url": "postgresql://u:p@127.0.0.1:5443/db",
            "ollama_url": "http://ollama.test",
            **values,
        }
    )


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace every service's two functions with recorders."""
    called: list[str] = []
    for name, module in [
        ("ollama", ollama_embeddings),
        ("workers_ai", workers_ai_embeddings),
    ]:

        def problem(model: EmbeddingModel, *args: object, _name: str = name) -> str:
            called.append(f"{_name}.model_problem")
            return _name

        def make(model: EmbeddingModel, *args: object, _name: str = name) -> Embeddings:
            called.append(f"{_name}.for_model")
            return DeterministicFakeEmbedding(size=model.dimensions)

        monkeypatch.setattr(module, "model_problem", problem)
        monkeypatch.setattr(module, "for_model", make)
    return called


def test_a_workers_ai_model_goes_to_workers_ai(calls: list[str]) -> None:
    assert embedders.model_problem(BGE_M3, settings()) == "workers_ai"
    embedders.for_model(BGE_M3, settings())
    assert calls == ["workers_ai.model_problem", "workers_ai.for_model"]


def test_an_ollama_model_goes_to_ollama(calls: list[str]) -> None:
    assert embedders.model_problem(NOMIC_V15, settings()) == "ollama"
    embedders.for_model(NOMIC_V15, settings())
    assert calls == ["ollama.model_problem", "ollama.for_model"]


def test_the_credentials_come_from_the_settings() -> None:
    configured = settings(PROFILE_RAG_CF_ACCOUNT_ID="acct", PROFILE_RAG_CF_API_TOKEN=SECRET)
    assert embedders.model_problem(BGE_M3, configured) is None
    missing = embedders.model_problem(BGE_M3, settings())
    assert missing is not None
    assert "PROFILE_RAG_CF_API_TOKEN" in missing
    assert SECRET not in repr(configured)


def test_demo_check_asks_cloudflare_credentials_not_ollama_for_a_workers_ai_model() -> None:
    def no_ollama(_url: str) -> dict[str, str]:
        raise AssertionError("must not ask Ollama")

    problems = demo_check.check_embedder(BGE_M3, settings(), no_ollama)
    assert len(problems) == 1
    assert "PROFILE_RAG_CF_ACCOUNT_ID" in problems[0]
    configured = settings(PROFILE_RAG_CF_ACCOUNT_ID="acct", PROFILE_RAG_CF_API_TOKEN=SECRET)
    assert demo_check.check_embedder(BGE_M3, configured, no_ollama) == []


def test_the_token_setting_is_secret() -> None:
    configured = settings(PROFILE_RAG_CF_API_TOKEN=SECRET)
    assert isinstance(configured.cf_api_token, SecretStr)
    assert configured.cf_api_token.get_secret_value() == SECRET
