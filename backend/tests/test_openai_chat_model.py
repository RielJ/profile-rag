"""Tests for generation/openai_chat_model.py and generation/chat_models.py.

The provider is an httpx.MockTransport, so no request leaves the machine.
"""

import json
from collections.abc import Callable
from typing import Any, override

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from pydantic import PostgresDsn, SecretStr
from pydantic_settings import SettingsConfigDict

from profile_rag import rag
from profile_rag.generation import chat_models, openai_chat_model
from profile_rag.generation.errors import ModelError, ModelTimeout
from profile_rag.generation.openai_chat_model import OpenAIChatModel
from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import PiProcessError, PiTimeout
from profile_rag.settings import Settings

Handler = Callable[[httpx.Request], httpx.Response]

LLM_VARIABLES = (
    "PROFILE_RAG_LLM",
    "PROFILE_RAG_LLM_BASE_URL",
    "PROFILE_RAG_LLM_MODEL",
    "PROFILE_RAG_LLM_API_KEY",
    "PROFILE_RAG_LLM_TIMEOUT_SECONDS",
)


def sse(pieces: list[str], model: str = "provider/served-model") -> bytes:
    """An OpenAI chat completions stream that writes `pieces` in order."""
    events: list[dict[str, Any]] = [
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
        }
        for piece in pieces
    ]
    events.append(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
    )
    lines = [f"data: {json.dumps(event)}\n\n" for event in events] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def model_with(handler: Handler) -> OpenAIChatModel:
    return OpenAIChatModel(
        model="openai/gpt-oss-120b",
        base_url="https://llm.test/v1",
        api_key=SecretStr("test-key"),
        temperature=0,
        streaming=True,
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class Tokens(BaseCallbackHandler):
    def __init__(self) -> None:
        self.tokens: list[str] = []

    @override
    def on_llm_new_token(self, token: str | list[str | dict[str, Any]], **kwargs: Any) -> None:
        if isinstance(token, str):
            self.tokens.append(token)


def test_the_reply_streams_to_callbacks_and_names_the_served_model() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer test-key"
        body = sse(['{"status": "unknown", ', '"reason": "not in the evidence"}'])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    tokens = Tokens()
    result = model_with(handler).invoke("question?", config={"callbacks": [tokens]})

    assert isinstance(result, AIMessage)
    assert result.text == '{"status": "unknown", "reason": "not in the evidence"}'
    assert result.response_metadata["model_name"] == "provider/served-model"
    assert "".join(tokens.tokens) == result.text
    assert sent[0]["model"] == "openai/gpt-oss-120b"
    assert sent[0]["stream"] is True


def test_drafts_reach_the_api_through_draft_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = sse(['{"status": "answered", "answer": "He built ', 'Cashflowy."', "}"])
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    drafts: list[str] = []
    model_with(handler).invoke("q", config={"callbacks": [rag.DraftText(drafts.append)]})

    assert "".join(drafts) == "He built Cashflowy."


def test_an_http_error_raises_model_error_without_the_response_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "secret prompt echo"}})

    with pytest.raises(ModelError) as caught:
        model_with(handler).invoke("q")

    assert not isinstance(caught.value, ModelTimeout)
    assert "429" in str(caught.value)
    assert "secret prompt echo" not in str(caught.value)
    assert rag.failure_code(caught.value) == "pi_failed"


def test_a_timeout_raises_model_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(ModelTimeout) as caught:
        model_with(handler).invoke("q")

    assert rag.failure_code(caught.value) == "pi_timeout"


def test_a_connection_failure_raises_model_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ModelError) as caught:
        model_with(handler).invoke("q")

    assert rag.failure_code(caught.value) == "pi_failed"


def test_pi_errors_keep_their_failure_codes() -> None:
    assert rag.failure_code(PiTimeout(1.0)) == "pi_timeout"
    assert rag.failure_code(PiProcessError("x", returncode=1, stderr_tail="")) == "pi_failed"


class EnvironmentOnly(Settings):
    """Settings that ignore the repository's .env file."""

    model_config = SettingsConfigDict(env_file=None)


@pytest.fixture
def load_settings(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Settings]:
    for name in LLM_VARIABLES:
        monkeypatch.delenv(name, raising=False)

    def load(**environment: str) -> Settings:
        for name, value in environment.items():
            monkeypatch.setenv(name, value)
        return EnvironmentOnly(database_url=PostgresDsn("postgresql://u:p@127.0.0.1:5443/db"))

    return load


def test_pi_is_the_default_backend(load_settings: Callable[..., Settings]) -> None:
    assert isinstance(chat_models.from_settings(load_settings()), PiChatModel)


def test_the_openai_backend_reads_the_llm_settings(
    load_settings: Callable[..., Settings],
) -> None:
    settings = load_settings(
        PROFILE_RAG_LLM="openai",
        PROFILE_RAG_LLM_BASE_URL="https://openrouter.ai/api/v1",
        PROFILE_RAG_LLM_MODEL="some/model:free",
        PROFILE_RAG_LLM_API_KEY="k",
        PROFILE_RAG_LLM_TIMEOUT_SECONDS="5",
    )

    model = chat_models.from_settings(settings)

    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "some/model:free"
    assert model.openai_api_base == "https://openrouter.ai/api/v1"
    assert model.request_timeout == 5.0
    assert model.streaming is True


def test_the_openai_backend_needs_an_api_key(load_settings: Callable[..., Settings]) -> None:
    with pytest.raises(ValueError, match="PROFILE_RAG_LLM_API_KEY"):
        openai_chat_model.from_settings(load_settings(PROFILE_RAG_LLM="openai"))
