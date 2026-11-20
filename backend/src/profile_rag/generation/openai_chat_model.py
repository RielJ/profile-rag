"""`OpenAIChatModel`: the answer model over an OpenAI-compatible HTTP API.

For the deployed demo, where Pi cannot run: Groq, OpenRouter, and other
providers that speak the OpenAI chat completions API. It is `ChatOpenAI`
with streaming on, so each piece of the reply reaches the run's callbacks as
`on_llm_new_token` (the API's drafts, rag.DraftText), and the reply's
`response_metadata["model_name"]` names the model the provider used.

The only change to `ChatOpenAI`: errors from the `openai` library become
`errors.ModelTimeout` or `errors.ModelError`, so `rag.failure_code` reports
them as `pi_timeout` or `pi_failed`, the same as a failed Pi call. Their
messages name the error class and HTTP status only, never the provider's
response body, which may echo the prompt.

Tests: backend/tests/test_openai_chat_model.py.
"""

from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any, override

import openai
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

from profile_rag.generation.errors import ModelError, ModelTimeout
from profile_rag.settings import Settings


class OpenAIChatModel(ChatOpenAI):
    """`ChatOpenAI` whose failures raise `ModelError` and `ModelTimeout`."""

    @property
    @override
    def _llm_type(self) -> str:
        return "openai-compatible"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with _model_errors():
            return super()._generate(messages, stop, run_manager, **kwargs)

    # ChatOpenAI declares the stream methods as (*args, **kwargs).
    @override
    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        with _model_errors():
            yield from super()._stream(*args, **kwargs)

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        with _model_errors():
            return await super()._agenerate(messages, stop, run_manager, **kwargs)

    @override
    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        with _model_errors():
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk


@contextmanager
def _model_errors() -> Iterator[None]:
    try:
        yield
    except openai.APITimeoutError as error:
        raise ModelTimeout("the answer model did not reply in time") from error
    except openai.APIStatusError as error:
        raise ModelError(
            f"the answer model's API returned HTTP {error.status_code} ({type(error).__name__})"
        ) from error
    except openai.APIError as error:
        raise ModelError(f"the answer model's API failed ({type(error).__name__})") from error


def from_settings(settings: Settings) -> OpenAIChatModel:
    """The model that the `PROFILE_RAG_LLM_*` settings configure."""
    if settings.llm_api_key is None:
        raise ValueError("PROFILE_RAG_LLM_API_KEY is required when PROFILE_RAG_LLM=openai")
    return OpenAIChatModel(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=settings.llm_timeout_seconds,
        temperature=0,
        streaming=True,
        max_retries=1,
    )
