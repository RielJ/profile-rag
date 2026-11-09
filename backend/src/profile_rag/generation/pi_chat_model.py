"""`PiChatModel`: a LangChain chat model that answers through Pi.

`_generate` joins the messages into one prompt, each under a role marker
such as `=== SYSTEM ===`, sends it with `pi_rpc.run_prompt`, and returns a
`ChatResult` with one `AIMessage`. Its `response_metadata` holds the model
Pi used (`model_name`, as `provider/model`), the model the provider
reported (`pi_response_model`), and the sections of Pi's system prompt
(`pi_system_sections`). Errors from
`pi_rpc` (`PiError` and its subclasses) pass through unchanged, so callers
can tell a timeout from a crash.

While Pi writes, each piece of reply text goes to the run's callbacks as
`on_llm_new_token`, so a caller can show a draft before the reply is
complete. The returned message is still the whole, final reply.

Pi runs one request per process and has no stop sequences: `stop` is
rejected rather than ignored.

Tests: backend/tests/test_pi_chat_model.py.
"""

from collections.abc import Sequence
from typing import override

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from profile_rag.generation.pi_rpc import PiConfig, run_prompt
from profile_rag.settings import Settings

ROLE_NAMES = {"system": "SYSTEM", "human": "USER", "ai": "ASSISTANT"}


def join_messages(messages: Sequence[BaseMessage]) -> str:
    """All messages as one prompt: a role marker line, then the message text.

    The prompt always starts with a marker, never with `/`, so Pi cannot
    read it as a slash command.
    """
    parts: list[str] = []
    for message in messages:
        role = ROLE_NAMES.get(message.type, message.type.upper())
        parts.append(f"=== {role} ===\n{message.text}")
    return "\n\n".join(parts)


class PiChatModel(BaseChatModel):
    """A chat model backed by one `pi --mode rpc` process per call."""

    config: PiConfig

    @property
    @override
    def _llm_type(self) -> str:
        return "pi-rpc"

    @property
    @override
    def _identifying_params(self) -> dict[str, object]:
        return {"model": self.config.model, "thinking": self.config.thinking}

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: object,
    ) -> ChatResult:
        if stop:
            raise ValueError("PiChatModel does not support stop sequences")
        on_text = run_manager.on_llm_new_token if run_manager is not None else None
        reply = run_prompt(join_messages(messages), self.config, on_text)
        message = AIMessage(
            content=reply.text,
            response_metadata={
                "model_name": reply.model,
                "pi_response_model": reply.response_model,
                "pi_system_sections": sorted(reply.system_sections),
            },
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def from_settings(settings: Settings) -> PiChatModel:
    """The Pi chat model that `.env` configures: model, Pi binary, and timeout."""
    config = PiConfig(
        model=settings.pi_model,
        executable=(settings.pi_bin,),
        timeout=settings.pi_timeout_seconds,
    )
    return PiChatModel(config=config)
