"""The answer model that the settings choose: Pi or an OpenAI-compatible API.

`PROFILE_RAG_LLM=pi` (the default) runs Pi, for local use with your own
logins. `PROFILE_RAG_LLM=openai` calls an OpenAI-compatible HTTP API, for a
deployed server. Both raise `errors.ModelError` subclasses on failure.

Tests: backend/tests/test_openai_chat_model.py.
"""

from langchain_core.language_models import BaseChatModel

from profile_rag.generation import openai_chat_model, pi_chat_model
from profile_rag.settings import Settings


def from_settings(settings: Settings) -> BaseChatModel:
    """The answer model of `settings.llm`."""
    match settings.llm:
        case "pi":
            return pi_chat_model.from_settings(settings)
        case "openai":
            return openai_chat_model.from_settings(settings)
