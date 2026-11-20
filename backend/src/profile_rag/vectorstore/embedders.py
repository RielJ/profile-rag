"""Which service embeds for a model: Ollama, fastembed, or Workers AI.

Each `EmbeddingModel` (models.py) is an `OllamaModel` or a `WorkersAIModel`.
A `WorkersAIModel` always goes to Workers AI, because nothing else has its
weights. An `OllamaModel` goes to `local`: Ollama, the CLI's only local
embedder, or fastembed, which the API can use instead (settings.py,
`PROFILE_RAG_EMBEDDER`).

- `model_problem(model, settings, local=...)` says what stops the model
  from being used, or returns None. For Ollama it asks the server for the
  digest; for Workers AI it checks only that the credentials are set.
- `for_model(model, settings, local=...)` returns the model's embeddings,
  with its prefixes. It does not call any service.

The functions call `ollama_embeddings.for_model` and the others through
their modules, so tests that replace those still work through this module.

Tests: backend/tests/test_embedders.py.
"""

from typing import Literal

from langchain_core.embeddings import Embeddings

from profile_rag.settings import Settings
from profile_rag.vectorstore import (
    fastembed_embeddings,
    ollama_embeddings,
    workers_ai_embeddings,
)
from profile_rag.vectorstore.models import EmbeddingModel, WorkersAIModel

type LocalEmbedder = Literal["ollama", "fastembed"]


def model_problem(
    model: EmbeddingModel, settings: Settings, *, local: LocalEmbedder = "ollama"
) -> str | None:
    """What stops `model` from being embedded with these settings, or None."""
    if isinstance(model, WorkersAIModel):
        return workers_ai_embeddings.model_problem(
            model, settings.cf_account_id, settings.cf_api_token
        )
    match local:
        case "ollama":
            return ollama_embeddings.model_problem(model, settings.ollama_url)
        case "fastembed":
            return fastembed_embeddings.model_problem(model)


def for_model(
    model: EmbeddingModel, settings: Settings, *, local: LocalEmbedder = "ollama"
) -> Embeddings:
    """The embeddings of `model`, with its prefixes, from the service that runs it."""
    if isinstance(model, WorkersAIModel):
        return workers_ai_embeddings.for_model(model, settings.cf_account_id, settings.cf_api_token)
    match local:
        case "ollama":
            return ollama_embeddings.for_model(model, settings.ollama_url)
        case "fastembed":
            return fastembed_embeddings.for_model(model)
