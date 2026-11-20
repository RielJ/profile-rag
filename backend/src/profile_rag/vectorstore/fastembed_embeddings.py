"""The fastembed side of an embedding model: embeddings computed in-process.

For a deployed server, where no Ollama runs. fastembed runs the model's
ONNX weights on the CPU, so no request leaves the server and nothing is
rate-limited. `PROFILE_RAG_EMBEDDER=fastembed` makes the API use it
(api/search.py); indexing still runs through Ollama.

The vectors must match the ones in the table, which Ollama computed. For
`nomic-embed-text-v1.5` they do: on 2026-11-12, fastembed 0.9.1 and Ollama's
`nomic-embed-text:v1.5` gave a cosine similarity of 1.0000 for every test
text. The quantized `-Q` variant gave about 0.97, too far off to search a
table Ollama filled, so models.py names the full model only.

fastembed downloads the weights from Hugging Face on first use, into
`FASTEMBED_CACHE_PATH` (a temporary directory if unset). The Docker image
downloads them at build time, so the server never needs the network for it.

- `model_problem(model)` is None when `model` has a fastembed name.
- `for_model(model)` loads the weights and returns the model's embeddings,
  with the query and document prefixes it needs (prefixed.py).

Tests: backend/tests/test_fastembed_embeddings.py.
"""

from typing import override

from fastembed import TextEmbedding
from langchain_core.embeddings import Embeddings

from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.prefixed import PrefixedEmbeddings


class FastEmbedEmbeddings(Embeddings):
    """LangChain `Embeddings` over a fastembed `TextEmbedding`, texts sent unchanged."""

    def __init__(self, model: TextEmbedding) -> None:
        self.model = model

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self.model.embed(texts)]

    @override
    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def model_problem(model: EmbeddingModel) -> str | None:
    """What stops `model` from being used through fastembed, or None."""
    if model.fastembed_name is None:
        return (
            f"{model.key} has no fastembed model in vectorstore/models.py. Set"
            " PROFILE_RAG_EMBEDDER=ollama, or choose a model that has one."
        )
    return None


def for_model(model: EmbeddingModel) -> Embeddings:
    """The embeddings of `model` through fastembed, with its prefixes.

    Loads the weights, downloading them first if they are not in the cache.
    Raises ValueError if `model` has no fastembed name.
    """
    if model.fastembed_name is None:
        raise ValueError(f"{model.key} has no fastembed model")
    return PrefixedEmbeddings(FastEmbedEmbeddings(TextEmbedding(model.fastembed_name)), model)
