"""Tests for vectorstore/fastembed_embeddings.py and the API's embedder choice.

The unit tests replace fastembed's `TextEmbedding`, so nothing is downloaded.
The `live` test loads the real weights and compares them with Ollama's:

    uv --directory backend run pytest tests/test_fastembed_embeddings.py --live
"""

import math
from collections.abc import Iterable, Iterator

import numpy as np
import pytest
from fastembed import TextEmbedding
from langchain_ollama import OllamaEmbeddings

from profile_rag.api import search
from profile_rag.settings import get_settings
from profile_rag.vectorstore import fastembed_embeddings, ollama_embeddings
from profile_rag.vectorstore.fastembed_embeddings import FastEmbedEmbeddings
from profile_rag.vectorstore.models import (
    MODELS,
    NOMIC_V15,
    QWEN3_06B,
    EmbeddingModel,
    OllamaModel,
)
from profile_rag.vectorstore.prefixed import PrefixedEmbeddings


class FakeTextEmbedding:
    """Records each text and returns a vector of the model's size per text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.seen: list[str] = []

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        for text in texts:
            self.seen.append(text)
            yield np.full(NOMIC_V15.dimensions, 0.5, dtype=np.float32)


@pytest.fixture
def fake_text_embedding(monkeypatch: pytest.MonkeyPatch) -> list[FakeTextEmbedding]:
    made: list[FakeTextEmbedding] = []

    def make(model_name: str) -> FakeTextEmbedding:
        made.append(FakeTextEmbedding(model_name))
        return made[-1]

    monkeypatch.setattr(fastembed_embeddings, "TextEmbedding", make)
    return made


def test_the_query_and_documents_get_the_models_prefixes(
    fake_text_embedding: list[FakeTextEmbedding],
) -> None:
    embeddings = fastembed_embeddings.for_model(NOMIC_V15)

    query = embeddings.embed_query("who?")
    documents = embeddings.embed_documents(["a chunk"])

    (model,) = fake_text_embedding
    assert model.model_name == "nomic-ai/nomic-embed-text-v1.5"
    assert model.seen == ["search_query: who?", "search_document: a chunk"]
    assert isinstance(embeddings, PrefixedEmbeddings)
    assert len(query) == NOMIC_V15.dimensions
    assert all(isinstance(value, float) for value in query)
    assert len(documents) == 1


def test_a_model_without_fastembed_weights_is_a_setup_problem() -> None:
    problem = fastembed_embeddings.model_problem(QWEN3_06B)
    assert problem is not None
    assert "PROFILE_RAG_EMBEDDER=ollama" in problem
    assert fastembed_embeddings.model_problem(NOMIC_V15) is None
    with pytest.raises(ValueError, match="qwen3-06b"):
        fastembed_embeddings.for_model(QWEN3_06B)


def test_the_api_with_fastembed_never_asks_ollama(
    monkeypatch: pytest.MonkeyPatch, fake_text_embedding: list[FakeTextEmbedding]
) -> None:
    def no_ollama(*args: object) -> None:
        raise AssertionError("Ollama was asked")

    monkeypatch.setattr(ollama_embeddings, "model_problem", no_ollama)
    monkeypatch.setattr(ollama_embeddings, "for_model", no_ollama)
    settings = get_settings().model_copy(update={"embedder": "fastembed"})

    assert search._embedder_problem(settings, NOMIC_V15) is None
    assert isinstance(search._embeddings(settings, NOMIC_V15), PrefixedEmbeddings)
    assert search._embedder_problem(settings, QWEN3_06B) is not None


def test_the_api_uses_ollama_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    asked: list[str] = []

    def record(model: EmbeddingModel, url: str) -> None:
        asked.append(url)

    monkeypatch.setattr(ollama_embeddings, "model_problem", record)
    settings = get_settings()
    assert settings.embedder == "ollama"
    search._embedder_problem(settings, NOMIC_V15)
    assert asked == [settings.ollama_url]


TEXTS = [
    "search_query: What did he build with React Native?",
    "search_document: Cashflowy is a personal finance app built with React Native.",
    "search_document: He led the migration of the payments service to Kubernetes.",
]


@pytest.mark.live
@pytest.mark.parametrize(
    "model",
    [m for m in MODELS.values() if isinstance(m, OllamaModel) and m.fastembed_name],
    ids=lambda m: m.key,
)
def test_fastembed_vectors_match_ollamas(model: OllamaModel) -> None:
    assert model.fastembed_name is not None
    ours = FastEmbedEmbeddings(TextEmbedding(model.fastembed_name)).embed_documents(TEXTS)
    theirs = OllamaEmbeddings(
        model=model.ollama_tag, base_url=get_settings().ollama_url
    ).embed_documents(TEXTS)
    for a, b in zip(ours, theirs, strict=True):
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert math.isclose(cosine, 1.0, abs_tol=1e-3), cosine
