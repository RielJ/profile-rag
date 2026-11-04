"""Opt-in live tests: one real embedding per model, through a running Ollama.

Skipped unless pytest runs with `--live`:

    uv --directory backend run pytest tests/test_ollama_live.py --live

Each model must be pulled first (`ollama pull nomic-embed-text:v1.5`, and
`ollama pull qwen3-embedding:0.6b`). These tests call `OllamaEmbeddings`
directly, not `PrefixedEmbeddings`, so they check the server and the models
alone.
"""

import math

import ollama
import pytest
from langchain_ollama import OllamaEmbeddings

from profile_rag.settings import get_settings
from profile_rag.vectorstore.models import MODELS, EmbeddingModel

pytestmark = pytest.mark.live


@pytest.mark.parametrize("model", list(MODELS.values()), ids=lambda m: m.key)
def test_the_pulled_model_has_the_expected_digest(model: EmbeddingModel) -> None:
    listed = ollama.Client(host=get_settings().ollama_url).list()
    digests = {entry.model: (entry.digest or "").removeprefix("sha256:") for entry in listed.models}
    assert model.ollama_tag in digests, f"run `ollama pull {model.ollama_tag}`"
    assert digests[model.ollama_tag] == model.digest


@pytest.mark.parametrize("model", list(MODELS.values()), ids=lambda m: m.key)
def test_one_embedding_has_the_expected_dimensions(model: EmbeddingModel) -> None:
    embeddings = OllamaEmbeddings(model=model.ollama_tag, base_url=get_settings().ollama_url)
    vector = embeddings.embed_query(model.query_prefix + "Who is Dana Whitlock?")
    assert len(vector) == model.dimensions
    assert all(math.isfinite(value) for value in vector)
