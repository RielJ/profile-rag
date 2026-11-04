"""Tests for `PrefixedEmbeddings` in vectorstore/prefixed.py.

No Ollama: a recording fake stands in for `OllamaEmbeddings`. It records every
call with the texts it received and returns made-up vectors, so each test can
check which prefix reached which method, and what happens to a bad vector.
"""

import asyncio
import math

import pytest
from langchain_core.embeddings import Embeddings

from profile_rag.vectorstore.models import NOMIC_V15, QWEN3_06B, EmbeddingModel
from profile_rag.vectorstore.prefixed import EmbeddingError, PrefixedEmbeddings

# A small model, so the fake's vectors stay short and readable.
TINY = EmbeddingModel(
    key="tiny",
    ollama_tag="tiny:1",
    digest="0" * 64,
    dimensions=3,
    table_name="chunks_tiny",
    document_prefix="doc: ",
    query_prefix="query: ",
    max_words=5,
)


class Recorder(Embeddings):
    """Records each call and returns one vector per text.

    `dimensions` sets the vector length. `count_delta` returns that many
    more (or, if negative, fewer) vectors than texts. `bad_value` replaces
    the first value of the vector at `bad_at`.
    """

    def __init__(
        self,
        dimensions: int,
        *,
        count_delta: int = 0,
        bad_at: int | None = None,
        bad_value: float = math.nan,
    ) -> None:
        self.dimensions = dimensions
        self.count_delta = count_delta
        self.bad_at = bad_at
        self.bad_value = bad_value
        self.calls: list[tuple[str, list[str]]] = []

    def _vectors(self, count: int) -> list[list[float]]:
        vectors = [[0.5 + i] * self.dimensions for i in range(count + self.count_delta)]
        if self.bad_at is not None:
            vectors[self.bad_at][0] = self.bad_value
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed_documents", list(texts)))
        return self._vectors(len(texts))

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(("embed_query", [text]))
        return self._vectors(1)[0]


def wrap(recorder: Recorder, model: EmbeddingModel = TINY) -> PrefixedEmbeddings:
    return PrefixedEmbeddings(recorder, model)


# --- Prefixes ---------------------------------------------------------------


def test_documents_get_the_document_prefix_in_one_call() -> None:
    recorder = Recorder(3)
    vectors = wrap(recorder).embed_documents(["Dana builds queues.", "She writes runbooks."])
    assert recorder.calls == [
        ("embed_documents", ["doc: Dana builds queues.", "doc: She writes runbooks."])
    ]
    assert vectors == [[0.5, 0.5, 0.5], [1.5, 1.5, 1.5]]


def test_query_gets_the_query_prefix_through_embed_query() -> None:
    recorder = Recorder(3)
    vector = wrap(recorder).embed_query("Who is Dana?")
    assert recorder.calls == [("embed_query", ["query: Who is Dana?"])]
    assert vector == [0.5, 0.5, 0.5]


def test_nomic_prefixes() -> None:
    recorder = Recorder(NOMIC_V15.dimensions)
    embeddings = wrap(recorder, NOMIC_V15)
    embeddings.embed_documents(["Dana builds queues."])
    embeddings.embed_query("Who is Dana?")
    assert recorder.calls == [
        ("embed_documents", ["search_document: Dana builds queues."]),
        ("embed_query", ["search_query: Who is Dana?"]),
    ]


def test_qwen3_documents_are_unchanged_and_queries_get_the_instruction() -> None:
    recorder = Recorder(QWEN3_06B.dimensions)
    embeddings = wrap(recorder, QWEN3_06B)
    embeddings.embed_documents(["Dana builds queues."])
    embeddings.embed_query("Who is Dana?")
    task = "Given a web search query, retrieve relevant passages that answer the query"
    assert recorder.calls == [
        ("embed_documents", ["Dana builds queues."]),
        ("embed_query", [f"Instruct: {task}\nQuery:Who is Dana?"]),
    ]


def test_async_methods_use_the_same_prefixes() -> None:
    recorder = Recorder(3)
    embeddings = wrap(recorder)
    asyncio.run(embeddings.aembed_documents(["a"]))
    asyncio.run(embeddings.aembed_query("b"))
    assert recorder.calls == [("embed_documents", ["doc: a"]), ("embed_query", ["query: b"])]


def test_an_empty_list_returns_nothing_without_a_call() -> None:
    recorder = Recorder(3)
    assert wrap(recorder).embed_documents([]) == []
    assert recorder.calls == []


def test_the_input_list_is_not_changed() -> None:
    texts = ["Dana builds queues."]
    wrap(Recorder(3)).embed_documents(texts)
    assert texts == ["Dana builds queues."]


# --- Word limit -------------------------------------------------------------


def test_a_text_at_the_word_limit_is_embedded() -> None:
    recorder = Recorder(3)
    wrap(recorder).embed_documents(["one two three four five"])
    assert len(recorder.calls) == 1


def test_the_prefix_does_not_count_toward_the_limit() -> None:
    # "query: " is one more word; five words of text must still pass.
    recorder = Recorder(3)
    wrap(recorder).embed_query("one two three four five")
    assert len(recorder.calls) == 1


def test_an_over_long_document_is_refused_before_any_call() -> None:
    recorder = Recorder(3)
    long_text = "one two three four five six"
    with pytest.raises(EmbeddingError) as caught:
        wrap(recorder).embed_documents(["short", long_text])
    assert caught.value.position == 1
    assert caught.value.text == long_text
    assert recorder.calls == []


def test_an_over_long_query_is_refused_before_any_call() -> None:
    recorder = Recorder(3)
    with pytest.raises(EmbeddingError) as caught:
        wrap(recorder).embed_query("one two three four five six")
    assert caught.value.position == 0
    assert recorder.calls == []


# --- Checks on the result ---------------------------------------------------


def test_a_wrong_number_of_vectors_has_no_position() -> None:
    with pytest.raises(EmbeddingError) as caught:
        wrap(Recorder(3, count_delta=-1)).embed_documents(["a", "b"])
    assert caught.value.position is None
    assert caught.value.text is None


def test_extra_vectors_are_refused_too() -> None:
    with pytest.raises(EmbeddingError):
        wrap(Recorder(3, count_delta=1)).embed_documents(["a", "b"])


def test_a_wrong_dimension_names_the_text() -> None:
    with pytest.raises(EmbeddingError) as caught:
        wrap(Recorder(4)).embed_documents(["a", "b"])
    assert caught.value.position == 0
    assert caught.value.text == "a"


def test_a_wrong_dimension_in_a_query_is_refused() -> None:
    with pytest.raises(EmbeddingError) as caught:
        wrap(Recorder(2)).embed_query("Who is Dana?")
    assert caught.value.position == 0
    assert caught.value.text == "Who is Dana?"


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_a_value_that_is_not_finite_names_the_text(bad_value: float) -> None:
    with pytest.raises(EmbeddingError) as caught:
        wrap(Recorder(3, bad_at=1, bad_value=bad_value)).embed_documents(["a", "b", "c"])
    assert caught.value.position == 1
    assert caught.value.text == "b"


def test_a_query_with_nan_is_refused() -> None:
    with pytest.raises(EmbeddingError):
        wrap(Recorder(3, bad_at=0)).embed_query("Who is Dana?")


def test_embedding_error_is_a_value_error() -> None:
    # The store and the CLI may catch ValueError; EmbeddingError is one.
    assert issubclass(EmbeddingError, ValueError)
    assert str(EmbeddingError("too long", position=2, text="x")) == "text 2: too long"


# --- The two models ----------------------------------------------------------


@pytest.mark.parametrize("model", [NOMIC_V15, QWEN3_06B], ids=lambda m: m.key)
def test_each_model_entry_is_complete(model: EmbeddingModel) -> None:
    assert ":" in model.ollama_tag  # an explicit tag, never an implicit :latest
    assert len(model.digest) == 64
    assert all(c in "0123456789abcdef" for c in model.digest)
    assert model.table_name == f"chunks_{model.key.replace('-', '_')}"
    assert model.max_words > 200  # above the splitter's default chunk size
