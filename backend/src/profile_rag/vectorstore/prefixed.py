"""Add each model's prefixes, and check every vector before the store sees it.

`OllamaEmbeddings` sends the text unchanged and returns whatever the server
answers: no prefix, no check of the vector count, the vector length, or the
values. `PrefixedEmbeddings` wraps it (or any other `Embeddings`) and is the
only place where the document path and the query path differ.

`PGVectorStore` calls `embed_documents` when it stores chunks and
`embed_query` when it searches. The async methods call these two by default.

Words are checked before the call, so a long text never reaches Ollama:
`OllamaEmbeddings` does not pass Ollama's `truncate` option, and the
server's default cuts long input without an error. After the call, the
number of vectors, each vector's length, and each value are checked.

Tests: backend/tests/test_prefixed_embeddings.py.
"""

import math

from langchain_core.embeddings import Embeddings

from profile_rag.vectorstore.models import EmbeddingModel


class EmbeddingError(ValueError):
    """One text could not be embedded safely.

    `position` is the text's index in the list passed to `embed_documents`
    (0 for `embed_query`), or `None` when the problem is not one text's,
    such as a wrong number of vectors. `text` is the text as the caller
    passed it, without the prefix, or `None` together with `position`. The
    CLI uses `text` to name the chunk.
    """

    def __init__(
        self, reason: str, *, position: int | None = None, text: str | None = None
    ) -> None:
        where = "" if position is None else f"text {position}: "
        super().__init__(f"{where}{reason}")
        self.reason = reason
        self.position = position
        self.text = text


def _check_words(text: str, max_words: int, position: int | None = None) -> None:
    if max_words < len(text.split()):
        raise EmbeddingError("too many words", position=position, text=text)


def _check_vector(
    vector: list[float], dimensions: int, text: str | None = None, position: int | None = None
) -> None:
    if len(vector) != dimensions:
        raise EmbeddingError("wrong vector length", position=position, text=text)
    for value in vector:
        if not math.isfinite(value):
            raise EmbeddingError("non-finite value", position=position, text=text)


class PrefixedEmbeddings(Embeddings):
    """`Embeddings` that adds `model`'s prefixes and checks every result."""

    def __init__(self, inner: Embeddings, model: EmbeddingModel) -> None:
        self.inner = inner
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks for storage, one vector per text, in order.

        1. If a text has more than `model.max_words` words (`len(text.split())`,
           prefix not counted), raise `EmbeddingError` with its `position` and
           `text`. The inner object is not called.
        2. An empty `texts` returns `[]` without calling the inner object.
        3. Call `inner.embed_documents` once, with `model.document_prefix`
           before every text.
        4. If the number of vectors differs from the number of texts, raise
           `EmbeddingError` with `position=None`.
        5. If a vector does not have `model.dimensions` values, or one of its
           values is not finite (`NaN`, `inf`), raise `EmbeddingError` with
           that text's `position` and `text`.

        Return the inner object's vectors unchanged.
        """
        for position, text in enumerate(texts):
            _check_words(text, self.model.max_words, position)
        if texts == []:
            return []
        vectors = self.inner.embed_documents([self.model.document_prefix + t for t in texts])
        if len(vectors) != len(texts):
            raise EmbeddingError("wrong number of vectors", position=None)
        for position, (text, vector) in enumerate(zip(texts, vectors, strict=True)):
            _check_vector(vector, self.model.dimensions, text, position)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a question for search.

        The same word check and vector checks as `embed_documents`, with
        `position=0`. Call `inner.embed_query` once, with `model.query_prefix`
        before the text. Return its vector unchanged.
        """
        _check_words(text, self.model.max_words, 0)
        vector = self.inner.embed_query(self.model.query_prefix + text)
        _check_vector(vector, self.model.dimensions, text, 0)
        return vector
