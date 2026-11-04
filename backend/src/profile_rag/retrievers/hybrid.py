"""Hybrid retrieval: dense search and keyword search, fused by rank.

Dense search finds chunks that mean something close to the question, even
with other words. Keyword search finds chunks that contain the question's
words, such as a project name the embedding model has never seen.

The dense half is the store's own similarity search. The keyword half is
`keyword.keyword_search`, which matches any word of the question.
`PGVectorStore` can run a keyword search too, but its query joins every word
with AND (`plainto_tsquery`), so a whole question almost never matches (0 of
14 example questions did). Version 0.0.19 has no option to change that.

Both lists go through reciprocal rank fusion here, in Python.

Tests: backend/tests/test_retrievers.py.
"""

from collections.abc import Mapping
from typing import override

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVectorStore

from profile_rag.retrievers.keyword import keyword_search
from profile_rag.vectorstore.models import EmbeddingModel

# The constant of reciprocal rank fusion: a chunk at rank r (0 for the first)
# of one list adds 1 / (r + RRF_K) to its score. 60 is the usual value.
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], *, k: int, rrf_k: int = RRF_K
) -> list[Document]:
    """The `k` best chunks of all lists by summed `1 / (rank + rrf_k)`, rank from 0.

    Chunks are the same when their `chunk_id` metadata is. A chunk in only
    one list still counts. Equal scores keep the order in which the chunks
    first appear, going through the lists in turn.
    """
    scores: dict[str, float] = {}
    first: dict[str, Document] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.metadata["chunk_id"]
            first.setdefault(key, doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rrf_k)
    order = sorted(scores, key=lambda key: scores[key], reverse=True)
    return [first[key] for key in order[:k]]


class HybridRetriever(BaseRetriever):
    """Dense and keyword search on one model's table, fused with reciprocal rank fusion.

    `BaseRetriever` is a Pydantic model, so the fields below are its
    constructor arguments:
    `HybridRetriever(store=store, conninfo=url, model=model, k=8)`.
    """

    store: PGVectorStore  # the model's table, opened for search (`open_search_store`)
    conninfo: str  # the same database, for psycopg (keyword search)
    model: EmbeddingModel  # names the table that keyword search reads
    k: int = 8  # chunks returned after fusion
    k_dense: int = 20  # candidates from dense search
    k_keyword: int = 20  # candidates from keyword search
    # Metadata filter applied to both searches, such as `dense.VISIBLE`.
    # `None` means no filter.
    filter: Mapping[str, object] | None = None

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return at most `k` chunks, best fused score first.

        A chunk that only one search found still counts. If no chunk shares
        a word with `query` (or `query` has only stopwords), the keyword list
        is empty and the result is the dense order. Called by `invoke`;
        `run_manager` is LangChain's callback hook, unused here.
        """
        filter = dict(self.filter) if self.filter is not None else None
        dense = self.store.similarity_search(query, k=self.k_dense, filter=filter)
        keyword = keyword_search(
            self.conninfo, self.model, query, k=self.k_keyword, filter=self.filter
        )
        return reciprocal_rank_fusion([dense, keyword], k=self.k)
