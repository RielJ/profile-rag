"""Hybrid retrieval: dense search and keyword search, fused by rank.

Dense search finds chunks that mean something close to the question, even
with other words. Keyword search finds chunks that contain the question's
words, such as a project name the embedding model has never seen.
`PGVectorStore` runs both when it gets a `HybridSearchConfig`, and a fusion
function merges the two lists.

`PGVectorStore` writes the first query into a config's `fts_query` field and
`fetch_top_k` into its `fusion_function_parameters`. A reused config then
searches every later query with the first query's keywords. So
`HybridRetriever` builds a new config on every call and passes it to a store
opened without a config of its own.

Tests: backend/tests/test_retrievers.py.
"""

from collections.abc import Mapping
from typing import override

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVectorStore
from langchain_postgres.v2.hybrid_search_config import (
    HybridSearchConfig,
    reciprocal_rank_fusion,
)

from profile_rag.vectorstore.store import TSV_COLUMN

# The constant of reciprocal rank fusion: a chunk at rank r (0 for the first)
# of one list adds 1 / (r + RRF_K) to its score. 60 is the usual value.
RRF_K = 60


class HybridRetriever(BaseRetriever):
    """Dense and keyword search on one store, fused with reciprocal rank fusion.

    `BaseRetriever` is a Pydantic model, so the fields below are its
    constructor arguments: `HybridRetriever(store=store, k=8)`.
    """

    store: PGVectorStore  # opened without a HybridSearchConfig (`open_search_store`)
    k: int = 8  # chunks returned after fusion
    k_dense: int = 20  # candidates from dense search (`primary_top_k`)
    k_keyword: int = 20  # candidates from keyword search (`secondary_top_k`)
    # Metadata filter applied to both searches, such as `dense.VISIBLE`.
    # `None` means no filter.
    filter: Mapping[str, object] | None = None

    def _config(self) -> HybridSearchConfig:
        # A new object on every call: the store writes the query into it.
        return HybridSearchConfig(  # pyrefly: ignore[direct-abstract-base-instantiation]
            tsv_column=TSV_COLUMN,
            fusion_function=reciprocal_rank_fusion,
            fusion_function_parameters={"rrf_k": RRF_K},
            primary_top_k=self.k_dense,
            secondary_top_k=self.k_keyword,
        )

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """Return at most `k` chunks, best fused score first.

        A chunk that only one search found still counts. Keyword search uses
        `plainto_tsquery`, which joins the words with AND. If no chunk
        contains every word of `query`, the keyword list is empty and the
        result is the dense order. Called by `invoke`; `run_manager` is
        LangChain's callback hook, unused here.
        """
        return self.store.similarity_search(
            query,
            k=self.k,
            filter=dict(self.filter) if self.filter is not None else None,
            hybrid_search_config=self._config(),
        )
