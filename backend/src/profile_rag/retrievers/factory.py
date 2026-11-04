"""Build the retriever for a search mode.

`profile-rag search` and `profile-rag eval` both call `build_retriever`, so
an evaluation measures the same retriever, with the same settings, that a
search runs.

Tests: backend/tests/test_cli_search.py and backend/tests/test_eval_cli.py.
"""

from typing import Literal

from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVectorStore

from profile_rag.retrievers import dense, hybrid
from profile_rag.vectorstore.models import EmbeddingModel

SearchMode = Literal["dense", "hybrid", "mmr"]

# Candidates each search looks at before it keeps `k`: MMR's `fetch_k`, and
# hybrid search's dense and keyword lists. Never fewer than `k`.
CANDIDATES = 20
MMR_LAMBDA = 0.5


def build_retriever(
    mode: SearchMode, store: PGVectorStore, k: int, *, conninfo: str, model: EmbeddingModel
) -> BaseRetriever:
    """The retriever for `mode` on `store`, returning at most `k` visible chunks.

    `store` is `model`'s table, opened for search
    (`vectorstore.store.open_search_store`). Hybrid mode also reads that
    table with plain SQL through `conninfo`. Every mode applies the
    `dense.VISIBLE` filter.
    """
    candidates = max(CANDIDATES, k)
    if mode == "dense":
        return dense.dense_retriever(store, k=k, filter=dense.VISIBLE)
    if mode == "mmr":
        return dense.mmr_retriever(
            store, k=k, fetch_k=candidates, lambda_mult=MMR_LAMBDA, filter=dense.VISIBLE
        )
    return hybrid.HybridRetriever(
        store=store,
        conninfo=conninfo,
        model=model,
        k=k,
        k_dense=candidates,
        k_keyword=candidates,
        filter=dense.VISIBLE,
    )
