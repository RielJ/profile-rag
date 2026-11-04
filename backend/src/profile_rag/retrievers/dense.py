"""Dense and MMR retrievers over a model's `PGVectorStore`.

A retriever takes a question and returns `list[Document]` without scores.
Both retrievers here wrap the store's own vector search, so each one is a
`VectorStoreRetriever`. To see the scores, call
`store.similarity_search_with_score(question, k=k)`. With the default cosine
strategy the score is a cosine distance: lower is closer.

Tests: backend/tests/test_retrievers.py.
"""

from collections.abc import Mapping

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_postgres import PGVectorStore

# Rows a visitor may see. The registry accepts only these two values, so
# every indexed chunk passes today. The filter keeps that true in the query
# itself if a third value is ever added.
VISIBLE: dict[str, object] = {"visibility": {"$in": ["public", "sanitized"]}}


def dense_retriever(
    store: PGVectorStore, *, k: int, filter: Mapping[str, object] | None = None
) -> VectorStoreRetriever:
    """A retriever that returns the `k` chunks closest to the question.

    The chunks come in order of cosine distance, closest first. `filter` is
    a metadata filter such as `VISIBLE`. Rows it rejects are never returned,
    even if fewer than `k` rows are left. `None` means no filter.
    """
    search_kwargs: dict[str, object] = {"k": k}
    if filter is not None:
        search_kwargs["filter"] = dict(filter)
    return store.as_retriever(search_type="similarity", search_kwargs=search_kwargs)


def mmr_retriever(
    store: PGVectorStore,
    *,
    k: int,
    fetch_k: int,
    lambda_mult: float,
    filter: Mapping[str, object] | None = None,
) -> VectorStoreRetriever:
    """A retriever that returns `k` chunks by maximal marginal relevance.

    The store fetches the `fetch_k` closest chunks, then picks `k` of them
    one at a time. Each pick weighs closeness to the question
    (`lambda_mult`) against likeness to the chunks already picked
    (`1 - lambda_mult`). `lambda_mult=1` gives the dense order; lower values
    give a more varied set. Do not pass 0: the store treats 0 as "use the
    default" and runs 0.5. `filter` works as in `dense_retriever` and applies
    to the `fetch_k` candidates.
    """
    if not 0 < lambda_mult <= 1:
        raise ValueError(f"lambda_mult must be in (0, 1], got {lambda_mult}")
    if fetch_k < k:
        raise ValueError(f"fetch_k ({fetch_k}) must be at least k ({k})")
    search_kwargs: dict[str, object] = {"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult}
    if filter is not None:
        search_kwargs["filter"] = dict(filter)
    return store.as_retriever(search_type="mmr", search_kwargs=search_kwargs)
