"""Tests for retrievers/dense.py and retrievers/hybrid.py.

The table is built once for this file, in a throwaway database, by the
`fill_search_table` fixture in conftest.py. The tests open it with
`store.open_search_store`, without a `HybridSearchConfig`, as the CLI does.
No Ollama is needed.

The vectors come from `DeterministicFakeEmbedding`, which hashes each text
to made-up values. Its "closest" chunk has nothing to do with meaning, so
these tests never expect a dense ranking. They check counts, filters,
keyword hits, and fusion. Identical texts get identical vectors, which the
MMR test uses.
"""

import asyncio
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_postgres import PGEngine, PGVectorStore

from profile_rag.retrievers.dense import VISIBLE, dense_retriever, mmr_retriever
from profile_rag.retrievers.hybrid import HybridRetriever
from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.store import open_search_store

SMALL = EmbeddingModel(
    key="small-8",
    ollama_tag="small:8",
    digest="0" * 64,
    dimensions=8,
    table_name="chunks_small_8",
    document_prefix="",
    query_prefix="",
    max_words=1000,
)

FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]


def chunk(chunk_id: str, text: str, *, visibility: str = "public") -> Document:
    source_id = chunk_id.split(":")[0]
    return Document(
        page_content=text,
        metadata={
            "source_id": source_id,
            "chunk_id": chunk_id,
            "attribution": "self_authored",
            "visibility": visibility,
            "title": source_id,
            "heading_path": ["Dana Whitlock"],
            "ordinal": 0,
        },
    )


DUPLICATE = "Dana mentors two junior engineers every week."
CHUNKS = [
    chunk("resume:v1:a", "Dana Whitlock is a backend engineer who writes Python and Go."),
    chunk("resume:v1:b", "Moved four services from cron jobs to a PostgreSQL queue."),
    chunk("resume:v1:c", DUPLICATE),
    chunk("resume:v1:d", DUPLICATE),
    chunk("lantern:v1:a", "lantern-queue is a job queue for PostgreSQL written in Go."),
    chunk("lantern:v1:b", "Install lantern-queue with pip, then start one worker."),
    chunk("talks:v1:a", "A talk on Kubernetes autoscaling.", visibility="sanitized"),
    # The registry never allows this value. The row is here to be filtered out.
    chunk("drafts:v1:a", "Draft notes on a Terraform migration.", visibility="internal"),
]
ALL_IDS = {doc.metadata["chunk_id"] for doc in CHUNKS}


def ids(docs: list[Document]) -> list[str]:
    return [doc.metadata["chunk_id"] for doc in docs]


@pytest.fixture(scope="module")
def search_store(
    temporary_database_factory: Callable[[], AbstractContextManager[str]],
    fill_search_table: FillSearchTable,
    url_of: Callable[[str], str],
) -> Iterator[PGVectorStore]:
    """One filled table for every test here. The tests only read it."""
    with temporary_database_factory() as conninfo:
        embeddings = DeterministicFakeEmbedding(size=SMALL.dimensions)
        fill_search_table(conninfo, SMALL, embeddings, CHUNKS)
        engine = PGEngine.from_connection_string(url=url_of(conninfo))
        try:
            yield open_search_store(engine, SMALL, embeddings)
        finally:
            asyncio.run(engine.close())


# --- dense_retriever ----------------------------------------------------------------


def test_dense_returns_k_chunks_with_their_metadata(search_store: PGVectorStore) -> None:
    docs = dense_retriever(search_store, k=3).invoke("Who is Dana?")
    assert len(docs) == 3
    for doc in docs:
        assert doc.metadata["chunk_id"] in ALL_IDS
        assert doc.metadata["heading_path"] == ["Dana Whitlock"]


def test_dense_follows_the_store_distance_order(search_store: PGVectorStore) -> None:
    question = "Where does Dana use PostgreSQL?"
    scored = search_store.similarity_search_with_score(question, k=4)
    docs = dense_retriever(search_store, k=4).invoke(question)
    # Contents, not IDs: the two duplicate rows tie, in either order.
    assert [doc.page_content for doc in docs] == [doc.page_content for doc, _ in scored]


def test_dense_without_a_filter_returns_every_row(search_store: PGVectorStore) -> None:
    docs = dense_retriever(search_store, k=50).invoke("Who is Dana?")
    assert set(ids(docs)) == ALL_IDS


def test_the_visible_filter_drops_other_visibility(search_store: PGVectorStore) -> None:
    docs = dense_retriever(search_store, k=50, filter=VISIBLE).invoke("Who is Dana?")
    assert set(ids(docs)) == ALL_IDS - {"drafts:v1:a"}
    assert {doc.metadata["visibility"] for doc in docs} == {"public", "sanitized"}


def test_a_filter_on_one_value(search_store: PGVectorStore) -> None:
    only = {"visibility": {"$eq": "sanitized"}}
    docs = dense_retriever(search_store, k=50, filter=only).invoke("Who is Dana?")
    assert ids(docs) == ["talks:v1:a"]


# --- mmr_retriever ------------------------------------------------------------------


def test_mmr_returns_k_different_chunks(search_store: PGVectorStore) -> None:
    retriever = mmr_retriever(search_store, k=3, fetch_k=8, lambda_mult=0.5)
    assert retriever.search_type == "mmr"
    docs = retriever.invoke("Who is Dana?")
    assert len(docs) == 3
    assert len(set(ids(docs))) == 3


def test_mmr_skips_a_duplicate_that_dense_search_keeps(search_store: PGVectorStore) -> None:
    duplicates = {"resume:v1:c", "resume:v1:d"}
    # The question is the duplicated text, so both copies are at distance 0.
    dense = dense_retriever(search_store, k=2).invoke(DUPLICATE)
    assert set(ids(dense)) == duplicates

    # Below 0.5, a copy of a picked chunk always loses to any other chunk.
    # Not 0: the store treats lambda_mult=0 as "use the default 0.5".
    mmr = mmr_retriever(search_store, k=2, fetch_k=8, lambda_mult=0.25).invoke(DUPLICATE)
    assert len(mmr) == 2
    assert len(set(ids(mmr)) & duplicates) == 1


# --- HybridRetriever ----------------------------------------------------------------


def test_hybrid_returns_k_chunks(search_store: PGVectorStore) -> None:
    docs = HybridRetriever(store=search_store, k=3).invoke("Kubernetes")
    assert len(docs) == 3
    assert len(set(ids(docs))) == 3


def test_hybrid_puts_the_keyword_hit_first(search_store: PGVectorStore) -> None:
    # Every row is a dense candidate (k_dense=20 > 8 rows), so each one scores
    # at most 1/60 from dense search. Only the keyword hit adds a second 1/60.
    docs = HybridRetriever(store=search_store, k=3).invoke("Kubernetes")
    assert ids(docs)[0] == "talks:v1:a"


def test_two_different_hybrid_queries_in_a_row(search_store: PGVectorStore) -> None:
    retriever = HybridRetriever(store=search_store, k=3)
    assert ids(retriever.invoke("Kubernetes"))[0] == "talks:v1:a"
    # A reused HybridSearchConfig still searches for "Kubernetes" here.
    assert ids(retriever.invoke("pip"))[0] == "lantern:v1:b"
    assert ids(retriever.invoke("Terraform"))[0] == "drafts:v1:a"


def test_a_question_no_chunk_fully_matches_gives_the_dense_order(
    search_store: PGVectorStore,
) -> None:
    # plainto_tsquery joins the words with AND: 'languag' & 'dana' & 'use' & 'work'.
    question = "What languages does Dana use at work?"
    hybrid = HybridRetriever(store=search_store, k=4).invoke(question)
    dense = dense_retriever(search_store, k=4).invoke(question)
    assert [doc.page_content for doc in hybrid] == [doc.page_content for doc in dense]
