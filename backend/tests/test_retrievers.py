"""Tests for retrievers/dense.py, retrievers/keyword.py, and retrievers/hybrid.py.

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
import dataclasses
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from typing import NamedTuple

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_postgres import PGEngine, PGVectorStore

from profile_rag.retrievers.dense import VISIBLE, dense_retriever, mmr_retriever
from profile_rag.retrievers.hybrid import RRF_K, HybridRetriever, reciprocal_rank_fusion
from profile_rag.retrievers.keyword import allowed_values, keyword_search, or_query
from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.store import TABLE_NAMES, open_search_store

# Keyword search only reads tables in TABLE_NAMES, so this test model borrows
# a real table name. The database is a throwaway, so the name is free.
SMALL = EmbeddingModel(
    key="small-8",
    ollama_tag="small:8",
    digest="0" * 64,
    dimensions=8,
    table_name=min(TABLE_NAMES),
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


class SearchTable(NamedTuple):
    conninfo: str
    store: PGVectorStore


@pytest.fixture(scope="module")
def search_table(
    temporary_database_factory: Callable[[], AbstractContextManager[str]],
    fill_search_table: FillSearchTable,
    url_of: Callable[[str], str],
) -> Iterator[SearchTable]:
    """One filled table for every test here. The tests only read it."""
    with temporary_database_factory() as conninfo:
        embeddings = DeterministicFakeEmbedding(size=SMALL.dimensions)
        fill_search_table(conninfo, SMALL, embeddings, CHUNKS)
        engine = PGEngine.from_connection_string(url=url_of(conninfo))
        try:
            yield SearchTable(conninfo, open_search_store(engine, SMALL, embeddings))
        finally:
            asyncio.run(engine.close())


@pytest.fixture
def search_store(search_table: SearchTable) -> PGVectorStore:
    return search_table.store


@pytest.fixture
def conninfo(search_table: SearchTable) -> str:
    return search_table.conninfo


@pytest.fixture
def hybrid(search_table: SearchTable) -> Callable[..., HybridRetriever]:
    """`hybrid(k=3, ...)`: a HybridRetriever on the test table."""

    def make(**fields: object) -> HybridRetriever:
        return HybridRetriever.model_validate(
            {"store": search_table.store, "conninfo": search_table.conninfo, "model": SMALL}
            | fields
        )

    return make


def keyword(conninfo: str, question: str, filter: Mapping[str, object] | None = None) -> list[str]:
    return ids(keyword_search(conninfo, SMALL, question, k=50, filter=filter))


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


# --- keyword_search -----------------------------------------------------------------


def test_keyword_matches_any_word_of_a_question(conninfo: str) -> None:
    # No chunk has every word; each of these has at least one ("Go", "PostgreSQL").
    found = keyword(conninfo, "Does Dana write Go or PostgreSQL code?")
    assert set(found) == {"resume:v1:a", "resume:v1:b", "lantern:v1:a"} | {
        "resume:v1:c",
        "resume:v1:d",
    }


def test_keyword_ranks_chunks_with_more_question_words_first(conninfo: str) -> None:
    found = keyword(conninfo, "Which job queue is written in Go for PostgreSQL?")
    assert found[0] == "lantern:v1:a"


def test_keyword_stems_like_the_index(conninfo: str) -> None:
    # "services" and "jobs" are stored as 'servic' and 'job'.
    assert keyword(conninfo, "service job") == ["resume:v1:b", "lantern:v1:a"]


def test_a_stopword_only_question_has_no_keyword_hits(conninfo: str) -> None:
    assert keyword(conninfo, "What is it, and where was it?") == []
    assert keyword(conninfo, "") == []
    assert keyword(conninfo, "?!") == []


def test_keyword_applies_the_visibility_filter(conninfo: str) -> None:
    assert "drafts:v1:a" in keyword(conninfo, "Terraform migration")
    assert keyword(conninfo, "Terraform migration", filter=VISIBLE) == []
    assert keyword(conninfo, "Kubernetes", filter=VISIBLE) == ["talks:v1:a"]
    only_public = {"visibility": "public"}
    assert keyword(conninfo, "Kubernetes", filter=only_public) == []


GO = {"resume:v1:a", "lantern:v1:a"}  # the chunks with the word "Go"
PYTHON = {"resume:v1:a"}
KUBERNETES = {"talks:v1:a"}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("'; DROP TABLE chunks; --", set()),
        ("Robert'); DELETE FROM pg_class; --", set()),
        ("Go' | 'Kubernetes", GO | KUBERNETES),
        ("Go & !Python <-> (Kubernetes)", GO | PYTHON | KUBERNETES),
        ("back\\slash 'quote' \"double\" Go:*", GO),
        ("%s %(name)s {table} Go", GO),
    ],
)
def test_question_text_is_only_ever_data(conninfo: str, question: str, expected: set[str]) -> None:
    # Operators and quotes in the question are not tsquery or SQL syntax: the
    # result is what its plain words match, every one ORed.
    assert set(keyword(conninfo, question, filter=VISIBLE)) == expected
    assert keyword(conninfo, "Kubernetes") == ["talks:v1:a"]  # the table is still there


def test_or_query_quotes_each_lexeme() -> None:
    assert or_query(["go", "o'brien", "a\\b"]) == "'go' | 'o''brien' | 'a\\\\b'"


def test_keyword_refuses_a_table_outside_the_allowlist(conninfo: str) -> None:
    other = dataclasses.replace(SMALL, table_name="pg_class")
    with pytest.raises(ValueError, match="unknown chunk table"):
        keyword_search(conninfo, other, "Go", k=3)


def test_keyword_refuses_filters_it_cannot_apply() -> None:
    assert allowed_values(VISIBLE)["visibility"] == ["public", "sanitized"]
    assert allowed_values({"source_id": "resume"})["source_id"] == ["resume"]
    with pytest.raises(ValueError, match="cannot filter on 'title'"):
        allowed_values({"title": "x"})
    with pytest.raises(ValueError, match="does not support '\\$ne'"):
        allowed_values({"visibility": {"$ne": "internal"}})
    with pytest.raises(ValueError, match="strings only"):
        allowed_values({"visibility": {"$in": [1]}})


# --- reciprocal_rank_fusion ---------------------------------------------------------


def test_fusion_adds_the_reciprocal_ranks() -> None:
    a, b, c = (chunk(f"s:v1:{name}", name) for name in "abc")
    # a: 1/60 + 1/62; b: 1/61 + 1/60 (more than a, as 1/61 > 1/62); c: 1/61.
    fused = reciprocal_rank_fusion([[a, b], [b, c, a]], k=3)
    assert ids(fused) == ["s:v1:b", "s:v1:a", "s:v1:c"]
    assert RRF_K == 60


def test_fusion_keeps_k_and_breaks_ties_by_first_appearance() -> None:
    a, b = chunk("s:v1:a", "a"), chunk("s:v1:b", "b")
    assert ids(reciprocal_rank_fusion([[a], [b]], k=5)) == ["s:v1:a", "s:v1:b"]
    assert ids(reciprocal_rank_fusion([[a], [b]], k=1)) == ["s:v1:a"]
    assert reciprocal_rank_fusion([[], []], k=3) == []


# --- HybridRetriever ----------------------------------------------------------------


def test_hybrid_returns_k_chunks(hybrid: Callable[..., HybridRetriever]) -> None:
    docs = hybrid(k=3).invoke("Kubernetes")
    assert len(docs) == 3
    assert len(set(ids(docs))) == 3


def test_hybrid_puts_the_keyword_hit_first(hybrid: Callable[..., HybridRetriever]) -> None:
    # Every row is a dense candidate (k_dense=20 > 8 rows), so each one scores
    # at most 1/60 from dense search. Only the keyword hit adds a second 1/60.
    docs = hybrid(k=3).invoke("Kubernetes")
    assert ids(docs)[0] == "talks:v1:a"


def test_two_different_hybrid_queries_in_a_row(hybrid: Callable[..., HybridRetriever]) -> None:
    retriever = hybrid(k=3)
    assert ids(retriever.invoke("Kubernetes"))[0] == "talks:v1:a"
    # A retriever that kept state from the first query would still find "Kubernetes".
    assert ids(retriever.invoke("pip"))[0] == "lantern:v1:b"
    assert ids(retriever.invoke("Terraform"))[0] == "drafts:v1:a"


def test_a_question_spread_over_a_chunk_gets_keyword_hits(
    hybrid: Callable[..., HybridRetriever], conninfo: str
) -> None:
    # The old AND query needed 'instal' & 'lantern' & 'worker' & 'pip' &
    # 'kubernet' in one chunk. No chunk has them all; lantern:v1:b has four.
    question = "Can I install the lantern worker with pip on Kubernetes?"
    assert keyword(conninfo, question)[0] == "lantern:v1:b"
    docs = hybrid(k=2, k_dense=1).invoke(question)
    assert "lantern:v1:b" in ids(docs)


def test_a_stopword_only_question_gives_the_dense_order(
    hybrid: Callable[..., HybridRetriever], search_store: PGVectorStore
) -> None:
    question = "What is it, and where was it?"
    docs = hybrid(k=4).invoke(question)
    dense = dense_retriever(search_store, k=4).invoke(question)
    assert [doc.page_content for doc in docs] == [doc.page_content for doc in dense]


def test_hybrid_applies_the_filter_to_the_keyword_half(
    hybrid: Callable[..., HybridRetriever],
) -> None:
    # Unfiltered, the only keyword hit wins; filtered, it is gone from both halves.
    assert ids(hybrid(k=3).invoke("Terraform"))[0] == "drafts:v1:a"
    docs = hybrid(k=50, filter=VISIBLE).invoke("Terraform")
    assert "drafts:v1:a" not in ids(docs)
    assert len(docs) == len(ALL_IDS) - 1
