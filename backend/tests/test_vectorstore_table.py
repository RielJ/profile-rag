"""Tests for `create_table` and the two stores in vectorstore/store.py.

Each test gets its own throwaway database and a model entry whose vectors
have 8 values, so `DeterministicFakeEmbedding(size=8)` can fill it. No
Ollama is needed.

The shape test reads `information_schema.columns`. If a langchain-postgres
upgrade changes the table it creates, this file fails instead of retrieval
breaking without an error.
"""

import asyncio
from collections.abc import Iterator
from typing import ClassVar

import psycopg
import pytest
import sqlalchemy.exc
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_postgres import PGEngine

from profile_rag.vectorstore import store
from profile_rag.vectorstore.models import EmbeddingModel

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

URL = "postgresql+psycopg://db.test/x"
FAKE = DeterministicFakeEmbedding(size=SMALL.dimensions)

# PGVectorStore keeps its settings on a private inner store. One test reads it.
INNER_STORE = "_PGVectorStore__vs"


@pytest.fixture
def engine(empty_database_url: str) -> Iterator[PGEngine]:
    engine = PGEngine.from_connection_string(url=empty_database_url)
    yield engine
    asyncio.run(engine.close())


@pytest.fixture
def embeddings() -> DeterministicFakeEmbedding:
    return DeterministicFakeEmbedding(size=SMALL.dimensions)


def created(engine: PGEngine) -> PGEngine:
    """Create SMALL's table. Called in the test body, not a fixture, so a
    failure in `create_table` is reported as the test's own failure."""
    store.create_table(engine, SMALL, DeterministicFakeEmbedding(size=SMALL.dimensions))
    return engine


def chunk(text: str, chunk_id: str, **extra: object) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_id": "resume-2026",
            "chunk_id": chunk_id,
            "attribution": "self_authored",
            "visibility": "public",
            "title": "Resume",
            "heading_path": ["Dana Whitlock"],
            "ordinal": 0,
            **extra,
        },
    )


def test_table_has_the_planned_columns_in_order(engine: PGEngine, empty_database: str) -> None:
    created(engine)
    with psycopg.connect(empty_database) as conn:
        rows = conn.execute(
            "SELECT column_name, data_type, udt_name, is_nullable"
            " FROM information_schema.columns WHERE table_name = %s ORDER BY ordinal_position",
            (SMALL.table_name,),
        ).fetchall()
    assert rows == [
        ("langchain_id", "text", "text", "NO"),
        ("content", "text", "text", "NO"),
        ("embedding", "USER-DEFINED", "vector", "NO"),
        ("content_tsv", "tsvector", "tsvector", "NO"),
        ("source_id", "text", "text", "NO"),
        ("chunk_id", "text", "text", "NO"),
        ("attribution", "text", "text", "NO"),
        ("visibility", "text", "text", "NO"),
        ("langchain_metadata", "json", "json", "YES"),
    ]


def test_embedding_column_has_the_model_dimensions(engine: PGEngine, empty_database: str) -> None:
    created(engine)
    with psycopg.connect(empty_database) as conn:
        row = conn.execute(
            "SELECT format_type(atttypid, atttypmod) FROM pg_attribute"
            " WHERE attrelid = %s::regclass AND attname = 'embedding'",
            (SMALL.table_name,),
        ).fetchone()
    assert row == ("vector(8)",)


def test_primary_key_and_gin_index(engine: PGEngine, empty_database: str) -> None:
    created(engine)
    with psycopg.connect(empty_database) as conn:
        indexes = dict(
            conn.execute(
                "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
                (SMALL.table_name,),
            ).fetchall()
        )
    assert set(indexes) == {f"{SMALL.table_name}_pkey", store.tsv_index_name(SMALL)}
    assert "(langchain_id)" in indexes[f"{SMALL.table_name}_pkey"]
    assert "USING gin (content_tsv)" in indexes[store.tsv_index_name(SMALL)]


def test_table_exists_after_create(engine: PGEngine, empty_database: str) -> None:
    created(engine)
    assert store.table_exists(empty_database, SMALL.table_name)
    assert not store.table_exists(empty_database, "chunks_other")


def test_a_second_create_raises(engine: PGEngine, embeddings: DeterministicFakeEmbedding) -> None:
    ready = created(engine)
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        store.create_table(ready, SMALL, embeddings)


def test_index_store_writes_columns_metadata_and_tsv(
    engine: PGEngine, embeddings: DeterministicFakeEmbedding, empty_database: str
) -> None:
    ready = created(engine)
    index_store = store.open_index_store(ready, SMALL, embeddings)
    index_store.add_documents([chunk("Dana Whitlock builds job queues.", "resume-2026:v1:a")])
    with psycopg.connect(empty_database) as conn:
        row = conn.execute(
            "SELECT source_id, chunk_id, attribution, visibility, langchain_metadata,"
            " content_tsv @@ to_tsquery('english', 'queue') FROM chunks_small_8"
        ).fetchone()
    assert row == (
        "resume-2026",
        "resume-2026:v1:a",
        "self_authored",
        "public",
        {"title": "Resume", "heading_path": ["Dana Whitlock"], "ordinal": 0},
        True,
    )


def test_index_store_refuses_a_chunk_without_a_column_value(
    engine: PGEngine, embeddings: DeterministicFakeEmbedding
) -> None:
    ready = created(engine)
    index_store = store.open_index_store(ready, SMALL, embeddings)
    bad = chunk("No visibility.", "resume-2026:v1:b")
    del bad.metadata["visibility"]
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        index_store.add_documents([bad])


def test_search_store_finds_rows_and_has_no_hybrid_config(
    engine: PGEngine, embeddings: DeterministicFakeEmbedding
) -> None:
    ready = created(engine)
    index_store = store.open_index_store(ready, SMALL, embeddings)
    index_store.add_documents([chunk("Dana Whitlock builds job queues.", "resume-2026:v1:a")])
    search_store = store.open_search_store(ready, SMALL, embeddings)
    found = search_store.similarity_search("Dana Whitlock builds job queues.", k=1)
    assert [doc.metadata["chunk_id"] for doc in found] == ["resume-2026:v1:a"]
    # Without a HybridSearchConfig the store leaves content_tsv empty, and
    # the NOT NULL constraint rejects the insert.
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        search_store.add_documents([chunk("Another chunk.", "resume-2026:v1:c")])


def test_opening_a_missing_table_raises(
    engine: PGEngine, embeddings: DeterministicFakeEmbedding
) -> None:
    with pytest.raises(ValueError):
        store.open_index_store(engine, SMALL, embeddings)
    with pytest.raises(ValueError):
        store.open_search_store(engine, SMALL, embeddings)


def test_each_index_store_gets_its_own_hybrid_config(
    engine: PGEngine, embeddings: DeterministicFakeEmbedding
) -> None:
    ready = created(engine)
    # A search writes its query into the config; a shared one would leak it.
    first = store.open_index_store(ready, SMALL, embeddings)
    second = store.open_index_store(ready, SMALL, embeddings)
    first_config = getattr(first, INNER_STORE).hybrid_search_config
    second_config = getattr(second, INNER_STORE).hybrid_search_config
    assert first_config is not None
    assert first_config is not second_config


class _RecordingEngine:
    """Stands in for `PGEngine` in the lifecycle tests: records `close()`."""

    opened: ClassVar[list[_RecordingEngine]] = []

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False

    @classmethod
    def from_connection_string(cls, url: str) -> _RecordingEngine:
        engine = cls(url)
        cls.opened.append(engine)
        return engine

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def recording_engine(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingEngine]:
    _RecordingEngine.opened = []
    monkeypatch.setattr(store, "PGEngine", _RecordingEngine)
    return _RecordingEngine.opened


def test_connected_closes_the_engine_when_the_block_ends(
    recording_engine: list[_RecordingEngine],
) -> None:
    with store.connected(URL) as engine:
        assert not recording_engine[0].closed
    assert engine is recording_engine[0]
    assert engine.closed


def test_connected_closes_the_engine_when_the_block_raises(
    recording_engine: list[_RecordingEngine],
) -> None:
    with pytest.raises(RuntimeError, match="boom"), store.connected(URL):
        raise RuntimeError("boom")
    assert recording_engine[0].closed


def test_searching_closes_its_engine_when_opening_the_store_fails(
    recording_engine: list[_RecordingEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing_table(*_args: object, **_kwargs: object) -> None:
        raise ValueError("no table")

    monkeypatch.setattr(store, "open_search_store", missing_table)
    with (
        pytest.raises(ValueError, match="no table"),
        store.searching(URL, SMALL, FAKE),
    ):
        pass
    assert recording_engine[0].closed


def test_searching_yields_a_search_store_on_a_real_table(
    empty_database_url: str, embeddings: DeterministicFakeEmbedding
) -> None:
    with store.connected(empty_database_url) as engine:
        store.create_table(engine, SMALL, embeddings)
    with store.searching(empty_database_url, SMALL, embeddings, ef_search=64) as search_store:
        assert search_store.similarity_search("anything", k=1) == []
