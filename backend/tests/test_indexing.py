"""Tests for `index_corpus` in vectorstore/indexing.py.

Each test gets its own throwaway database, `DeterministicFakeEmbedding`, and
a record manager from `open_record_manager`. The table comes from
`store.create_table`. No Ollama is needed.
"""

import asyncio
import re
from collections.abc import Iterator

import psycopg
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_postgres import PGEngine, PGVectorStore

from profile_rag.vectorstore import indexing, store
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
OTHER = EmbeddingModel(
    key="other-8",
    ollama_tag="other:8",
    digest="1" * 64,
    dimensions=8,
    table_name="chunks_other_8",
    document_prefix="",
    query_prefix="",
    max_words=1000,
)

Counts = dict[str, int]


def _new_store(engine: PGEngine, model: EmbeddingModel, embeddings: Embeddings) -> PGVectorStore:
    store.create_table(engine, model, embeddings)
    return store.open_index_store(engine, model, embeddings)


class Table:
    """One model's store and record manager, plus ways to read its rows."""

    def __init__(self, engine: PGEngine, url: str, model: EmbeddingModel) -> None:
        self.url = url
        self.model = model
        embeddings = DeterministicFakeEmbedding(size=model.dimensions)
        self.store = _new_store(engine, model, embeddings)
        self.record_manager = indexing.open_record_manager(model, url)

    def run(self, docs: list[Document]) -> Counts:
        result = indexing.index_corpus(docs, self.store, self.record_manager)
        return {
            "num_added": result["num_added"],
            "num_updated": result["num_updated"],
            "num_skipped": result["num_skipped"],
            "num_deleted": result["num_deleted"],
        }

    def rows(self) -> list[tuple[str, str, str]]:
        """(langchain_id, source_id, chunk_id) of every row, by chunk_id."""
        with psycopg.connect(self.url) as conn:
            return conn.execute(  # pyrefly: ignore[no-matching-overload]
                f"SELECT langchain_id, source_id, chunk_id FROM {self.model.table_name}"
                " ORDER BY chunk_id"
            ).fetchall()

    def chunk_ids(self) -> list[str]:
        return [chunk_id for _, _, chunk_id in self.rows()]


@pytest.fixture
def engine(empty_database_url: str) -> Iterator[PGEngine]:
    engine = PGEngine.from_connection_string(url=empty_database_url)
    yield engine
    asyncio.run(engine.close())


@pytest.fixture
def table(engine: PGEngine, empty_database_url: str) -> Table:
    return Table(engine, empty_database_url, SMALL)


def chunk(source_id: str, letter: str, text: str) -> Document:
    chunk_id = f"{source_id}:v1:{letter}"
    return Document(
        page_content=text,
        metadata={
            "source_id": source_id,
            "chunk_id": chunk_id,
            "attribution": "self_authored",
            "visibility": "public",
            "title": source_id,
            "heading_path": [],
            "ordinal": ord(letter) - ord("a"),
        },
        id=chunk_id,
    )


RESUME = [
    chunk("resume-2026", "a", "Dana Whitlock, backend engineer."),
    chunk("resume-2026", "b", "Moved four services to a PostgreSQL queue."),
]
README = [
    chunk("lantern-queue", "a", "lantern-queue is a job queue on PostgreSQL."),
    chunk("lantern-queue", "b", "Install it with pip."),
]
CORPUS = RESUME + README


def counts(added: int = 0, updated: int = 0, skipped: int = 0, deleted: int = 0) -> Counts:
    return {
        "num_added": added,
        "num_updated": updated,
        "num_skipped": skipped,
        "num_deleted": deleted,
    }


def test_first_run_adds_every_chunk(table: Table) -> None:
    assert table.run(CORPUS) == counts(added=4)
    assert table.chunk_ids() == sorted(doc.metadata["chunk_id"] for doc in CORPUS)


def test_second_run_adds_nothing(table: Table) -> None:
    table.run(CORPUS)
    before = table.rows()
    assert table.run(CORPUS) == counts(skipped=4)
    assert table.rows() == before


def test_row_ids_are_sha256_hashes_not_chunk_ids(table: Table) -> None:
    table.run(CORPUS)
    for row_id, _, chunk_id in table.rows():
        assert re.fullmatch(r"[0-9a-f]{64}", row_id), row_id
        assert row_id != chunk_id


def test_no_sha1_warning(table: Table, recwarn: pytest.WarningsRecorder) -> None:
    table.run(CORPUS)
    assert not [w for w in recwarn if "SHA-1" in str(w.message)]


def test_an_edited_chunk_replaces_one_row(table: Table) -> None:
    table.run(CORPUS)
    old = dict((chunk_id, row_id) for row_id, _, chunk_id in table.rows())
    edited = chunk("lantern-queue", "a", "lantern-queue is a job queue for PostgreSQL 18.")
    assert table.run([*RESUME, edited, README[1]]) == counts(added=1, skipped=3, deleted=1)
    new = dict((chunk_id, row_id) for row_id, _, chunk_id in table.rows())
    assert set(new) == set(old)
    assert new["lantern-queue:v1:a"] != old["lantern-queue:v1:a"]
    assert {k: v for k, v in new.items() if k != "lantern-queue:v1:a"} == {
        k: v for k, v in old.items() if k != "lantern-queue:v1:a"
    }


def test_changed_metadata_replaces_the_row(table: Table) -> None:
    table.run(CORPUS)
    retitled = chunk("lantern-queue", "b", "Install it with pip.")
    retitled.metadata["title"] = "README"
    assert table.run([*RESUME, README[0], retitled]) == counts(added=1, skipped=3, deleted=1)


def test_a_source_that_loses_approval_loses_its_rows(table: Table) -> None:
    table.run(CORPUS)
    assert table.run(RESUME) == counts(skipped=2, deleted=2)
    assert {source_id for _, source_id, _ in table.rows()} == {"resume-2026"}


def test_a_new_source_is_added_without_touching_the_others(table: Table) -> None:
    table.run(RESUME)
    assert table.run(CORPUS) == counts(added=2, skipped=2)


def test_an_empty_corpus_empties_the_table(table: Table) -> None:
    table.run(CORPUS)
    assert table.run([]) == counts(deleted=4)
    assert table.rows() == []


def test_two_models_do_not_delete_each_other(
    engine: PGEngine, empty_database_url: str, table: Table
) -> None:
    other = Table(engine, empty_database_url, OTHER)
    table.run(CORPUS)
    assert other.run(RESUME) == counts(added=2)
    assert len(table.rows()) == 4
    assert table.run(CORPUS) == counts(skipped=4)


def test_record_namespace_names_the_table() -> None:
    assert indexing.record_namespace(SMALL) == "pgvector/chunks_small_8"
