"""Tests for corpus/sources_table.py: `sync_sources`.

Each test gets a throwaway database with every migration applied, and reads
the example registry in corpus-example/sources.yaml.
"""

from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from profile_rag.corpus.registry import Registry, load_registry
from profile_rag.corpus.sources_table import sync_sources
from profile_rag.db.migrate import migrate

EXAMPLE = Path(__file__).resolve().parents[2] / "corpus-example" / "sources.yaml"
APPROVED = ["blog-postgres-queues", "lantern-queue-readme", "resume-2026"]


@pytest.fixture
def conn(empty_database: str) -> Iterator[psycopg.Connection[TupleRow]]:
    migrate(empty_database)
    with psycopg.connect(empty_database) as conn:
        yield conn


@pytest.fixture
def registry() -> Registry:
    return load_registry(EXAMPLE)


def ids(conn: psycopg.Connection[TupleRow]) -> list[str]:
    return [row[0] for row in conn.execute("SELECT id FROM sources ORDER BY id").fetchall()]


def without(registry: Registry, source_id: str) -> Registry:
    """The registry with one source moved back to `rejected`."""
    data = registry.model_dump()
    for entry in data["sources"]:
        if entry["id"] == source_id:
            entry.update(
                state="rejected",
                path=None,
                revision=None,
                content_sha256=None,
                approved_at=None,
                approved_by=None,
                rejection_reason="test",
            )
    return Registry.model_validate(data)


def test_only_approved_sources_are_written(
    conn: psycopg.Connection[TupleRow], registry: Registry
) -> None:
    result = sync_sources(conn, registry)
    assert sorted(result.upserted) == APPROVED
    assert result.deleted == []
    assert ids(conn) == APPROVED


def test_columns_come_from_the_registry(
    conn: psycopg.Connection[TupleRow], registry: Registry
) -> None:
    sync_sources(conn, registry)
    row = conn.execute(
        "SELECT kind, title, url, visibility, attribution, revision, content_sha256,"
        " approved_by FROM sources WHERE id = 'resume-2026'"
    ).fetchone()
    entry = next(e for e in registry.sources if e.id == "resume-2026")
    assert row == (
        entry.kind,
        entry.title,
        entry.url,
        entry.visibility,
        entry.attribution,
        entry.revision,
        entry.content_sha256,
        entry.approved_by,
    )


def test_a_second_run_changes_nothing(
    conn: psycopg.Connection[TupleRow], registry: Registry
) -> None:
    sync_sources(conn, registry)
    before = conn.execute("SELECT * FROM sources ORDER BY id").fetchall()
    assert sync_sources(conn, registry).deleted == []
    assert conn.execute("SELECT * FROM sources ORDER BY id").fetchall() == before


def test_a_source_that_loses_approval_is_deleted(
    conn: psycopg.Connection[TupleRow], registry: Registry
) -> None:
    sync_sources(conn, registry)
    result = sync_sources(conn, without(registry, "lantern-queue-readme"))
    assert result.deleted == ["lantern-queue-readme"]
    assert ids(conn) == ["blog-postgres-queues", "resume-2026"]


def test_an_edited_entry_is_updated(conn: psycopg.Connection[TupleRow], registry: Registry) -> None:
    sync_sources(conn, registry)
    data = registry.model_dump()
    data["sources"][0]["title"] = "Resume (2026, edited)"
    sync_sources(conn, Registry.model_validate(data))
    row = conn.execute("SELECT title FROM sources WHERE id = 'resume-2026'").fetchone()
    assert row == ("Resume (2026, edited)",)


def test_an_empty_registry_empties_the_table(
    conn: psycopg.Connection[TupleRow], registry: Registry
) -> None:
    sync_sources(conn, registry)
    assert sorted(sync_sources(conn, Registry()).deleted) == APPROVED
    assert ids(conn) == []
