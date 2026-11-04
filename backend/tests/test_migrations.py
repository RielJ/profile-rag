"""Tests for the migration runner and for migrations 0001 and 0002.

Three groups:

- Runner tests use small migration files written to a temporary folder.
  They test db/migrate.py.
- Schema tests for 0001 apply only db/migrations/0001_initial.sql to a
  throwaway database. 0001 runs first on every database, so its schema
  still matters.
- Schema tests for 0002 apply every file in db/migrations/ and check what
  is left: `sources` only. PGVectorStore owns the chunk tables
  (docs/ARCHITECTURE.md, "Tables").

Run only the schema tests with:

    uv --directory backend run pytest tests/test_migrations.py -k schema
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from pathlib import Path

import psycopg
import pytest
from psycopg.rows import TupleRow

from profile_rag.db.migrate import MIGRATIONS_DIR, MigrationError, discover, migrate

Conn = psycopg.Connection[TupleRow]
# The type of conftest's `temporary_database_factory` fixture.
TemporaryDatabase = Callable[[], AbstractContextManager[str]]

FIRST_MIGRATION = MIGRATIONS_DIR / "0001_initial.sql"


# --- Runner ---------------------------------------------------------------


def write(directory: Path, filename: str, sql: str) -> None:
    (directory / filename).write_text(sql)


def table_exists(conninfo: str, table: str) -> bool:
    with psycopg.connect(conninfo) as conn:
        row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table,)).fetchone()
    assert row is not None
    return bool(row[0])


def test_runner_applies_files_in_version_order(tmp_path: Path, empty_database: str) -> None:
    # 0002 references 0001's table, so it only works if 0001 runs first.
    write(tmp_path, "0002_child.sql", "CREATE TABLE child (id int REFERENCES parent(id));")
    write(tmp_path, "0001_parent.sql", "CREATE TABLE parent (id int PRIMARY KEY);")

    applied = migrate(empty_database, tmp_path)

    assert [m.version for m in applied] == ["0001", "0002"]
    assert table_exists(empty_database, "child")


def test_runner_second_run_applies_nothing(tmp_path: Path, empty_database: str) -> None:
    write(tmp_path, "0001_a.sql", "CREATE TABLE a (id int);")
    write(tmp_path, "0002_b.sql", "CREATE TABLE b (id int);")

    migrate(empty_database, tmp_path)
    assert migrate(empty_database, tmp_path) == []

    with psycopg.connect(empty_database) as conn:
        rows = conn.execute(
            "SELECT version, name, length(checksum) FROM schema_migrations ORDER BY version"
        ).fetchall()
    assert rows == [("0001", "a", 64), ("0002", "b", 64)]


def test_runner_rolls_back_a_failing_file_completely(tmp_path: Path, empty_database: str) -> None:
    write(tmp_path, "0001_ok.sql", "CREATE TABLE ok (id int);")
    write(tmp_path, "0002_broken.sql", "CREATE TABLE half (id int);\nSELECT 1 / 0;")

    with pytest.raises(psycopg.errors.DivisionByZero):
        migrate(empty_database, tmp_path)

    assert table_exists(empty_database, "ok"), "0001 succeeded and must stay applied"
    assert not table_exists(empty_database, "half"), "0002 failed, so none of it may remain"
    with psycopg.connect(empty_database) as conn:
        versions = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert versions == [("0001",)]


def test_runner_refuses_an_edited_applied_file(tmp_path: Path, empty_database: str) -> None:
    write(tmp_path, "0001_a.sql", "CREATE TABLE a (id int);")
    migrate(empty_database, tmp_path)

    write(tmp_path, "0001_a.sql", "CREATE TABLE a (id bigint);")

    with pytest.raises(MigrationError, match="changed after it was applied"):
        migrate(empty_database, tmp_path)


def test_runner_rejects_a_malformed_filename(tmp_path: Path) -> None:
    write(tmp_path, "1_initial.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match=r"NNNN_description\.sql"):
        discover(tmp_path)


def test_runner_rejects_a_duplicate_version(tmp_path: Path) -> None:
    write(tmp_path, "0001_a.sql", "SELECT 1;")
    write(tmp_path, "0001_b.sql", "SELECT 1;")
    with pytest.raises(MigrationError, match="version 0001 is also used"):
        discover(tmp_path)


def test_runner_ignores_files_that_are_not_sql(tmp_path: Path) -> None:
    write(tmp_path, "0001_a.sql", "SELECT 1;")
    (tmp_path / "README.md").write_text("notes")
    assert [m.path.name for m in discover(tmp_path)] == ["0001_a.sql"]


# --- Schema (0001_initial.sql) --------------------------------------------


def only_0001(directory: Path) -> Path:
    """Copy 0001_initial.sql alone into `directory` and return it.

    The runner applies every file in a folder, so a folder with one file
    stops the schema at 0001.
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FIRST_MIGRATION.name).write_bytes(FIRST_MIGRATION.read_bytes())
    return directory


def apply_or_fail(conninfo: str, directory: Path = MIGRATIONS_DIR) -> None:
    failure = None
    try:
        migrate(conninfo, directory)
    except (psycopg.Error, MigrationError) as error:
        failure = str(error)
    # Failing outside `except` prints the database error once, not chained.
    if failure is not None:
        pytest.fail(f"{directory.name} did not apply: {failure}", pytrace=False)


@pytest.fixture(scope="module")
def database_at_0001(
    temporary_database_factory: TemporaryDatabase, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[str]:
    """A throwaway database with only 0001_initial.sql applied."""
    with temporary_database_factory() as conninfo:
        apply_or_fail(conninfo, only_0001(tmp_path_factory.mktemp("only-0001")))
        yield conninfo


@pytest.fixture(scope="module")
def migrated_database(temporary_database_factory: TemporaryDatabase) -> Iterator[str]:
    """A throwaway database with every file in db/migrations/ applied."""
    with temporary_database_factory() as conninfo:
        apply_or_fail(conninfo)
        yield conninfo


@pytest.fixture
def conn(database_at_0001: str) -> Iterator[Conn]:
    """A connection to the 0001 schema whose changes are rolled back after each test."""
    with psycopg.connect(database_at_0001) as connection:
        yield connection
        connection.rollback()


SOURCE: dict[str, str | None] = {
    "id": "resume-2026",
    "kind": "resume",
    "title": "Resume",
    "url": "https://example.com/resume",
    "visibility": "public",
    "attribution": "self_authored",
    "revision": "2026-10-01",
    "content_sha256": "0" * 64,
    "approved_at": "2026-10-01T00:00:00Z",
    "approved_by": "riel",
}


def insert_source(conn: Conn, **overrides: str | None) -> None:
    conn.execute(
        """
        INSERT INTO sources (id, kind, title, url, visibility, attribution,
                             revision, content_sha256, approved_at, approved_by)
        VALUES (%(id)s, %(kind)s, %(title)s, %(url)s, %(visibility)s, %(attribution)s,
                %(revision)s, %(content_sha256)s, %(approved_at)s, %(approved_by)s)
        """,
        {**SOURCE, **overrides},
    )


def insert_chunk(
    conn: Conn, chunk_id: str = "c1", *, ordinal: int = 0, text: str = "Built things."
) -> None:
    conn.execute(
        """
        INSERT INTO chunks (id, source_id, chunker_version, ordinal, text, char_start, char_end)
        VALUES (%s, %s, 'v1', %s, %s, 0, %s)
        """,
        (chunk_id, SOURCE["id"], ordinal, text, len(text)),
    )


def insert_config(conn: Conn, config_id: str = "nomic-v1.5-768", dimensions: int = 768) -> None:
    conn.execute(
        """
        INSERT INTO embedding_configs
            (id, model, model_digest, dimensions, query_template, doc_template)
        VALUES (%s, 'nomic-embed-text:v1.5', 'sha256:abc', %s,
                'search_query: {text}', 'search_document: {text}')
        """,
        (config_id, dimensions),
    )


def insert_embedding(conn: Conn, chunk_id: str, config_id: str, vector: list[float]) -> None:
    conn.execute(
        "INSERT INTO chunk_embeddings (chunk_id, config_id, embedding) VALUES (%s, %s, %s::vector)",
        (chunk_id, config_id, str(vector)),
    )


def first_value(cursor: psycopg.Cursor[TupleRow]) -> object:
    """Return the first column of the first row of an executed query."""
    row = cursor.fetchone()
    assert row is not None, "query returned no rows"
    return row[0]


def test_schema_installs_the_vector_extension(conn: Conn) -> None:
    assert (
        first_value(conn.execute("SELECT count(*) FROM pg_extension WHERE extname = 'vector'")) == 1
    )


# (column, type as Postgres prints it, nullable)
EXPECTED_COLUMNS: dict[str, set[tuple[str, str, bool]]] = {
    "sources": {
        ("id", "text", False),
        ("kind", "text", False),
        ("title", "text", False),
        ("url", "text", True),
        ("visibility", "text", False),
        ("attribution", "text", False),
        ("revision", "text", False),
        ("content_sha256", "text", False),
        ("approved_at", "timestamp with time zone", False),
        ("approved_by", "text", False),
    },
    "chunks": {
        ("id", "text", False),
        ("source_id", "text", False),
        ("chunker_version", "text", False),
        ("ordinal", "integer", False),
        ("heading_path", "text[]", False),
        ("text", "text", False),
        ("char_start", "integer", False),
        ("char_end", "integer", False),
        ("tsv", "tsvector", True),
    },
    "embedding_configs": {
        ("id", "text", False),
        ("model", "text", False),
        ("model_digest", "text", False),
        ("dimensions", "integer", False),
        ("query_template", "text", False),
        ("doc_template", "text", False),
        ("created_at", "timestamp with time zone", False),
    },
    "chunk_embeddings": {
        ("chunk_id", "text", False),
        ("config_id", "text", False),
        ("embedding", "vector", False),
    },
}


@pytest.mark.parametrize("table", EXPECTED_COLUMNS)
def test_schema_table_has_expected_columns(conn: Conn, table: str) -> None:
    rows = conn.execute(
        """
        SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull
        FROM pg_attribute a
        WHERE a.attrelid = to_regclass(%s) AND a.attnum > 0 AND NOT a.attisdropped
        """,
        (table,),
    ).fetchall()
    assert rows, f"table {table} does not exist"
    assert set(rows) == EXPECTED_COLUMNS[table]


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("sources", ["id"]),
        ("chunks", ["id"]),
        ("embedding_configs", ["id"]),
        ("chunk_embeddings", ["chunk_id", "config_id"]),
    ],
)
def test_schema_primary_keys(conn: Conn, table: str, columns: list[str]) -> None:
    rows = conn.execute(
        """
        SELECT a.attname
        FROM pg_index i
        JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY (i.indkey)
        WHERE i.indrelid = to_regclass(%s) AND i.indisprimary
        ORDER BY array_position(i.indkey, a.attnum)
        """,
        (table,),
    ).fetchall()
    assert [r[0] for r in rows] == columns


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [("kind", "podcast"), ("visibility", "private"), ("attribution", "rumor")],
)
def test_schema_sources_rejects_unknown_values(conn: Conn, column: str, bad_value: str) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        insert_source(conn, **{column: bad_value})


def test_schema_sources_accepts_a_null_url_for_private_notes(conn: Conn) -> None:
    insert_source(conn, kind="note", visibility="sanitized", url=None)


def test_schema_chunk_needs_an_existing_source(conn: Conn) -> None:
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        insert_chunk(conn)


def test_schema_chunk_position_is_unique_per_source_and_chunker(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn, "c1", ordinal=0)
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_chunk(conn, "c2", ordinal=0)


def test_schema_chunk_defaults(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn)
    assert first_value(conn.execute("SELECT heading_path FROM chunks WHERE id = 'c1'")) == []


def test_schema_tsv_is_generated_from_text(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn, text="Deployed three services to production.")
    # Stemming: "deploying" in the query matches "Deployed" in the text.
    matched = first_value(
        conn.execute("SELECT tsv @@ plainto_tsquery('english', 'deploying') FROM chunks")
    )
    assert matched is True


def test_schema_tsv_cannot_be_written_directly(conn: Conn) -> None:
    insert_source(conn)
    with pytest.raises(psycopg.errors.GeneratedAlways):
        conn.execute(
            """
            INSERT INTO chunks (id, source_id, chunker_version, ordinal, text,
                                char_start, char_end, tsv)
            VALUES ('c1', 'resume-2026', 'v1', 0, 'x', 0, 1, to_tsvector('x'))
            """
        )


def test_schema_chunks_tsv_has_a_gin_index(conn: Conn) -> None:
    indexdef = first_value(
        conn.execute("SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_tsv_idx'")
    )
    assert "USING gin (tsv)" in str(indexdef)


def test_schema_embedding_configs_created_at_defaults_to_now(conn: Conn) -> None:
    insert_config(conn)
    assert first_value(conn.execute("SELECT created_at = now() FROM embedding_configs")) is True


def test_schema_embeddings_accept_any_dimension(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn)
    insert_config(conn, "small-3", dimensions=3)
    insert_config(conn, "small-5", dimensions=5)
    insert_embedding(conn, "c1", "small-3", [1, 0, 0])
    insert_embedding(conn, "c1", "small-5", [1, 0, 0, 0, 0])
    dims = first_value(
        conn.execute("SELECT array_agg(vector_dims(embedding) ORDER BY 1) FROM chunk_embeddings")
    )
    assert dims == [3, 5]


def test_schema_one_embedding_per_chunk_and_config(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn)
    insert_config(conn, "small-3", dimensions=3)
    insert_embedding(conn, "c1", "small-3", [1, 0, 0])
    with pytest.raises(psycopg.errors.UniqueViolation):
        insert_embedding(conn, "c1", "small-3", [0, 1, 0])


def test_schema_deleting_a_source_deletes_its_chunks_and_embeddings(conn: Conn) -> None:
    insert_source(conn)
    insert_chunk(conn)
    insert_config(conn, "small-3", dimensions=3)
    insert_embedding(conn, "c1", "small-3", [1, 0, 0])

    conn.execute("DELETE FROM sources WHERE id = 'resume-2026'")

    assert first_value(conn.execute("SELECT count(*) FROM chunks")) == 0
    assert first_value(conn.execute("SELECT count(*) FROM chunk_embeddings")) == 0


def test_schema_a_config_in_use_cannot_be_deleted(conn: Conn) -> None:
    # No ON DELETE CASCADE here on purpose: deleting a config must not
    # silently throw away every embedding computed with it.
    insert_source(conn)
    insert_chunk(conn)
    insert_config(conn, "small-3", dimensions=3)
    insert_embedding(conn, "c1", "small-3", [1, 0, 0])
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute("DELETE FROM embedding_configs WHERE id = 'small-3'")


# --- Schema after 0002_drop_phase1_chunks.sql ---------------------------------


def public_tables(conninfo: str) -> set[str]:
    with psycopg.connect(conninfo) as connection:
        rows = connection.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ).fetchall()
    return {str(row[0]) for row in rows}


def test_schema_0002_leaves_only_sources(migrated_database: str) -> None:
    assert public_tables(migrated_database) == {"schema_migrations", "sources"}


def test_schema_0002_keeps_sources_and_the_vector_extension(migrated_database: str) -> None:
    with psycopg.connect(migrated_database) as connection:
        rows = connection.execute(
            """
            SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass('sources') AND a.attnum > 0 AND NOT a.attisdropped
            """
        ).fetchall()
        extensions = connection.execute(
            "SELECT count(*) FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    assert set(rows) == EXPECTED_COLUMNS["sources"]
    assert extensions == (1,)


def test_schema_0002_applies_after_0001_with_rows_in_every_table(
    tmp_path: Path, empty_database: str
) -> None:
    # The order an older database sees: 0001, rows written, then 0002.
    apply_or_fail(empty_database, only_0001(tmp_path))
    with psycopg.connect(empty_database) as connection:
        insert_source(connection)
        insert_chunk(connection)
        insert_config(connection, "small-3", dimensions=3)
        insert_embedding(connection, "c1", "small-3", [1, 0, 0])

    applied = migrate(empty_database)

    assert [m.path.name for m in applied] == ["0002_drop_phase1_chunks.sql"]
    assert public_tables(empty_database) == {"schema_migrations", "sources"}
    with psycopg.connect(empty_database) as connection:
        kept = connection.execute("SELECT id FROM sources").fetchall()
    assert kept == [("resume-2026",)]


def test_schema_second_migration_run_changes_nothing(migrated_database: str) -> None:
    assert migrate(migrated_database) == []
