"""The approximate nearest-neighbour (HNSW) index on a model's chunk table.

Without this index, a dense search compares the question with every row (a
sequential scan): exact, and fast enough for a few thousand chunks. An HNSW
index answers from a graph of the vectors instead. It reads far fewer rows,
but it can miss some of the true nearest chunks.

- `apply_hnsw` creates the index `<table>_hnsw` with pgvector's
  `vector_cosine_ops`, the operator class of the `<=>` distance the store
  searches with. Running it again with the same settings changes nothing.
- `drop_hnsw` removes it. Search keeps working, by sequential scan.
- `hnsw_status` reads the index from the catalog.
- `ef_search` (search time) is how many candidates the graph search keeps.
  Higher finds more of the true nearest chunks and reads more rows.
  `store.open_search_store(..., ef_search=N)` makes the store run
  `SET LOCAL hnsw.ef_search = N` in the transaction of each dense query.

Postgres decides per query whether to use the index. For a small table a
sequential scan is cheaper, so the index is not used at all. `open_engine`
with `plan="index"` turns sequential scans off for its connections, so the
planner takes the index whenever it can; `plan="exact"` turns index scans
off, so every dense query is exact. evaluation/ann_compare.py uses both.

Tests: backend/tests/test_hnsw.py.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import psycopg
from langchain_postgres import PGEngine, PGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex
from psycopg import sql

from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.store import TABLE_NAMES

DEFAULT_M = 16  # graph links per node
DEFAULT_EF_CONSTRUCTION = 64  # candidates kept while the graph is built
DEFAULT_EF_SEARCH = 40  # pgvector's default `hnsw.ef_search`
# The range pgvector accepts for `hnsw.ef_search`.
EF_SEARCH_MIN = 1
EF_SEARCH_MAX = 1000

# `default`: the planner chooses. `index`: sequential scans off, so the HNSW
# index is used when it can be. `exact`: index scans off, so no HNSW.
Plan = Literal["default", "index", "exact"]
_PLAN_OPTIONS: dict[Plan, str | None] = {
    "default": None,
    "index": "-c enable_seqscan=off",
    "exact": "-c enable_indexscan=off",
}

ApplyResult = Literal["created", "unchanged", "rebuilt"]


def hnsw_index_name(model: EmbeddingModel) -> str:
    return f"{model.table_name}_hnsw"


@dataclass(frozen=True)
class HnswStatus:
    """The index as the catalog describes it."""

    name: str
    definition: str  # `pg_indexes.indexdef`: the CREATE INDEX statement
    valid: bool  # `pg_index.indisvalid`: false after a failed concurrent build
    m: int | None
    ef_construction: int | None
    size_bytes: int


def _options(reloptions: list[str] | None) -> dict[str, str]:
    """`['m=16', 'ef_construction=64']` as a dict."""
    return dict(option.split("=", 1) for option in reloptions or [])


def hnsw_status(conninfo: str, model: EmbeddingModel) -> HnswStatus | None:
    """The model's HNSW index, or None if the table has none."""
    with psycopg.connect(conninfo) as conn:
        row = conn.execute(
            """
            SELECT i.indexdef, x.indisvalid, c.reloptions, pg_relation_size(c.oid)
            FROM pg_indexes i
            JOIN pg_namespace n ON n.nspname = i.schemaname
            JOIN pg_class c ON c.relname = i.indexname AND c.relnamespace = n.oid
            JOIN pg_index x ON x.indexrelid = c.oid
            WHERE i.schemaname = current_schema() AND i.tablename = %s AND i.indexname = %s
            """,
            (model.table_name, hnsw_index_name(model)),
        ).fetchone()
    if row is None:
        return None
    definition, valid, reloptions, size_bytes = row
    options = _options(reloptions)
    return HnswStatus(
        name=hnsw_index_name(model),
        definition=definition,
        valid=valid,
        m=int(options["m"]) if "m" in options else None,
        ef_construction=(int(options["ef_construction"]) if "ef_construction" in options else None),
        size_bytes=size_bytes,
    )


def apply_hnsw(
    store: PGVectorStore,
    model: EmbeddingModel,
    *,
    conninfo: str,
    m: int = DEFAULT_M,
    ef_construction: int = DEFAULT_EF_CONSTRUCTION,
) -> ApplyResult:
    """Make sure `model`'s table has a valid HNSW index with these settings.

    `store` is the model's table (`store.open_search_store`); `conninfo` is
    the same database, for reading the catalog. Returns "unchanged" if a valid
    index with the same `m` and `ef_construction` exists, "rebuilt" if an
    invalid one or one with other settings was replaced, else "created".
    """
    current = hnsw_status(conninfo, model)
    if (
        current is not None
        and current.valid
        and (current.m, current.ef_construction) == (m, ef_construction)
    ):
        return "unchanged"
    name = hnsw_index_name(model)
    if current is not None:
        store.drop_vector_index(name)
    store.apply_vector_index(HNSWIndex(name=name, m=m, ef_construction=ef_construction))
    return "created" if current is None else "rebuilt"


def drop_hnsw(store: PGVectorStore, model: EmbeddingModel) -> None:
    """Drop the model's HNSW index. Does nothing if there is none."""
    store.drop_vector_index(hnsw_index_name(model))


def open_engine(url: str, *, plan: Plan = "default") -> PGEngine:
    """A `PGEngine` whose connections plan dense queries as `plan` says.

    The setting is a connection option (`-c ...`), so it holds for every
    query on the engine's connections and for no other connection.
    """
    options = _PLAN_OPTIONS[plan]
    if options is None:
        return PGEngine.from_connection_string(url=url)
    return PGEngine.from_connection_string(url=url, connect_args={"options": options})


# One fixed query per chunk table, built from the allowlist `TABLE_NAMES`.
# Every value is a query parameter.
_EXPLAIN_DENSE = {
    table: sql.SQL(
        "EXPLAIN SELECT langchain_id, content, embedding, source_id, chunk_id, attribution,"
        " visibility, langchain_metadata,"
        " cosine_distance(embedding, %(q)s::vector) AS distance"
        " FROM {table} WHERE visibility = ANY(%(visibility)s)"
        " ORDER BY embedding <=> %(q)s::vector LIMIT %(k)s"
    ).format(table=sql.Identifier(table))
    for table in TABLE_NAMES
}


def explain_dense(
    conninfo: str,
    model: EmbeddingModel,
    embedding: list[float],
    *,
    k: int,
    visibility: Sequence[str],
    plan: Plan = "default",
) -> list[str]:
    """The query plan of the store's dense query for `embedding`, one line per row.

    The query is the one `PGVectorStore` 0.0.19 runs for a dense search with
    the visibility filter, as `PGEngine.from_connection_string(..., echo=True)`
    logs it: `... WHERE visibility = ANY(...) ORDER BY "embedding" <=> q
    LIMIT k`. `plan` applies the same setting as `open_engine`, with SET LOCAL
    in the EXPLAIN's transaction.
    """
    if model.table_name not in TABLE_NAMES:
        raise ValueError(f"unknown chunk table {model.table_name!r}")
    vector = "[" + ",".join(repr(float(value)) for value in embedding) + "]"
    # Fixed SQL text. The table name is allowlisted above and quoted as an
    # identifier; every value is a query parameter.
    with psycopg.connect(conninfo) as conn, conn.transaction(), conn.cursor() as cursor:
        if plan == "index":
            cursor.execute("SET LOCAL enable_seqscan = off")
        elif plan == "exact":
            cursor.execute("SET LOCAL enable_indexscan = off")
        rows = cursor.execute(
            _EXPLAIN_DENSE[model.table_name], {"visibility": list(visibility), "q": vector, "k": k}
        ).fetchall()
    return [str(row[0]) for row in rows]
