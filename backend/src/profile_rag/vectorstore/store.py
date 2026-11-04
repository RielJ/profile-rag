"""Create each model's table, and open it as a `PGVectorStore`.

`PGVectorStore` owns the chunk tables: one per embedding model, created by
`PGEngine.init_vectorstore_table()`. docs/ARCHITECTURE.md shows the shape.
backend/tests/test_vectorstore_table.py checks it.

The store used for indexing has a `HybridSearchConfig` for `TSV_COLUMN`, so
its inserts fill that column. The store used for search has none: a search
writes the query into the config it is given, so each hybrid search builds
its own (retrievers/hybrid.py).
"""

import psycopg
from langchain_core.embeddings import Embeddings
from langchain_postgres import Column, PGEngine, PGVectorStore
from langchain_postgres.v2.hybrid_search_config import HybridSearchConfig
from langchain_postgres.v2.indexes import HNSWQueryOptions
from psycopg import sql

from profile_rag.vectorstore.models import MODELS, EmbeddingModel

# The only tables `chunk_ids` reads. Table names come from code, not input.
TABLE_NAMES = frozenset(model.table_name for model in MODELS.values())

# Metadata keys that get their own NOT NULL text column. Every other key of a
# chunk's metadata (title, url, heading_path, ordinal) goes to the JSON column
# `langchain_metadata`. Retrievers filter on these four.
METADATA_COLUMNS = ("source_id", "chunk_id", "attribution", "visibility")

# A text primary key: `index()` with key_encoder="sha256" writes 64 hex
# characters, which the default uuid column rejects.
ID_COLUMN = "langchain_id"

# The tsvector column for keyword search, filled by the store on every insert.
TSV_COLUMN = "content_tsv"

# PGVectorStore's default names for the chunk text and the JSON metadata.
# retrievers/keyword.py reads rows with plain SQL and needs them.
CONTENT_COLUMN = "content"
METADATA_JSON_COLUMN = "langchain_metadata"

# The text search configuration that fills TSV_COLUMN: the default `tsv_lang`
# of HybridSearchConfig. Keyword search must parse questions with the same one.
TS_CONFIG = "pg_catalog.english"


def tsv_index_name(model: EmbeddingModel) -> str:
    """The name of the table's GIN index on `TSV_COLUMN`."""
    return f"{model.table_name}_tsv_idx"


def table_exists(conninfo: str, table_name: str) -> bool:
    """True if `table_name` exists in the database's search path."""
    with psycopg.connect(conninfo) as conn:
        row = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (table_name,)).fetchone()
    return bool(row and row[0])


def chunk_ids(conninfo: str, model: EmbeddingModel) -> set[str]:
    """The `chunk_id` of every row in the model's table.

    The table name comes from `models.MODELS`, never from user input, and is
    quoted as an identifier.
    """
    if model.table_name not in TABLE_NAMES:
        raise ValueError(f"unknown chunk table {model.table_name!r}")
    with psycopg.connect(conninfo) as conn:
        rows = conn.execute(_SELECT_CHUNK_IDS[model.table_name])
        return {row[0] for row in rows}


# One fixed query per chunk table, built from the allowlist `TABLE_NAMES`.
_SELECT_CHUNK_IDS = {
    table: sql.SQL("SELECT chunk_id FROM {}").format(sql.Identifier(table)) for table in TABLE_NAMES
}


def _hybrid_config(model: EmbeddingModel) -> HybridSearchConfig:
    # A new object every call: PGVectorStore writes the query into it on search.
    # Pyrefly reports HybridSearchConfig as abstract. It has no abstract
    # methods, so instantiating it works.
    return HybridSearchConfig(  # pyrefly: ignore[direct-abstract-base-instantiation]
        tsv_column=TSV_COLUMN, tsv_lang=TS_CONFIG, index_name=tsv_index_name(model)
    )


def create_table(engine: PGEngine, model: EmbeddingModel, embeddings: Embeddings) -> None:
    """Create `model.table_name` and its GIN index.

    The table has, in this order: `ID_COLUMN` (text, primary key),
    `content` (text), `embedding` (vector(model.dimensions)), `TSV_COLUMN`
    (tsvector), one NOT NULL text column per name in METADATA_COLUMNS, and
    `langchain_metadata` (json). The GIN index is named
    `tsv_index_name(model)`. `embeddings` is only needed to open the store
    that creates the index; nothing is embedded.

    Raises `sqlalchemy.exc.ProgrammingError` if the table exists.
    """
    engine.init_vectorstore_table(
        table_name=model.table_name,
        vector_size=model.dimensions,
        metadata_columns=[Column(name, "TEXT", nullable=False) for name in METADATA_COLUMNS],
        id_column=Column(ID_COLUMN, "TEXT", nullable=False),
        hybrid_search_config=_hybrid_config(model),
    )

    store = PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=model.table_name,
        metadata_columns=list(METADATA_COLUMNS),
        hybrid_search_config=_hybrid_config(model),
    )

    store.apply_hybrid_search_index()


def open_index_store(
    engine: PGEngine, model: EmbeddingModel, embeddings: Embeddings
) -> PGVectorStore:
    """Open the model's table for writing.

    Metadata columns are METADATA_COLUMNS. The store has its own new
    `HybridSearchConfig` for `TSV_COLUMN`, so its inserts fill that column.
    Raises `ValueError` if the table or one of its columns is missing.
    """
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=model.table_name,
        metadata_columns=list(METADATA_COLUMNS),
        hybrid_search_config=_hybrid_config(model),
    )


def open_search_store(
    engine: PGEngine,
    model: EmbeddingModel,
    embeddings: Embeddings,
    *,
    ef_search: int | None = None,
) -> PGVectorStore:
    """Open the model's table for search: like `open_index_store`, without a hybrid config.

    With `ef_search`, each dense query runs after `SET LOCAL hnsw.ef_search =
    <ef_search>` in its own transaction. It matters only when Postgres plans
    the query with the HNSW index (vectorstore/hnsw.py). None keeps the
    server's setting (pgvector's default is 40).
    """
    return PGVectorStore.create_sync(
        engine=engine,
        embedding_service=embeddings,
        table_name=model.table_name,
        metadata_columns=list(METADATA_COLUMNS),
        index_query_options=None if ef_search is None else HNSWQueryOptions(ef_search=ef_search),
    )
