"""Keep a model's table in step with the approved corpus.

LangChain's `index()` hashes each chunk's content and metadata, asks a record
manager which hashes it already wrote, embeds only the new chunks, and
deletes the rows the run did not produce. The record manager is the table
`upsertion_record`, shared by every model, with one namespace per table.

`index()` ignores `Document.id`. The hash of the content and metadata
becomes the row ID, which is why `chunk_id` is a metadata column. An edited
chunk is one row deleted and one added, never an update.

`cleanup="full"` deletes any row the run did not produce, including every
row of a source that lost approval. `cleanup="incremental"` would delete
only within the sources that appear in the run, so a source that disappears
completely would keep its rows. The default key encoder, sha1, warns on
every call, so this module uses sha256.

Tests: backend/tests/test_indexing.py.
"""

from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import IndexingResult, index
from langchain_postgres import PGVectorStore

from profile_rag.vectorstore.models import EmbeddingModel


def record_namespace(model: EmbeddingModel) -> str:
    """The record manager namespace of the model's table."""
    return f"pgvector/{model.table_name}"


def open_record_manager(model: EmbeddingModel, database_url: str) -> SQLRecordManager:
    """A record manager for the model's table, with its table created.

    `create_schema()` creates `upsertion_record` only if it is missing, so
    this is safe on every run.
    """
    record_manager = SQLRecordManager(record_namespace(model), db_url=database_url)
    record_manager.create_schema()
    return record_manager


def index_corpus(
    docs: list[Document], store: PGVectorStore, record_manager: SQLRecordManager
) -> IndexingResult:
    """Make the store hold exactly the rows for `docs`, and return the counts.

    `docs` is every chunk of every approved source. Each chunk's metadata has
    `source_id`. Unchanged chunks are skipped, new and changed chunks are
    embedded and added, and every other row of the record manager's
    namespace is deleted, so an empty `docs` empties the table. The row IDs
    are SHA-256 hex digests.

    The result is a dict with `num_added`, `num_updated`, `num_skipped`, and
    `num_deleted`.
    """
    return index(
        docs, record_manager, store, cleanup="full", source_id_key="source_id", key_encoder="sha256"
    )
