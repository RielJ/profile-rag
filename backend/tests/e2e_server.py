"""The API server for the browser tests (frontend/tests/e2e/). Never a real server.

    PROFILE_RAG_TEST_MODE=1 uv run python tests/e2e_server.py --port 8765

It serves the real routes (api/routes.py) and the real answer chain
(rag.py), with three test doubles in place of the outside services:

- Database: a throwaway `profile_rag_tmp_*` database, created on the
  TEST_DATABASE_URL server at startup and dropped at shutdown
  (conftest.temporary_database). DATABASE_URL is never read.
- Embeddings: `DeterministicFakeEmbedding`, so Ollama is not needed. The
  vectors are hash-based, so retrieval order is arbitrary but stable.
- Pi: e2e_fake_pi.py, which answers from the prompt without a model.
  PROFILE_RAG_PI_BIN is never read.

The corpus is always corpus-example/ (sources and chunks), whatever
CORPUS_DIR says. Tracing is off: importing conftest turns it off, as it
does for every test.

Safeguards, so this cannot run by accident or against real data:

- It lives in backend/tests/, outside the installed package. The production
  app (`uvicorn profile_rag.api.app:app`) has no switch for fake services.
- It refuses to start unless PROFILE_RAG_TEST_MODE=1 is set.
- It refuses any database whose name does not start with
  `profile_rag_tmp_`, so it never writes to the main or shared test database.
- It listens on 127.0.0.1 only.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import psycopg
import uvicorn
from conftest import TEMPORARY_DATABASE_PREFIX, as_url, temporary_database
from e2e_fake_pi import E2E_FAKE_PI
from fastapi import FastAPI
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import Runnable
from langchain_postgres import PGEngine
from psycopg.conninfo import conninfo_to_dict

from profile_rag.api.app import create_app
from profile_rag.corpus import documents, registry, sources_table, splitting
from profile_rag.db.migrate import migrate
from profile_rag.db.pool import create_pool
from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import PiConfig
from profile_rag.rag import ChainCache
from profile_rag.retrievers.factory import SearchMode, build_retriever
from profile_rag.tracing import EXAMPLE_CORPUS
from profile_rag.vectorstore import store
from profile_rag.vectorstore.models import NOMIC_V15

TEST_MODE_VARIABLE = "PROFILE_RAG_TEST_MODE"
MODEL = NOMIC_V15
FAKE_PI_MODEL = "e2e/fake-model"


class RefusedToStart(SystemExit):
    def __init__(self, reason: str) -> None:
        super().__init__(f"e2e_server: {reason}")


def require_test_mode() -> None:
    if os.environ.get(TEST_MODE_VARIABLE) != "1":
        raise RefusedToStart(f"set {TEST_MODE_VARIABLE}=1 to start the fake-service test server.")


def require_throwaway(conninfo: str) -> None:
    name = str(conninfo_to_dict(conninfo).get("dbname", ""))
    if not name.startswith(TEMPORARY_DATABASE_PREFIX):
        raise RefusedToStart(f"{name!r} is not a throwaway {TEMPORARY_DATABASE_PREFIX}* database.")


def example_chunks() -> tuple[registry.Registry, list[Document]]:
    current = registry.load_registry(EXAMPLE_CORPUS / registry.REGISTRY_FILENAME)
    loaded = documents.load_documents(current, EXAMPLE_CORPUS)
    return current, splitting.split_documents(loaded)


def fill(conninfo: str, engine: PGEngine, embeddings: DeterministicFakeEmbedding) -> None:
    """Create the schema and load corpus-example into the throwaway database."""
    require_throwaway(conninfo)
    migrate(conninfo)
    current, chunks = example_chunks()
    with psycopg.connect(conninfo) as conn:
        sources_table.sync_sources(conn, current)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    store.create_table(engine, MODEL, embeddings)
    index_store = store.open_index_store(engine, MODEL, embeddings)
    index_store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])


def chains(conninfo: str, engine: PGEngine, embeddings: DeterministicFakeEmbedding) -> ChainCache:
    database_url = as_url(conninfo)
    search_store = store.open_search_store(engine, MODEL, embeddings)

    def make_retriever(mode: SearchMode, k: int) -> Runnable[str, list[Document]]:
        return build_retriever(mode, search_store, k, conninfo=database_url, model=MODEL)

    config = PiConfig(
        model=FAKE_PI_MODEL, executable=(sys.executable, str(E2E_FAKE_PI)), timeout=10.0
    )
    model = PiChatModel(config=config)
    return ChainCache(make_retriever, model, default_mode="hybrid", default_k=5)


@asynccontextmanager
async def e2e_lifespan(app: FastAPI) -> AsyncGenerator[None]:
    with temporary_database() as conninfo:
        require_throwaway(conninfo)
        embeddings = DeterministicFakeEmbedding(size=MODEL.dimensions)
        engine = PGEngine.from_connection_string(url=as_url(conninfo))
        pool = create_pool(conninfo)
        try:
            await asyncio.to_thread(fill, conninfo, engine, embeddings)
            app.state.pool = pool
            app.state.chains = chains(conninfo, engine, embeddings)
            yield
        finally:
            pool.close()
            await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="The API server for the browser tests.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    require_test_mode()
    uvicorn.run(create_app(e2e_lifespan), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
