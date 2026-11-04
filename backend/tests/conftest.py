"""Shared pytest fixtures.

A fixture is a function pytest runs to build a value a test asks for by
parameter name. `scope="session"` builds it once for the whole test run.
"""

import asyncio
import logging
import os
from collections.abc import Callable, Generator, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import langsmith
import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGEngine
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import PoolTimeout
from sqlalchemy.engine import make_url

from profile_rag.api.app import create_app
from profile_rag.api.deps import get_pool
from profile_rag.db.pool import Pool, create_pool
from profile_rag.settings import get_settings
from profile_rag.tracing import TRACING_SWITCHES
from profile_rag.vectorstore import store
from profile_rag.vectorstore.models import EmbeddingModel


def _tracing_off() -> None:
    """No test sends a trace, whatever the parent environment or .env says.

    Environment variables take precedence over .env, so LANGSMITH_TRACING=false
    here also overrides a LANGSMITH_TRACING=true in .env. Without an API key,
    a client could not authenticate either. Tests that turn tracing on
    (test_tracing.py) inject a client that captures requests instead.
    """
    for name in TRACING_SWITCHES:
        os.environ.pop(name, None)
    os.environ["LANGSMITH_TRACING"] = "false"
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "LANGSMITH_ENDPOINT"):
        os.environ.pop(name, None)
    langsmith.configure(client=None, enabled=False)


_tracing_off()


@pytest.fixture(autouse=True)
def tracing_stays_off() -> Iterator[None]:
    """Turn tracing off again after each test, in case a test turned it on."""
    yield
    _tracing_off()


# Port 9 ("discard") has no database listening, so connections fail fast.
UNREACHABLE_DSN = "postgresql://nobody:nothing@127.0.0.1:9/none"

# Prefix of throwaway databases. If a test run is killed before cleanup, find
# leftovers with `\l profile_rag_tmp_*` in `make psql-test` and drop them.
TEMPORARY_DATABASE_PREFIX = "profile_rag_tmp_"


def _test_database_url() -> str:
    url = get_settings().test_database_url
    if url is None:
        pytest.fail("TEST_DATABASE_URL is not set. Run `make setup` to create .env.")
    return str(url)


@contextmanager
def temporary_database() -> Generator[str]:
    """Create an empty database, yield its connection string, then drop it.

    Each caller gets a database no other test touches, so tests that create
    or break a schema cannot affect each other or the shared test database.
    """
    admin_url = _test_database_url()
    name = f"{TEMPORARY_DATABASE_PREFIX}{uuid4().hex[:12]}"
    # CREATE and DROP DATABASE cannot run inside a transaction: autocommit.
    # A database name is an identifier, not a value, so it cannot be a %s
    # parameter; sql.Identifier quotes it safely instead.
    with psycopg.connect(admin_url, autocommit=True) as admin:
        admin.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    try:
        yield make_conninfo(admin_url, dbname=name)
    finally:
        with psycopg.connect(admin_url, autocommit=True) as admin:
            # FORCE ends any connection a failed test left open.
            admin.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )


def _is_project_frame(path: Path | str, root: Path) -> bool:
    """True for files in this project, false for the stdlib and .venv."""
    try:
        relative = Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return relative.parts[:1] != (".venv",)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, object]:
    """Show only this project's frames in failure tracebacks.

    A failure through FastAPI's test client otherwise prints ~80 lines of
    Starlette and anyio frames. The exception line itself is kept. Run
    `pytest --full-trace` to see every frame.
    """
    excinfo = call.excinfo
    if excinfo is not None and not item.config.getoption("fulltrace"):
        _drop_library_frames(excinfo.tb, item.config.rootpath.resolve())
        # Rebuild pytest's cached view from the relinked chain.
        excinfo.traceback = type(excinfo.traceback)(excinfo.tb)
    # Code before the yield runs before pytest builds the report, so the
    # report shows the shortened traceback.
    yield


def _drop_library_frames(head: TracebackType, root: Path) -> None:
    """Unlink stdlib and .venv frames from a raw traceback chain.

    pytest rebuilds tracebacks from `tb_next` links, so filtering its
    Traceback view is not enough; the links themselves are rewritten.
    The head (a pytest frame) is kept and pytest cuts it later.
    """
    tail = head
    current = head.tb_next
    while current is not None:
        if _is_project_frame(current.tb_frame.f_code.co_filename, root):
            tail.tb_next = current
            tail = current
        current = current.tb_next
    tail.tb_next = None


TemporaryDatabase = Callable[[], AbstractContextManager[str]]


@pytest.fixture(scope="session")
def temporary_database_factory() -> TemporaryDatabase:
    """`with temporary_database_factory() as conninfo:` makes a throwaway database.

    A fixture, not an import, so module-scoped fixtures can use it too.
    """
    return temporary_database


@pytest.fixture
def empty_database() -> Iterator[str]:
    """Connection string of an empty throwaway database, dropped after the test."""
    with temporary_database() as conninfo:
        yield conninfo


def as_url(conninfo: str) -> str:
    """The test database URL with the database of `conninfo`.

    `PGEngine` and `SQLRecordManager` take a URL, not a `key=value` string.
    """
    name = conninfo_to_dict(conninfo)["dbname"]
    url = make_url(_test_database_url()).set(database=str(name))
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def url_of() -> Callable[[str], str]:
    """`url_of(conninfo)` is the URL that `PGEngine` needs for that database."""
    return as_url


@pytest.fixture
def empty_database_url(empty_database: str) -> str:
    """URL of an empty throwaway database, dropped after the test."""
    return as_url(empty_database)


@pytest.fixture(scope="session")
def test_pool() -> Iterator[Pool]:
    """A pool connected to the test database (profile_rag_test)."""
    pool = create_pool(_test_database_url())
    try:
        pool.wait(timeout=5)
    except PoolTimeout:
        pool.close()
        pytest.fail("Cannot reach the test database. Run `make db-up` first.")
    yield pool
    pool.close()


@pytest.fixture
def unreachable_pool(caplog: pytest.LogCaptureFixture) -> Iterator[Pool]:
    """An open pool pointing at an address where no database listens."""
    # The pool logs a WARNING on every failed reconnect. Here failure is the
    # point, so those lines are noise in test reports. caplog restores the
    # level after the test.
    caplog.set_level(logging.ERROR, logger="psycopg.pool")
    pool = create_pool(UNREACHABLE_DSN)
    yield pool
    pool.close()


@pytest.fixture
def closed_pool() -> Pool:
    """A pool that was never opened, like one after application shutdown."""
    return create_pool(UNREACHABLE_DSN, open_now=False)


@pytest.fixture
def make_client() -> Callable[[Pool], TestClient]:
    """Build an API test client whose routes receive the given pool.

    The client is not used as a context manager, so the application's
    lifespan (which connects to the real database) does not run.
    """

    def build(pool: Pool) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_pool] = lambda: pool
        return TestClient(app)

    return build


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--live",
        action="store_true",
        help="Also run tests marked `live`, which call a real Ollama or Pi.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests marked `live` unless pytest runs with `--live`."""
    if config.getoption("live"):
        return
    skip = pytest.mark.skip(reason="calls a real service: run with --live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


# --- A filled chunk table for the retriever and `search` tests ---------------


def _fill_search_table(
    conninfo: str, model: EmbeddingModel, embeddings: Embeddings, chunks: list[Document]
) -> None:
    """Create `model.table_name` in the database of `conninfo` and write `chunks`.

    Uses the same code as `profile-rag index`: `store.create_table` and the
    index store, which fills the tsvector column. Rows are added directly, not
    through `index()`, so no record manager table is needed.
    """
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    engine = PGEngine.from_connection_string(url=as_url(conninfo))
    try:
        store.create_table(engine, model, embeddings)
        index_store = store.open_index_store(engine, model, embeddings)
        index_store.add_documents(chunks, ids=[chunk.metadata["chunk_id"] for chunk in chunks])
    finally:
        asyncio.run(engine.close())


FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]


@pytest.fixture(scope="session")
def fill_search_table() -> FillSearchTable:
    """`fill_search_table(conninfo, model, embeddings, chunks)` creates and fills a chunk table."""
    return _fill_search_table
