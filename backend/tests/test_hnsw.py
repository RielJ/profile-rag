"""Tests for vectorstore/hnsw.py, evaluation/ann_compare.py, and the HNSW options of the CLI.

Each test gets a throwaway database whose `chunks_nomic_v15` holds the
corpus-example/ chunks, as `index` splits them, with vectors from
`DeterministicFakeEmbedding`. No Ollama: the CLI tests replace
`ollama_embeddings.ollama_digests` and
`ollama_embeddings.for_model`, as test_eval_cli.py does.

How the tests know `ef_search` reaches the database session: pgvector's
HNSW index scan returns at most `hnsw.ef_search` rows (iterative scans are
off by default). With the index forced (`enable_seqscan = off`), a dense
search for 5 chunks returns 1 row at `ef_search = 1` and 5 rows at the
default. A second test reads the SQL that the engine logs.
"""

import asyncio
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path

import psycopg
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_postgres import PGEngine, PGVectorStore
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.corpus import documents, registry, splitting
from profile_rag.evaluation.ann_compare import overlap_at_k, scan_line
from profile_rag.evaluation.runner import RunConfig
from profile_rag.retrievers.dense import VISIBLE, VISIBLE_VALUES, dense_retriever
from profile_rag.settings import get_settings
from profile_rag.vectorstore import hnsw, ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15, EmbeddingModel
from profile_rag.vectorstore.store import open_search_store

FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "corpus-example"
EXAMPLE_REGISTRY = registry.load_registry(EXAMPLE_ROOT / "sources.yaml")
CHUNKS = splitting.split_documents(documents.load_documents(EXAMPLE_REGISTRY, EXAMPLE_ROOT))
EMBEDDINGS = DeterministicFakeEmbedding(size=NOMIC_V15.dimensions)
INDEX = "chunks_nomic_v15_hnsw"
QUESTIONS: list[dict[str, object]] = [
    {
        "id": "t-degree",
        "question": "Northbridge",
        "expected": [{"source_id": "resume-2026", "span": "Northbridge University"}],
        "tags": ["direct"],
    },
    {"id": "t-none", "question": "Kubernetes", "expected": [], "tags": ["unanswerable"]},
]


@pytest.fixture
def database(
    empty_database: str,
    empty_database_url: str,
    fill_search_table: FillSearchTable,
    monkeypatch: pytest.MonkeyPatch,
) -> str:
    """URL of a throwaway database with the example chunks; the CLI points at it too."""
    fill_search_table(empty_database, NOMIC_V15, EMBEDDINGS, CHUNKS)
    settings = get_settings().model_copy(
        update={"database_url": empty_database_url, "ollama_url": "http://ollama.test"}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return empty_database_url


@pytest.fixture
def engine(database: str) -> Iterator[PGEngine]:
    engine = PGEngine.from_connection_string(url=database)
    yield engine
    asyncio.run(engine.close())


@pytest.fixture
def search_store(engine: PGEngine) -> PGVectorStore:
    return open_search_store(engine, NOMIC_V15, EMBEDDINGS)


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def listed(base_url: str) -> dict[str, str]:
        return {NOMIC_V15.ollama_tag: NOMIC_V15.digest}

    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", listed)
    monkeypatch.setattr(ollama_embeddings, "for_model", make)


def indexes(url: str) -> dict[str, str]:
    """Index name -> definition of every index on the nomic table, from pg_indexes."""
    with psycopg.connect(url) as conn:
        rows = conn.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename = %s",
            (NOMIC_V15.table_name,),
        ).fetchall()
    return dict(rows)


def dense_count(url: str, *, plan: hnsw.Plan, ef_search: int | None, k: int = 5) -> int:
    """How many chunks a dense search for `k` returns on an engine with `plan`."""
    with hnsw.open_engine(url, plan=plan) as engine:
        store = open_search_store(engine, NOMIC_V15, EMBEDDINGS, ef_search=ef_search)
        return len(dense_retriever(store, k=k, filter=VISIBLE).invoke("Go job queue"))


def cli_run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, list(args))


# --- apply, drop, status -------------------------------------------------------------


def test_apply_creates_the_index_with_the_planned_definition(
    database: str, search_store: PGVectorStore
) -> None:
    assert hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database) == "created"
    assert indexes(database)[INDEX] == (
        f"CREATE INDEX {INDEX} ON public.chunks_nomic_v15 USING hnsw"
        " (embedding vector_cosine_ops) WITH (m='16', ef_construction='64')"
    )
    status = hnsw.hnsw_status(database, NOMIC_V15)
    assert status is not None
    assert (status.name, status.valid, status.m, status.ef_construction) == (INDEX, True, 16, 64)
    assert status.size_bytes > 0


def test_apply_twice_changes_nothing(database: str, search_store: PGVectorStore) -> None:
    hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database)
    before = indexes(database)
    assert hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database) == "unchanged"
    assert indexes(database) == before


def test_apply_with_other_settings_rebuilds(database: str, search_store: PGVectorStore) -> None:
    hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database)
    result = hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database, m=8, ef_construction=32)
    assert result == "rebuilt"
    assert "WITH (m='8', ef_construction='32')" in indexes(database)[INDEX]


def test_status_is_none_without_an_index(database: str) -> None:
    assert hnsw.hnsw_status(database, NOMIC_V15) is None


def test_drop_then_search_still_works(database: str, search_store: PGVectorStore) -> None:
    hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database)
    hnsw.drop_hnsw(search_store, NOMIC_V15)
    hnsw.drop_hnsw(search_store, NOMIC_V15)  # absent: no error
    assert INDEX not in indexes(database)
    assert hnsw.hnsw_status(database, NOMIC_V15) is None
    found = dense_retriever(search_store, k=3, filter=VISIBLE).invoke("Go job queue")
    assert len(found) == 3


# --- planner and ef_search ---------------------------------------------------------


def explain(url: str, plan: hnsw.Plan) -> str:
    vector = EMBEDDINGS.embed_query("Go job queue")
    return scan_line(
        hnsw.explain_dense(url, NOMIC_V15, vector, k=5, visibility=VISIBLE_VALUES, plan=plan)
    )


def test_the_dense_query_uses_the_index_only_when_seqscan_is_off(
    database: str, search_store: PGVectorStore
) -> None:
    hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database)
    assert explain(database, "index") == f"Index Scan using {INDEX} on chunks_nomic_v15"
    assert explain(database, "exact").startswith("Seq Scan on chunks_nomic_v15")
    # 15 rows: a sequential scan is cheaper, so the default plan skips the index.
    assert explain(database, "default").startswith("Seq Scan on chunks_nomic_v15")


def test_ef_search_reaches_the_session(database: str, search_store: PGVectorStore) -> None:
    hnsw.apply_hnsw(search_store, NOMIC_V15, conninfo=database)
    # The index scan stops after ef_search candidates.
    assert dense_count(database, plan="index", ef_search=1) == 1
    assert dense_count(database, plan="index", ef_search=None) == 5
    assert dense_count(database, plan="index", ef_search=40) == 5
    # The exact plan does not use the index, so ef_search has no effect.
    assert dense_count(database, plan="exact", ef_search=1) == 5


def test_the_store_sets_ef_search_in_the_query_transaction(
    database: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = PGEngine.from_connection_string(url=database, echo=True)
    try:
        store = open_search_store(engine, NOMIC_V15, EMBEDDINGS, ef_search=17)
        with caplog.at_level(logging.INFO, logger="sqlalchemy.engine.Engine"):
            dense_retriever(store, k=3, filter=VISIBLE).invoke("Go job queue")
    finally:
        asyncio.run(engine.close())
    statements = [record.getMessage() for record in caplog.records]
    set_at = statements.index("SET LOCAL hnsw.ef_search = 17;")
    assert statements[set_at - 1] == "BEGIN (implicit)"
    assert any("ORDER BY" in s and "<=>" in s for s in statements[set_at + 1 :])


def test_without_ef_search_the_store_sets_nothing(
    database: str, caplog: pytest.LogCaptureFixture
) -> None:
    engine = PGEngine.from_connection_string(url=database, echo=True)
    try:
        store = open_search_store(engine, NOMIC_V15, EMBEDDINGS)
        with caplog.at_level(logging.INFO, logger="sqlalchemy.engine.Engine"):
            dense_retriever(store, k=3, filter=VISIBLE).invoke("Go job queue")
    finally:
        asyncio.run(engine.close())
    assert not any("hnsw.ef_search" in record.getMessage() for record in caplog.records)


# --- overlap -------------------------------------------------------------------------


def test_overlap_counts_shared_chunks_of_the_exact_list() -> None:
    assert overlap_at_k(["a", "b", "c"], ["c", "b", "a"]) == 1.0
    assert overlap_at_k(["a", "x"], ["a", "b", "c", "d"]) == 0.25
    assert overlap_at_k([], ["a"]) == 0.0
    assert overlap_at_k([], []) == 1.0


def test_an_ef_search_suffix_names_the_run() -> None:
    assert RunConfig("example", "nomic-v15", "dense", 8, 15).name == "example-nomic-v15-dense-k8"
    assert RunConfig("example", "nomic-v15", "dense", 8, 15, 40).name == (
        "example-nomic-v15-dense-k8-ef40"
    )


# --- CLI ------------------------------------------------------------------------------


@pytest.mark.usefixtures("database")
def test_cli_apply_status_drop() -> None:
    first = cli_run("hnsw", "apply", "--model", "nomic-v15")
    assert first.exit_code == 0, first.output
    assert first.output.startswith(f"created: {INDEX}: valid, m 16, ef_construction 64,")
    again = cli_run("hnsw", "apply", "--model", "nomic-v15")
    assert again.output.startswith("unchanged:")
    status = cli_run("hnsw", "status", "--model", "nomic-v15")
    assert status.exit_code == 0
    assert "USING hnsw (embedding vector_cosine_ops)" in status.output
    dropped = cli_run("hnsw", "drop", "--model", "nomic-v15")
    assert dropped.output == f"dropped: {INDEX}\n"
    assert cli_run("hnsw", "drop", "--model", "nomic-v15").output == f"absent: {INDEX}\n"
    assert cli_run("hnsw", "status", "--model", "nomic-v15").output == (
        "chunks_nomic_v15: no HNSW index\n"
    )


@pytest.mark.usefixtures("database")
def test_cli_apply_rejects_ef_construction_below_twice_m() -> None:
    result = cli_run("hnsw", "apply", "--m", "16", "--ef-construction", "20")
    assert result.exit_code == 2
    assert "at least 2 * --m (32)" in result.output


@pytest.mark.usefixtures("database")
def test_cli_search_takes_ef_search() -> None:
    assert cli_run("hnsw", "apply").exit_code == 0
    result = cli_run("search", "Go job queue", "-k", "3", "--ef-search", "20")
    assert result.exit_code == 0, result.output
    assert result.output.startswith("chunks_nomic_v15, dense: 3 chunks")
    assert cli_run("hnsw", "drop").exit_code == 0
    assert cli_run("search", "Go job queue", "-k", "3").exit_code == 0


def test_cli_search_rejects_ef_search_out_of_range() -> None:
    result = cli_run("search", "q", "--ef-search", "0")
    assert result.exit_code == 2
    assert "--ef-search" in result.output


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point the example set at a temporary question file, and the outputs at tmp_path."""
    questions = tmp_path / "questions.jsonl"
    questions.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    sets = {name: (questions, EXAMPLE_ROOT) for name in ("example", "dev", "heldout")}
    monkeypatch.setattr(cli, "EVAL_SETS", sets)
    monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "EVAL_DOCS_DIR", tmp_path / "docs")
    return {"results": tmp_path / "results", "docs": tmp_path / "docs"}


@pytest.mark.usefixtures("database")
def test_cli_eval_with_ef_search_writes_its_own_report(paths: dict[str, Path]) -> None:
    result = cli_run("eval", "--set", "example", "-k", "3", "--ef-search", "40")
    assert result.exit_code == 0, result.output
    report = paths["docs"] / "example-nomic-v15-dense-k3-ef40.md"
    assert "| example | nomic-v15 | dense (ef_search 40) | 3 |" in report.read_text()
    comparison = (paths["docs"] / "example-comparison.md").read_text()
    assert "| nomic-v15 | dense (ef_search 40) | 3 |" in comparison


@pytest.mark.usefixtures("database")
def test_cli_compare_needs_the_index(paths: dict[str, Path]) -> None:
    result = cli_run("hnsw", "compare", "--set", "example", "-k", "3")
    assert result.exit_code == 2
    assert "no HNSW index on chunks_nomic_v15" in result.output
    assert not paths["docs"].exists()


@pytest.mark.usefixtures("database")
def test_cli_compare_rejects_a_bad_ef_list(paths: dict[str, Path]) -> None:
    result = cli_run("hnsw", "compare", "--set", "example", "--ef-search", "10,ten")
    assert result.exit_code == 2
    assert "'ten' is not a whole number" in result.output


@pytest.mark.usefixtures("database")
def test_cli_compare_writes_numbers_only(paths: dict[str, Path]) -> None:
    assert cli_run("hnsw", "apply").exit_code == 0
    result = cli_run("hnsw", "compare", "--set", "example", "-k", "3", "--ef-search", "1,100")
    assert result.exit_code == 0, result.output
    report = (paths["docs"] / "hnsw-example-nomic-v15.md").read_text()
    assert f"| enable_seqscan = off | Index Scan using {INDEX} on chunks_nomic_v15 |" in report
    assert "| enable_indexscan = off | Seq Scan on chunks_nomic_v15 |" in report
    # ef_search 1 returns one chunk of three; ef_search 100 sees all 15 rows.
    assert "| hnsw | 1 | 0.333 | 0.333 | 2 |" in report
    assert "| hnsw | 100 | 1.000 | 1.000 | 0 |" in report
    assert "ef_search 100: overlap@3 mean 1.000, min 1.000, short 0" in result.output
    # No question, span, or chunk text in the report or the output.
    for text in (report, result.output):
        for question in QUESTIONS:
            assert str(question["question"]) not in text
        assert "Northbridge University" not in text
        for doc in CHUNKS:
            assert doc.page_content[:40] not in text
    assert not paths["results"].exists()


@pytest.mark.usefixtures("database")
def test_cli_search_warns_when_ef_search_is_below_the_dense_rows() -> None:
    low = cli_run("search", "Go job queue", "-k", "3", "--ef-search", "2")
    assert low.exit_code == 0, low.output
    assert "warning: --ef-search 2 is below the 3 rows that dense search" in low.output
    hybrid = cli_run("search", "Go job queue", "-k", "3", "--mode", "hybrid", "--ef-search", "10")
    assert "warning: --ef-search 10 is below the 20 rows that hybrid search" in hybrid.output
    assert "warning" not in cli_run("search", "Go job queue", "-k", "3", "--ef-search", "3").output
