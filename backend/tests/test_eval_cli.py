"""Tests for cli.py: `profile-rag eval`.

No Ollama: the tests replace `ollama_embeddings.ollama_digests` and
`ollama_embeddings.for_model`, as
test_cli_search.py does. The question file and the output folders are
temporary; the corpus is corpus-example/, split as `index` splits it, and
written to a throwaway database with the `fill_search_table` fixture.

`DeterministicFakeEmbedding` vectors are hash-based, so the exact-number
tests use hybrid search with one-word questions: keyword search puts the
only chunk containing the word first.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.corpus import documents, registry, splitting
from profile_rag.settings import get_settings
from profile_rag.vectorstore import ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15, EmbeddingModel

FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "corpus-example"
EXAMPLE_REGISTRY = registry.load_registry(EXAMPLE_ROOT / "sources.yaml")
CHUNKS = splitting.split_documents(documents.load_documents(EXAMPLE_REGISTRY, EXAMPLE_ROOT))

QUESTIONS: list[dict[str, object]] = [
    {
        "id": "t-degree",
        "question": "Northbridge",
        "expected": [{"source_id": "resume-2026", "span": "Northbridge University"}],
        "tags": ["direct"],
    },
    {
        "id": "t-metrics",
        "question": "Prometheus",
        "expected": [
            {"source_id": "lantern-queue-readme", "span": "exported in Prometheus format"}
        ],
        "tags": ["attribution_trap"],
    },
    {"id": "t-none", "question": "Kubernetes", "expected": [], "tags": ["unanswerable"]},
]


@pytest.fixture
def paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point the example set at a temporary question file, and the outputs at tmp_path."""
    questions = tmp_path / "questions.jsonl"
    questions.write_text("".join(json.dumps(q) + "\n" for q in QUESTIONS))
    monkeypatch.setattr(
        cli,
        "EVAL_SETS",
        {
            "example": (questions, EXAMPLE_ROOT),
            "dev": (tmp_path / "absent.jsonl", EXAMPLE_ROOT),
            "heldout": (tmp_path / "absent.jsonl", EXAMPLE_ROOT),
        },
    )
    monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "EVAL_DOCS_DIR", tmp_path / "docs")
    return {"questions": questions, "results": tmp_path / "results", "docs": tmp_path / "docs"}


@pytest.fixture
def database(empty_database_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """An empty throwaway database that the CLI's settings point at."""
    settings = get_settings().model_copy(
        update={"database_url": empty_database_url, "ollama_url": "http://ollama.test"}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return empty_database_url


@pytest.fixture
def fill(
    empty_database: str, database: str, fill_search_table: FillSearchTable
) -> Callable[[list[Document]], None]:
    """`fill(chunks)` writes `chunks` to `chunks_nomic_v15`."""
    embeddings = DeterministicFakeEmbedding(size=NOMIC_V15.dimensions)

    def write(chunks: list[Document]) -> None:
        fill_search_table(empty_database, NOMIC_V15, embeddings, chunks)

    return write


@pytest.fixture(autouse=True)
def fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    def listed(base_url: str) -> dict[str, str]:
        return {NOMIC_V15.ollama_tag: NOMIC_V15.digest}

    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", listed)
    monkeypatch.setattr(ollama_embeddings, "for_model", make)


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["eval", *args])


# --- Checks before the run -----------------------------------------------------------


def test_an_unknown_set_is_an_error() -> None:
    result = run("--set", "prod")
    assert result.exit_code == 2
    assert "--set" in result.output


@pytest.mark.usefixtures("paths")
def test_a_missing_question_file_is_an_error() -> None:
    result = run("--set", "dev")
    assert result.exit_code == 2
    assert "absent.jsonl: cannot read" in result.output


def test_an_invalid_question_file_names_the_line(paths: dict[str, Path]) -> None:
    paths["questions"].write_text(json.dumps(QUESTIONS[0]) + "\n{broken\n")
    result = run("--set", "example")
    assert result.exit_code == 2
    assert "questions.jsonl:2: not JSON" in result.output


def test_label_errors_stop_the_run_without_printing_spans(paths: dict[str, Path]) -> None:
    bad = {**QUESTIONS[0], "expected": [{"source_id": "resume-2026", "span": "Zanzibar"}]}
    paths["questions"].write_text(json.dumps(bad) + "\n")
    result = run("--set", "example")
    assert result.exit_code == 2
    assert "t-degree label 0 (resume-2026): the span is in no chunk of the source" in (
        result.output
    )
    assert "1 label errors in questions.jsonl" in result.output
    assert "Zanzibar" not in result.output
    assert not paths["docs"].exists()


@pytest.mark.usefixtures("paths", "database")
def test_a_missing_table_is_an_error() -> None:
    result = run("--set", "example")
    assert result.exit_code == 2
    assert "no table chunks_nomic_v15" in result.output


@pytest.mark.usefixtures("paths")
def test_a_table_of_another_corpus_is_an_error(fill: Callable[[list[Document]], None]) -> None:
    fill(CHUNKS[:-1])
    result = run("--set", "example")
    assert result.exit_code == 2
    assert "chunks_nomic_v15 does not hold this set's corpus (corpus-example/)" in result.output
    assert "1 chunks missing, 0 extra" in result.output
    assert "make index CORPUS_DIR=corpus-example MODEL=nomic-v15" in result.output


# --- Running ---------------------------------------------------------------------------


def test_a_run_prints_the_metrics_and_writes_three_files(
    paths: dict[str, Path], fill: Callable[[list[Document]], None]
) -> None:
    fill(CHUNKS)
    result = run("--set", "example", "--mode", "hybrid", "-k", "3")
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == (
        "example-nomic-v15-hybrid-k3: recall@3 1.000, MRR 1.000, n 2, unanswerable 1"
    )
    assert lines[1] == "misses: none"

    detail = paths["results"] / "example-nomic-v15-hybrid-k3.jsonl"
    report = paths["docs"] / "example-nomic-v15-hybrid-k3.md"
    comparison = paths["docs"] / "example-comparison.md"
    assert [line.removeprefix("wrote ") for line in lines[2:]] == [
        str(detail),
        str(report),
        str(comparison),
    ]
    records = [json.loads(line) for line in detail.read_text().splitlines()]
    assert [r["id"] for r in records] == ["t-degree", "t-metrics", "t-none"]
    assert [len(r["retrieved"]) for r in records] == [3, 3, 3]
    assert "| example | nomic-v15 | hybrid | 3 | 1.000 | 1.000 | 2 | 1 | 15 |" in report.read_text()
    assert "| nomic-v15 | hybrid | 3 | 1.000 | 1.000 | 2 | 1 | none |" in comparison.read_text()

    for path in (report, comparison):
        text = path.read_text()
        for question in QUESTIONS:
            assert str(question["question"]) not in text
