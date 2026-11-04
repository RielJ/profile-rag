"""Tests for cli.py: `profile-rag index`.

No Ollama: the tests replace `cli._ollama_digests` and `cli._embeddings`, so
the digest check sees a fixed list and the vectors come from
`DeterministicFakeEmbedding`. The database is a throwaway one with every
migration applied.
"""

from pathlib import Path
from typing import override

import psycopg
import pytest
import yaml
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.db.migrate import migrate
from profile_rag.settings import get_settings
from profile_rag.vectorstore.models import NOMIC_V15, EmbeddingModel
from profile_rag.vectorstore.prefixed import EmbeddingError

EXAMPLE_ROOT = Path(__file__).resolve().parents[2] / "corpus-example"


@pytest.fixture
def database(empty_database: str, empty_database_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """A migrated throwaway database that the CLI's settings point at."""
    migrate(empty_database)
    settings = get_settings().model_copy(
        update={"database_url": empty_database_url, "ollama_url": "http://ollama.test"}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return empty_database


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """What the fake Ollama has pulled. Starts with the expected nomic model."""
    pulled = {NOMIC_V15.ollama_tag: NOMIC_V15.digest}

    def listed(base_url: str) -> dict[str, str]:
        return pulled

    monkeypatch.setattr(cli, "_ollama_digests", listed)
    return pulled


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(cli, "_embeddings", make)


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["index", *args])


def copy_example(tmp_path: Path) -> Path:
    for path in [EXAMPLE_ROOT / "sources.yaml", *(EXAMPLE_ROOT / "approved").iterdir()]:
        target = tmp_path / path.relative_to(EXAMPLE_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    return tmp_path


def reject(root: Path, source_id: str) -> None:
    path = root / "sources.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in data["sources"]:
        if entry["id"] == source_id:
            for key in ("path", "revision", "content_sha256", "approved_at", "approved_by"):
                entry.pop(key, None)
            entry.update(state="rejected", rejection_reason="test")
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def counts_line(result: Result) -> str:
    return result.output.strip().splitlines()[-1]


def chunk_count(database: str, source_id: str | None = None) -> int:
    with psycopg.connect(database) as conn:
        if source_id is None:
            row = conn.execute("SELECT count(*) FROM chunks_nomic_v15").fetchone()
        else:
            row = conn.execute(
                "SELECT count(*) FROM chunks_nomic_v15 WHERE source_id = %s", (source_id,)
            ).fetchone()
    assert row is not None
    return row[0]


# --- The model check ---------------------------------------------------------


def test_an_unknown_model_is_an_error() -> None:
    result = run("--model", "nomic")
    assert result.exit_code == 2
    assert "unknown model 'nomic'" in result.output
    assert "nomic-v15" in result.output


@pytest.mark.usefixtures("database")
def test_a_model_ollama_lacks_is_an_error(digests: dict[str, str]) -> None:
    digests.clear()
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "ollama pull nomic-embed-text:v1.5" in result.output


@pytest.mark.usefixtures("database")
def test_a_different_digest_is_an_error(digests: dict[str, str]) -> None:
    digests[NOMIC_V15.ollama_tag] = "f" * 64
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert f"has digest {'f' * 12}" in result.output
    assert f"expects {NOMIC_V15.digest[:12]}" in result.output


@pytest.mark.usefixtures("database")
def test_ollama_not_running_is_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(base_url: str) -> dict[str, str]:
        raise ConnectionError("Failed to connect to Ollama.")

    monkeypatch.setattr(cli, "_ollama_digests", refuse)
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "cannot reach Ollama at http://ollama.test" in result.output


# --- Indexing -------------------------------------------------------------------


@pytest.mark.usefixtures("digests", "fake_embeddings")
def test_first_run_adds_every_chunk_and_second_skips_them(database: str) -> None:
    first = run("--root", str(EXAMPLE_ROOT))
    assert first.exit_code == 0, first.output
    assert "sources: 3 approved, 0 removed" in first.output
    assert "created table chunks_nomic_v15" in first.output
    total = chunk_count(database)
    assert total > 0
    assert counts_line(first) == (
        f"chunks_nomic_v15: {total} chunks, {total} added, 0 updated, 0 skipped, 0 deleted"
    )

    second = run("--root", str(EXAMPLE_ROOT))
    assert second.exit_code == 0, second.output
    assert "created table" not in second.output
    assert counts_line(second) == (
        f"chunks_nomic_v15: {total} chunks, 0 added, 0 updated, {total} skipped, 0 deleted"
    )


@pytest.mark.usefixtures("digests", "fake_embeddings")
def test_a_rejected_source_loses_its_rows_and_its_sources_row(
    database: str, tmp_path: Path
) -> None:
    root = copy_example(tmp_path)
    assert run("--root", str(root)).exit_code == 0
    total = chunk_count(database)
    readme = chunk_count(database, "lantern-queue-readme")
    assert readme > 0

    reject(root, "lantern-queue-readme")
    result = run("--root", str(root))
    assert result.exit_code == 0, result.output
    assert "sources: 2 approved, 1 removed" in result.output
    assert "removed lantern-queue-readme" in result.output
    kept = total - readme
    assert counts_line(result) == (
        f"chunks_nomic_v15: {kept} chunks, 0 added, 0 updated, {kept} skipped, {readme} deleted"
    )
    assert chunk_count(database, "lantern-queue-readme") == 0


@pytest.mark.usefixtures("digests")
def test_an_embedding_error_names_the_chunk(database: str, monkeypatch: pytest.MonkeyPatch) -> None:
    class Refuses(DeterministicFakeEmbedding):
        @override
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise EmbeddingError("too long", position=0, text=texts[0])

    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return Refuses(size=model.dimensions)

    monkeypatch.setattr(cli, "_embeddings", make)
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "error: chunk " in result.output
    assert result.output.rstrip().endswith(": too long")
