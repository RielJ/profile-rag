"""Tests for cli.py: `profile-rag index`.

No Ollama: the tests replace `ollama_embeddings.ollama_digests` and
`ollama_embeddings.for_model`, so
the digest check sees a fixed list and the vectors come from
`DeterministicFakeEmbedding`. No Cloudflare either: the Workers AI tests
give `workers_ai_embeddings.for_model` an `httpx.MockTransport`. The
database is a throwaway one with every migration applied.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import override

import httpx
import psycopg
import pytest
import yaml
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from pydantic import SecretStr
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.db.migrate import migrate
from profile_rag.settings import get_settings
from profile_rag.vectorstore import ollama_embeddings, workers_ai_embeddings
from profile_rag.vectorstore.models import BGE_M3, NOMIC_V15, EmbeddingModel, WorkersAIModel
from profile_rag.vectorstore.prefixed import EmbeddingError, PrefixedEmbeddings

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

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", listed)
    return pulled


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(ollama_embeddings, "for_model", make)


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

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", refuse)
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "Ollama does not answer at http://ollama.test" in result.output


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

    monkeypatch.setattr(ollama_embeddings, "for_model", make)
    result = run("--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "error: chunk " in result.output
    assert result.output.rstrip().endswith(": too long")


# --- A Workers AI model -----------------------------------------------------------

CF_TOKEN = "cf-test-token-not-real"


@pytest.fixture
def cf_settings(database: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The `database` settings plus Cloudflare credentials; Ollama refuses."""
    settings = cli.get_settings().model_copy(
        update={"cf_account_id": "acct", "cf_api_token": SecretStr(CF_TOKEN)}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)

    def no_ollama(base_url: str) -> dict[str, str]:
        raise AssertionError("a Workers AI model must not ask Ollama")

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", no_ollama)
    return database


def cloudflare(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> None:
    """Route `workers_ai_embeddings.for_model` through `handler` instead of the network."""
    real = workers_ai_embeddings.for_model

    def make(
        model: WorkersAIModel, account_id: str | None, api_token: SecretStr | None
    ) -> PrefixedEmbeddings:
        return real(model, account_id, api_token, transport=httpx.MockTransport(handler))

    monkeypatch.setattr(workers_ai_embeddings, "for_model", make)


def bge_rows(database: str) -> int:
    with psycopg.connect(database) as conn:
        row = conn.execute("SELECT count(*) FROM chunks_bge_m3").fetchone()
    assert row is not None
    return row[0]


@pytest.mark.usefixtures("database")
def test_a_workers_ai_model_without_credentials_is_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = cli.get_settings().model_copy(update={"cf_account_id": None, "cf_api_token": None})
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    result = run("--model", "bge-m3", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "PROFILE_RAG_CF_ACCOUNT_ID" in result.output
    assert "PROFILE_RAG_CF_API_TOKEN" in result.output


def test_a_workers_ai_model_indexes_through_cloudflare(
    cf_settings: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[str] = []
    fake = DeterministicFakeEmbedding(size=BGE_M3.dimensions)

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == f"Bearer {CF_TOKEN}"
        texts = json.loads(request.content)["text"]
        sent.extend(texts)
        data = fake.embed_documents(texts)
        return httpx.Response(200, json={"success": True, "result": {"data": data}})

    cloudflare(monkeypatch, answer)
    result = run("--model", "bge-m3", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 0, result.output
    assert "created table chunks_bge_m3" in result.output
    total = bge_rows(cf_settings)
    assert total == len(sent) > 0
    assert counts_line(result) == (
        f"chunks_bge_m3: {total} chunks, {total} added, 0 updated, 0 skipped, 0 deleted"
    )


def test_a_cloudflare_400_stops_the_index_and_skips_nothing(
    cf_settings: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_long(request: httpx.Request) -> httpx.Response:
        error = {"code": 3010, "message": "Sequence too long: 9000 > 8192"}
        return httpx.Response(400, json={"success": False, "errors": [error], "result": None})

    cloudflare(monkeypatch, too_long)
    result = run("--model", "bge-m3", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "error: Workers AI @cf/baai/bge-m3, texts 0-" in result.output
    assert "HTTP 400 (3010: Sequence too long: 9000 > 8192)" in result.output
    assert "skipped" not in result.output
    assert CF_TOKEN not in result.output
    assert bge_rows(cf_settings) == 0
