"""Tests for cli.py: `profile-rag search`.

No Ollama: the tests replace `ollama_embeddings.ollama_digests` and
`ollama_embeddings.for_model`, as
test_cli_index.py does. The chunk table is built by the `fill_search_table`
fixture in conftest.py.

`DeterministicFakeEmbedding` vectors are hash-based, so no test expects a
dense ranking. They check the line format, counts, the visibility filter,
and the keyword hit that hybrid search puts first.
"""

from collections.abc import Callable

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.settings import get_settings
from profile_rag.vectorstore import ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15, EmbeddingModel

FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]


def chunk(chunk_id: str, text: str, heading_path: list[str], visibility: str) -> Document:
    source_id = chunk_id.split(":")[0]
    metadata: dict[str, object] = {
        "source_id": source_id,
        "chunk_id": chunk_id,
        "attribution": "self_authored",
        "visibility": visibility,
        "title": source_id,
        "heading_path": heading_path,
        "ordinal": 0,
    }
    return Document(page_content=text, metadata=metadata)


CHUNKS = [
    chunk("resume:v1:a", "Dana Whitlock is a backend engineer.", [], "public"),
    chunk(
        "resume:v1:b",
        "Moved four services to a PostgreSQL queue.",
        ["Dana Whitlock", "Experience"],
        "public",
    ),
    chunk("lantern:v1:a", "Install lantern-queue with pip.", ["lantern-queue"], "public"),
    chunk("talks:v1:a", "A talk on Kubernetes autoscaling.", ["Talks", "2026"], "sanitized"),
    # The registry never allows this value. The row is here to be filtered out.
    chunk("drafts:v1:a", "Draft notes on a Terraform migration.", ["Drafts"], "internal"),
]


@pytest.fixture
def database(empty_database_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """An empty throwaway database that the CLI's settings point at."""
    settings = get_settings().model_copy(
        update={"database_url": empty_database_url, "ollama_url": "http://ollama.test"}
    )
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    return empty_database_url


@pytest.fixture
def table(empty_database: str, database: str, fill_search_table: FillSearchTable) -> None:
    """`chunks_nomic_v15` with `CHUNKS`."""
    embeddings = DeterministicFakeEmbedding(size=NOMIC_V15.dimensions)
    fill_search_table(empty_database, NOMIC_V15, embeddings, CHUNKS)


@pytest.fixture
def digests(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """What the fake Ollama has pulled. Starts with the expected nomic model."""
    pulled = {NOMIC_V15.ollama_tag: NOMIC_V15.digest}

    def listed(base_url: str) -> dict[str, str]:
        return pulled

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", listed)
    return pulled


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch: pytest.MonkeyPatch) -> None:
    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(ollama_embeddings, "for_model", make)


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["search", *args])


def rows(result: Result) -> list[tuple[int, float, str, str]]:
    """(rank, distance, chunk_id, heading path) of each printed chunk."""
    lines = result.output.strip().splitlines()
    assert lines[0].startswith("chunks_nomic_v15, ")
    parsed: list[tuple[int, float, str, str]] = []
    for line in lines[2:]:
        rank, distance, chunk_id, path = line.split(maxsplit=3)
        parsed.append((int(rank), float(distance), chunk_id, path))
    return parsed


def chunk_by_id(chunk_id: str) -> Document:
    return next(doc for doc in CHUNKS if doc.metadata["chunk_id"] == chunk_id)


# --- Checks before the search ------------------------------------------------------


def test_an_unknown_model_is_an_error() -> None:
    result = run("Who is Dana?", "--model", "nomic")
    assert result.exit_code == 2
    assert "unknown model 'nomic'" in result.output


def test_an_unknown_mode_is_an_error() -> None:
    result = run("Who is Dana?", "--mode", "sparse")
    assert result.exit_code == 2
    assert "--mode" in result.output


def test_k_below_one_is_an_error() -> None:
    result = run("Who is Dana?", "-k", "0")
    assert result.exit_code == 2
    assert "-k" in result.output


@pytest.mark.usefixtures("database")
def test_a_model_ollama_lacks_is_an_error(digests: dict[str, str]) -> None:
    digests.clear()
    result = run("Who is Dana?")
    assert result.exit_code == 2
    assert "ollama pull nomic-embed-text:v1.5" in result.output


@pytest.mark.usefixtures("database", "digests")
def test_a_missing_table_is_an_error() -> None:
    result = run("Who is Dana?")
    assert result.exit_code == 2
    assert "no table chunks_nomic_v15. Run `profile-rag index --model nomic-v15` first." in (
        result.output
    )


# --- Searching ------------------------------------------------------------------------


@pytest.mark.usefixtures("table", "digests")
def test_dense_prints_k_ranked_lines_closest_first() -> None:
    result = run("Who is Dana?", "-k", "3")
    assert result.exit_code == 0, result.output
    assert result.output.startswith("chunks_nomic_v15, dense: 3 chunks\n")
    printed = rows(result)
    assert [rank for rank, *_ in printed] == [1, 2, 3]
    distances = [distance for _, distance, *_ in printed]
    assert distances == sorted(distances)
    for _, _, chunk_id, path in printed:
        heading_path = chunk_by_id(chunk_id).metadata["heading_path"]
        assert path == (" > ".join(heading_path) or "(before the first heading)")


@pytest.mark.usefixtures("table", "digests")
def test_dense_never_prints_a_hidden_chunk() -> None:
    result = run("Draft notes on a Terraform migration.", "-k", "50")
    assert result.exit_code == 0, result.output
    printed = {chunk_id for _, _, chunk_id, _ in rows(result)}
    assert printed == {doc.metadata["chunk_id"] for doc in CHUNKS} - {"drafts:v1:a"}


@pytest.mark.usefixtures("table", "digests")
def test_hybrid_prints_the_keyword_hit_first() -> None:
    result = run("Kubernetes", "--mode", "hybrid", "-k", "3")
    assert result.exit_code == 0, result.output
    assert result.output.startswith("chunks_nomic_v15, hybrid: 3 chunks\n")
    _, _, first, path = rows(result)[0]
    assert (first, path) == ("talks:v1:a", "Talks > 2026")


@pytest.mark.usefixtures("table", "digests")
def test_mmr_prints_k_different_chunks() -> None:
    result = run("Who is Dana?", "--mode", "mmr", "-k", "4")
    assert result.exit_code == 0, result.output
    chunk_ids = [chunk_id for _, _, chunk_id, _ in rows(result)]
    assert len(chunk_ids) == len(set(chunk_ids)) == 4
