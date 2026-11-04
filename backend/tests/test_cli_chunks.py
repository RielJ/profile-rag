"""Tests for cli.py: `profile-rag chunks show`.

The command calls corpus/documents.py and corpus/splitting.py.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.corpus import documents, splitting
from profile_rag.corpus.registry import Registry
from profile_rag.corpus.splitting import SPLITTER_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "corpus-example"


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["chunks", "show", *args])


def test_example_source_prints_numbered_chunks() -> None:
    result = run("resume-2026", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("resume-2026: ")
    assert f"({SPLITTER_VERSION}, at most 200 words, overlap 40)" in lines[0]
    count = int(lines[0].split(": ")[1].split()[0])
    assert count > 0
    for ordinal in range(count):
        assert any(line.startswith(f"[{ordinal}] ") for line in lines), ordinal
    assert "id resume-2026:" in result.output


def test_each_chunk_shows_path_size_id_and_text(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_split(docs: list[Document], *, chunk_words: int, overlap_words: int) -> list[Document]:
        return [
            Document(
                page_content="Dana Whitlock, backend engineer.",
                metadata={"heading_path": [], "ordinal": 0, "chunk_id": "x:0"},
            ),
            Document(
                page_content="Python, Go.\nPostgreSQL.",
                metadata={
                    "heading_path": ["Dana Whitlock", "Skills"],
                    "ordinal": 1,
                    "chunk_id": "x:1",
                },
            ),
        ]

    monkeypatch.setattr(splitting, "split_documents", fake_split)
    result = run("resume-2026", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[1:] == [
        "",
        "[0] (before the first heading)",
        "    4 words, id x:0",
        "    | Dana Whitlock, backend engineer.",
        "",
        "[1] Dana Whitlock > Skills",
        "    3 words, id x:1",
        "    | Python, Go.",
        "    | PostgreSQL.",
    ]


def test_sizes_are_passed_to_the_splitter(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, int] = {}

    def fake_split(docs: list[Document], *, chunk_words: int, overlap_words: int) -> list[Document]:
        seen.update(chunk_words=chunk_words, overlap_words=overlap_words, docs=len(docs))
        return []

    monkeypatch.setattr(splitting, "split_documents", fake_split)
    result = run(
        "resume-2026", "--root", str(EXAMPLE_ROOT), "--chunk-words", "50", "--overlap-words", "5"
    )
    assert result.exit_code == 0, result.output
    assert seen == {"chunk_words": 50, "overlap_words": 5, "docs": 1}
    assert "at most 50 words, overlap 5" in result.output


def test_bad_sizes_are_an_error() -> None:
    result = run("resume-2026", "--root", str(EXAMPLE_ROOT), "--overlap-words", "200")
    assert result.exit_code == 2
    assert result.output.startswith("error: ")


def test_an_unknown_source_is_an_error() -> None:
    result = run("no-such-source", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert "no-such-source" in result.output


@pytest.mark.parametrize(
    ("source_id", "state"),
    [("linkedin-experience", "candidate"), ("note-harborlight-migration", "rejected")],
)
def test_an_unapproved_source_is_refused(source_id: str, state: str) -> None:
    result = run(source_id, "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert f"{source_id} is {state}, not approved" in result.output


def test_a_changed_file_is_an_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def changed(registry: Registry, corpus_dir: Path) -> list[Document]:
        raise documents.ApprovedSnapshotError("resume-2026: approved/resume-2026.md changed")

    monkeypatch.setattr(documents, "load_documents", changed)
    result = run("resume-2026", "--root", str(EXAMPLE_ROOT))
    assert result.exit_code == 2
    assert result.output == "error: resume-2026: approved/resume-2026.md changed\n"


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "sources.yaml").write_text(
        (EXAMPLE_ROOT / "sources.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    result = run("resume-2026", "--root", str(tmp_path))
    assert result.exit_code == 2
    assert result.output.startswith("error: resume-2026: ")
