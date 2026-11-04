"""Tests for corpus/documents.py.

Run them with:

    uv --directory backend run pytest tests/test_documents.py
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from profile_rag.corpus.documents import ApprovedSnapshotError, load_documents
from profile_rag.corpus.registry import Registry, load_registry

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "corpus-example"
APPROVED_AT = datetime(2026, 10, 1, 9, 0, tzinfo=UTC)
METADATA_KEYS = {"source_id", "title", "attribution", "url", "visibility"}


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "approved").mkdir()
    return tmp_path


def write(corpus: Path, source_id: str, content: str | bytes) -> str:
    """Write approved/<source_id>.md and return the SHA-256 of its bytes."""
    data = content.encode("utf-8") if isinstance(content, str) else content
    (corpus / "approved" / f"{source_id}.md").write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def entry(source_id: str, *, state: str = "approved", **fields: object) -> dict[str, object]:
    """A valid registry entry. Approved entries get every approval field."""
    item: dict[str, object] = {
        "id": source_id,
        "kind": "note",
        "title": f"Title of {source_id}",
        "attribution": "self_authored",
        "visibility": "public",
        "state": state,
    }
    if state in ("approved", "in_review"):
        item |= {"path": f"approved/{source_id}.md", "revision": "2026-10-01"}
    if state == "approved":
        item |= {"content_sha256": "0" * 64, "approved_at": APPROVED_AT, "approved_by": "dana"}
    if state == "rejected":
        item["rejection_reason"] = "Not public."
    return item | fields


def make_registry(*entries: dict[str, object]) -> Registry:
    return Registry.model_validate({"sources": list(entries)})


# --- corpus-example -------------------------------------------------------------------


def test_example_corpus_gives_the_approved_sources_in_registry_order() -> None:
    docs = load_documents(load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT)
    # linkedin-experience (candidate) and note-harborlight-migration
    # (rejected) are skipped.
    assert [doc.metadata["source_id"] for doc in docs] == [
        "resume-2026",
        "lantern-queue-readme",
        "blog-postgres-queues",
    ]


def test_example_metadata_is_copied_from_the_entry() -> None:
    docs = load_documents(load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT)
    assert docs[0].metadata == {
        "source_id": "resume-2026",
        "title": "Resume (2026)",
        "attribution": "self_report",
        "url": "https://example.com/dana/resume",
        "visibility": "public",
    }
    assert docs[1].metadata["attribution"] == "project_doc"
    assert docs[2].metadata["attribution"] == "self_authored"


def test_example_metadata_values_are_plain_strings() -> None:
    docs = load_documents(load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT)
    for doc in docs:
        assert set(doc.metadata) == METADATA_KEYS
        assert {type(value) for value in doc.metadata.values()} == {str}, doc.metadata


def test_example_page_content_is_the_file_text_unchanged() -> None:
    docs = load_documents(load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT)
    for doc, name in zip(
        docs,
        ["resume-2026.md", "lantern-queue-readme.md", "blog-postgres-queues.md"],
        strict=True,
    ):
        expected = (EXAMPLE_ROOT / "approved" / name).read_text(encoding="utf-8")
        assert doc.page_content == expected


def test_documents_have_no_id_yet() -> None:
    docs = load_documents(load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT)
    assert [doc.id for doc in docs] == [None, None, None]


# --- small registries -------------------------------------------------------------------


def test_an_empty_registry_gives_no_documents(corpus: Path) -> None:
    assert load_documents(make_registry(), corpus) == []


def test_unapproved_entries_are_skipped_even_when_their_file_exists(corpus: Path) -> None:
    sha = write(corpus, "kept", "# Kept\nPublic text.\n")
    write(corpus, "waiting", "Not reviewed yet.\n")
    registry = make_registry(
        entry("candidate-note", state="candidate"),
        entry("waiting", state="in_review"),
        entry("kept", content_sha256=sha),
        entry("rejected-note", state="rejected"),
    )
    docs = load_documents(registry, corpus)
    assert [doc.metadata["source_id"] for doc in docs] == ["kept"]
    assert docs[0].page_content == "# Kept\nPublic text.\n"


def test_unapproved_entries_are_never_read(corpus: Path) -> None:
    # The in-review entry's file does not exist. Reading it would raise.
    registry = make_registry(entry("waiting", state="in_review"))
    assert load_documents(registry, corpus) == []


def test_an_entry_without_url_has_url_none(corpus: Path) -> None:
    sha = write(corpus, "private-note", "Sanitized notes.\n")
    registry = make_registry(
        entry("private-note", content_sha256=sha, visibility="sanitized", attribution="third_party")
    )
    (doc,) = load_documents(registry, corpus)
    assert doc.metadata == {
        "source_id": "private-note",
        "title": "Title of private-note",
        "attribution": "third_party",
        "url": None,
        "visibility": "sanitized",
    }


def test_non_ascii_text_is_read_as_utf8(corpus: Path) -> None:
    sha = write(corpus, "cafe", "# Café\nÜber Postgres.\n")
    (doc,) = load_documents(make_registry(entry("cafe", content_sha256=sha)), corpus)
    assert doc.page_content == "# Café\nÜber Postgres.\n"


# --- snapshot checks --------------------------------------------------------------------


def test_a_missing_approved_file_raises_snapshot_error_naming_the_source(corpus: Path) -> None:
    with pytest.raises(ApprovedSnapshotError) as caught:
        load_documents(make_registry(entry("gone-note")), corpus)
    assert str(caught.value).startswith("gone-note")
    assert isinstance(caught.value.__cause__, OSError)


def test_a_changed_file_raises_snapshot_error_naming_the_source(corpus: Path) -> None:
    sha = write(corpus, "resume-old", "Senior Backend Engineer.\n")
    write(corpus, "resume-old", "Staff Backend Engineer.\n")
    with pytest.raises(ApprovedSnapshotError) as caught:
        load_documents(make_registry(entry("resume-old", content_sha256=sha)), corpus)
    assert str(caught.value).startswith("resume-old")


def test_the_hash_covers_bytes_not_text(corpus: Path) -> None:
    # Same text after read_text, different bytes: the CRLF file was approved,
    # then saved with LF line endings.
    sha = write(corpus, "notes", b"Line one.\r\nLine two.\r\n")
    write(corpus, "notes", b"Line one.\nLine two.\n")
    with pytest.raises(ApprovedSnapshotError, match=r"^notes"):
        load_documents(make_registry(entry("notes", content_sha256=sha)), corpus)


def test_one_bad_entry_fails_the_whole_load(corpus: Path) -> None:
    sha = write(corpus, "good", "Fine.\n")
    registry = make_registry(entry("good", content_sha256=sha), entry("missing"))
    with pytest.raises(ApprovedSnapshotError, match=r"^missing"):
        load_documents(registry, corpus)
