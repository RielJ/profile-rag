"""Tests for corpus/snapshot.py. No network: downloads use a fake fetch."""

import subprocess
from datetime import date
from pathlib import Path

import pytest

from profile_rag.corpus.snapshot import Download, Fetch, Snapshot, SnapshotError, take_snapshot

TODAY = date(2026, 10, 1)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    return corpus


@pytest.fixture
def outside(tmp_path: Path) -> Path:
    """A folder outside the corpus, standing in for ~/Documents."""
    folder = tmp_path / "outside"
    folder.mkdir()
    return folder


def fetch_returning(content: bytes, content_type: str | None, calls: list[str]) -> Fetch:
    """A fake `fetch_url` that records each URL in `calls`."""

    def fetch(url: str) -> Download:
        calls.append(url)
        return Download(content, content_type)

    return fetch


def git(directory: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def make_repo(directory: Path) -> None:
    git(directory, "init", "--quiet")
    git(directory, "config", "user.email", "test@example.com")
    git(directory, "config", "user.name", "Test")
    git(directory, "config", "commit.gpgsign", "false")


def test_copies_a_local_file_byte_for_byte(root: Path, outside: Path) -> None:
    original = outside / "resume.md"
    original.write_bytes(b"# Resume\r\ncaf\xc3\xa9\r\n")

    taken = take_snapshot("resume-2026", str(original), root, today=TODAY)

    assert taken == Snapshot("approved/resume-2026.md", "2026-10-01", str(original), False)
    assert (root / "approved" / "resume-2026.md").read_bytes() == original.read_bytes()


def test_an_explicit_revision_wins(root: Path, outside: Path) -> None:
    original = outside / "notes.txt"
    original.write_text("text")
    taken = take_snapshot("notes", str(original), root, revision="v2", today=TODAY)
    assert taken.revision == "v2"
    assert taken.path == "approved/notes.txt"


def test_snapshot_again_replaces_the_copy(root: Path, outside: Path) -> None:
    original = outside / "resume.md"
    original.write_text("old")
    take_snapshot("resume", str(original), root, today=TODAY)
    original.write_text("new")
    take_snapshot("resume", str(original), root, today=TODAY)
    assert (root / "approved" / "resume.md").read_text() == "new"
    assert not list((root / "approved").glob(".*")), "temporary file left behind"


def test_a_clean_tracked_file_records_the_commit_sha(root: Path, outside: Path) -> None:
    make_repo(outside)
    readme = outside / "README.md"
    readme.write_text("# Project\n")
    git(outside, "add", "README.md")
    git(outside, "commit", "--quiet", "-m", "init")
    head = git(outside, "rev-parse", "HEAD")

    assert take_snapshot("project", str(readme), root, today=TODAY).revision == head

    readme.write_text("# Project, edited\n")  # now HEAD no longer describes it
    assert take_snapshot("project", str(readme), root, today=TODAY).revision == "2026-10-01"


def test_an_untracked_file_in_a_repository_records_the_date(root: Path, outside: Path) -> None:
    make_repo(outside)
    draft = outside / "draft.md"
    draft.write_text("draft")
    assert take_snapshot("draft", str(draft), root, today=TODAY).revision == "2026-10-01"


def test_downloads_a_url_and_picks_the_suffix(root: Path) -> None:
    calls: list[str] = []
    page = fetch_returning(b"<h1>Post</h1>", "text/html", calls)

    taken = take_snapshot(
        "blog-post", "https://example.com/blog/post", root, fetch=page, today=TODAY
    )

    assert calls == ["https://example.com/blog/post"]
    assert taken == Snapshot(
        "approved/blog-post.html", "2026-10-01", "https://example.com/blog/post", False
    )
    assert (root / "approved" / "blog-post.html").read_bytes() == b"<h1>Post</h1>"


@pytest.mark.parametrize(
    ("url", "content_type", "suffix"),
    [
        ("https://example.com/raw/README.md", "text/plain", ".md"),
        ("https://example.com/cv.pdf", "application/octet-stream", ".pdf"),
        ("https://example.com/about", "text/markdown", ".md"),
        ("https://example.com/about", None, ".html"),
    ],
)
def test_url_suffix_comes_from_the_path_then_the_content_type(
    root: Path, url: str, content_type: str | None, suffix: str
) -> None:
    fetch = fetch_returning(b"x", content_type, [])
    assert take_snapshot("page", url, root, fetch=fetch).path == f"approved/page{suffix}"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://example.com/x.md"])
def test_other_url_schemes_are_refused(root: Path, url: str) -> None:
    with pytest.raises(SnapshotError, match="only http and https"):
        take_snapshot("page", url, root)


def test_private_copies_go_to_private(root: Path, outside: Path) -> None:
    raw = outside / "meeting.md"
    raw.write_text("client names inside")
    taken = take_snapshot("note-meeting", str(raw), root, private=True, today=TODAY)
    assert taken.path == "private/note-meeting.md"
    assert taken.private
    assert not (root / "approved").exists()


def test_raw_private_material_cannot_be_snapshot_into_approved(root: Path) -> None:
    raw = root / "private" / "meeting.md"
    raw.parent.mkdir()
    raw.write_text("client names inside")
    with pytest.raises(SnapshotError, match="sanitized"):
        take_snapshot("note-meeting", str(raw), root)
    # A relative path is resolved against the corpus root, and refused too.
    with pytest.raises(SnapshotError, match="sanitized"):
        take_snapshot("note-meeting", "private/meeting.md", root)


def test_a_file_already_in_approved_stays_in_place(root: Path) -> None:
    written = root / "approved" / "linkedin.md"
    written.parent.mkdir()
    written.write_text("copied by hand")

    taken = take_snapshot("linkedin-experience", "approved/linkedin.md", root, today=TODAY)

    assert taken.path == "approved/linkedin.md"
    assert sorted(p.name for p in written.parent.iterdir()) == ["linkedin.md"]


def test_missing_origin_and_folders_are_errors(root: Path, outside: Path) -> None:
    with pytest.raises(SnapshotError, match="no such file"):
        take_snapshot("x", str(outside / "missing.md"), root)
    with pytest.raises(SnapshotError, match="not a file"):
        take_snapshot("x", str(outside), root)


@pytest.mark.parametrize("source_id", ["../escape", "a/b", ".hidden", "", "a\\b"])
def test_ids_that_would_change_the_folder_are_refused(
    root: Path, outside: Path, source_id: str
) -> None:
    original = outside / "x.md"
    original.write_text("x")
    with pytest.raises(SnapshotError, match="file name"):
        take_snapshot(source_id, str(original), root)
