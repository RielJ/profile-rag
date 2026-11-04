"""Copy or download a source into the corpus, and work out its revision.

A snapshot is the exact bytes that get reviewed and, once approved,
ingested. Taking a copy (instead of reading the original in place) means the
reviewed content can't change behind your back: if you edit the original
later, nothing happens until you snapshot it again and review the new copy.

Where the copy goes:

    origin                         destination
    -----------------------------  -------------------------------------------
    a local file or http(s) URL    approved/<id><suffix>
    the same, with private=True    private/<id><suffix> (raw, never ingested)
    a file already in approved/    left where it is (hand-written or sanitized)
    a file in private/             refused: write a sanitized copy instead

How the revision is chosen, first match wins:

1. The `revision` argument, for example a commit SHA you looked up yourself.
2. For a local file tracked by Git with no uncommitted changes: the commit
   SHA of `HEAD` in its repository.
3. Otherwise: today's date (UTC), as `YYYY-MM-DD`. For a URL this is the
   retrieval date.

Relative local paths are resolved against the current directory first, then
against the corpus root. So `approved/linkedin.md` works from anywhere.

HTML pages are saved as they are. The loader (documents.py) reads
only Markdown and plain text, so convert a page to Markdown before you
approve it. For a GitHub file, use its raw URL
(https://raw.githubusercontent.com/...), or snapshot your local checkout to
record the commit SHA.
"""

import os
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from profile_rag.corpus.registry import APPROVED_DIR, PRIVATE_DIR

# Downloads larger than this are refused. A profile page or README is far
# smaller; anything this big is probably not what you meant to fetch.
MAX_DOWNLOAD_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30

# File types the loader can read. Other suffixes are kept as they are.
_SUFFIX_BY_CONTENT_TYPE = {
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/markdown": ".md",
    "text/x-markdown": ".md",
    "text/plain": ".txt",
    "application/pdf": ".pdf",
}
_KNOWN_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".html", ".htm", ".pdf"})


class SnapshotError(Exception):
    """The snapshot could not be taken. The message says why."""


@dataclass(frozen=True)
class Download:
    content: bytes
    content_type: str | None  # media type only, e.g. "text/html"


Fetch = Callable[[str], Download]


@dataclass(frozen=True)
class Snapshot:
    """Where the snapshot is and what it was taken from."""

    path: str  # relative to the corpus root, with "/" separators
    revision: str
    origin: str  # the URL, or the absolute path of the local file
    private: bool


def fetch_url(url: str) -> Download:
    """Download `url` with a size limit and a timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": "profile-rag-snapshot"})
    try:
        # The scheme is checked by the caller: only http and https get here.
        with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            content: bytes = response.read(MAX_DOWNLOAD_BYTES + 1)
            content_type: str | None = response.headers.get_content_type()
    except OSError as error:  # URLError and timeouts are OSErrors
        raise SnapshotError(f"cannot download {url}: {error}") from error
    if len(content) > MAX_DOWNLOAD_BYTES:
        raise SnapshotError(f"{url} is larger than {MAX_DOWNLOAD_BYTES} bytes")
    return Download(content, content_type)


def take_snapshot(
    source_id: str,
    origin: str,
    root: Path,
    *,
    private: bool = False,
    revision: str | None = None,
    fetch: Fetch = fetch_url,
    today: date | None = None,
) -> Snapshot:
    """Copy or download `origin` into the corpus at `root`. See the module docstring.

    An existing snapshot at the destination is replaced. `fetch` and `today`
    exist so tests can avoid the network and the clock.
    """
    _check_id(source_id)
    day = (today or datetime.now(UTC).date()).isoformat()
    target_dir = PRIVATE_DIR if private else APPROVED_DIR

    if urlsplit(origin).scheme in ("http", "https"):
        download = fetch(origin)
        suffix = _url_suffix(origin, download.content_type)
        relative = f"{target_dir}/{source_id}{suffix}"
        _write_atomically(root / relative, download.content)
        return Snapshot(relative, revision or day, origin, private)
    if "://" in origin:
        raise SnapshotError(f"{origin}: only http and https URLs can be downloaded")

    source = _resolve_local(origin, root)
    approved, private_dir = (root / APPROVED_DIR).resolve(), (root / PRIVATE_DIR).resolve()
    if not private and source.is_relative_to(private_dir):
        raise SnapshotError(
            f"{source} is raw private material. Write a sanitized copy into "
            f"{APPROVED_DIR}/ and snapshot that instead."
        )
    revision = revision or _git_revision(source) or day

    if not private and source.is_relative_to(approved):
        # Already in place: written by hand, or a sanitized derivative.
        relative = source.relative_to(root.resolve()).as_posix()
        return Snapshot(relative, revision, str(source), private)

    relative = f"{target_dir}/{source_id}{source.suffix}"
    _write_atomically(root / relative, source.read_bytes())
    return Snapshot(relative, revision, str(source), private)


def _check_id(source_id: str) -> None:
    # The registry validates IDs fully. This only guarantees the ID can't
    # turn the destination into another folder (`../x`, `a/b`).
    if not source_id or source_id.startswith(".") or PurePosixPath(source_id).name != source_id:
        raise SnapshotError(f"{source_id!r} cannot be used as a file name")
    if "\\" in source_id:
        raise SnapshotError(f"{source_id!r} cannot be used as a file name")


def _resolve_local(origin: str, root: Path) -> Path:
    candidate = Path(origin).expanduser()
    for path in (candidate, root / candidate):
        if path.is_file():
            return path.resolve()
        if path.exists():
            raise SnapshotError(f"{path} is not a file")
        if candidate.is_absolute():
            break
    raise SnapshotError(f"{origin}: no such file")


def _url_suffix(url: str, content_type: str | None) -> str:
    suffix = PurePosixPath(urlsplit(url).path).suffix.lower()
    if suffix in _KNOWN_SUFFIXES:
        return suffix
    return _SUFFIX_BY_CONTENT_TYPE.get(content_type or "", ".html")


def _git(directory: Path, *args: str) -> str | None:
    """Run git in `directory`. Return its output, or None if git fails."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _git_revision(path: Path) -> str | None:
    """HEAD's commit SHA if `path` is tracked and unchanged, else None."""
    directory, name = path.parent, path.name
    if _git(directory, "ls-files", "--error-unmatch", "--", name) is None:
        return None  # not in a repository, or not tracked
    if _git(directory, "status", "--porcelain", "--", name) != "":
        return None  # uncommitted changes: HEAD does not describe this content
    return _git(directory, "rev-parse", "HEAD")


def _write_atomically(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)
