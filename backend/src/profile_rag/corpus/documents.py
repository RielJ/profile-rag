"""Load the approved snapshots of the registry as LangChain `Document`s.

A `Document` is the unit every later LangChain step passes around: the
splitter cuts it into smaller `Document`s, an embedding model reads its
`page_content`, and a vector store saves its `metadata` next to the vector.
Whatever a citation, a filter, or the cleanup of a removed source needs later
has to be in the metadata from the start, because each chunk copies the
metadata of the document it came from.

Only approved sources become documents. A candidate, in-review, or rejected
entry is never read, even when its file exists. An approved file is used only
if its bytes still match the hash recorded at approval.

Tests: backend/tests/test_documents.py.
"""

from pathlib import Path

from langchain_core.documents import Document

from profile_rag.corpus.registry import Registry, file_sha256


class ApprovedSnapshotError(Exception):
    """An approved snapshot is missing, unreadable, or changed since approval.

    Not the same as `corpus.snapshot.SnapshotError`, which means a snapshot
    could not be taken.

    The message starts with the source ID, so the person who reads it knows
    which entry to fix.
    """


def load_documents(registry: Registry, corpus_dir: Path) -> list[Document]:
    """Return one Document per approved source, in registry order.

    `corpus_dir` is the corpus root; each entry's `path` is relative to it.
    Entries in any other state are skipped without reading their file.

    Each Document has:

    - `page_content`: the file's text, read as UTF-8, unchanged.
    - `metadata`: exactly these five keys, copied from the entry:
      `source_id` (the entry's `id`), `title`, `attribution`, `url` (None
      when the entry has no URL), and `visibility`. Every value is a plain
      `str`, or None for a missing `url`.
    - `id`: None. Chunk IDs are set by the splitter (splitting.py).

    Raises ApprovedSnapshotError, with a message that starts with the source ID, when
    an approved file is missing or can't be read, or when its `file_sha256`
    differs from the entry's `content_sha256`.
    """
    docs: list[Document] = []

    for entry in registry.sources:
        if entry.state != "approved":
            continue
        assert entry.path is not None
        path = corpus_dir / entry.path.lstrip("/")
        try:
            actual = file_sha256(path)
            data = path.read_bytes()
        except OSError as error:
            raise ApprovedSnapshotError(
                f"{entry.id}: cannot read {path}: {error.strerror}"
            ) from error

        if actual != entry.content_sha256:
            raise ApprovedSnapshotError(
                f"{entry.id}: {entry.path} changed since approval."
                f"expected {entry.content_sha256}, got {actual}"
            )

        docs.append(
            Document(
                page_content=data.decode("utf-8"),
                metadata={
                    "source_id": entry.id,
                    "title": entry.title,
                    "attribution": entry.attribution,
                    "url": entry.url,
                    "visibility": entry.visibility,
                },
            )
        )
    return docs
