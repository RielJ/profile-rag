"""Split source Documents into chunks with LangChain's text splitters.

A chunk is the unit the app embeds, searches, and quotes. Where its
boundaries fall decides what can be found: a resume bullet cut off from its
job heading no longer says which job it belongs to. So the split runs in two
steps:

1. `MarkdownHeaderTextSplitter` cuts each document at its headings. Each
   section knows the headings above it.
2. `RecursiveCharacterTextSplitter` cuts only the sections that are too long,
   with sizes counted in words (`count_words`), not characters.

`chunk_words` is a maximum, not a size every chunk is filled up to. A short
section stays one short chunk.

Headings are words a question may use ("What did Dana do at Acme?"), but a
chunk's body rarely repeats them. So each chunk's text starts with its
heading path on one line, then a blank line, then the body:

    Dana Whitlock > Experience > Staff Engineer, Acme

    - Led the migration of four services ...

Embedding and keyword search both see that line. `heading_path` stays in the
metadata too, and `chunk_body` returns the text without the line.

Each chunk gets a `chunk_id` that stays the same while its source, its
headings, and its text stay the same. Indexing uses it to update only the
chunks that changed.

Tests: backend/tests/test_splitting.py.
"""

import hashlib
import json
from collections import Counter

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Names the splitting rules below. It is part of every chunk ID, so chunks
# made by different rules never share an ID. Change it whenever
# `split_documents` would return different chunks for the same documents and
# the same sizes (a new rule or a new splitter setting, not a size change).
#
# v1: chunk text was the body only.
# v2: chunk text starts with the heading path line (`heading_line`).
SPLITTER_VERSION = "v2"

# Joins the headings of a path on the first line of a chunk.
HEADING_SEPARATOR = " > "

# Sizes used by `profile-rag chunks show` unless you pass others. Read your
# own chunks and adjust.
DEFAULT_CHUNK_WORDS = 200
DEFAULT_OVERLAP_WORDS = 40


def count_words(text: str) -> int:
    """The number of words in `text`.

    A word is a run of characters that are not whitespace, so `- Led` is two
    words and `PostgreSQL-backed` is one. Sizes in this module always mean
    this count. Pass it to the splitter as `length_function`.
    """
    return len(text.split())


def heading_line(heading_path: list[str]) -> str:
    """The first line of a chunk under `heading_path`: the headings joined by " > ".

    An empty path (text before the first heading) gives "".
    """
    return HEADING_SEPARATOR.join(heading_path)


def chunk_text(heading_path: list[str], body: str) -> str:
    """The text of a chunk: its heading line, a blank line, then `body`.

    With an empty heading path the text is `body` alone.
    """
    line = heading_line(heading_path)
    return f"{line}\n\n{body}" if line else body


def chunk_body(chunk: Document) -> str:
    """The chunk's text without the heading line that `chunk_text` put first.

    Reads `heading_path` from the metadata. Raises ValueError if the text
    does not start with that path's line, which means the chunk was not made
    by this splitter version.
    """
    line = heading_line(chunk.metadata["heading_path"])
    if not line:
        return chunk.page_content
    prefix = f"{line}\n\n"
    if not chunk.page_content.startswith(prefix):
        raise ValueError(f"chunk {chunk.metadata.get('chunk_id')} does not start with its headings")
    return chunk.page_content.removeprefix(prefix)


def chunk_id(source_id: str, heading_path: list[str], text: str) -> str:
    """The ID of a chunk, before duplicates are numbered.

    The format is `<source_id>:<SPLITTER_VERSION>:<hash>`, where `<hash>` is
    the first 12 lowercase hex characters of a SHA-256 over the heading path
    and the text together. `text` is the body, without the heading line.
    The same three inputs always give the same ID. A different heading path
    or a different text gives a different ID.
    """
    hash = hashlib.sha256(bytes(json.dumps([heading_path, text]), "utf-8")).hexdigest()[:12]
    return f"{source_id}:{SPLITTER_VERSION}:{hash}"


def split_documents(
    docs: list[Document],
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[Document]:
    """Split each Markdown Document into chunks, in order.

    Raises ValueError unless `chunk_words >= 2` and
    `0 <= overlap_words < chunk_words`, before any document is read.

    Rules:

    1. Headings. Lines starting with `#` to `######` and a space are headings,
       except inside ``` or ~~~ fenced code. Heading lines are not part of any
       chunk's body. Text before the first heading is a section too.
    2. Size. Each section is split with `RecursiveCharacterTextSplitter`, using
       `chunk_words` as `chunk_size`, `overlap_words` as `chunk_overlap`, and
       `count_words` as `length_function`. Keep the other settings at their
       defaults. No chunk body has more than `chunk_words` words; the
       heading line comes on top. Chunks of different sections never
       overlap.
    3. Text. Each chunk's `page_content` is `chunk_text(heading_path, body)`:
       the heading line, a blank line, then the body. `chunk_body` undoes it.
    4. Metadata. Each chunk's metadata is a copy of its source Document's
       metadata plus exactly three keys:
       - `heading_path`: a list of the heading texts above the chunk,
         outermost first. `# A` then `### C` gives `["A", "C"]`; a following
         `## D` gives `["A", "D"]`. Text before the first heading has `[]`.
       - `ordinal`: the chunk's position within its source, from 0. It is
         for reading only; the ID does not use it.
       - `chunk_id`: `chunk_id(source_id, heading_path, body)`. The
         second chunk of a source with that same ID gets `:2` appended, the
         third `:3`, and so on, so IDs are unique within a source.
       The chunk's `Document.id` is the same `chunk_id`. No other keys are
       added (no heading keys such as `h1`, no `start_index`).

    A document without words (empty, or only headings) gives no chunks. The
    input Documents are not changed.
    """
    if chunk_words < 2:
        raise ValueError("chunk_words must be at least 2")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("0 <= overlap_words < chunk_words")

    headers = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
    )
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_words,
        chunk_overlap=overlap_words,
        length_function=count_words,
    )

    chunks: list[Document] = []
    for doc in docs:
        sections = md_splitter.split_text(doc.page_content)
        for section in sections:
            section.metadata = {**doc.metadata, **section.metadata}
        seen: Counter[str] = Counter()
        for ordinal, chunk in enumerate(rec_splitter.split_documents(sections)):
            path = [chunk.metadata.pop(key) for _, key in headers if key in chunk.metadata]
            base = chunk_id(doc.metadata["source_id"], path, chunk.page_content)
            chunk.page_content = chunk_text(path, chunk.page_content)
            seen[base] += 1
            cid = base if seen[base] == 1 else f"{base}:{seen[base]}"
            chunk.metadata |= {"chunk_id": cid, "heading_path": path, "ordinal": ordinal}
            chunk.id = cid
            chunks.append(chunk)
    return chunks
