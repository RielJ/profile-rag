"""Tests for corpus/splitting.py.

The example-corpus tests read the files in corpus-example/approved/
themselves, so they do not depend on corpus/documents.py.

Run them with:

    uv --directory backend run pytest tests/test_splitting.py
"""

import copy
import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pytest
from langchain_core.documents import Document

from profile_rag.corpus.splitting import (
    DEFAULT_CHUNK_WORDS,
    DEFAULT_OVERLAP_WORDS,
    SPLITTER_VERSION,
    chunk_body,
    chunk_id,
    chunk_text,
    count_words,
    heading_line,
    split_documents,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_APPROVED = REPO_ROOT / "corpus-example" / "approved"
EXAMPLE_IDS = ["resume-2026", "lantern-queue-readme", "blog-postgres-queues"]

HEADING = re.compile(r"^#{1,6} ")
FENCE = re.compile(r"^(```|~~~)")

NESTED = """\
Intro before any heading.

# A
Text under A.
### C
Text under C.
## D
Text under D.
```bash
# install the queue
pip install lantern-queue
```
~~~
## not a heading either
~~~
#### Deep
Text under deep.
"""

# One line of 30 words.
LONG_LINE = (
    "Dana Whitlock led the migration of four services from RabbitMQ to a PostgreSQL"
    " queue, wrote the runbook, trained the on-call rotation, and cut the monthly"
    " infrastructure bill by a third."
)


def source(text: str, source_id: str = "note-dana", **metadata: object) -> Document:
    return Document(
        page_content=text,
        metadata={"source_id": source_id, "title": f"Title of {source_id}", **metadata},
    )


def example_documents() -> list[Document]:
    return [
        source((EXAMPLE_APPROVED / f"{name}.md").read_text(encoding="utf-8"), name)
        for name in EXAMPLE_IDS
    ]


def words_outside_headings(text: str) -> list[str]:
    """Every word of `text` except those on heading lines outside fences."""
    found: list[str] = []
    in_fence = False
    for line in text.split("\n"):
        if FENCE.match(line):
            in_fence = not in_fence
        elif not in_fence and HEADING.match(line):
            continue
        found.extend(line.split())
    return found


# --- constants --------------------------------------------------------------------------


def test_public_constants() -> None:
    assert (DEFAULT_CHUNK_WORDS, DEFAULT_OVERLAP_WORDS) == (200, 40)
    assert SPLITTER_VERSION == "v2"
    assert count_words("- Led the  PostgreSQL-backed\nqueue") == 5


# --- sizes ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_words", "overlap_words"),
    [(1, 0), (0, 0), (-5, 0), (10, 10), (10, 11), (10, -1)],
)
def test_invalid_sizes_raise_before_reading(chunk_words: int, overlap_words: int) -> None:
    with pytest.raises(ValueError):
        split_documents([], chunk_words=chunk_words, overlap_words=overlap_words)


def test_the_smallest_valid_sizes_work() -> None:
    chunks = split_documents([source("one two three")], chunk_words=2, overlap_words=1)
    assert [chunk.page_content for chunk in chunks] == ["one two", "two three"]


@pytest.mark.parametrize(("chunk_words", "overlap_words"), [(200, 40), (50, 10), (20, 5), (8, 0)])
def test_no_chunk_is_longer_than_chunk_words(chunk_words: int, overlap_words: int) -> None:
    chunks = split_documents(
        example_documents(), chunk_words=chunk_words, overlap_words=overlap_words
    )
    assert chunks
    assert max(count_words(chunk_body(chunk)) for chunk in chunks) <= chunk_words


def test_a_small_section_stays_one_small_chunk() -> None:
    text = "## Skills\nPython, Go, PostgreSQL.\n## Talks\n" + LONG_LINE + "\n"
    chunks = split_documents([source(text)], chunk_words=200, overlap_words=40)
    assert [chunk_body(chunk) for chunk in chunks] == ["Python, Go, PostgreSQL.", LONG_LINE]


def test_a_long_line_is_cut_with_overlap() -> None:
    chunks = split_documents([source(LONG_LINE)], chunk_words=10, overlap_words=3)
    texts = [chunk.page_content.split() for chunk in chunks]
    assert len(texts) > 1
    for previous, current in pairwise(texts):
        assert current[:3] == previous[-3:]


def test_chunks_of_different_sections_never_overlap() -> None:
    text = "## Harborlight\n" + LONG_LINE + "\n## Copperfield\nBuilt the tracking API.\n"
    chunks = split_documents([source(text)], chunk_words=10, overlap_words=3)
    assert chunk_body(chunks[-1]) == "Built the tracking API."
    assert chunks[-1].metadata["heading_path"] == ["Copperfield"]


# --- headings ---------------------------------------------------------------------------


def test_heading_paths_follow_the_headings_above_each_chunk() -> None:
    chunks = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    assert [chunk.metadata["heading_path"] for chunk in chunks] == [
        [],
        ["A"],
        ["A", "C"],
        ["A", "D"],
        ["A", "D", "Deep"],
    ]


def test_hash_lines_inside_fences_are_text_not_headings() -> None:
    chunks = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    under_d = chunk_body(chunks[3])
    assert "# install the queue" in under_d
    assert "## not a heading either" in under_d


def test_heading_lines_are_not_in_any_chunk() -> None:
    chunks = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    for chunk in chunks:
        assert not any(HEADING.match(line) for line in chunk_body(chunk).split("\n")), chunk


def test_example_resume_heading_paths() -> None:
    (resume,) = [doc for doc in example_documents() if doc.metadata["source_id"] == "resume-2026"]
    paths = [chunk.metadata["heading_path"] for chunk in split_documents([resume])]
    assert ["Dana Whitlock", "Summary"] in paths
    assert [
        "Dana Whitlock",
        "Experience",
        "Senior Backend Engineer, Harborlight Analytics (2023 to present)",
    ] in paths
    assert ["Dana Whitlock", "Skills"] in paths


@pytest.mark.parametrize("text", ["", "\n\n", "# Only\n## Headings\n"])
def test_a_document_without_words_gives_no_chunks(text: str) -> None:
    assert split_documents([source(text)]) == []


# --- the heading line -------------------------------------------------------------------


def test_each_chunk_starts_with_its_heading_path() -> None:
    chunks = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    assert [chunk.page_content.split("\n\n")[0] for chunk in chunks[1:]] == [
        "A",
        "A > C",
        "A > D",
        "A > D > Deep",
    ]
    assert chunks[2].page_content == "A > C\n\nText under C."


def test_text_before_the_first_heading_has_no_heading_line() -> None:
    (first, *_) = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    assert first.metadata["heading_path"] == []
    assert first.page_content == "Intro before any heading."
    assert chunk_body(first) == first.page_content


def test_example_resume_chunks_name_the_job() -> None:
    (resume,) = [doc for doc in example_documents() if doc.metadata["source_id"] == "resume-2026"]
    job = "Senior Backend Engineer, Harborlight Analytics (2023 to present)"
    under_job = [c for c in split_documents([resume]) if job in c.metadata["heading_path"]]
    assert under_job
    for chunk in under_job:
        assert chunk.page_content.startswith(f"Dana Whitlock > Experience > {job}\n\n")
        assert job not in chunk_body(chunk)


def test_chunk_text_and_chunk_body_round_trip() -> None:
    path = ["Dana Whitlock", "Skills"]
    assert heading_line(path) == "Dana Whitlock > Skills"
    assert heading_line([]) == ""
    text = chunk_text(path, "Python, Go.")
    assert text == "Dana Whitlock > Skills\n\nPython, Go."
    doc = Document(page_content=text, metadata={"heading_path": path})
    assert chunk_body(doc) == "Python, Go."


def test_chunk_body_refuses_text_without_its_heading_line() -> None:
    metadata = {"heading_path": ["Skills"], "chunk_id": "x"}
    doc = Document(page_content="Python, Go.", metadata=metadata)
    with pytest.raises(ValueError, match="does not start with its headings"):
        chunk_body(doc)


# --- coverage ---------------------------------------------------------------------------


@pytest.mark.parametrize(("chunk_words", "overlap_words"), [(200, 40), (20, 5), (8, 0)])
def test_every_word_outside_headings_is_in_a_chunk(chunk_words: int, overlap_words: int) -> None:
    docs = [*example_documents(), source(NESTED, "nested")]
    chunks = split_documents(docs, chunk_words=chunk_words, overlap_words=overlap_words)
    for doc in docs:
        expected = Counter(words_outside_headings(doc.page_content))
        found = Counter(
            word
            for chunk in chunks
            if chunk.metadata["source_id"] == doc.metadata["source_id"]
            for word in chunk_body(chunk).split()
        )
        assert expected - found == Counter(), doc.metadata["source_id"]


# --- metadata ---------------------------------------------------------------------------


def test_source_metadata_is_copied_and_three_keys_are_added() -> None:
    doc = source(NESTED, kind="note", url="https://example.com/dana/notes")
    chunks = split_documents([doc], chunk_words=200, overlap_words=0)
    for chunk in chunks:
        assert set(chunk.metadata) == set(doc.metadata) | {"heading_path", "ordinal", "chunk_id"}
        for key, value in doc.metadata.items():
            assert chunk.metadata[key] == value


def test_ordinals_count_from_zero_per_source() -> None:
    chunks = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    for name in EXAMPLE_IDS:
        ordinals = [c.metadata["ordinal"] for c in chunks if c.metadata["source_id"] == name]
        assert ordinals == list(range(len(ordinals))), name
        assert len(ordinals) > 1, name


def test_chunks_come_out_in_source_order() -> None:
    chunks = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    order = [chunk.metadata["source_id"] for chunk in chunks]
    assert list(dict.fromkeys(order)) == EXAMPLE_IDS


def test_input_documents_are_not_changed() -> None:
    docs = example_documents()
    before = copy.deepcopy(docs)
    split_documents(docs, chunk_words=50, overlap_words=10)
    assert docs == before


# --- chunk_id ---------------------------------------------------------------------------


def test_chunk_id_format() -> None:
    value = chunk_id("resume-2026", ["Dana Whitlock", "Skills"], "Python, Go.")
    assert re.fullmatch(rf"resume-2026:{SPLITTER_VERSION}:[0-9a-f]{{12}}", value), value


def test_chunk_id_is_the_same_for_the_same_inputs() -> None:
    first = chunk_id("resume-2026", ["Skills"], "Python, Go.")
    assert chunk_id("resume-2026", ["Skills"], "Python, Go.") == first


def test_chunk_id_depends_on_source_heading_path_and_text() -> None:
    ids = [
        chunk_id("resume-2026", ["Skills"], "Python, Go."),
        chunk_id("resume-2025", ["Skills"], "Python, Go."),
        chunk_id("resume-2026", ["Talks"], "Python, Go."),
        chunk_id("resume-2026", [], "Python, Go."),
        chunk_id("resume-2026", ["Skills"], "Python, Go, SQL."),
        chunk_id("resume-2026", ["Dana Whitlock", "Skills"], "Python, Go."),
        chunk_id("resume-2026", ["Dana Whitlock Skills"], "Python, Go."),
    ]
    assert len(set(ids)) == len(ids), ids


# --- IDs on chunks ----------------------------------------------------------------------


def test_chunk_id_is_in_metadata_and_on_the_document() -> None:
    chunks = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    for chunk in chunks:
        metadata = chunk.metadata
        expected = chunk_id(metadata["source_id"], metadata["heading_path"], chunk_body(chunk))
        assert metadata["chunk_id"] == expected
        assert chunk.id == expected


def test_ids_are_stable_across_runs() -> None:
    first = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    second = split_documents(example_documents(), chunk_words=50, overlap_words=10)
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)


def test_changing_the_text_changes_only_that_chunks_id() -> None:
    before = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    edited = NESTED.replace("Text under C.", "Text under C, edited.")
    after = split_documents([source(edited)], chunk_words=200, overlap_words=0)
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a.id != b.id]
    assert changed == [2]


def test_inserting_a_section_near_the_top_keeps_every_other_id() -> None:
    before = split_documents([source(NESTED)], chunk_words=200, overlap_words=0)
    edited = NESTED.replace("# A\n", "# New\nA new section.\n# A\n")
    after = split_documents([source(edited)], chunk_words=200, overlap_words=0)
    # Every chunk after the new one has moved by one position ...
    assert len(after) == len(before) + 1
    # ... and kept its ID.
    assert {chunk.id for chunk in before} < {chunk.id for chunk in after}


def test_the_same_text_under_other_headings_gets_another_id() -> None:
    text = "## Harborlight\nWrote the runbook.\n## Copperfield\nWrote the runbook.\n"
    first, second = split_documents([source(text)])
    assert chunk_body(first) == chunk_body(second)
    assert first.id != second.id
    assert second.id == chunk_id("note-dana", ["Copperfield"], "Wrote the runbook.")


# MarkdownHeaderTextSplitter joins neighbouring sections with the same
# headings into one, so the repeats below have another section between them.
REPEATED = (
    "## Notes\nWrote the runbook.\n"
    "## Talks\nQueues without a broker.\n"
    "## Notes\nWrote the runbook.\n"
    "## Talks\nQueues without a broker.\n"
    "## Notes\nWrote the runbook.\n"
)


def test_duplicate_chunks_in_one_source_get_a_count_suffix() -> None:
    chunks = split_documents([source(REPEATED)])
    notes = [chunk for chunk in chunks if chunk.metadata["heading_path"] == ["Notes"]]
    base = chunk_id("note-dana", ["Notes"], "Wrote the runbook.")
    assert [chunk.id for chunk in notes] == [base, f"{base}:2", f"{base}:3"]
    assert [chunk.metadata["chunk_id"] for chunk in notes] == [base, f"{base}:2", f"{base}:3"]
    assert len({chunk.id for chunk in chunks}) == len(chunks)


def test_the_count_starts_again_in_each_source() -> None:
    chunks = split_documents([source(REPEATED, "one"), source(REPEATED, "two")])
    for name in ("one", "two"):
        base = chunk_id(name, ["Notes"], "Wrote the runbook.")
        ids = [c.id for c in chunks if c.metadata["source_id"] == name]
        assert [base, f"{base}:2", f"{base}:3"] == [i for i in ids if i and i.startswith(base)]
