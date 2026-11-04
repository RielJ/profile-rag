"""Turn each label (a source ID and a quoted span) into the chunks that hold it.

A label names text, not chunk IDs, so it stays valid when the splitter
settings change: re-splitting moves the span into other chunks, and the
label resolves to those.

Rules:

- Matching normalises whitespace on both sides: every run of spaces, tabs,
  and line breaks becomes one space, and leading and trailing whitespace is
  dropped. Nothing else changes. Case, punctuation, and Markdown marks such
  as `**` must match the source file.
- A label resolves to every chunk of its source whose text contains the
  whole span. Neighbouring chunks overlap, so a span inside an overlap
  resolves to two chunks; retrieving either one counts as finding it.
- A span must fit inside one chunk. A span that crosses a chunk boundary
  is a label error, as is a span that includes a heading line (headings are
  metadata, not chunk text) or a span the source does not contain. Shorten
  the span to the sentence that proves the answer.
- A label for a source with no chunks (unknown, not approved, or empty) is
  a label error.

Tests: backend/tests/test_labels.py.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from langchain_core.documents import Document

from profile_rag.evaluation.questions import Question


def normalise(text: str) -> str:
    """`text` with each whitespace run replaced by one space, and trimmed."""
    return " ".join(text.split())


@dataclass(frozen=True)
class LabelError:
    """A label that matches no chunk. `label` is its position in `expected`, from 0."""

    question_id: str
    label: int
    source_id: str
    reason: str

    def describe(self) -> str:
        """One line without the span text, so it is safe to print for private sets."""
        return f"{self.question_id} label {self.label} ({self.source_id}): {self.reason}"


@dataclass(frozen=True)
class ResolvedQuestion:
    """A question with, for each label in order, the IDs of the chunks that hold its span."""

    question: Question
    relevant: tuple[frozenset[str], ...]


@dataclass(frozen=True)
class Resolution:
    """The questions whose labels all resolved, and every label error."""

    questions: list[ResolvedQuestion]
    errors: list[LabelError]


def resolve_labels(
    questions: Sequence[Question],
    chunks: Sequence[Document],
    documents: Sequence[Document] = (),
) -> Resolution:
    """Resolve every label of `questions` against `chunks`.

    `chunks` come from `splitting.split_documents`; each has `source_id` and
    `chunk_id` in its metadata. `documents` are the unsplit sources. They are
    optional and only make error messages more precise: with them, a span
    the source contains but no single chunk does is reported as crossing a
    boundary or a heading, not as missing.

    A question with any label error is left out of `questions`. A question
    without labels (unanswerable) always resolves.
    """
    by_source: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk.metadata["source_id"]].append(
            (chunk.metadata["chunk_id"], normalise(chunk.page_content))
        )
    source_text = {doc.metadata["source_id"]: normalise(doc.page_content) for doc in documents}

    resolved: list[ResolvedQuestion] = []
    errors: list[LabelError] = []
    for question in questions:
        relevant: list[frozenset[str]] = []
        failed = False
        for position, label in enumerate(question.expected):
            span = normalise(label.span)
            source_chunks = by_source.get(label.source_id)
            if not source_chunks:
                reason = "the source has no chunks (unknown ID, not approved, or empty)"
            else:
                ids = frozenset(cid for cid, text in source_chunks if span in text)
                if ids:
                    relevant.append(ids)
                    continue
                if span in source_text.get(label.source_id, ""):
                    reason = "the span crosses a chunk boundary or a heading; shorten it"
                else:
                    reason = "the span is in no chunk of the source"
            errors.append(LabelError(question.id, position, label.source_id, reason))
            failed = True
        if not failed:
            resolved.append(ResolvedQuestion(question, tuple(relevant)))
    return Resolution(resolved, errors)
