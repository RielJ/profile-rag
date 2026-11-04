"""The question file: one labeled question per line, in JSON (JSONL).

Example line:

    {"id": "ex-01", "question": "Where does Dana work now?",
     "expected": [{"source_id": "resume-2026", "span": "Harborlight Analytics"}],
     "tags": ["direct"]}

- `id`: unique within the file. Reports name questions by ID only, so an ID
  must not quote the question. Letters, digits, `-` and `_`.
- `question`: the text sent to the retriever.
- `expected`: the labels. Each one is a source ID and a short span copied
  verbatim from that source. `labels.py` turns a label into the chunks that
  contain the span. An empty list means the corpus cannot answer the
  question.
- `tags`: optional, from `Tag`. `unanswerable` is required exactly when
  `expected` is empty.

Blank lines are skipped. Any other problem stops the load with an error that
names the line.

Tests: backend/tests/test_labels.py.
"""

import json
from pathlib import Path
from typing import Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Tag = Literal[
    "direct", "paraphrase", "multi_source", "attribution_trap", "unanswerable", "injection"
]
TAGS: tuple[Tag, ...] = get_args(Tag)


class QuestionFileError(Exception):
    """The question file is missing or invalid. The message names the file and line."""


class Label(BaseModel):
    """Evidence for an answer: a span copied verbatim from one source."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    span: str = Field(min_length=1)

    @model_validator(mode="after")
    def _span_has_words(self) -> Self:
        if not self.span.strip():
            raise ValueError("span is only whitespace")
        return self


class Question(BaseModel):
    """One labeled question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", max_length=64)
    question: str = Field(min_length=1)
    expected: tuple[Label, ...]
    tags: tuple[Tag, ...] = ()

    @model_validator(mode="after")
    def _check_tags(self) -> Self:
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("tags repeat a value")
        unanswerable = "unanswerable" in self.tags
        if unanswerable and self.expected:
            raise ValueError("an unanswerable question has no expected labels")
        if not unanswerable and not self.expected:
            raise ValueError("a question without expected labels needs the tag 'unanswerable'")
        return self

    @property
    def answerable(self) -> bool:
        return bool(self.expected)


def parse_questions(text: str, *, name: str = "<questions>") -> list[Question]:
    """Parse JSONL text. `name` is used in error messages, such as the file path."""
    questions: list[Question] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            question = Question.model_validate(json.loads(line))
        except json.JSONDecodeError as error:
            raise QuestionFileError(f"{name}:{number}: not JSON: {error.msg}") from error
        except ValidationError as error:
            problems = "; ".join(
                f"{'.'.join(str(part) for part in problem['loc']) or 'line'}: "
                f"{problem['msg'].removeprefix('Value error, ')}"
                for problem in error.errors()
            )
            raise QuestionFileError(f"{name}:{number}: {problems}") from error
        if question.id in seen:
            raise QuestionFileError(
                f"{name}:{number}: id {question.id!r} repeats line {seen[question.id]}"
            )
        seen[question.id] = number
        questions.append(question)
    return questions


def load_questions(path: Path) -> list[Question]:
    """Read and validate a JSONL question file. Raises QuestionFileError."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise QuestionFileError(f"{path}: cannot read: {error.strerror}") from error
    return parse_questions(text, name=str(path))
