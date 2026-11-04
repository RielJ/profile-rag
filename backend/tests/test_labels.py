"""Tests for evaluation/questions.py (the JSONL format) and evaluation/labels.py."""

import json
from pathlib import Path

import pytest
from langchain_core.documents import Document

from profile_rag.corpus import documents, registry, splitting
from profile_rag.evaluation.labels import normalise, resolve_labels
from profile_rag.evaluation.questions import (
    TAGS,
    Label,
    Question,
    QuestionFileError,
    load_questions,
    parse_questions,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "corpus-example"
EXAMPLE_QUESTIONS = REPO_ROOT / "eval-example" / "dev.jsonl"


def line(**fields: object) -> str:
    base: dict[str, object] = {
        "id": "q1",
        "question": "Where does Dana work?",
        "expected": [{"source_id": "resume", "span": "Harborlight"}],
    }
    return json.dumps(base | fields)


def question(*labels: tuple[str, str], qid: str = "q1") -> Question:
    expected = tuple(Label(source_id=s, span=t) for s, t in labels)
    return Question(
        id=qid, question="?", expected=expected, tags=() if expected else ("unanswerable",)
    )


def doc(
    source_id: str, text: str, chunk_id: str | None = None, heading_path: list[str] | None = None
) -> Document:
    """A chunk (with `chunk_id`) or a source (without). A chunk's text gets its heading line."""
    if chunk_id is None:
        return Document(page_content=text, metadata={"source_id": source_id})
    path = heading_path or []
    metadata = {"source_id": source_id, "chunk_id": chunk_id, "heading_path": path}
    return Document(page_content=splitting.chunk_text(path, text), metadata=metadata)


# --- The question file -------------------------------------------------------------


def test_a_valid_file_parses_and_skips_blank_lines() -> None:
    text = line() + "\n\n" + line(id="q2", expected=[], tags=["unanswerable"]) + "\n"
    parsed = parse_questions(text)
    assert [q.id for q in parsed] == ["q1", "q2"]
    assert parsed[0].expected == (Label(source_id="resume", span="Harborlight"),)
    assert parsed[0].answerable
    assert not parsed[1].answerable


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"expected": []}, "needs the tag 'unanswerable'"),
        ({"tags": ["unanswerable"]}, "has no expected labels"),
        ({"tags": ["typo"]}, "tags.0"),
        ({"tags": ["direct", "direct"]}, "tags repeat"),
        ({"id": "has spaces"}, "id:"),
        ({"question": ""}, "question:"),
        ({"expected": [{"source_id": "resume", "span": "  "}]}, "only whitespace"),
        ({"expected": [{"source_id": "resume"}]}, "expected.0.span"),
        ({"extra": 1}, "extra:"),
    ],
)
def test_an_invalid_line_names_the_line_and_field(fields: dict[str, object], message: str) -> None:
    with pytest.raises(QuestionFileError, match=r"^questions\.jsonl:2: ") as caught:
        parse_questions(line(id="q0") + "\n" + line(**fields), name="questions.jsonl")
    assert message in str(caught.value)


def test_a_line_that_is_not_json_is_an_error() -> None:
    with pytest.raises(QuestionFileError, match=r"^f:1: not JSON"):
        parse_questions("{nope", name="f")


def test_a_repeated_id_is_an_error() -> None:
    with pytest.raises(QuestionFileError, match=r"^f:2: id 'q1' repeats line 1"):
        parse_questions(line() + "\n" + line(), name="f")


def test_a_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(QuestionFileError, match="cannot read"):
        load_questions(tmp_path / "absent.jsonl")


def test_the_tag_list_is_complete() -> None:
    assert set(TAGS) == {
        "direct",
        "paraphrase",
        "multi_source",
        "attribution_trap",
        "unanswerable",
        "injection",
    }


# --- Resolving labels ---------------------------------------------------------------


def test_normalise_collapses_whitespace_only() -> None:
    assert normalise("  a\n  b\t\tC.  ") == "a b C."


def test_a_span_resolves_to_the_chunk_that_contains_it() -> None:
    chunks = [doc("s", "First part.", "s:1"), doc("s", "Second part.", "s:2")]
    resolution = resolve_labels([question(("s", "Second part."))], chunks)
    assert resolution.errors == []
    assert resolution.questions[0].relevant == (frozenset({"s:2"}),)


def test_whitespace_differences_do_not_matter() -> None:
    chunks = [doc("s", "- Built the API,\nserving about 40 requests\nper second.", "s:1")]
    label = ("s", "serving  about 40\n  requests per second")
    resolution = resolve_labels([question(label)], chunks)
    assert resolution.questions[0].relevant == (frozenset({"s:1"}),)


def test_case_and_punctuation_must_match() -> None:
    chunks = [doc("s", "Led the move.", "s:1")]
    resolution = resolve_labels([question(("s", "led the move"))], chunks)
    assert [e.reason for e in resolution.errors] == ["the span is in no chunk of the source"]


def test_a_span_in_an_overlap_resolves_to_both_chunks() -> None:
    chunks = [
        doc("s", "one two three four five", "s:1"),
        doc("s", "four five six seven", "s:2"),
    ]
    resolution = resolve_labels([question(("s", "four five"))], chunks)
    assert resolution.questions[0].relevant == (frozenset({"s:1", "s:2"}),)


def test_a_span_at_the_start_of_the_body_resolves_under_a_heading_line() -> None:
    chunks = [doc("s", "Led the move to Go.", "s:1", ["Dana", "Acme"])]
    resolution = resolve_labels([question(("s", "Led the move"))], chunks)
    assert resolution.questions[0].relevant == (frozenset({"s:1"}),)


def test_a_span_does_not_match_the_heading_line() -> None:
    source = doc("s", "# Dana\n## Acme\nLed the move to Go.")
    chunks = [doc("s", "Led the move to Go.", "s:1", ["Dana", "Acme"])]
    resolution = resolve_labels([question(("s", "Acme Led the move"))], chunks, [source])
    assert "crosses a chunk boundary or a heading" in resolution.errors[0].reason
    resolution = resolve_labels([question(("s", "Dana > Acme"))], chunks, [source])
    assert resolution.errors[0].reason == "the span is in no chunk of the source"


def test_a_span_only_matches_chunks_of_its_own_source() -> None:
    chunks = [doc("a", "shared words", "a:1"), doc("b", "shared words", "b:1")]
    resolution = resolve_labels([question(("b", "shared words"))], chunks)
    assert resolution.questions[0].relevant == (frozenset({"b:1"}),)


def test_a_span_across_a_chunk_boundary_is_an_error() -> None:
    source = doc("s", "one two three four five six")
    chunks = [doc("s", "one two three", "s:1"), doc("s", "four five six", "s:2")]
    resolution = resolve_labels([question(("s", "three four"))], chunks, [source])
    assert resolution.questions == []
    [error] = resolution.errors
    assert (error.question_id, error.label, error.source_id) == ("q1", 0, "s")
    assert "crosses a chunk boundary" in error.reason


def test_a_span_across_a_boundary_without_the_source_is_reported_missing() -> None:
    chunks = [doc("s", "one two three", "s:1"), doc("s", "four five six", "s:2")]
    resolution = resolve_labels([question(("s", "three four"))], chunks)
    assert resolution.errors[0].reason == "the span is in no chunk of the source"


def test_a_missing_span_is_an_error() -> None:
    chunks = [doc("s", "Something else.", "s:1")]
    source = doc("s", "Something else.")
    resolution = resolve_labels([question(("s", "Not here."))], chunks, [source])
    assert resolution.questions == []
    assert resolution.errors[0].reason == "the span is in no chunk of the source"


def test_an_unknown_source_is_an_error() -> None:
    resolution = resolve_labels([question(("ghost", "anything"))], [doc("s", "text", "s:1")])
    assert "has no chunks" in resolution.errors[0].reason


def test_one_bad_label_drops_the_question_and_reports_its_position() -> None:
    chunks = [doc("s", "alpha", "s:1")]
    resolution = resolve_labels(
        [question(("s", "alpha"), ("s", "beta")), question(("s", "alpha"), qid="q2")], chunks
    )
    assert [r.question.id for r in resolution.questions] == ["q2"]
    assert [(e.question_id, e.label) for e in resolution.errors] == [("q1", 1)]


def test_an_unanswerable_question_resolves_to_no_labels() -> None:
    resolution = resolve_labels([question()], [])
    assert resolution.errors == []
    assert resolution.questions[0].relevant == ()


def test_describe_does_not_print_the_span() -> None:
    resolution = resolve_labels([question(("s", "secret span text"))], [doc("s", "x", "s:1")])
    described = resolution.errors[0].describe()
    assert described.startswith("q1 label 0 (s): ")
    assert "secret" not in described


# --- The example set ---------------------------------------------------------------


def test_every_example_label_resolves_against_corpus_example() -> None:
    loaded = documents.load_documents(
        registry.load_registry(EXAMPLE_ROOT / "sources.yaml"), EXAMPLE_ROOT
    )
    chunks = splitting.split_documents(loaded)
    questions = load_questions(EXAMPLE_QUESTIONS)
    resolution = resolve_labels(questions, chunks, loaded)
    assert resolution.errors == []
    assert len(resolution.questions) == len(questions)


def test_the_example_set_covers_every_tag() -> None:
    questions = load_questions(EXAMPLE_QUESTIONS)
    assert {tag for q in questions for tag in q.tags} == set(TAGS)
