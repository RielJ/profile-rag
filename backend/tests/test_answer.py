"""Tests for generation/answer.py: the answer models and check_citations."""

import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import ValidationError

from profile_rag.generation.answer import (
    Answer,
    Answered,
    InventedCitation,
    Unknown,
    check_citations,
    draft_text,
)

parser = PydanticOutputParser(pydantic_object=Answer)


def doc(chunk_id: str) -> Document:
    return Document(page_content="text", metadata={"chunk_id": chunk_id})


def test_an_answered_reply_parses_to_answered() -> None:
    answer = parser.parse('{"status": "answered", "answer": "Rust.", "citations": ["a:1"]}')
    assert answer.root == Answered(status="answered", answer="Rust.", citations=["a:1"])


def test_an_unknown_reply_parses_to_unknown() -> None:
    answer = parser.parse('{"status": "unknown", "reason": "Not in the evidence."}')
    assert answer.root == Unknown(status="unknown", reason="Not in the evidence.")


def test_a_json_fence_is_accepted() -> None:
    answer = parser.parse('```json\n{"status": "unknown", "reason": "No evidence."}\n```')
    assert isinstance(answer.root, Unknown)


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("The answer is Rust.", id="prose"),
        pytest.param('{"status": "maybe", "reason": "x"}', id="unknown status"),
        pytest.param('{"answer": "Rust.", "citations": ["a:1"]}', id="no status"),
        pytest.param(
            '{"status": "answered", "answer": "Rust.", "citations": ["a:1"], "confidence": 1}',
            id="extra field",
        ),
        pytest.param('{"status": "answered", "answer": "Rust.", "citations": []}', id="no cites"),
        pytest.param('{"status": "answered", "answer": "  ", "citations": ["a:1"]}', id="blank"),
        pytest.param('{"status": "answered", "answer": "R", "citations": [" "]}', id="blank id"),
        pytest.param('{"status": "unknown", "reason": ""}', id="empty reason"),
        pytest.param(
            '{"status": "unknown", "reason": "x", "citations": ["a:1"]}', id="unknown with cites"
        ),
    ],
)
def test_a_reply_of_the_wrong_shape_is_rejected(reply: str) -> None:
    with pytest.raises(OutputParserException):
        parser.parse(reply)


def test_the_models_reject_blank_text_directly() -> None:
    with pytest.raises(ValidationError):
        Unknown(status="unknown", reason="\n\t")


def test_citations_of_retrieved_chunks_pass_unchanged() -> None:
    answer = Answer(Answered(status="answered", answer="Rust.", citations=["a:1", "b:2"]))
    assert check_citations(answer, [doc("a:1"), doc("b:2"), doc("c:3")]) is answer


def test_an_invented_citation_raises_and_lists_each_unknown_id_once() -> None:
    answer = Answer(
        Answered(status="answered", answer="Rust.", citations=["a:1", "x:9", "x:9", "y:8"])
    )
    with pytest.raises(InventedCitation) as caught:
        check_citations(answer, [doc("a:1")])
    assert caught.value.invented == ["x:9", "y:8"]


def test_an_unknown_answer_has_no_citations_to_check() -> None:
    answer = Answer(Unknown(status="unknown", reason="No evidence."))
    assert check_citations(answer, []) is answer


@pytest.mark.parametrize(
    ("partial", "draft"),
    [
        ("", ""),
        ('{"sta', ""),
        ('{"status": "answered", "ans', ""),
        ('{"status": "answered", "answer": "He cut', "He cut"),
        ('{"status": "answered", "answer": "a \\"quoted', 'a "quoted'),
        ('{"status": "answered", "answer": "line\\nnext', "line\nnext"),
        ('{"status": "answered", "answer": "done.", "citations": ["c', "done."),
        ('```json\n{"status": "unknown", "reason": "Not in', "Not in"),
        ('{"status": "answered", "answer": 3', ""),
        ("[1, 2", ""),
        ("not json at all", ""),
    ],
)
def test_draft_text_reads_the_answer_or_reason_from_a_cut_off_reply(
    partial: str, draft: str
) -> None:
    assert draft_text(partial) == draft
