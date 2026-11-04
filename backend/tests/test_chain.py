"""Tests for generation/chain.py, with FakeListChatModel and a fake retriever.

No database, Ollama, or Pi is needed.
"""

import json
from dataclasses import dataclass
from typing import Any, override

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from profile_rag.generation.answer import Answered, InventedCitation, Unknown
from profile_rag.generation.chain import NO_EVIDENCE_REASON, GroundedAnswer, build_chain

DOCS = [
    Document(
        page_content="Dana maintains Rivet.",
        metadata={"chunk_id": "rivet:1", "title": "Rivet README", "attribution": "project_doc"},
    ),
    Document(
        page_content="Dana wrote a parser in Rust.",
        metadata={"chunk_id": "resume:2", "title": "Resume", "attribution": "self_report"},
    ),
]


class Recorder:
    """A fake retriever that returns fixed docs and records each query."""

    def __init__(self, docs: list[Document]) -> None:
        self.docs = docs
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[Document]:
        self.queries.append(query)
        return self.docs


class PromptRecorder(BaseCallbackHandler):
    """Records the prompt text of every chat model call."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @override
    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self.prompts.extend("\n".join(m.text for m in batch) for batch in messages)


@dataclass
class Setup:
    retriever: Recorder
    model: FakeListChatModel
    recorder: PromptRecorder
    chain: Runnable[dict[str, str], GroundedAnswer]

    def ask(self, question: str) -> GroundedAnswer:
        return self.chain.invoke({"question": question}, {"callbacks": [self.recorder]})


def chain_with(docs: list[Document], *replies: str) -> Setup:
    retriever = Recorder(docs)
    model = FakeListChatModel(responses=list(replies))
    chain = build_chain(RunnableLambda(retriever), model)
    return Setup(retriever, model, PromptRecorder(), chain)


def test_an_answered_reply_is_parsed_checked_and_returned_with_the_docs() -> None:
    reply = json.dumps(
        {"status": "answered", "answer": "Dana wrote a parser in Rust.", "citations": ["resume:2"]}
    )
    setup = chain_with(DOCS, reply)

    result = setup.ask("What did Dana write in Rust?")

    assert setup.retriever.queries == ["What did Dana write in Rust?"]
    assert result.answer.root == Answered(
        status="answered", answer="Dana wrote a parser in Rust.", citations=["resume:2"]
    )
    assert result.docs == DOCS
    assert result.cited_docs() == [DOCS[1]]
    (prompt,) = setup.recorder.prompts
    assert "chunk_id: rivet:1" in prompt
    assert "chunk_id: resume:2" in prompt
    assert prompt.endswith("Question: What did Dana write in Rust?")


def test_an_unknown_reply_is_returned_as_unknown() -> None:
    reply = '```json\n{"status": "unknown", "reason": "No salary in the evidence."}\n```'
    result = chain_with(DOCS, reply).ask("What is Dana's salary?")
    assert result.answer.root == Unknown(status="unknown", reason="No salary in the evidence.")
    assert result.cited_docs() == []


def test_empty_retrieval_returns_unknown_without_calling_the_model() -> None:
    setup = chain_with([], "this reply must not be used")
    result = setup.ask("Anything?")
    assert result.answer.root == Unknown(status="unknown", reason=NO_EVIDENCE_REASON)
    assert result.docs == []
    assert setup.recorder.prompts == []
    assert setup.model.i == 0


def test_an_invented_citation_raises() -> None:
    reply = json.dumps({"status": "answered", "answer": "Rust.", "citations": ["made-up:9"]})
    with pytest.raises(InventedCitation) as caught:
        chain_with(DOCS, reply).ask("Q?")
    assert caught.value.invented == ["made-up:9"]


@pytest.mark.parametrize(
    "reply",
    [
        pytest.param("Dana wrote a parser.", id="prose"),
        pytest.param('{"status": "answered", "answer": "x", "citations": []}', id="no citations"),
        pytest.param('{"status": "unknown", "reason": "x", "extra": 1}', id="extra field"),
    ],
)
def test_a_reply_that_is_not_a_valid_answer_raises_output_parser_exception(reply: str) -> None:
    with pytest.raises(OutputParserException):
        chain_with(DOCS, reply).ask("Q?")
