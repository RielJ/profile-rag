"""Tests for rag.py: `ask` maps the chain's result to an `AskResponse`.

`ask` runs the real chain (generation/chain.py) through `rag.ChainCache`,
with a fake retriever and a fake chat model, so no database, Ollama, or Pi
is needed. `to_response` and the excerpt rules are the reference the Worker
is checked against (answers.py).
"""

import json
import logging
from collections.abc import Callable
from typing import Any, override

import psycopg
import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    GenericFakeChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from profile_rag import rag
from profile_rag.answers import FailedCode
from profile_rag.corpus.splitting import chunk_text
from profile_rag.generation.chain import NO_EVIDENCE_REASON
from profile_rag.generation.pi_rpc import PiProcessError, PiProtocolError, PiTimeout
from profile_rag.retrievers.factory import SearchMode
from profile_rag.vectorstore.prefixed import EmbeddingError

# Stands in for a model reply or Pi's stderr: it must never reach a response or a log.
SECRET = "SECRET-REPLY-TEXT-7f3a"

LONG_TEXT = "Dana Whitlock maintains lantern-queue, a Postgres-backed job queue. " * 8


def chunk(chunk_id: str, text: str, title: str, url: str | None) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_id": chunk_id.split(":")[0],
            "chunk_id": chunk_id,
            "attribution": "project_doc",
            "visibility": "public",
            "title": title,
            "url": url,
            "heading_path": [],
            "ordinal": 0,
        },
    )


DOCS = [
    chunk("lantern:v1:a", LONG_TEXT, "lantern-queue README", "https://git.example.com/lantern"),
    chunk("notes:v1:a", "Dana  moved\nthe queue to Postgres.", "Migration notes", None),
    chunk("local:v1:a", "Dana writes Rust.", "Local copy", "approved/local.md"),
]


class Retriever:
    """A fake retriever: fixed docs, or an error. Records each (mode, k) it is built for."""

    def __init__(self, docs: list[Document], error: Exception | None = None) -> None:
        self.docs = docs
        self.error = error
        self.built: list[tuple[SearchMode, int]] = []

    def make(self, mode: SearchMode, k: int) -> Runnable[str, list[Document]]:
        self.built.append((mode, k))
        return RunnableLambda(self._search)

    def _search(self, query: str) -> list[Document]:
        if self.error is not None:
            raise self.error
        return self.docs


class RaisingModel(BaseChatModel):
    """A chat model that raises the given error, as PiChatModel does for Pi failures."""

    error: Exception

    @property
    @override
    def _llm_type(self) -> str:
        return "raising"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise self.error


def chains_for(retriever: Retriever, model: BaseChatModel) -> rag.ChainCache:
    return rag.ChainCache(retriever.make, model, default_mode="hybrid", default_k=8)


def replying(*replies: str, docs: list[Document] = DOCS) -> rag.ChainCache:
    return chains_for(Retriever(docs), FakeListChatModel(responses=list(replies)))


def answered(*citations: str, answer: str = "Dana maintains lantern-queue.") -> str:
    return json.dumps({"status": "answered", "answer": answer, "citations": list(citations)})


def ask(
    chains: rag.ChainCache, question: str = "What does Dana maintain?", request_id: str = "r1"
) -> dict[str, Any]:
    """`rag.ask` on the default chain, as the JSON the Worker returns."""
    return rag.ask(question, chains.get(), request_id=request_id).model_dump(mode="json")


# --- answered and unknown -----------------------------------------------------


def test_answered_has_one_citation_per_cited_chunk_with_its_source() -> None:
    body = ask(replying(answered("lantern:v1:a", "notes:v1:a", "local:v1:a", "lantern:v1:a")))

    assert body["status"] == "answered"
    assert body["answer"] == "Dana maintains lantern-queue."
    assert body["model"] is None  # FakeListChatModel reports no model name
    assert body["citations"][1:] == [
        {
            "chunk_id": "notes:v1:a",
            "source_id": "notes",
            "source_title": "Migration notes",
            "section": None,
            "url": None,
            "attribution": "project_doc",
            "excerpt": "Dana moved the queue to Postgres.",
        },
        {
            "chunk_id": "local:v1:a",
            "source_id": "local",
            "source_title": "Local copy",
            "section": None,
            "url": None,  # not an http(s) link, so not shown
            "attribution": "project_doc",
            "excerpt": "Dana writes Rust.",
        },
    ]
    first = body["citations"][0]
    assert first["chunk_id"] == "lantern:v1:a"
    assert first["url"] == "https://git.example.com/lantern"
    assert len(first["excerpt"]) <= rag.EXCERPT_CHARS
    assert first["excerpt"].endswith("…")
    assert LONG_TEXT.startswith(first["excerpt"].removesuffix("…"))
    assert len(body["citations"]) == 3, "a repeated citation is listed once"


def test_a_citation_names_its_section_and_its_excerpt_starts_after_the_headings() -> None:
    path = ["Dana Whitlock - Resume", "Experience", "Staff Engineer, Acme"]
    text = (
        "Moved four services\nto Kubernetes.\n\n"
        "- Led the queue migration.\n"
        "- Cut deploy time\n  from 40 to 6 minutes.\n"
        "1. First item\n\n"
        "Run a worker:\n```bash\nlantern-queue worker\n  --concurrency 4\n```"
    )
    doc = chunk("resume:v2:a", chunk_text(path, text), "Resume", None)
    doc.metadata["heading_path"] = path

    (citation,) = ask(replying(answered("resume:v2:a"), docs=[doc]))["citations"]
    assert citation["section"] == "Experience > Staff Engineer, Acme"
    assert citation["excerpt"] == (
        "Moved four services to Kubernetes.\n"
        "- Led the queue migration.\n"
        "- Cut deploy time from 40 to 6 minutes.\n"
        "1. First item\n"
        "Run a worker:\n"
        "lantern-queue worker\n"
        "--concurrency 4"
    )


def test_a_citation_under_only_the_top_heading_has_no_section() -> None:
    path = ["Dana Whitlock - Resume"]
    doc = chunk("resume:v2:b", chunk_text(path, "Dana writes Rust."), "Resume", None)
    doc.metadata["heading_path"] = path
    (citation,) = ask(replying(answered("resume:v2:b"), docs=[doc]))["citations"]
    assert citation["section"] is None
    assert citation["excerpt"] == "Dana writes Rust."


def test_answered_reports_the_model_named_in_the_reply() -> None:
    reply = AIMessage(
        content=answered("notes:v1:a"), response_metadata={"model_name": "test/fake-model"}
    )
    body = ask(chains_for(Retriever(DOCS), GenericFakeChatModel(messages=iter([reply]))))
    assert body["status"] == "answered"
    assert body["model"] == "test/fake-model"


def test_unknown_returns_the_models_reason() -> None:
    chains = replying('{"status": "unknown", "reason": "No salary in the evidence."}')
    assert ask(chains, "What is Dana's salary?") == {
        "status": "unknown",
        "reason": "No salary in the evidence.",
        "model": None,
    }


def test_empty_retrieval_is_unknown_without_a_model_call() -> None:
    # No replies: a model call would raise.
    chains = chains_for(Retriever([]), FakeListChatModel(responses=[]))
    assert ask(chains) == {"status": "unknown", "reason": NO_EVIDENCE_REASON, "model": None}


def test_the_chain_cache_builds_each_mode_and_k_once_with_the_defaults() -> None:
    retriever = Retriever(DOCS)
    chains = chains_for(retriever, FakeListChatModel(responses=[]))
    assert chains.get() is chains.get("hybrid", 8)
    chains.get("dense", 3)
    chains.get("dense", 3)
    assert retriever.built == [("hybrid", 8), ("dense", 3)]


# --- failed -------------------------------------------------------------------


def model_error(error: Exception) -> rag.ChainCache:
    return chains_for(Retriever(DOCS), RaisingModel(error=error))


def retrieval_error(error: Exception) -> rag.ChainCache:
    return chains_for(Retriever(DOCS, error=error), FakeListChatModel(responses=[]))


FAILURES: list[tuple[FailedCode, Callable[[], rag.ChainCache]]] = [
    ("retrieval_failed", lambda: retrieval_error(psycopg.OperationalError(SECRET))),
    ("retrieval_failed", lambda: retrieval_error(EmbeddingError(SECRET, text=SECRET))),
    ("retrieval_failed", lambda: retrieval_error(ConnectionError(SECRET))),
    ("pi_timeout", lambda: model_error(PiTimeout(1.0))),
    ("pi_failed", lambda: model_error(PiProcessError(SECRET, returncode=1, stderr_tail=SECRET))),
    ("pi_failed", lambda: model_error(PiProtocolError(SECRET))),
    ("bad_reply", lambda: replying(f"not JSON: {SECRET}")),
    ("bad_reply", lambda: replying(f'{{"status": "answered", "answer": "{SECRET}"}}')),
    ("invented_citation", lambda: replying(answered(f"{SECRET}:v1:a", answer=SECRET))),
    ("internal_error", lambda: retrieval_error(RuntimeError(SECRET))),
]


@pytest.mark.parametrize(
    ("code", "make_chains"),
    FAILURES,
    ids=[f"{code}-{i}" for i, (code, _) in enumerate(FAILURES)],
)
def test_each_failure_is_a_failed_response_with_its_code(
    code: FailedCode,
    make_chains: Callable[[], rag.ChainCache],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    body = ask(make_chains(), request_id="req-42")

    assert body == {
        "status": "failed",
        "error": code,
        "message": rag.FAILURES[code],
        "request_id": "req-42",
    }
    # Neither the reply nor Pi's stderr reaches the response or any log record.
    assert SECRET not in json.dumps(body)
    assert caplog.records, "the failure was not logged"
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in (record.exc_text or "")
        assert record.exc_info is None
    failure_lines = [r.getMessage() for r in caplog.records if r.name == "profile_rag.rag"]
    assert len(failure_lines) == 1
    assert "request_id=req-42" in failure_lines[0]
    assert f"code={code}" in failure_lines[0]
