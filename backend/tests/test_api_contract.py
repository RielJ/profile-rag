"""Tests for the API contract: POST /api/ask, GET /api/sources/{id}, /openapi.json.

`POST /api/ask` runs the real chain (generation/chain.py) through
`rag.ChainCache`, with a fake retriever and a fake chat model, so no
database, Ollama, or Pi is needed. The source tests use a throwaway
database filled from corpus-example/sources.yaml.
"""

import inspect
import json
import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, override

import httpx2
import psycopg
import pytest
from fastapi.testclient import TestClient
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
from profile_rag.api import app as app_module
from profile_rag.api import routes, search
from profile_rag.api.app import create_app
from profile_rag.api.deps import get_chains, get_pool
from profile_rag.api.models import FailedCode
from profile_rag.corpus.registry import load_registry
from profile_rag.corpus.sources_table import sync_sources
from profile_rag.db.migrate import migrate
from profile_rag.db.pool import Pool, create_pool
from profile_rag.generation.chain import NO_EVIDENCE_REASON
from profile_rag.generation.pi_rpc import PiProcessError, PiProtocolError, PiTimeout
from profile_rag.retrievers.factory import SearchMode
from profile_rag.settings import Settings
from profile_rag.vectorstore.prefixed import EmbeddingError

EXAMPLE_REGISTRY = Path(__file__).resolve().parents[2] / "corpus-example" / "sources.yaml"

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


def client_for(retriever: Retriever, model: BaseChatModel) -> TestClient:
    """A client whose /api/ask runs the real chain on `retriever` and `model`.

    Not used as a context manager, so the lifespan (real database, Ollama)
    does not run.
    """
    chains = rag.ChainCache(retriever.make, model, default_mode="hybrid", default_k=8)
    app = create_app()
    app.dependency_overrides[get_chains] = lambda: chains
    return TestClient(app)


def replying(*replies: str, docs: list[Document] = DOCS) -> TestClient:
    return client_for(Retriever(docs), FakeListChatModel(responses=list(replies)))


def answered(*citations: str, answer: str = "Dana maintains lantern-queue.") -> str:
    return json.dumps({"status": "answered", "answer": answer, "citations": list(citations)})


def post(
    client: TestClient, question: str = "What does Dana maintain?", **extra: object
) -> httpx2.Response:
    return client.post("/api/ask", json={"question": question, **extra})


# --- answered and unknown -----------------------------------------------------


def test_answered_has_one_citation_per_cited_chunk_with_its_source() -> None:
    client = replying(answered("lantern:v1:a", "notes:v1:a", "local:v1:a", "lantern:v1:a"))
    response = post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["answer"] == "Dana maintains lantern-queue."
    assert body["model"] is None  # FakeListChatModel reports no model name
    assert body["citations"][1:] == [
        {
            "chunk_id": "notes:v1:a",
            "source_id": "notes",
            "source_title": "Migration notes",
            "url": None,
            "attribution": "project_doc",
            "excerpt": "Dana moved the queue to Postgres.",
        },
        {
            "chunk_id": "local:v1:a",
            "source_id": "local",
            "source_title": "Local copy",
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


def test_answered_reports_the_model_named_in_the_reply() -> None:
    reply = AIMessage(
        content=answered("notes:v1:a"), response_metadata={"model_name": "test/fake-model"}
    )
    client = client_for(Retriever(DOCS), GenericFakeChatModel(messages=iter([reply])))
    body = post(client).json()
    assert body["status"] == "answered"
    assert body["model"] == "test/fake-model"


def test_unknown_returns_the_models_reason() -> None:
    client = replying('{"status": "unknown", "reason": "No salary in the evidence."}')
    response = post(client, "What is Dana's salary?")
    assert response.status_code == 200
    assert response.json() == {
        "status": "unknown",
        "reason": "No salary in the evidence.",
        "model": None,
    }


def test_empty_retrieval_is_unknown_without_a_model_call() -> None:
    # No replies: a model call would raise.
    client = client_for(Retriever([]), FakeListChatModel(responses=[]))
    response = post(client)
    assert response.status_code == 200
    assert response.json() == {"status": "unknown", "reason": NO_EVIDENCE_REASON, "model": None}


def test_mode_and_k_select_the_chain_and_default_to_the_settings() -> None:
    retriever = Retriever(DOCS)
    replies = [answered("notes:v1:a")] * 3
    client = client_for(retriever, FakeListChatModel(responses=replies))
    post(client)
    post(client, mode="dense", k=3)
    post(client)
    assert retriever.built == [("hybrid", 8), ("dense", 3)], "each chain is built once"


# --- failed -------------------------------------------------------------------


def model_error(error: Exception) -> TestClient:
    return client_for(Retriever(DOCS), RaisingModel(error=error))


def retrieval_error(error: Exception) -> TestClient:
    return client_for(Retriever(DOCS, error=error), FakeListChatModel(responses=[]))


FAILURES: list[tuple[FailedCode, Callable[[], TestClient], int]] = [
    ("retrieval_failed", lambda: retrieval_error(psycopg.OperationalError(SECRET)), 503),
    ("retrieval_failed", lambda: retrieval_error(EmbeddingError(SECRET, text=SECRET)), 503),
    ("retrieval_failed", lambda: retrieval_error(ConnectionError(SECRET)), 503),
    ("pi_timeout", lambda: model_error(PiTimeout(1.0)), 504),
    (
        "pi_failed",
        lambda: model_error(PiProcessError(SECRET, returncode=1, stderr_tail=SECRET)),
        502,
    ),
    ("pi_failed", lambda: model_error(PiProtocolError(SECRET)), 502),
    ("bad_reply", lambda: replying(f"not JSON: {SECRET}"), 502),
    ("bad_reply", lambda: replying(f'{{"status": "answered", "answer": "{SECRET}"}}'), 502),
    ("invented_citation", lambda: replying(answered(f"{SECRET}:v1:a", answer=SECRET)), 502),
    ("internal_error", lambda: retrieval_error(RuntimeError(SECRET)), 500),
]


@pytest.mark.parametrize(
    ("code", "make_client", "http_status"),
    FAILURES,
    ids=[f"{code}-{i}" for i, (code, _, _) in enumerate(FAILURES)],
)
def test_each_failure_is_a_failed_body_with_its_code_and_status(
    code: FailedCode,
    make_client: Callable[[], TestClient],
    http_status: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    response = post(make_client())

    assert response.status_code == http_status
    request_id = response.headers["X-Request-ID"]
    assert response.json() == {
        "status": "failed",
        "error": code,
        "message": rag.FAILURES[code],
        "request_id": request_id,
    }
    # Neither the reply nor Pi's stderr reaches the response or any log record.
    assert SECRET not in response.text
    assert caplog.records, "the failure was not logged"
    for record in caplog.records:
        assert SECRET not in record.getMessage()
        assert SECRET not in (record.exc_text or "")
        assert record.exc_info is None
    failure_lines = [r.getMessage() for r in caplog.records if r.name == "profile_rag.rag"]
    assert len(failure_lines) == 1
    assert request_id in failure_lines[0]
    assert f"code={code}" in failure_lines[0]


def test_each_request_gets_its_own_request_id() -> None:
    client = model_error(PiTimeout(1.0))
    first = post(client).json()["request_id"]
    second = post(client).json()["request_id"]
    assert first != second


# --- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   \n"},
        {"question": "x" * 1001},
        {"question": "ok", "mode": "fuzzy"},
        {"question": "ok", "k": 0},
        {"question": "ok", "k": 21},
        {"question": "ok", "extra": True},
    ],
    ids=["missing", "empty", "blank", "too-long", "bad-mode", "k-0", "k-21", "extra-field"],
)
def test_an_invalid_request_is_422(body: dict[str, object]) -> None:
    # No replies and no docs: the chain must not run.
    client = client_for(Retriever([], error=AssertionError("ran")), FakeListChatModel(responses=[]))
    response = client.post("/api/ask", json=body)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_the_longest_question_is_accepted() -> None:
    response = post(replying(answered("notes:v1:a")), "x" * 1000)
    assert response.status_code == 200


def test_the_ask_route_runs_in_the_thread_pool() -> None:
    # A plain `def` route runs in a worker thread, so the blocking Pi call
    # does not stop the event loop.
    assert not inspect.iscoroutinefunction(routes.ask)
    assert not inspect.iscoroutinefunction(routes.get_source)


# --- sources ------------------------------------------------------------------


@pytest.fixture
def source_pool(empty_database: str) -> Iterator[Pool]:
    """A pool on a migrated database whose `sources` hold the example registry."""
    migrate(empty_database)
    with psycopg.connect(empty_database) as conn:
        sync_sources(conn, load_registry(EXAMPLE_REGISTRY))
    pool = create_pool(empty_database)
    pool.wait(timeout=5)
    yield pool
    pool.close()


def source_client(pool: Pool) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_pool] = lambda: pool
    return TestClient(app)


def test_an_approved_source_returns_its_metadata_only(source_pool: Pool) -> None:
    entry = next(e for e in load_registry(EXAMPLE_REGISTRY).sources if e.id == "resume-2026")
    assert entry.approved_at is not None

    response = source_client(source_pool).get("/api/sources/resume-2026")

    assert response.status_code == 200
    body = response.json()
    assert body.pop("approved_at").replace("Z", "+00:00") == entry.approved_at.isoformat()
    assert body == {
        "id": "resume-2026",
        "kind": entry.kind,
        "title": entry.title,
        "url": "https://example.com/dana/resume",
        "attribution": entry.attribution,
        "visibility": "public",
    }


def test_a_source_url_that_is_not_http_is_null(source_pool: Pool) -> None:
    with source_pool.connection() as conn:
        conn.execute("UPDATE sources SET url = 'approved/resume.md' WHERE id = 'resume-2026'")
    body = source_client(source_pool).get("/api/sources/resume-2026").json()
    assert body["url"] is None


@pytest.mark.parametrize(
    "source_id",
    ["linkedin-experience", "note-harborlight-migration", "no-such-source", "RESUME-2026"],
    ids=["candidate", "rejected", "unknown", "other-case"],
)
def test_an_unknown_or_unapproved_source_is_404(source_pool: Pool, source_id: str) -> None:
    response = source_client(source_pool).get(f"/api/sources/{source_id}")
    assert response.status_code == 404
    assert response.json() == {"detail": "source not found"}


def test_sources_is_503_when_the_database_is_down(unreachable_pool: Pool) -> None:
    response = source_client(unreachable_pool).get("/api/sources/resume-2026")
    assert response.status_code == 503


# --- startup -----------------------------------------------------------------


class FakeEngine:
    closed = False

    async def close(self) -> None:
        self.closed = True


def test_the_lifespan_opens_the_chains_once_and_closes_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chains = rag.ChainCache(
        Retriever(DOCS).make,
        FakeListChatModel(responses=[answered("notes:v1:a")] * 2),
        default_mode="hybrid",
        default_k=8,
    )
    engine = FakeEngine()
    opened: list[Settings] = []

    def fake_open(settings: Settings) -> search.OpenChains:
        opened.append(settings)
        return search.OpenChains(chains=chains, engine=engine)

    monkeypatch.setattr(app_module, "open_chains", fake_open)

    def closed_pool(conninfo: str) -> Pool:
        return create_pool(conninfo, open_now=False)  # never connects: no database needed

    monkeypatch.setattr(app_module, "create_pool", closed_pool)
    with TestClient(create_app()) as client:
        assert post(client).json()["status"] == "answered"
        assert post(client).json()["status"] == "answered"
        assert not engine.closed
    assert len(opened) == 1
    assert engine.closed


def test_startup_refuses_an_unknown_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_RAG_API_MODEL", "no-such-model")
    settings = Settings()
    with pytest.raises(search.SearchSetupError, match="PROFILE_RAG_API_MODEL"):
        search.open_chains(settings)


# --- OpenAPI ------------------------------------------------------------------


def test_openapi_has_the_status_discriminator_and_the_failed_statuses() -> None:
    schema = TestClient(create_app()).get("/openapi.json").json()
    schemas = schema["components"]["schemas"]

    ask_response = schemas["AskResponse"]
    assert ask_response["discriminator"] == {
        "propertyName": "status",
        "mapping": {
            "answered": "#/components/schemas/AnsweredResponse",
            "failed": "#/components/schemas/FailedResponse",
            "unknown": "#/components/schemas/UnknownResponse",
        },
    }
    assert len(ask_response["oneOf"]) == 3
    assert schemas["FailedResponse"]["properties"]["error"]["enum"] == list(rag.FAILURES)

    responses = schema["paths"]["/api/ask"]["post"]["responses"]
    ok = responses["200"]["content"]["application/json"]["schema"]
    assert ok == {"$ref": "#/components/schemas/AskResponse"}
    for http_status in ("500", "502", "503", "504"):
        failed = responses[http_status]["content"]["application/json"]["schema"]
        assert failed == {"$ref": "#/components/schemas/FailedResponse"}
    assert "422" in responses
