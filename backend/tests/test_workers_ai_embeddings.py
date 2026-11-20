"""Tests for vectorstore/workers_ai_embeddings.py.

No network: `httpx.MockTransport` answers in place of Cloudflare and
records each request. One opt-in live test embeds a question with the real
API (`--live`, with PROFILE_RAG_CF_ACCOUNT_ID and PROFILE_RAG_CF_API_TOKEN
set).
"""

import json
import math
import os
from collections.abc import Callable

import httpx
import pytest
from pydantic import SecretStr

from profile_rag.vectorstore import workers_ai_embeddings
from profile_rag.vectorstore.models import BGE_M3, MODELS, WorkersAIModel
from profile_rag.vectorstore.prefixed import EmbeddingError, PrefixedEmbeddings
from profile_rag.vectorstore.workers_ai_embeddings import (
    MAX_BATCH,
    WorkersAIError,
    for_model,
    model_problem,
)

ACCOUNT = "acct123"
TOKEN = "cf-secret-token-value"
TINY = WorkersAIModel(
    key="tiny-cf",
    workers_ai_name="@cf/test/tiny",
    dimensions=3,
    table_name="chunks_tiny_cf",
    document_prefix="doc: ",
    query_prefix="q: ",
    max_words=1000,
)

Handler = Callable[[httpx.Request], httpx.Response]


def vectors_for(texts: list[str], dimensions: int = 3) -> list[list[float]]:
    """A different vector per position, so order mistakes show."""
    return [[float(i), 1.0, 0.5][:dimensions] + [0.0] * (dimensions - 3) for i in range(len(texts))]


def answering(requests: list[httpx.Request], dimensions: int = 3) -> Handler:
    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        texts = json.loads(request.content)["text"]
        data = vectors_for(texts, dimensions)
        return httpx.Response(
            200,
            json={"success": True, "result": {"shape": [len(data), dimensions], "data": data}},
        )

    return handle


def embeddings(
    handler: Handler, model: WorkersAIModel = TINY, *, token: str | None = TOKEN
) -> PrefixedEmbeddings:
    return for_model(
        model,
        ACCOUNT,
        None if token is None else SecretStr(token),
        transport=httpx.MockTransport(handler),
    )


def sent_texts(request: httpx.Request) -> list[str]:
    return json.loads(request.content)["text"]


# --- Requests ------------------------------------------------------------------


def test_documents_go_to_the_model_url_with_the_token_and_prefix() -> None:
    requests: list[httpx.Request] = []
    vectors = embeddings(answering(requests)).embed_documents(["one", "two"])

    assert vectors == [[0.0, 1.0, 0.5], [1.0, 1.0, 0.5]]
    [request] = requests
    assert request.method == "POST"
    assert str(request.url) == (
        f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/test/tiny"
    )
    assert request.headers["authorization"] == f"Bearer {TOKEN}"
    assert sent_texts(request) == ["doc: one", "doc: two"]


def test_a_query_gets_the_query_prefix_and_one_vector() -> None:
    requests: list[httpx.Request] = []
    vector = embeddings(answering(requests)).embed_query("Who is Dana?")
    assert vector == [0.0, 1.0, 0.5]
    assert sent_texts(requests[0]) == ["q: Who is Dana?"]


def test_many_texts_are_sent_in_batches_and_kept_in_order() -> None:
    requests: list[httpx.Request] = []
    texts = [f"text {i}" for i in range(2 * MAX_BATCH + 7)]
    vectors = embeddings(answering(requests)).embed_documents(texts)

    assert [len(sent_texts(r)) for r in requests] == [MAX_BATCH, MAX_BATCH, 7]
    assert [t for r in requests for t in sent_texts(r)] == [f"doc: {t}" for t in texts]
    # Each batch numbers its vectors from 0 (vectors_for), so position i
    # gets the value i % MAX_BATCH.
    assert [v[0] for v in vectors] == [float(i % MAX_BATCH) for i in range(len(texts))]


def test_no_texts_make_no_request() -> None:
    requests: list[httpx.Request] = []
    assert embeddings(answering(requests)).embed_documents([]) == []
    assert requests == []


# --- Errors --------------------------------------------------------------------


def test_a_400_is_an_error_with_cloudflares_message_and_no_token() -> None:
    def too_long(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "success": False,
                "errors": [{"code": 3010, "message": "Sequence too long: 9000 > 8192"}],
                "result": None,
            },
        )

    with pytest.raises(WorkersAIError) as caught:
        embeddings(too_long).embed_documents(["a", "b"])
    message = str(caught.value)
    assert "HTTP 400" in message
    assert "3010: Sequence too long: 9000 > 8192" in message
    assert "texts 0-1" in message
    assert TOKEN not in message
    # An EmbeddingError without a position: the CLI stops instead of naming
    # one chunk, and no chunk is skipped.
    assert isinstance(caught.value, EmbeddingError)
    assert caught.value.position is None
    assert caught.value.text is None


def test_the_failing_batch_is_named() -> None:
    calls = 0

    def second_fails(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            return httpx.Response(500, json={"success": False, "errors": []})
        texts = sent_texts(request)
        return httpx.Response(200, json={"success": True, "result": {"data": vectors_for(texts)}})

    texts = [f"t{i}" for i in range(MAX_BATCH + 1)]
    with pytest.raises(WorkersAIError, match=rf"texts {MAX_BATCH}-{MAX_BATCH}: HTTP 500"):
        embeddings(second_fails).embed_documents(texts)


@pytest.mark.parametrize(
    ("response", "detail"),
    [
        (httpx.Response(502, text="<html>bad gateway</html>"), "HTTP 502 (not JSON)"),
        (httpx.Response(200, json={"success": False, "errors": []}), "HTTP 200 (no result.data)"),
        (httpx.Response(200, json={"success": True, "result": {}}), "HTTP 200 (no result.data)"),
        (
            httpx.Response(200, json={"success": True, "result": {"data": "x"}}),
            "HTTP 200 (no result.data)",
        ),
        (
            httpx.Response(
                401, json={"success": False, "errors": [{"code": 10000, "message": "Auth"}]}
            ),
            "HTTP 401 (10000: Auth)",
        ),
    ],
)
def test_an_answer_without_vectors_is_an_error(response: httpx.Response, detail: str) -> None:
    with pytest.raises(WorkersAIError, match=detail.replace("(", r"\(").replace(")", r"\)")):
        embeddings(lambda request: response).embed_query("q")


def test_a_network_error_names_only_its_class() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"cannot reach {request.url} with {TOKEN}")

    with pytest.raises(WorkersAIError) as caught:
        embeddings(refuse).embed_query("q")
    assert "request failed (ConnectError)" in str(caught.value)
    assert TOKEN not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__


def test_a_wrong_vector_length_is_caught() -> None:
    requests: list[httpx.Request] = []
    with pytest.raises(EmbeddingError, match="wrong vector length"):
        embeddings(answering(requests, dimensions=4)).embed_query("q")


def test_missing_credentials_fail_before_any_request() -> None:
    requests: list[httpx.Request] = []
    with pytest.raises(WorkersAIError, match="PROFILE_RAG_CF_API_TOKEN"):
        embeddings(answering(requests), token=None).embed_documents(["a"])
    assert requests == []


def test_the_token_is_not_in_the_objects_repr() -> None:
    inner = embeddings(answering([])).inner
    assert TOKEN not in repr(inner)
    assert TOKEN not in repr(vars(inner))


# --- model_problem -------------------------------------------------------------


@pytest.mark.parametrize(("account", "token"), [(None, "t"), (ACCOUNT, None), (None, None)])
def test_a_missing_credential_is_a_problem(account: str | None, token: str | None) -> None:
    problem = model_problem(TINY, account, None if token is None else SecretStr(token))
    assert problem is not None
    assert "PROFILE_RAG_CF_ACCOUNT_ID" in problem
    assert "PROFILE_RAG_CF_API_TOKEN" in problem


def test_both_credentials_are_no_problem() -> None:
    assert model_problem(TINY, ACCOUNT, SecretStr(TOKEN)) is None


# --- The models ----------------------------------------------------------------


def test_bge_m3_entry() -> None:
    assert MODELS["bge-m3"] is BGE_M3
    assert BGE_M3.workers_ai_name == "@cf/baai/bge-m3"
    assert BGE_M3.dimensions == 1024
    assert BGE_M3.table_name == "chunks_bge_m3"
    assert (BGE_M3.document_prefix, BGE_M3.query_prefix) == ("", "")
    assert BGE_M3.max_words > 200  # above the splitter's default chunk size


@pytest.mark.live
@pytest.mark.parametrize(
    "model",
    [m for m in MODELS.values() if isinstance(m, WorkersAIModel)],
    ids=lambda m: m.key,
)
def test_one_live_embedding_has_the_expected_dimensions(model: WorkersAIModel) -> None:
    account = os.environ.get("PROFILE_RAG_CF_ACCOUNT_ID")
    token = os.environ.get("PROFILE_RAG_CF_API_TOKEN")
    if not account or not token:
        pytest.skip("set PROFILE_RAG_CF_ACCOUNT_ID and PROFILE_RAG_CF_API_TOKEN")
    vector = workers_ai_embeddings.for_model(model, account, SecretStr(token)).embed_query(
        "Who is Dana Whitlock?"
    )
    assert len(vector) == model.dimensions
    assert all(math.isfinite(value) for value in vector)
