"""Embeddings from Cloudflare Workers AI, through its REST API.

The deployed API runs on a Cloudflare Worker, which can call Workers AI but
neither Ollama nor fastembed. A model the Worker uses must therefore be
indexed with the same Workers AI model (models.py, `workers_ai_name`). This
module is the indexer's side, and the CLI's for `search` and `eval`.

One request per batch of at most `MAX_BATCH` texts:

    POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}
    Authorization: Bearer <token>
    {"text": ["...", "..."]}

The answer is `{"success": true, "result": {"shape": [n, d], "data": [[...], ...]}}`.
The vector count, the vector length, and the values are checked by
`PrefixedEmbeddings` (prefixed.py), as for every model.

Every failure raises `WorkersAIError`, an `EmbeddingError` without a
position: an HTTP error, a non-JSON answer, `success: false` (such as HTTP
400 "Sequence too long" for a text over the model's token limit), or an
answer without `result.data`. So the CLI's `index` stops with the message
and no chunk is skipped. The message holds the HTTP status and Cloudflare's
error codes and messages, never the token or the request headers.

- `model_problem(model, account_id, api_token)` is None when both
  credentials are set. It does not call Cloudflare.
- `for_model(model, account_id, api_token)` returns the model's embeddings,
  with its prefixes. It does not call Cloudflare either, so commands that
  never embed (`hnsw apply`) can use it without credentials; the first
  embedding then fails with a `WorkersAIError`.

Tests: backend/tests/test_workers_ai_embeddings.py, with an
`httpx.MockTransport`; one opt-in live test (`--live`).
"""

from collections.abc import Sequence
from typing import override

import httpx
from langchain_core.embeddings import Embeddings
from pydantic import SecretStr

from profile_rag.vectorstore.models import WorkersAIModel
from profile_rag.vectorstore.prefixed import EmbeddingError, PrefixedEmbeddings

API_BASE = "https://api.cloudflare.com/client/v4"
# The `maxItems` of the text array in the input schema of Workers AI's
# embedding models.
MAX_BATCH = 100
# One batch of 100 chunks takes a few seconds; a minute means it is stuck.
TIMEOUT_SECONDS = 60
# Cut Cloudflare's error text in messages, so a long answer stays one line.
_MAX_DETAIL = 300


class WorkersAIError(EmbeddingError):
    """Workers AI did not return embeddings for a batch."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason, position=None)


def _errors(body: object) -> str:
    """Cloudflare's `errors` list as "code: message; ...", or ""."""
    if not isinstance(body, dict):
        return ""
    errors = body.get("errors")
    if not isinstance(errors, list):
        return ""
    parts = []
    for error in errors:
        if isinstance(error, dict):
            parts.append(f"{error.get('code', '?')}: {error.get('message', '')}".strip())
    return "; ".join(parts)[:_MAX_DETAIL]


def _vectors(body: object) -> list[list[float]] | None:
    """`result.data` of a successful answer, or None if the answer has no such list."""
    if not isinstance(body, dict) or body.get("success") is not True:
        return None
    result = body.get("result")
    if not isinstance(result, dict):
        return None
    data = result.get("data")
    if not isinstance(data, list) or not all(isinstance(vector, list) for vector in data):
        return None
    return data


class WorkersAIEmbeddings(Embeddings):
    """Raw Workers AI embeddings: no prefixes and no checks of the vectors.

    Use `for_model`, which wraps this in `PrefixedEmbeddings`. The token is
    only in the `httpx.Client`'s headers, not in an attribute, so neither
    `repr` nor a traceback's local variables show it.
    """

    def __init__(
        self,
        model_name: str,
        account_id: str | None,
        api_token: SecretStr | None,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model_name = model_name
        self.has_credentials = account_id is not None and api_token is not None
        self.client = httpx.Client(
            base_url=f"{API_BASE}/accounts/{account_id}/ai/run/",
            headers=(
                {"Authorization": f"Bearer {api_token.get_secret_value()}"}
                if api_token is not None
                else {}
            ),
            timeout=TIMEOUT_SECONDS,
            transport=transport,
        )

    def _run(self, texts: Sequence[str], first: int) -> list[list[float]]:
        """Embed one batch. `first` is the batch's first position, for messages."""
        where = f"Workers AI {self.model_name}, texts {first}-{first + len(texts) - 1}"
        if not self.has_credentials:
            raise WorkersAIError(
                f"{where}: set PROFILE_RAG_CF_ACCOUNT_ID and PROFILE_RAG_CF_API_TOKEN."
            )
        try:
            response = self.client.post(self.model_name, json={"text": list(texts)})
        except httpx.HTTPError as error:
            # The class name only: the message of some httpx errors names the
            # request, and nothing from the request belongs in the output.
            raise WorkersAIError(f"{where}: request failed ({type(error).__name__})") from None
        try:
            body: object = response.json()
        except ValueError:
            body = None
        vectors = _vectors(body)
        if response.is_success and vectors is not None:
            return vectors
        detail = _errors(body) or ("not JSON" if body is None else "no result.data")
        raise WorkersAIError(f"{where}: HTTP {response.status_code} ({detail})")

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for first in range(0, len(texts), MAX_BATCH):
            vectors.extend(self._run(texts[first : first + MAX_BATCH], first))
        return vectors

    @override
    def embed_query(self, text: str) -> list[float]:
        vectors = self._run([text], 0)
        if len(vectors) != 1:
            raise WorkersAIError(f"Workers AI {self.model_name}: wrong number of vectors")
        return vectors[0]


def model_problem(
    model: WorkersAIModel, account_id: str | None, api_token: SecretStr | None
) -> str | None:
    """What stops `model` from being used through Workers AI, or None."""
    if account_id is None or api_token is None:
        return (
            f"{model.key} runs on Cloudflare Workers AI. Set PROFILE_RAG_CF_ACCOUNT_ID and"
            " PROFILE_RAG_CF_API_TOKEN (a token with the Workers AI: Read permission)."
        )
    return None


def for_model(
    model: WorkersAIModel,
    account_id: str | None,
    api_token: SecretStr | None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> PrefixedEmbeddings:
    """The embeddings of `model` through Workers AI, with its prefixes.

    `transport` replaces the network in tests.
    """
    inner = WorkersAIEmbeddings(model.workers_ai_name, account_id, api_token, transport=transport)
    return PrefixedEmbeddings(inner, model)
