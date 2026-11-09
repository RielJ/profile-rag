"""The API routes: `POST /api/ask`, `POST /api/ask/stream`, `GET /api/sources/{id}`,
and `GET /api/health`.

Status codes of `POST /api/ask` (docs/ARCHITECTURE.md, "API"):

- 200: `answered` or `unknown`.
- 5xx: `failed`, with the code in `FAILED_STATUS`. The body is still the
  `FailedResponse` variant of `AskResponse`.
- 422: the request body is invalid (FastAPI's validation error body).

Every `POST /api/ask` response carries an `X-Request-ID` header. A failed
response repeats it in `request_id`, and the server log line for the failure
names it.

`POST /api/ask/stream` always answers 200 once the request is valid: a
failure arrives as the `failed` response in the last line, because the
status line is sent before the answer exists. The request ID is in the
`X-Request-ID` header, as for `POST /api/ask`.

`POST /api/ask` and `GET /api/sources/{id}` are plain `def` functions, so
FastAPI runs them in its thread pool: the Pi call and the database queries
block a worker thread, not the event loop. `POST /api/ask/stream` runs
`rag.ask` in a worker thread too, and passes its draft pieces to the event
loop through a queue. If the client disconnects, the next draft piece raises
`rag.Abandoned` in the worker, which stops the Pi process.
"""

import asyncio
import logging
import threading
import uuid
from collections.abc import AsyncIterator
from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, HTTPException, Path, Response, status
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row

from profile_rag import rag
from profile_rag.api import health
from profile_rag.api.deps import ChainsDep, PoolDep
from profile_rag.api.models import (
    AskRequest,
    AskResponse,
    AskStreamEvent,
    DraftEvent,
    FailedCode,
    FailedResponse,
    ResultEvent,
    SourceResponse,
)
from profile_rag.db.pool import Pool
from profile_rag.retrievers.dense import VISIBLE_VALUES
from profile_rag.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()
router.include_router(health.router)

FAILED_STATUS: dict[FailedCode, int] = {
    "retrieval_failed": status.HTTP_503_SERVICE_UNAVAILABLE,
    "pi_timeout": status.HTTP_504_GATEWAY_TIMEOUT,
    "pi_failed": status.HTTP_502_BAD_GATEWAY,
    "bad_reply": status.HTTP_502_BAD_GATEWAY,
    "invented_citation": status.HTTP_502_BAD_GATEWAY,
    "internal_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
}

_FAILED_DOC: dict[int | str, dict[str, Any]] = {
    code: {"model": FailedResponse, "description": "The question could not be answered."}
    for code in sorted(set(FAILED_STATUS.values()))
}


@router.post("/ask", tags=["ask"], response_model=AskResponse, responses=_FAILED_DOC)
def ask(body: AskRequest, chains: ChainsDep, response: Response) -> AskResponse:
    """Answer a question from the indexed evidence, with citations."""
    request_id = uuid.uuid4().hex
    response.headers["X-Request-ID"] = request_id
    result = rag.ask(body.question, chains.get(body.mode, body.k), request_id=request_id)
    if isinstance(result.root, FailedResponse):
        response.status_code = FAILED_STATUS[result.root.error]
    return result


class NDJSONResponse(StreamingResponse):
    """A stream of newline-delimited JSON. Named so OpenAPI shows its media type."""

    media_type = "application/x-ndjson"


_STREAM_DOC: dict[int | str, dict[str, Any]] = {
    200: {
        "model": AskStreamEvent,
        "description": "Newline-delimited JSON: `draft` events, then one `result` event.",
    }
}


@router.post("/ask/stream", tags=["ask"], response_class=NDJSONResponse, responses=_STREAM_DOC)
async def ask_stream(body: AskRequest, chains: ChainsDep) -> NDJSONResponse:
    """Answer a question as `POST /api/ask` does, streaming the draft answer first."""
    request_id = uuid.uuid4().hex
    chain = chains.get(body.mode, body.k)
    return NDJSONResponse(
        _ask_events(body.question, chain, request_id),
        headers={"X-Request-ID": request_id, "Cache-Control": "no-store"},
    )


async def _ask_events(question: str, chain: rag.AnswerChain, request_id: str) -> AsyncIterator[str]:
    loop = asyncio.get_running_loop()
    lines: asyncio.Queue[DraftEvent | ResultEvent] = asyncio.Queue()
    gone = threading.Event()

    def on_draft(text: str) -> None:  # runs in the worker thread
        if gone.is_set():
            raise rag.Abandoned
        loop.call_soon_threadsafe(lines.put_nowait, DraftEvent(type="draft", text=text))

    def run() -> None:
        try:
            result = rag.ask(question, chain, request_id=request_id, on_draft=on_draft)
        except rag.Abandoned:
            logger.info("ask abandoned, client disconnected: request_id=%s", request_id)
            return
        loop.call_soon_threadsafe(lines.put_nowait, ResultEvent(type="result", response=result))

    worker = loop.run_in_executor(None, run)
    try:
        while True:
            event = await lines.get()
            yield event.model_dump_json() + "\n"
            if isinstance(event, ResultEvent):
                break
        await worker
    finally:
        gone.set()  # the client left early, or the stream is done: drafts are no longer wanted


def find_source(pool: Pool, source_id: str, *, timeout: float) -> SourceResponse | None:
    """The approved, visible source `source_id`, or None.

    The `sources` table holds approved sources only (corpus/sources_table.py),
    so a candidate, in-review, or rejected source is never found. A URL that
    is not http(s) is returned as None, as in citations.
    """
    with pool.connection(timeout=timeout) as conn, conn.cursor(row_factory=dict_row) as cur:
        row = cur.execute(
            "SELECT id, kind, title, url, attribution, visibility, approved_at FROM sources"
            " WHERE id = %s AND visibility = ANY(%s)",
            (source_id, list(VISIBLE_VALUES)),
        ).fetchone()
    if row is None:
        return None
    return SourceResponse.model_validate({**row, "url": rag.public_url(row["url"])})


@router.get(
    "/sources/{source_id}",
    tags=["sources"],
    response_model=SourceResponse,
    responses={404: {"description": "No approved, visible source has this ID."}},
)
def get_source(source_id: Annotated[str, Path(max_length=200)], pool: PoolDep) -> SourceResponse:
    """The metadata of one approved source. Never its text."""
    try:
        source = find_source(pool, source_id, timeout=get_settings().db_connect_timeout)
    except psycopg.Error as error:
        logger.warning("source lookup failed: error=%s", type(error).__name__)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "database unavailable") from None
    if source is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "source not found")
    return source
