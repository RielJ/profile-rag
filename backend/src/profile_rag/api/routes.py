"""The API routes: `POST /api/ask`, `GET /api/sources/{id}`, and `GET /api/health`.

Status codes of `POST /api/ask` (docs/ARCHITECTURE.md, "API"):

- 200: `answered` or `unknown`.
- 5xx: `failed`, with the code in `FAILED_STATUS`. The body is still the
  `FailedResponse` variant of `AskResponse`.
- 422: the request body is invalid (FastAPI's validation error body).

Every `POST /api/ask` response carries an `X-Request-ID` header. A failed
response repeats it in `request_id`, and the server log line for the failure
names it.

Both new routes are plain `def` functions, so FastAPI runs them in its
thread pool: the Pi call and the database queries block a worker thread,
not the event loop.
"""

import logging
import uuid
from typing import Annotated, Any

import psycopg
from fastapi import APIRouter, HTTPException, Path, Response, status
from psycopg.rows import dict_row

from profile_rag import rag
from profile_rag.api import health
from profile_rag.api.deps import ChainsDep, PoolDep
from profile_rag.api.models import (
    AskRequest,
    AskResponse,
    FailedCode,
    FailedResponse,
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
