"""FastAPI application.

`uvicorn profile_rag.api.app:app` imports the `app` object below.

The lifespan function runs once at startup (code before `yield`) and once at
shutdown (code after `yield`). It owns long-lived resources, so they are
created once and closed cleanly: the connection pool, and the answer chains
with their search store (api/search.py). If the chains cannot be opened
(Ollama, the chunk table, or the database is missing), the server does not
start.
"""

import asyncio
import uuid
from collections.abc import AsyncGenerator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from profile_rag import rag, tracing
from profile_rag.api import access, routes
from profile_rag.api.log_config import configure_logging
from profile_rag.api.models import FailedResponse
from profile_rag.api.search import open_chains
from profile_rag.db.pool import create_pool
from profile_rag.settings import TracingSettings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    pool = create_pool(str(settings.database_url))
    app.state.pool = pool
    try:
        # Blocking checks and queries: run them off the event loop.
        opened = await asyncio.to_thread(open_chains, settings)
        app.state.chains = opened.chains
        try:
            yield
        finally:
            await opened.close()
    finally:
        pool.close()


Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_app(app_lifespan: Lifespan = lifespan) -> FastAPI:
    """The application. Only the e2e test server (tests/e2e_server.py) passes its own lifespan."""
    # Before anything logs. Raises a ValidationError for a bad PROFILE_RAG_LOG_LEVEL.
    configure_logging()
    # Before any LangChain object is made. Raises TracingRefused, which stops
    # the server from starting, if LANGSMITH_TRACING=true outside corpus-example.
    tracing.configure_tracing(TracingSettings(), tracing.corpus_dir_from_env())
    app = FastAPI(
        title="profile-rag",
        version="0.1.0",
        lifespan=app_lifespan,
    )
    app.include_router(routes.router, prefix="/api")
    app.add_exception_handler(access.RateLimited, _rate_limited)
    return app


async def _rate_limited(request: Request, error: Exception) -> JSONResponse:
    """A question over a limit (api/access.py): 429 with a `rate_limited` failure."""
    request_id = uuid.uuid4().hex
    body = FailedResponse(
        status="failed",
        error="rate_limited",
        message=rag.FAILURES["rate_limited"],
        request_id=request_id,
    )
    return JSONResponse(
        body.model_dump(),
        status_code=routes.FAILED_STATUS["rate_limited"],
        headers={"X-Request-ID": request_id},
    )


app = create_app()
