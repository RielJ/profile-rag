"""FastAPI application.

`uvicorn profile_rag.api.app:app` imports the `app` object below.

The lifespan function runs once at startup (code before `yield`) and once at
shutdown (code after `yield`). It owns long-lived resources such as the
connection pool, so they are created once and closed cleanly.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from profile_rag import tracing
from profile_rag.api import health
from profile_rag.db.pool import create_pool
from profile_rag.settings import TracingSettings, get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    settings = get_settings()
    pool = create_pool(str(settings.database_url))
    app.state.pool = pool
    try:
        yield
    finally:
        pool.close()


def create_app() -> FastAPI:
    # Before any LangChain object is made. Raises TracingRefused, which stops
    # the server from starting, if LANGSMITH_TRACING=true outside corpus-example.
    tracing.configure_tracing(TracingSettings(), tracing.corpus_dir_from_env())
    app = FastAPI(
        title="profile-rag",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/api")
    return app


app = create_app()
