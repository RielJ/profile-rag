"""FastAPI dependencies.

A dependency is a function FastAPI calls to provide a value to a route.
Routes declare what they need (`pool: PoolDep`) instead of reaching for
globals, and tests replace dependencies with `app.dependency_overrides`.
"""

from typing import Annotated

from fastapi import Depends, Request

from profile_rag.db.pool import Pool


def get_pool(request: Request) -> Pool:
    """Return the pool created in the application's lifespan."""
    pool: Pool = request.app.state.pool
    return pool


PoolDep = Annotated[Pool, Depends(get_pool)]
