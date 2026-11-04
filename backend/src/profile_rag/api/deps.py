"""FastAPI dependencies.

A dependency is a function FastAPI calls to provide a value to a route.
Routes declare what they need (`pool: PoolDep`) instead of reaching for
globals, and tests replace dependencies with `app.dependency_overrides`.
"""

from typing import Annotated

from fastapi import Depends, Request

from profile_rag.db.pool import Pool
from profile_rag.rag import ChainCache


def get_pool(request: Request) -> Pool:
    """Return the pool created in the application's lifespan."""
    pool: Pool = request.app.state.pool
    return pool


PoolDep = Annotated[Pool, Depends(get_pool)]


def get_chains(request: Request) -> ChainCache:
    """Return the answer chains opened in the application's lifespan (api/search.py)."""
    chains: ChainCache = request.app.state.chains
    return chains


ChainsDep = Annotated[ChainCache, Depends(get_chains)]
