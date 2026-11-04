"""Health endpoint: `GET /api/health`.

Reports whether the backend can reach its dependencies. Returns HTTP 200 when
everything is available and HTTP 503 (Service Unavailable) when something is
not, so scripts and load balancers can read the status code alone.
"""

from typing import Literal

import psycopg
from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from profile_rag.api.deps import PoolDep
from profile_rag.db.pool import Pool
from profile_rag.settings import get_settings

router = APIRouter(tags=["health"])

DependencyStatus = Literal["ok", "unavailable"]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: DependencyStatus


def check_database(pool: Pool, *, timeout: float) -> DependencyStatus:
    """Return "ok" if the database answers a trivial query, else "unavailable".

    Contract:
    - Never raises for database problems. A down, unreachable, or closed
      database is reported as "unavailable".
    - Returns within roughly `timeout` seconds even when the database is
      unreachable.
    - Does not hide programming errors: only database-related exceptions are
      treated as "unavailable".

    Tests: backend/tests/test_health.py
    """

    try:
        with pool.connection(timeout=timeout) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error:
        return "unavailable"
    return "ok"


@router.get("/health", response_model=HealthResponse)
def health(pool: PoolDep, response: Response) -> HealthResponse:
    database = check_database(pool, timeout=get_settings().db_connect_timeout)
    overall = "ok" if database == "ok" else "degraded"
    if overall != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status=overall, database=database)
