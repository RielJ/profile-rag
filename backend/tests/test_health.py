"""Tests for GET /api/health and check_database().

Run them with `make test-backend`, or a single test with:

    uv --directory backend run pytest tests/test_health.py -k unreachable
"""

import time
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from profile_rag.api.health import check_database
from profile_rag.db.pool import Pool

MakeClient = Callable[[Pool], TestClient]


def test_check_database_ok_when_database_answers(test_pool: Pool) -> None:
    assert check_database(test_pool, timeout=2) == "ok"


def test_check_database_unavailable_when_unreachable(unreachable_pool: Pool) -> None:
    assert check_database(unreachable_pool, timeout=1) == "unavailable"


def test_check_database_respects_timeout(unreachable_pool: Pool) -> None:
    started = time.monotonic()
    check_database(unreachable_pool, timeout=1)
    elapsed = time.monotonic() - started
    # Allow some slack over the 1-second timeout, but not an indefinite hang.
    assert elapsed < 3, f"check_database took {elapsed:.1f}s with a 1s timeout"


def test_check_database_unavailable_when_pool_closed(closed_pool: Pool) -> None:
    assert check_database(closed_pool, timeout=1) == "unavailable"


def test_check_database_does_not_hide_programming_errors() -> None:
    # Passing something that is not a pool is a bug in the caller, not a
    # database outage. It must raise, not report "unavailable".
    not_a_pool: Pool = object()  # pyrefly: ignore[bad-assignment]
    with pytest.raises(AttributeError):
        check_database(not_a_pool, timeout=1)


def test_health_endpoint_ok(test_pool: Pool, make_client: MakeClient) -> None:
    response = make_client(test_pool).get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_endpoint_degraded_when_database_down(
    unreachable_pool: Pool, make_client: MakeClient
) -> None:
    response = make_client(unreachable_pool).get("/api/health")
    assert response.status_code == 503
    assert response.json() == {"status": "degraded", "database": "unavailable"}
