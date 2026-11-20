"""Tests for api/access.py: the access key and the question limits.

The app runs without its lifespan, with a chain that answers `unknown`
without a model, so no database, Ollama, or Pi is needed.
"""

import json
from collections.abc import Callable

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.runnables import RunnableLambda
from pydantic import SecretStr

from profile_rag import rag
from profile_rag.api.access import QuestionLimiter, RateLimited
from profile_rag.api.app import create_app
from profile_rag.api.deps import get_chains, get_pool
from profile_rag.generation.chain import no_evidence_answer
from profile_rag.settings import get_settings

HOUR = 3600.0
DAY = 86400.0
QUESTION = {"question": "Who is Dana?"}


class Clock:
    def __init__(self, now: float = 10 * DAY) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_one_client_is_refused_after_its_hourly_limit() -> None:
    limiter = QuestionLimiter(per_client_per_hour=2, per_day=None, clock=Clock())
    limiter.take("a")
    limiter.take("a")
    with pytest.raises(RateLimited):
        limiter.take("a")
    limiter.take("b")  # another client still has its own allowance


def test_the_hourly_count_starts_again_in_the_next_hour() -> None:
    clock = Clock()
    limiter = QuestionLimiter(per_client_per_hour=1, per_day=None, clock=clock)
    limiter.take("a")
    clock.now += HOUR
    limiter.take("a")


def test_all_clients_share_the_daily_limit_until_the_next_utc_day() -> None:
    clock = Clock()
    limiter = QuestionLimiter(per_client_per_hour=None, per_day=2, clock=clock)
    limiter.take("a")
    limiter.take("b")
    with pytest.raises(RateLimited):
        limiter.take("c")
    clock.now += DAY
    limiter.take("c")


def test_a_refused_question_counts_toward_no_limit() -> None:
    limiter = QuestionLimiter(per_client_per_hour=1, per_day=2, clock=Clock())
    limiter.take("a")
    for _ in range(3):
        with pytest.raises(RateLimited):
            limiter.take("a")
    limiter.take("b")  # the daily count is 2 only after this one


def test_no_limits_never_refuses() -> None:
    limiter = QuestionLimiter(per_client_per_hour=None, per_day=None, clock=Clock())
    for _ in range(1000):
        limiter.take("a")


class UnknownChains:
    """A ChainCache stand-in whose chain answers `unknown` without a model."""

    def get(self, mode: object = None, k: object = None) -> RunnableLambda[object, object]:
        return RunnableLambda(lambda _inputs: no_evidence_answer())


def client(*, pool: Callable[[], object] | None = None, **settings: object) -> TestClient:
    base = get_settings()
    app = create_app()
    app.dependency_overrides[get_chains] = UnknownChains
    if pool is not None:
        app.dependency_overrides[get_pool] = pool
    app.dependency_overrides[get_settings] = lambda: base.model_copy(update=settings)
    return TestClient(app)


def test_without_settings_every_question_is_accepted() -> None:
    api = client(access_key=None, client_questions_per_hour=None, questions_per_day=None)
    for _ in range(5):
        assert api.post("/api/ask", json=QUESTION).status_code == 200


@pytest.mark.parametrize("path", ["/api/ask", "/api/ask/stream"])
def test_with_an_access_key_a_question_needs_the_header(path: str) -> None:
    api = client(access_key=SecretStr("k3y"))

    assert api.post(path, json=QUESTION).status_code == 401
    assert api.post(path, json=QUESTION, headers={"X-Access-Key": "wrong"}).status_code == 401
    assert api.post(path, json=QUESTION, headers={"X-Access-Key": "k3y"}).status_code == 200


class DownPool:
    """A pool whose every connection fails, so health reports the database down."""

    def connection(self, timeout: float) -> object:
        raise psycopg.OperationalError("down")


def test_the_health_check_needs_no_access_key() -> None:
    api = client(access_key=SecretStr("k3y"), questions_per_day=1, pool=DownPool)
    for _ in range(3):  # and is never rate-limited
        assert api.get("/api/health").json()["database"] == "unavailable"


@pytest.mark.parametrize("path", ["/api/ask", "/api/ask/stream"])
def test_a_question_over_the_limit_gets_a_429_failed_response(path: str) -> None:
    api = client(access_key=None, questions_per_day=1)
    assert api.post(path, json=QUESTION).status_code == 200

    refused = api.post(path, json=QUESTION)

    assert refused.status_code == 429
    body = refused.json()
    assert body == {
        "status": "failed",
        "error": "rate_limited",
        "message": rag.FAILURES["rate_limited"],
        "request_id": refused.headers["X-Request-ID"],
    }


def test_the_visitor_header_counts_per_visitor_only_with_the_access_key() -> None:
    keyed = client(access_key=SecretStr("k3y"), client_questions_per_hour=1)
    for visitor in ("198.51.100.1", "198.51.100.2"):
        headers = {"X-Access-Key": "k3y", "X-Visitor-IP": visitor}
        assert keyed.post("/api/ask", json=QUESTION, headers=headers).status_code == 200
    again = {"X-Access-Key": "k3y", "X-Visitor-IP": "198.51.100.1"}
    assert keyed.post("/api/ask", json=QUESTION, headers=again).status_code == 429

    # Without an access key, anyone could send the header: it is ignored, so
    # both "visitors" are the same connection and the second is refused.
    open_api = client(access_key=None, client_questions_per_hour=1)
    first = {"X-Visitor-IP": "198.51.100.1"}
    second = {"X-Visitor-IP": "198.51.100.2"}
    assert open_api.post("/api/ask", json=QUESTION, headers=first).status_code == 200
    assert open_api.post("/api/ask", json=QUESTION, headers=second).status_code == 429


def test_an_accepted_stream_still_ends_with_its_result() -> None:
    api = client(access_key=None, questions_per_day=5)
    response = api.post("/api/ask/stream", json=QUESTION)
    last = json.loads(response.text.strip().splitlines()[-1])
    assert last["type"] == "result"
    assert last["response"]["status"] == "unknown"
