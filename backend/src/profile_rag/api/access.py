"""Who may ask, and how often: the guard in front of the two ask routes.

A deployed demo spends a free model quota on every question, so the server
can refuse callers. Every check is off unless its setting is set, so a local
server behaves as before.

- `PROFILE_RAG_ACCESS_KEY`: the ask routes require an `X-Access-Key` header
  with this value, or answer 401. The portfolio's server-side proxy holds the
  key, so visitors reach the API through it only. `GET /api/health` stays
  open, so an uptime monitor can ping it.
- `PROFILE_RAG_CLIENT_QUESTIONS_PER_HOUR`: questions one client may ask per
  clock hour (UTC).
- `PROFILE_RAG_QUESTIONS_PER_DAY`: questions all clients together may ask per
  UTC day. Keep it below the model provider's daily free quota.

A refused question gets HTTP 429 and a `failed` response with the code
`rate_limited`, the same body shape as any other failure, so a client shows
its message.

The client is the `X-Visitor-IP` header, which the proxy sets to the
visitor's address, but only when the request carried the right access key:
anyone else could put any address there. Otherwise it is the address of the
connection.

The counts live in memory: one server process, reset by a restart. That fits
one free container; several replicas would each count on their own.

Tests: backend/tests/test_access.py.
"""

import hmac
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from profile_rag.settings import Settings, get_settings

ACCESS_KEY_HEADER = "X-Access-Key"
VISITOR_IP_HEADER = "X-Visitor-IP"

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


class RateLimited(Exception):
    """A question over one of the limits. api/app.py turns it into a 429 response."""


@dataclass
class QuestionLimiter:
    """Counts questions per client per hour and in total per day, in fixed UTC windows.

    `None` for a limit means no limit. `clock` returns seconds since the
    epoch; tests pass their own.
    """

    per_client_per_hour: int | None
    per_day: int | None
    clock: Callable[[], float] = time.time
    _hour: int = -1
    _day: int = -1
    _client_counts: dict[str, int] = field(default_factory=dict)
    _day_count: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, client: str) -> None:
        """Count one question from `client`. Raises RateLimited, counting nothing, if over."""
        now = self.clock()
        with self._lock:
            hour, day = int(now // SECONDS_PER_HOUR), int(now // SECONDS_PER_DAY)
            if hour != self._hour:
                self._hour = hour
                self._client_counts = {}
            if day != self._day:
                self._day, self._day_count = day, 0
            client_count = self._client_counts.get(client, 0)
            if self.per_client_per_hour is not None and client_count >= self.per_client_per_hour:
                raise RateLimited
            if self.per_day is not None and self._day_count >= self.per_day:
                raise RateLimited
            self._client_counts[client] = client_count + 1
            self._day_count += 1


def limiter_from_settings(settings: Settings) -> QuestionLimiter:
    return QuestionLimiter(
        per_client_per_hour=settings.client_questions_per_hour,
        per_day=settings.questions_per_day,
    )


def client_of(request: Request, *, trusted: bool) -> str:
    """The visitor's address: the proxy's header if `trusted`, else the connection's."""
    if trusted and (visitor := request.headers.get(VISITOR_IP_HEADER, "").strip()):
        return visitor[:100]
    return request.client.host if request.client else "unknown"


def has_access_key(request: Request, settings: Settings) -> bool:
    """True if the access key is set and the request carries it."""
    if settings.access_key is None:
        return False
    given = request.headers.get(ACCESS_KEY_HEADER, "")
    return hmac.compare_digest(given.encode(), settings.access_key.get_secret_value().encode())


def guard_question(request: Request, settings: Annotated[Settings, Depends(get_settings)]) -> None:
    """The dependency of the ask routes: the access key, then the limits."""
    trusted = has_access_key(request, settings)
    if settings.access_key is not None and not trusted:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or wrong access key")
    limiter_of(request, settings).take(client_of(request, trusted=trusted))


def limiter_of(request: Request, settings: Settings) -> QuestionLimiter:
    """The app's limiter, made on the first question.

    Made here rather than in create_app, so importing the app reads no settings.
    """
    state = request.app.state
    with _LIMITER_LOCK:
        if getattr(state, "limiter", None) is None:
            state.limiter = limiter_from_settings(settings)
        limiter: QuestionLimiter = state.limiter
    return limiter


_LIMITER_LOCK = threading.Lock()


QuestionGuard = Annotated[None, Depends(guard_question)]
