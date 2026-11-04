"""Logging for the API: the `profile_rag` loggers write to stderr.

uvicorn configures only its own loggers (`uvicorn`, `uvicorn.error`,
`uvicorn.access`). Without this module, a `profile_rag` log line reaches
stderr only through Python's last-resort handler, which drops everything
below WARNING and has no format.

`configure_logging()` gives the `profile_rag` logger one handler that
writes to the current `sys.stderr`, at the level that
`PROFILE_RAG_LOG_LEVEL` sets (default INFO). `create_app()` calls it, so
`uvicorn profile_rag.api.app:app` logs without extra flags. Calling it again
replaces the handler, so repeated app creation (tests) adds no duplicate
lines. Other libraries' loggers are left as they are, so their INFO lines
(for example, httpx's request lines) are not printed.

What is logged: the API code logs request IDs, exception class names, and
error codes. It never logs the question, the model's reply, or Pi's stderr
(rag.py and api/routes.py).

Tests: backend/tests/test_api_logging.py.
"""

import logging
import sys
from typing import Literal, override

from pydantic import Field
from pydantic_settings import BaseSettings

from profile_rag.settings import CONFIG

LOGGER_NAME = "profile_rag"
FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LogSettings(BaseSettings):
    """The log level. Separate from `Settings`, so it needs no DATABASE_URL."""

    model_config = CONFIG

    log_level: LogLevel = Field(default="INFO", validation_alias="PROFILE_RAG_LOG_LEVEL")


class StderrHandler(logging.Handler):
    """Writes each record to the `sys.stderr` of the moment it is emitted.

    `logging.StreamHandler` keeps the stream it was given at creation. When
    something replaces `sys.stderr` later (pytest's capture does), it would
    keep writing to the old, possibly closed, stream.
    """

    @override
    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


def configure_logging(settings: LogSettings | None = None) -> logging.Logger:
    """Send `profile_rag` log records at the configured level or above to stderr."""
    level = (settings or LogSettings()).log_level
    logger = logging.getLogger(LOGGER_NAME)
    for handler in [h for h in logger.handlers if isinstance(h, StderrHandler)]:
        logger.removeHandler(handler)
    handler = StderrHandler()
    handler.setFormatter(logging.Formatter(FORMAT))
    logger.addHandler(handler)
    logger.setLevel(level)
    # Records still reach the root logger's handlers (pytest's caplog is one).
    # uvicorn adds none there, so a line is not printed twice.
    logger.propagate = True
    return logger
