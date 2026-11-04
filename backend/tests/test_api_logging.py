"""Tests for the API's logging (api/log_config.py).

The failure line of rag.py must reach stderr through the app's own handler,
at the level PROFILE_RAG_LOG_LEVEL sets, and never hold the question, the
model's reply, or Pi's stderr. The requests use test_api_contract.py's
fake chain, so no database, Ollama, or Pi is needed.
"""

import logging
import logging.config
from collections.abc import Iterator

import pytest
import uvicorn.config
from pydantic import ValidationError
from test_api_contract import SECRET, model_error, post

from profile_rag.api.log_config import LOGGER_NAME, StderrHandler, configure_logging
from profile_rag.generation.pi_rpc import PiProcessError

QUESTION = "QUESTION-TEXT-91c2 What does Dana maintain?"


@pytest.fixture(autouse=True)
def restore_logger() -> Iterator[None]:
    logger = logging.getLogger(LOGGER_NAME)
    handlers, level, propagate = list(logger.handlers), logger.level, logger.propagate
    yield
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def failing_request() -> str:
    """POST a question that fails in Pi; return the response's request ID."""
    client = model_error(PiProcessError(SECRET, returncode=1, stderr_tail=SECRET))
    response = post(client, QUESTION)
    assert response.status_code == 502
    return response.headers["X-Request-ID"]


def test_the_failure_line_reaches_stderr_without_question_reply_or_pi_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_RAG_LOG_LEVEL", "INFO")
    request_id = failing_request()

    lines = [line for line in capsys.readouterr().err.splitlines() if "profile_rag.rag" in line]
    assert len(lines) == 1, lines
    assert " WARNING profile_rag.rag: ask failed: " in lines[0]
    assert f"request_id={request_id}" in lines[0]
    assert "error=PiProcessError" in lines[0]
    assert "code=pi_failed" in lines[0]
    assert SECRET not in lines[0]
    assert "QUESTION-TEXT" not in lines[0]


def test_the_line_survives_uvicorns_logging_config(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_RAG_LOG_LEVEL", "INFO")
    client = model_error(PiProcessError(SECRET, returncode=1, stderr_tail=SECRET))
    # What `uvicorn profile_rag.api.app:app` applies to the process.
    logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
    response = post(client, QUESTION)

    err = capsys.readouterr().err
    assert f"request_id={response.headers['X-Request-ID']}" in err
    assert SECRET not in err


def test_the_level_comes_from_profile_rag_log_level(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PROFILE_RAG_LOG_LEVEL", "ERROR")
    failing_request()  # model_error() creates the app, which reads the level
    assert "ask failed" not in capsys.readouterr().err
    assert logging.getLogger(LOGGER_NAME).level == logging.ERROR


def test_an_unknown_level_stops_the_app_at_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_RAG_LOG_LEVEL", "LOUD")
    with pytest.raises(ValidationError):
        configure_logging()


def test_configuring_twice_keeps_one_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROFILE_RAG_LOG_LEVEL", "INFO")
    configure_logging()
    logger = configure_logging()
    assert sum(isinstance(h, StderrHandler) for h in logger.handlers) == 1
    assert logger.level == logging.INFO
