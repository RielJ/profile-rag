"""Tests for tracing.py: tracing is off by default and, when on, sends no text.

No network: every test blocks socket connections, and the tests that turn
tracing on give the LangSmith client a session that records each request
instead of sending it. The payload tests run a real LangChain runnable with
a real `LangChainTracer`, then read the JSON bodies the client would send.
"""

import json
import os
import socket
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, override

import langsmith
import pytest
import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.context import _tracing_v2_is_enabled  # pyright: ignore[reportPrivateUsage]
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import run_trees
from langsmith.schemas import LangSmithInfo
from pydantic import SecretStr
from typer.testing import CliRunner

from profile_rag import cli
from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import PiConfig
from profile_rag.settings import TracingSettings
from profile_rag.tracing import (
    DEFAULT_CORPUS,
    EXAMPLE_CORPUS,
    REDACTED,
    TRACING_SWITCHES,
    TracingRefused,
    build_client,
    configure_tracing,
    redact_error,
    scrub_error,
    tracing_enabled,
)

SECRET_QUESTION = "SECRET-QUESTION-about-the-corpus"
SECRET_REPLY = "SECRET-REPLY-quoting-a-chunk"


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a test tried to open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def restore_hide_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    # configure_tracing writes os.environ; monkeypatch puts it back afterwards.
    monkeypatch.delenv("LANGSMITH_HIDE_INPUTS", raising=False)
    monkeypatch.delenv("LANGSMITH_HIDE_OUTPUTS", raising=False)


class RecordingSession(requests.Session):
    """A requests session that records every request and answers 200 `{}`."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, str, bytes]] = []

    @override
    def request(  # pyrefly: ignore[bad-override]
        self, method: str, url: str, *args: Any, **kwargs: Any
    ) -> requests.Response:
        data = kwargs.get("data") or b""
        self.requests.append((method, url, data if isinstance(data, bytes) else str(data).encode()))
        response = requests.Response()
        response.status_code = 200
        response._content = b"{}"  # pyright: ignore[reportPrivateUsage]
        return response

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(body) for _method, _url, body in self.requests if body]

    def text(self) -> str:
        return "\n".join(body.decode() for _method, _url, body in self.requests)


def capturing_client(session: RecordingSession) -> langsmith.Client:
    """The app's client, but sending each run synchronously to `session`."""
    return build_client(
        "not-a-real-key",
        api_url="http://langsmith.invalid",
        session=session,
        auto_batch_tracing=False,
    )


def on(api_key: str | None = "not-a-real-key") -> TracingSettings:
    return TracingSettings(
        langsmith_tracing=True,
        langsmith_api_key=SecretStr(api_key) if api_key else None,
    )


def off() -> TracingSettings:
    return TracingSettings(langsmith_tracing=False)


# --- Off ------------------------------------------------------------------------


def test_tracing_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in TRACING_SWITCHES:
        monkeypatch.delenv(name, raising=False)
    settings = TracingSettings.model_validate({})

    assert settings.langsmith_tracing is False
    assert configure_tracing(settings, DEFAULT_CORPUS) is False
    assert not tracing_enabled()


def test_the_test_run_has_tracing_off_whatever_the_parent_env_and_dotenv_say() -> None:
    # conftest.py sets LANGSMITH_TRACING=false, which beats .env, and drops the key.
    assert TracingSettings().langsmith_tracing is False
    assert TracingSettings().langsmith_api_key is None
    assert not tracing_enabled()
    assert not _tracing_v2_is_enabled()


@pytest.mark.parametrize("switch", ["LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2"])
def test_only_langsmith_tracing_turns_tracing_on(
    monkeypatch: pytest.MonkeyPatch, switch: str
) -> None:
    monkeypatch.setenv(switch, "true")
    langsmith.configure(enabled=None)  # back to reading the environment
    assert tracing_enabled(), "the library alone would trace"

    assert configure_tracing(off(), DEFAULT_CORPUS) is False

    assert not tracing_enabled()
    assert not _tracing_v2_is_enabled()


# --- On -------------------------------------------------------------------------


def test_on_sets_the_hide_variables_and_installs_the_scrubbing_client() -> None:
    assert configure_tracing(on(), EXAMPLE_CORPUS) is True

    assert os.environ["LANGSMITH_HIDE_INPUTS"] == "true"
    assert os.environ["LANGSMITH_HIDE_OUTPUTS"] == "true"
    assert tracing_enabled()
    client = run_trees.get_cached_client()
    assert client._hide_inputs is True  # pyright: ignore[reportPrivateUsage]
    assert client._hide_outputs is True  # pyright: ignore[reportPrivateUsage]
    assert client._anonymizer is scrub_error  # pyright: ignore[reportPrivateUsage]


def test_on_refuses_a_corpus_other_than_corpus_example() -> None:
    with pytest.raises(TracingRefused, match="corpus-example"):
        configure_tracing(on(), DEFAULT_CORPUS)

    assert "LANGSMITH_HIDE_INPUTS" not in os.environ
    assert not tracing_enabled()


def test_on_refuses_a_directory_named_corpus_example_elsewhere(tmp_path: Path) -> None:
    elsewhere = tmp_path / "corpus-example"
    elsewhere.mkdir()
    with pytest.raises(TracingRefused):
        configure_tracing(on(), elsewhere)


def test_on_refuses_without_an_api_key() -> None:
    with pytest.raises(TracingRefused, match="LANGSMITH_API_KEY"):
        configure_tracing(on(api_key=None), EXAMPLE_CORPUS)
    assert not tracing_enabled()


def test_the_cli_exits_nonzero_when_tracing_is_on_outside_corpus_example() -> None:
    result = CliRunner().invoke(
        cli.app,
        ["search", "anything"],
        env={"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "k", "CORPUS_DIR": ""},
    )

    assert result.exit_code == 2
    assert "corpus-example" in result.output
    assert not tracing_enabled()


# --- What a trace would send -----------------------------------------------------


def _fail_with_reply(_inputs: object) -> str:
    raise OutputParserException(f"Invalid json output: {SECRET_REPLY}", llm_output=SECRET_REPLY)


def _runs(session: RecordingSession, run: Callable[[], object]) -> list[dict[str, Any]]:
    with suppress(OutputParserException):
        run()
    wait_for_all_tracers()
    return session.bodies()


def _parse_failure() -> object:
    chain = RunnableLambda(_fail_with_reply).with_config(run_name="parse")
    return chain.invoke({"question": SECRET_QUESTION})


class SecretModel(SimpleChatModel):
    """A chat model that replies SECRET_REPLY.

    Unlike FakeListChatModel it keeps the reply out of its fields, as
    PiChatModel does: langsmith sends an llm run's `serialized` repr and
    invocation parameters, which list a model's fields.
    """

    @property
    @override
    def _llm_type(self) -> str:
        return "secret-model"

    @override
    def _call(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return SECRET_REPLY


def _model_answer() -> object:
    prompt = ChatPromptTemplate.from_messages([("human", "{question}")])
    chain = prompt | SecretModel() | StrOutputParser()
    return chain.invoke({"question": SECRET_QUESTION})


def test_a_failed_run_sends_no_question_reply_or_error_text() -> None:
    session = RecordingSession()
    configure_tracing(on(), EXAMPLE_CORPUS, client=capturing_client(session))

    bodies = _runs(session, _parse_failure)

    assert bodies, "the tracer sent nothing, so the test checks nothing"
    assert SECRET_QUESTION not in session.text()
    assert SECRET_REPLY not in session.text()
    errors = [body["error"] for body in bodies if body.get("error")]
    assert errors == [f"OutputParserException: {REDACTED}"]
    assert all(body.get("inputs") in (None, {}) for body in bodies)
    assert all(body.get("outputs") in (None, {}) for body in bodies)


def test_a_model_run_sends_no_question_or_reply() -> None:
    session = RecordingSession()
    configure_tracing(on(), EXAMPLE_CORPUS, client=capturing_client(session))

    bodies = _runs(session, _model_answer)

    names = {body.get("name") for body in bodies}
    assert "SecretModel" in names, "the model run was not traced"
    assert SECRET_QUESTION not in session.text()
    assert SECRET_REPLY not in session.text()


def test_hide_outputs_alone_does_not_hide_the_error_field() -> None:
    # The finding that build_client's anonymizer answers: LANGSMITH_HIDE_INPUTS
    # and LANGSMITH_HIDE_OUTPUTS leave the error text, and the reply in it, as is.
    session = RecordingSession()
    client = langsmith.Client(
        api_key="not-a-real-key",
        api_url="http://langsmith.invalid",
        session=session,
        auto_batch_tracing=False,
        info=LangSmithInfo(),
        hide_inputs=True,
        hide_outputs=True,
    )
    configure_tracing(on(), EXAMPLE_CORPUS, client=client)

    _runs(session, _parse_failure)

    assert SECRET_QUESTION not in session.text()
    assert SECRET_REPLY in session.text()


def test_pi_chat_model_parameters_hold_only_its_config() -> None:
    # What langsmith sends for an llm run besides inputs and outputs.
    model = PiChatModel(config=PiConfig(model="provider/model"))
    assert set(model._identifying_params) == {"model", "thinking"}  # pyright: ignore[reportPrivateUsage]
    assert set(PiChatModel.model_fields) - set(BaseChatModel.model_fields) == {"config"}


def test_redact_error_keeps_only_the_exception_class_name() -> None:
    text = f"OutputParserException('Invalid json output: {SECRET_REPLY}')\n\nTraceback ..."
    assert redact_error(text) == f"OutputParserException: {REDACTED}"
    assert redact_error(f"no class name {SECRET_REPLY}") == REDACTED
    assert scrub_error({"model": "m"}) == {"model": "m"}
    assert scrub_error({"error": 3}) == {"error": REDACTED}
