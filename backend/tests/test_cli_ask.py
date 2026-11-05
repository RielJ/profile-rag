"""Tests for cli.py: `profile-rag ask`.

No Ollama and no Pi. As in test_cli_search.py, `ollama_embeddings.ollama_digests` and
`ollama_embeddings.for_model` are fakes and the `fill_search_table` fixture builds the
chunk table. `cli._pi_model` is replaced with a FakeListChatModel, or with a
PiChatModel that runs the fake Pi in fake_pi.py.
"""

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import get_args

import psycopg
import pytest
import sqlalchemy.exc
from fake_pi import FAKE_PI, accepted, write_script
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import RunnableLambda
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.api.models import FailedCode
from profile_rag.generation.pi_chat_model import PiChatModel
from profile_rag.generation.pi_rpc import PiConfig, PiError
from profile_rag.settings import Settings, get_settings
from profile_rag.vectorstore import ollama_embeddings
from profile_rag.vectorstore.models import NOMIC_V15, EmbeddingModel

FillSearchTable = Callable[[str, EmbeddingModel, Embeddings, list[Document]], None]

# Never printed: an error message must not contain the model's reply.
SECRET = "REPLY-TEXT-THAT-MUST-NOT-BE-PRINTED"


def chunk(chunk_id: str, text: str, title: str, attribution: str) -> Document:
    metadata: dict[str, object] = {
        "source_id": chunk_id.split(":")[0],
        "chunk_id": chunk_id,
        "attribution": attribution,
        "visibility": "public",
        "title": title,
        "heading_path": [],
        "ordinal": 0,
    }
    return Document(page_content=text, metadata=metadata)


CHUNKS = [
    chunk("resume:v1:a", "Dana Whitlock is a backend engineer.", "Resume", "self_report"),
    chunk("talks:v1:a", "A talk on Kubernetes autoscaling.", "Talks", "self_authored"),
    chunk("lantern:v1:a", "Install lantern-queue with pip.", "lantern-queue", "project_doc"),
]


@pytest.fixture
def settings(empty_database_url: str, monkeypatch: pytest.MonkeyPatch) -> Settings:
    current = get_settings().model_copy(
        update={
            "database_url": empty_database_url,
            "ollama_url": "http://ollama.test",
            "pi_timeout_seconds": 10.0,
        }
    )
    monkeypatch.setattr(cli, "get_settings", lambda: current)
    return current


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch: pytest.MonkeyPatch) -> None:
    def make(model: EmbeddingModel, base_url: str) -> Embeddings:
        return DeterministicFakeEmbedding(size=model.dimensions)

    monkeypatch.setattr(ollama_embeddings, "for_model", make)

    def digests(base_url: str) -> dict[str, str]:
        return {NOMIC_V15.ollama_tag: NOMIC_V15.digest}

    monkeypatch.setattr(ollama_embeddings, "ollama_digests", digests)


@pytest.fixture
def table(empty_database: str, settings: Settings, fill_search_table: FillSearchTable) -> None:
    embeddings = DeterministicFakeEmbedding(size=NOMIC_V15.dimensions)
    fill_search_table(empty_database, NOMIC_V15, embeddings, CHUNKS)


def use_model(monkeypatch: pytest.MonkeyPatch, model: BaseChatModel) -> None:
    def make(settings: Settings) -> BaseChatModel:
        return model

    monkeypatch.setattr(cli, "_pi_model", make)


def replying(monkeypatch: pytest.MonkeyPatch, reply: str) -> FakeListChatModel:
    model = FakeListChatModel(responses=[reply])
    use_model(monkeypatch, model)
    return model


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["ask", *args])


@pytest.mark.usefixtures("table")
def test_an_answer_is_printed_with_each_citation_title_and_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reply = {
        "status": "answered",
        "answer": "Dana gave a talk on Kubernetes autoscaling.",
        "citations": ["talks:v1:a"],
    }
    replying(monkeypatch, json.dumps(reply))
    result = run("Kubernetes", "--mode", "hybrid", "-k", "3")
    assert result.exit_code == 0, result.output
    assert result.output == (
        "Dana gave a talk on Kubernetes autoscaling.\n"
        "\nCitations:\n"
        "  talks:v1:a  Talks (self_authored)\n"
    )


@pytest.mark.usefixtures("table")
def test_unknown_is_printed_with_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    replying(monkeypatch, '{"status": "unknown", "reason": "No salary in the evidence."}')
    result = run("What is Dana's salary?")
    assert result.exit_code == 0, result.output
    assert result.output == ("Unknown.\nReason: No salary in the evidence.\n(3 chunks retrieved)\n")


@pytest.mark.usefixtures("table")
def test_an_invented_citation_exits_4(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = {"status": "answered", "answer": SECRET, "citations": ["made-up:v1:z"]}
    replying(monkeypatch, json.dumps(reply))
    result = run("Who is Dana?")
    assert result.exit_code == 4
    assert "the model cited chunks it was not given: made-up:v1:z." in result.output
    assert SECRET not in result.output


@pytest.mark.usefixtures("table")
def test_a_reply_that_is_not_an_answer_exits_4_without_printing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replying(monkeypatch, f"Sure! {SECRET}")
    result = run("Who is Dana?")
    assert result.exit_code == 4
    assert "the model's reply is not a valid answer" in result.output
    assert SECRET not in result.output


@pytest.mark.usefixtures("table")
def test_a_pi_crash_exits_3_with_the_stderr_tail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    script = write_script(
        tmp_path / "s.json", [accepted(), {"stderr": "No API key for anthropic"}, {"exit": 1}]
    )
    executable = (sys.executable, str(FAKE_PI), str(script))
    use_model(monkeypatch, PiChatModel(config=PiConfig(model="m", executable=executable)))
    result = run("Who is Dana?")
    assert result.exit_code == 3
    assert "exited with code 1" in result.output
    assert "  No API key for anthropic" in result.output


@pytest.mark.usefixtures("table")
def test_a_missing_pi_binary_exits_3_and_names_pi_bin(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = settings.model_copy(update={"pi_bin": str(tmp_path / "no-pi")})
    monkeypatch.setattr(cli, "get_settings", lambda: missing)
    result = run("Who is Dana?")
    assert result.exit_code == 3
    assert "set PROFILE_RAG_PI_BIN in .env" in result.output


@pytest.mark.usefixtures("settings")
def test_a_missing_table_exits_2_before_any_model_call(monkeypatch: pytest.MonkeyPatch) -> None:
    model = replying(monkeypatch, "unused")
    result = run("Who is Dana?")
    assert result.exit_code == 2
    assert "no table chunks_nomic_v15" in result.output
    assert model.i == 0


def test_an_empty_question_exits_2() -> None:
    result = run("  ")
    assert result.exit_code == 2
    assert "the question is empty" in result.output


def chain_raising(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
    """Replace the chain with one that raises `error` when it is invoked."""

    def fail(_inputs: dict[str, str]) -> None:
        raise error

    def build(_retriever: object, _model: object) -> RunnableLambda[dict[str, str], None]:
        return RunnableLambda(fail)

    monkeypatch.setattr(cli, "build_chain", build)


def test_every_chain_failure_but_internal_error_has_an_exit_code() -> None:
    """A new FailedCode must get an exit code, the way the API gives it a status."""
    assert set(cli.ANSWER_EXIT) == set(get_args(FailedCode)) - {"internal_error"}


@pytest.mark.usefixtures("table")
def test_a_database_error_while_answering_exits_2_with_a_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replying(monkeypatch, "unused")
    chain_raising(monkeypatch, psycopg.OperationalError("server closed the connection"))
    result = run("Who is Dana?")
    assert result.exit_code == 2
    assert "the evidence search failed (OperationalError: server closed" in result.output


@pytest.mark.usefixtures("table")
def test_a_failed_search_prints_neither_the_sql_nor_its_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SQLAlchemy puts the statement and its parameters on the lines after the first."""
    replying(monkeypatch, "unused")
    error = sqlalchemy.exc.OperationalError(
        "SELECT content FROM chunks_nomic_v15 WHERE content @@ %(q)s",
        {"q": SECRET},
        psycopg.OperationalError("server closed the connection unexpectedly"),
    )
    chain_raising(monkeypatch, error)
    result = run("Who is Dana?")
    assert result.exit_code == 2
    assert "server closed the connection unexpectedly" in result.output
    assert "SELECT" not in result.output
    assert SECRET not in result.output


@pytest.mark.usefixtures("table")
def test_a_new_kind_of_pi_error_exits_3_without_a_cli_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PiProtocolError(PiError):
        pass

    replying(monkeypatch, "unused")
    chain_raising(monkeypatch, PiProtocolError("Pi sent an event it does not document"))
    result = run("Who is Dana?")
    assert result.exit_code == 3
    assert "Pi sent an event it does not document" in result.output


@pytest.mark.usefixtures("table")
def test_a_bug_in_the_chain_is_raised_not_turned_into_an_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replying(monkeypatch, "unused")
    chain_raising(monkeypatch, KeyError("answer"))
    result = run("Who is Dana?")
    assert isinstance(result.exception, KeyError)
    assert result.exit_code == 1
