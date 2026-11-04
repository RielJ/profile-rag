"""Tests for cli.py: `profile-rag agent` and `profile-rag compare-agent`.

The fixtures come from test_cli_ask.py: fake Ollama, a chunk table built by
`fill_search_table`, and `cli._pi_model` replaced with a FakeListChatModel.
"""

import json
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from test_cli_ask import (  # noqa: F401 (fixtures used by name)
    CHUNKS,
    SECRET,
    fake_ollama,
    replying,
    settings,
    table,
    use_model,
)
from typer.testing import CliRunner, Result

from profile_rag import cli
from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.questions import Label, Question
from profile_rag.vectorstore.models import NOMIC_V15

RELEVANT = json.dumps({"grade": "relevant"})
NOT_RELEVANT = json.dumps({"grade": "not_relevant"})
UNKNOWN = json.dumps({"status": "unknown", "reason": "No salary in the evidence."})
TALK = json.dumps(
    {"status": "answered", "answer": "A talk on autoscaling.", "citations": ["talks:v1:a"]}
)


def run(*args: str) -> Result:
    return CliRunner().invoke(cli.app, ["agent", *args])


def many(monkeypatch: pytest.MonkeyPatch, *replies: str) -> None:
    use_model(monkeypatch, FakeListChatModel(responses=list(replies)))


@pytest.mark.usefixtures("table")
def test_agent_prints_each_node_then_the_answer_and_the_pi_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    many(monkeypatch, RELEVANT, TALK)
    result = run("Kubernetes talk?", "-k", "3")
    assert result.exit_code == 0, result.output
    assert result.output == (
        "[retrieve] 3 chunks\n"
        "[grade] relevant (Pi calls so far: 1)\n"
        "[generate] answered (Pi calls so far: 2)\n"
        "\n"
        "A talk on autoscaling.\n"
        "\nCitations:\n"
        "  talks:v1:a  Talks (self_authored)\n"
        "\nPi calls: 2, rewrites: 0\n"
    )


@pytest.mark.usefixtures("table")
def test_agent_rewrites_once_and_prints_no_question_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    many(monkeypatch, NOT_RELEVANT, SECRET, NOT_RELEVANT, UNKNOWN)
    result = run("What is Dana's salary?")
    assert result.exit_code == 0, result.output
    assert "[rewrite] question rewritten (Pi calls so far: 2)\n" in result.output
    assert result.output.count("[grade] not_relevant") == 2
    assert "Unknown.\nReason: No salary in the evidence." in result.output
    assert result.output.endswith("Pi calls: 4, rewrites: 1\n")
    assert SECRET not in result.output
    assert "salary?" not in result.output


@pytest.mark.usefixtures("table")
def test_agent_exits_4_on_a_grader_reply_of_the_wrong_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replying(monkeypatch, f"maybe {SECRET}")
    result = run("Who is Dana?")
    assert result.exit_code == 4
    assert "the grader's reply is not JSON" in result.output
    assert SECRET not in result.output


@pytest.mark.usefixtures("table")
def test_compare_agent_writes_the_detail_and_the_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    question = Question(
        id="ex-01", question=SECRET, expected=(Label(source_id="talks", span="talk"),)
    )
    resolved = [ResolvedQuestion(question, (frozenset({"talks:v1:a"}),))]
    current = cli.get_settings()

    def inputs(eval_set: str, model: object) -> tuple[list[ResolvedQuestion], int, str, str]:
        return resolved, len(CHUNKS), str(current.database_url), current.ollama_url

    monkeypatch.setattr(cli, "_eval_inputs", inputs)
    monkeypatch.setattr(cli, "EVAL_RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(cli, "EVAL_DOCS_DIR", tmp_path / "docs")
    many(monkeypatch, TALK, RELEVANT, TALK)

    result = CliRunner().invoke(
        cli.app, ["compare-agent", "--set", "example", "-k", "3", "--max-pi-calls", "5"]
    )

    assert result.exit_code == 0, result.output
    assert "ex-01: chain answered (1), agent answered (2, rewrites 0)" in result.output
    assert "chain: answered 1, unknown 0, rejected 0, Pi calls 1" in result.output
    assert "agent: answered 1, unknown 0, rejected 0, Pi calls 2" in result.output
    report = tmp_path / "docs" / f"agent-vs-chain-example-{NOMIC_V15.key}.md"
    detail = tmp_path / "results" / f"agent-vs-chain-example-{NOMIC_V15.key}.jsonl"
    assert "| answers citing a labeled chunk | 1 | 1 |" in report.read_text()
    for path in (report, detail):
        assert SECRET not in path.read_text()
    assert SECRET not in result.output
