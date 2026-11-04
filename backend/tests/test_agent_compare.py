"""Tests for evaluation/agent_compare.py.

A fake retriever returns fixed chunks per question and a FakeListChatModel
plays Pi, so the chain and the graph run for real without a database or Pi.
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable, RunnableLambda

from profile_rag.agent.graph import AgentGraph, build_graph
from profile_rag.evaluation.agent_compare import (
    CompareConfig,
    compare,
    run_chain,
    run_graph,
    write_outputs,
)
from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.questions import Label, Question
from profile_rag.evaluation.runner import NOTES_MARKER
from profile_rag.generation.chain import GroundedAnswer, build_chain

RELEVANT = json.dumps({"grade": "relevant"})
NOT_RELEVANT = json.dumps({"grade": "not_relevant"})
UNKNOWN = json.dumps({"status": "unknown", "reason": "no"})
SECRET = "QUESTION-TEXT-THAT-MUST-NOT-BE-WRITTEN"


def answered(*citations: str) -> str:
    return json.dumps({"status": "answered", "answer": "a", "citations": list(citations)})


def doc(chunk_id: str) -> Document:
    metadata = {"chunk_id": chunk_id, "title": "T", "attribution": "project_doc"}
    return Document(page_content=f"text {chunk_id}", metadata=metadata)


def resolved(qid: str, relevant: set[str] | None) -> ResolvedQuestion:
    expected = (Label(source_id="s", span="span"),) if relevant else ()
    tags = () if relevant else ("unanswerable",)
    question = Question(id=qid, question=f"{SECRET} {qid}", expected=expected, tags=tags)
    return ResolvedQuestion(question, (frozenset(relevant),) if relevant else ())


def pipelines(
    replies: list[str],
) -> tuple[Runnable[dict[str, str], GroundedAnswer], AgentGraph]:
    def found(_question: str) -> list[Document]:
        return [doc("a:1"), doc("b:1")]

    model = FakeListChatModel(responses=replies)
    retriever = RunnableLambda(found)
    return build_chain(retriever, model), build_graph(retriever, model)


def test_run_chain_counts_one_pi_call_and_records_citations() -> None:
    chain, _graph = pipelines([answered("a:1")])
    run = run_chain(chain, "q")
    assert (run.outcome, run.citations, run.pi_calls) == ("answered", ("a:1",), 1)


def test_run_graph_counts_four_calls_and_one_rewrite() -> None:
    _chain, graph = pipelines([NOT_RELEVANT, "new q", RELEVANT, UNKNOWN])
    run = run_graph(graph, "q")
    assert (run.outcome, run.pi_calls, run.rewrites) == ("unknown", 4, 1)
    assert run.path == ("retrieve", "grade", "rewrite", "retrieve", "grade", "generate")


def test_a_rejected_reply_is_an_outcome_with_its_calls_counted() -> None:
    _chain, graph = pipelines([RELEVANT, answered("invented:9")])
    run = run_graph(graph, "q")
    assert (run.outcome, run.pi_calls, run.rejected_by) == ("rejected", 2, "InventedCitation")
    assert run.path == ("retrieve", "grade")

    chain, _graph = pipelines(["not json"])
    run = run_chain(chain, "q")
    assert (run.outcome, run.pi_calls, run.rejected_by) == (
        "rejected",
        1,
        "OutputParserException",
    )


def test_compare_scores_both_pipelines_against_the_labels(tmp_path: Path) -> None:
    questions = [resolved("q1", {"a:1"}), resolved("q2", None)]
    replies = [
        answered("a:1", "b:1"),  # chain q1: one labeled citation of two
        RELEVANT,
        answered("a:1"),  # agent q1
        answered("b:1"),  # chain q2: answers an unanswerable question
        NOT_RELEVANT,
        "q2 again",
        NOT_RELEVANT,
        UNKNOWN,  # agent q2
    ]
    chain, graph = pipelines(replies)

    output = compare(questions, chain, graph)

    assert output.chain.answered == 2
    assert output.chain.answered_unanswerable == 1
    assert output.chain.correct_unknown == 0
    assert output.chain.labeled_citation == 1
    assert output.chain.citation_precision == 0.5
    assert output.chain.pi_calls_total == 2
    assert output.agent.answered == 1
    assert output.agent.correct_unknown == 1
    assert output.agent.citation_precision == 1.0
    assert output.agent.pi_calls_total == 6
    assert output.agent.pi_calls_mean == 3.0
    assert output.agent.rewritten == 1
    assert output.agent.not_answered == []


def test_the_pi_call_cap_skips_questions_whose_worst_case_does_not_fit() -> None:
    questions = [resolved("q1", {"a:1"}), resolved("q2", {"a:1"})]
    chain, graph = pipelines([answered("a:1"), RELEVANT, answered("a:1")])

    # q1 fits (worst case 5); after it 3 are spent, and 3 + 5 > 7.
    output = compare(questions, chain, graph, max_pi_calls=7)

    assert [r.resolved.question.id for r in output.results] == ["q1"]
    assert output.skipped == ["q2"]


def test_outputs_hold_no_question_text_and_keep_notes(tmp_path: Path) -> None:
    questions = [resolved("q1", {"a:1"})]
    chain, graph = pipelines([answered("a:1"), RELEVANT, answered("a:1")])
    output = compare(questions, chain, graph)
    config = CompareConfig("example", "nomic-v15", "hybrid", 3, "provider/model")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / f"{config.name}.md").write_text(f"old\n{NOTES_MARKER}\n\nMy decision.\n")

    detail, report = write_outputs(config, output, results_dir=tmp_path / "eval", docs_dir=docs)

    for path in (detail, report):
        assert SECRET not in path.read_text()
    text = report.read_text()
    assert text.endswith("My decision.\n")
    assert "| answered | 1 | 1 |" in text
    assert "- q1: retrieve > grade > generate (2 Pi calls)" in text
    line = json.loads(detail.read_text())
    assert line["id"] == "q1"
    assert line["agent"]["path"] == ["retrieve", "grade", "generate"]
