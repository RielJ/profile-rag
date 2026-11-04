"""Tests for evaluation/runner.py with a fake retriever. No database, no Ollama."""

import json
from pathlib import Path
from typing import override

import pytest
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.questions import Label, Question, Tag
from profile_rag.evaluation.runner import (
    NOTES_MARKER,
    RunConfig,
    run_evaluation,
    run_questions,
    write_comparison,
)

SECRET_QUESTION = "Which secret project did the person lead?"


class FakeRetriever(BaseRetriever):
    """Returns the chunk IDs listed for each question text, in order."""

    answers: dict[str, list[str]]
    calls: list[str] = Field(default_factory=list[str])

    @override
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        self.calls.append(query)
        return [
            Document(
                page_content=f"text of {cid}",
                metadata={"chunk_id": cid, "source_id": cid.split(":")[0], "heading_path": []},
            )
            for cid in self.answers.get(query, [])
        ]


def resolved(
    qid: str, text: str, relevant: list[set[str]], tags: tuple[Tag, ...] = ()
) -> ResolvedQuestion:
    expected = tuple(Label(source_id="s", span=f"span {i}") for i in range(len(relevant)))
    if not relevant:
        tags = (*tags, "unanswerable")
    question = Question(id=qid, question=text, expected=expected, tags=tags)
    return ResolvedQuestion(question, tuple(frozenset(ids) for ids in relevant))


QUESTIONS = [
    resolved("q-hit", SECRET_QUESTION, [{"s:a"}], ("direct",)),
    resolved("q-second", "Which second fact?", [{"s:b"}], ("paraphrase",)),
    resolved("q-miss", "Which missed fact?", [{"s:c"}], ("paraphrase",)),
    resolved("q-none", "Which absent fact?", []),
]
ANSWERS = {
    SECRET_QUESTION: ["s:a", "s:x"],
    "Which second fact?": ["s:x", "s:b", "s:y"],
    "Which missed fact?": ["s:x", "s:y", "s:c"],  # relevant chunk at rank 3, beyond k=2
    "Which absent fact?": ["s:a"],
}


def config(mode: str = "dense", model: str = "fake", k: int = 2) -> RunConfig:
    return RunConfig(set_name="unit", model=model, mode=mode, k=k, chunks=6)


def test_run_questions_invokes_the_retriever_once_per_question_and_cuts_at_k() -> None:
    retriever = FakeRetriever(answers=ANSWERS, calls=[])
    results = run_questions(retriever, QUESTIONS, k=2)
    assert retriever.calls == [q.question.question for q in QUESTIONS]
    assert [r.chunk_ids for r in results] == [
        ["s:a", "s:x"],
        ["s:x", "s:b"],
        ["s:x", "s:y"],
        ["s:a"],
    ]
    assert [(r.score.recall, r.score.reciprocal_rank) for r in results] == [
        (1.0, 1.0),
        (1.0, 0.5),
        (0.0, 0.0),
        (None, None),
    ]


@pytest.fixture
def output(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "results", tmp_path / "docs"


def test_run_evaluation_writes_detail_report_and_comparison(output: tuple[Path, Path]) -> None:
    results_dir, docs_dir = output
    run = run_evaluation(
        FakeRetriever(answers=ANSWERS, calls=[]),
        QUESTIONS,
        config(),
        results_dir=results_dir,
        docs_dir=docs_dir,
    )
    assert run.summary.n == 3
    assert run.summary.unanswerable == 1
    assert run.summary.recall == pytest.approx(2 / 3)
    assert run.summary.mrr == pytest.approx(0.5)
    assert run.summary.misses == ("q-miss",)

    assert run.detail_path == results_dir / "unit-fake-dense-k2.jsonl"
    detail = [json.loads(line) for line in run.detail_path.read_text().splitlines()]
    assert [d["id"] for d in detail] == ["q-hit", "q-second", "q-miss", "q-none"]
    assert detail[0]["question"] == SECRET_QUESTION
    assert detail[1]["retrieved"][1] == {
        "rank": 2,
        "chunk_id": "s:b",
        "source_id": "s",
        "heading_path": [],
        "relevant": True,
    }
    assert detail[3]["recall"] is None

    assert run.report_path == docs_dir / "unit-fake-dense-k2.md"
    report = run.report_path.read_text()
    assert "| unit | fake | dense | 2 | 0.667 | 0.500 | 3 | 1 | 6 |" in report
    assert "Misses (recall@2 below 1): q-miss" in report
    assert "| paraphrase | 0.500 | 0.250 | 2 | 0 | q-miss |" in report
    assert "| unanswerable | n/a | n/a | 0 | 1 | none |" in report

    assert run.comparison_path == docs_dir / "unit-comparison.md"
    assert "| fake | dense | 2 | 0.667 | 0.500 | 3 | 1 | q-miss |" in (
        run.comparison_path.read_text()
    )


def test_committed_files_hold_no_question_span_or_chunk_text(output: tuple[Path, Path]) -> None:
    results_dir, docs_dir = output
    run_evaluation(
        FakeRetriever(answers=ANSWERS, calls=[]),
        QUESTIONS,
        config(),
        results_dir=results_dir,
        docs_dir=docs_dir,
    )
    for path in docs_dir.iterdir():
        text = path.read_text()
        for question in QUESTIONS:
            assert question.question.question not in text
        assert "span 0" not in text
        assert "text of" not in text


def test_the_comparison_has_one_row_per_run_sorted(output: tuple[Path, Path]) -> None:
    results_dir, docs_dir = output
    for model, mode in [("zeta", "mmr"), ("alpha", "hybrid"), ("alpha", "dense")]:
        run_evaluation(
            FakeRetriever(answers=ANSWERS, calls=[]),
            QUESTIONS,
            config(mode=mode, model=model),
            results_dir=results_dir,
            docs_dir=docs_dir,
        )
    rows = [
        line.split("|")[1:3]
        for line in (docs_dir / "unit-comparison.md").read_text().splitlines()
        if line.startswith("| ") and "---" not in line and "model" not in line
    ]
    assert [[c.strip() for c in row] for row in rows] == [
        ["alpha", "dense"],
        ["alpha", "hybrid"],
        ["zeta", "mmr"],
    ]


def test_a_rerun_replaces_its_row(output: tuple[Path, Path]) -> None:
    results_dir, docs_dir = output
    nothing: dict[str, list[str]] = {}
    for answers in (ANSWERS, nothing):
        run = run_evaluation(
            FakeRetriever(answers=answers, calls=[]),
            QUESTIONS,
            config(),
            results_dir=results_dir,
            docs_dir=docs_dir,
        )
    table = run.comparison_path.read_text()
    assert table.count("| fake | dense | 2 |") == 1
    assert "| fake | dense | 2 | 0.000 | 0.000 | 3 | 1 | q-hit, q-second, q-miss |" in table


def test_notes_below_the_marker_survive_a_rebuild(output: tuple[Path, Path]) -> None:
    results_dir, docs_dir = output
    docs_dir.mkdir(parents=True)
    path = docs_dir / "unit-comparison.md"
    path.write_text(f"old table\n\n{NOTES_MARKER}\n\n## Decision\n\nKeep this.\n")
    run_evaluation(
        FakeRetriever(answers=ANSWERS, calls=[]),
        QUESTIONS,
        config(),
        results_dir=results_dir,
        docs_dir=docs_dir,
    )
    text = path.read_text()
    assert "old table" not in text
    assert text.endswith(f"{NOTES_MARKER}\n## Decision\n\nKeep this.\n")
    assert text.count(NOTES_MARKER) == 1


def test_reports_of_other_sets_and_unreadable_files_are_ignored(tmp_path: Path) -> None:
    (tmp_path / "unit-broken-dense-k8.md").write_text("no data block")
    (tmp_path / "unitx-fake-dense-k8.md").write_text("```json\n{}\n```")
    path = write_comparison("unit", tmp_path)
    assert "| --- |" in path.read_text()
    assert "broken" not in path.read_text()
