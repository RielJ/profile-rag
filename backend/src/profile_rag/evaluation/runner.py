"""Run a question set through a retriever and write the results.

The runner accepts any LangChain `BaseRetriever`: it calls
`retriever.invoke(question)` and reads `metadata["chunk_id"]` from each
returned Document. It writes two kinds of output:

- Detail, one JSON line per question, to `eval/results/<run>.jsonl`. It has
  the question text and the retrieved chunk IDs, so it stays private
  (gitignored).
- An aggregate report to `docs/eval/<run>.md`, and a comparison table of all
  runs of the same set to `docs/eval/<set>-comparison.md`. Both are
  committed, so they hold only numbers, model, mode, k, sample sizes, tags,
  and question IDs. Never question, span, or chunk text.

`<run>` is `<set>-<model>-<mode>-k<k>`, plus `-ef<N>` for a run with
`--ef-search N`. Each report ends with a JSON block of
its numbers; the comparison table is rebuilt from those blocks. Text below
the `NOTES_MARKER` line of a comparison file is kept when the table is
rebuilt, so written conclusions survive later runs.

Tests: backend/tests/test_runner.py.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ValidationError

from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.metrics import Score, Summary, score, summarise
from profile_rag.evaluation.questions import TAGS

NOTES_MARKER = "<!-- Notes below this line are kept when the table is rebuilt. -->"
_SUMMARY_BLOCK = re.compile(r"```json\n(?P<body>.*?)\n```", re.DOTALL)


@dataclass(frozen=True)
class RunConfig:
    """What a run measured. `chunks` is the number of chunks in the set's corpus."""

    set_name: str
    model: str
    mode: str
    k: int
    chunks: int
    # `hnsw.ef_search` set for the dense queries; None keeps the server's.
    ef_search: int | None = None

    @property
    def name(self) -> str:
        suffix = "" if self.ef_search is None else f"-ef{self.ef_search}"
        return f"{self.set_name}-{self.model}-{self.mode}-k{self.k}{suffix}"


@dataclass(frozen=True)
class QuestionResult:
    """One question, the top-k Documents the retriever returned, and its score."""

    resolved: ResolvedQuestion
    retrieved: list[Document]
    score: Score

    @property
    def chunk_ids(self) -> list[str]:
        return [doc.metadata["chunk_id"] for doc in self.retrieved]


@dataclass(frozen=True)
class RunOutput:
    summary: Summary
    results: list[QuestionResult]
    detail_path: Path
    report_path: Path
    comparison_path: Path


def run_questions(
    retriever: BaseRetriever, questions: Sequence[ResolvedQuestion], k: int
) -> list[QuestionResult]:
    """Retrieve for each question and score the first `k` results, in question order."""
    results: list[QuestionResult] = []
    for resolved in questions:
        retrieved = retriever.invoke(resolved.question.question)[:k]
        ids = [doc.metadata["chunk_id"] for doc in retrieved]
        scored = score(resolved.question.id, ids, resolved.relevant, k)
        results.append(QuestionResult(resolved, retrieved, scored))
    return results


def _tag_summaries(results: Sequence[QuestionResult]) -> dict[str, Summary]:
    by_tag: dict[str, Summary] = {}
    for tag in (*TAGS, "untagged"):
        scores = [r.score for r in results if tag in (r.resolved.question.tags or ("untagged",))]
        if scores:
            by_tag[tag] = summarise(scores)
    return by_tag


def _detail_line(result: QuestionResult) -> str:
    question = result.resolved.question
    wanted = frozenset[str]().union(*result.resolved.relevant)
    record = {
        "id": question.id,
        "question": question.question,
        "tags": list(question.tags),
        "expected_sources": [label.source_id for label in question.expected],
        "relevant": [sorted(ids) for ids in result.resolved.relevant],
        "recall": result.score.recall,
        "reciprocal_rank": result.score.reciprocal_rank,
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": doc.metadata["chunk_id"],
                "source_id": doc.metadata.get("source_id"),
                "heading_path": doc.metadata.get("heading_path"),
                "relevant": doc.metadata["chunk_id"] in wanted,
            }
            for rank, doc in enumerate(result.retrieved, start=1)
        ],
    }
    return json.dumps(record, ensure_ascii=False)


def write_detail(results: Sequence[QuestionResult], config: RunConfig, results_dir: Path) -> Path:
    """Write one JSON line per question to `results_dir/<run>.jsonl`."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{config.name}.jsonl"
    path.write_text("".join(_detail_line(r) + "\n" for r in results), encoding="utf-8")
    return path


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _misses(summary: Summary) -> str:
    return ", ".join(summary.misses) or "none"


def _mode(config: RunConfig) -> str:
    """The mode, with `ef_search` if the run set it: `dense (ef_search 40)`."""
    if config.ef_search is None:
        return config.mode
    return f"{config.mode} (ef_search {config.ef_search})"


def render_report(config: RunConfig, summary: Summary, by_tag: dict[str, Summary]) -> str:
    """The Markdown report of one run. Holds no question or corpus text."""
    lines = [
        f"# Evaluation: {config.name}",
        "",
        "Generated by `profile-rag eval`. Do not edit by hand; rerun the command.",
        "",
        "| set | model | mode | k | recall@k | MRR | n | unanswerable | corpus chunks |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| {config.set_name} | {config.model} | {_mode(config)} | {config.k}"
        f" | {_number(summary.recall)} | {_number(summary.mrr)} | {summary.n}"
        f" | {summary.unanswerable} | {config.chunks} |",
        "",
        f"Misses (recall@{config.k} below 1): {_misses(summary)}",
        "",
        "## By tag",
        "",
        "A question with several tags counts under each of them.",
        "",
        "| tag | recall@k | MRR | n | unanswerable | misses |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for tag, tagged in by_tag.items():
        lines.append(
            f"| {tag} | {_number(tagged.recall)} | {_number(tagged.mrr)} | {tagged.n}"
            f" | {tagged.unanswerable} | {_misses(tagged)} |"
        )
    # Reports of runs without --ef-search keep the shape they had before it.
    settings = {
        key: value
        for key, value in asdict(config).items()
        if key != "ef_search" or value is not None
    }
    data = {"config": settings, "summary": asdict(summary)}
    lines += ["", "## Data", "", "```json", json.dumps(data, indent=2), "```", ""]
    return "\n".join(lines)


def write_report(
    config: RunConfig, summary: Summary, by_tag: dict[str, Summary], docs_dir: Path
) -> Path:
    """Write `docs_dir/<run>.md`."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / f"{config.name}.md"
    path.write_text(render_report(config, summary, by_tag), encoding="utf-8")
    return path


class _ReportData(BaseModel):
    """The JSON block at the end of a report."""

    config: RunConfig
    summary: Summary


def _read_report(path: Path) -> tuple[RunConfig, Summary] | None:
    """The config and summary in a report's JSON block, or None if it has no valid block."""
    match = _SUMMARY_BLOCK.search(path.read_text(encoding="utf-8"))
    if match is None:
        return None
    try:
        data = _ReportData.model_validate_json(match["body"])
    except ValidationError:
        return None
    return data.config, data.summary


def write_comparison(set_name: str, docs_dir: Path) -> Path:
    """Rebuild `docs_dir/<set>-comparison.md` from every report of the set.

    Rows are sorted by model, mode, k, and ef_search. Text below `NOTES_MARKER` in an
    existing file is kept.
    """
    runs: list[tuple[RunConfig, Summary]] = []
    for path in sorted(docs_dir.glob(f"{set_name}-*-k*.md")):
        found = _read_report(path)
        if found is not None and found[0].set_name == set_name:
            runs.append(found)
    runs.sort(key=lambda run: (run[0].model, run[0].mode, run[0].k, run[0].ef_search or 0))

    lines = [
        f"# Retrieval comparison: {set_name} set",
        "",
        "Rebuilt by `profile-rag eval` from the per-run reports in this folder.",
        "Recall@k and MRR are means over answerable questions (n); unanswerable",
        "questions are counted separately. A miss is a question with recall@k below 1.",
        "",
        "| model | mode | k | recall@k | MRR | n | unanswerable | misses |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for config, summary in runs:
        lines.append(
            f"| {config.model} | {_mode(config)} | {config.k} | {_number(summary.recall)}"
            f" | {_number(summary.mrr)} | {summary.n} | {summary.unanswerable}"
            f" | {_misses(summary)} |"
        )
    path = docs_dir / f"{set_name}-comparison.md"
    notes = ""
    if path.exists():
        _, marker, kept = path.read_text(encoding="utf-8").partition(NOTES_MARKER)
        if marker:
            notes = kept.lstrip("\n")
    lines += ["", NOTES_MARKER, ""]
    path.write_text("\n".join(lines) + notes, encoding="utf-8")
    return path


def run_evaluation(
    retriever: BaseRetriever,
    questions: Sequence[ResolvedQuestion],
    config: RunConfig,
    *,
    results_dir: Path,
    docs_dir: Path,
) -> RunOutput:
    """Run the questions, then write the detail, the report, and the comparison table."""
    results = run_questions(retriever, questions, config.k)
    summary = summarise(r.score for r in results)
    detail_path = write_detail(results, config, results_dir)
    report_path = write_report(config, summary, _tag_summaries(results), docs_dir)
    comparison_path = write_comparison(config.set_name, docs_dir)
    return RunOutput(summary, results, detail_path, report_path, comparison_path)
