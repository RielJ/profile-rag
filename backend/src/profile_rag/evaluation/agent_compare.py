"""Compare the answer chain with the agent graph on a labeled question set.

`compare(questions, chain, graph, ...)` runs each question through the chain
(generation/chain.py) and then the graph (agent/graph.py), and records for
each: the outcome (`answered`, `unknown`, or `rejected` when the reply
failed parsing or the citation check), the cited chunk IDs, the Pi calls,
and, for the graph, the rewrites and node path. Pi calls are counted with a
callback on every chat model start, so a rejected run counts its calls too.

Scoring against the labels (`ResolvedQuestion.relevant`):

- An answerable question should be answered; an unanswerable one should be
  `unknown`. `correct_unknown` and `answered_unanswerable` count the second
  kind.
- `labeled_citation`: answered answerable questions that cite at least one
  chunk holding an expected span. `citation_precision`: of all chunks those
  answers cite, the share that hold an expected span. A correct answer can
  cite a supporting chunk that holds no labeled span, so a low precision is
  a prompt to read the private detail file, not proof of a wrong citation.

`max_pi_calls` caps the run: a question starts only if its worst case
(`WORST_CASE_PI_CALLS`: 1 for the chain, 4 for the graph) still fits, so
the total never exceeds the cap. Questions that do not fit are skipped and
counted.

Outputs (`write_outputs`):

- `eval/results/agent-vs-chain-<set>-<model>.jsonl`, one line per question:
  ID, tags, labeled chunk IDs, and both runs. No question or answer text;
  still private (gitignored), like every file in eval/.
- `docs/eval/agent-vs-chain-<set>-<model>.md`: numbers, settings, and
  question IDs only. Text below `NOTES_MARKER` is kept on a rerun.

Pi failures, retrieval errors, and `AgentLoopError` stop the run: they are
setup problems or bugs, not answers.

Tests: backend/tests/test_agent_compare.py.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, override

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable

from profile_rag.agent.errors import GradeParseError, RewriteError
from profile_rag.agent.graph import AgentGraph, run_agent
from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.runner import NOTES_MARKER
from profile_rag.generation.answer import Answered, InventedCitation
from profile_rag.generation.chain import GroundedAnswer

Outcome = Literal["answered", "unknown", "rejected"]
Pipeline = Literal["chain", "agent"]

WORST_CASE_PI_CALLS = 1 + 4

# A reply the app rejects: counted as an outcome, not a failure of the run.
REJECTED_REPLIES = (OutputParserException, InventedCitation, GradeParseError, RewriteError)


@dataclass(frozen=True)
class CompareConfig:
    set_name: str
    model: str
    mode: str
    k: int
    pi_model: str
    max_pi_calls: int | None = None

    @property
    def name(self) -> str:
        return f"agent-vs-chain-{self.set_name}-{self.model}"


@dataclass(frozen=True)
class PipelineRun:
    outcome: Outcome
    citations: tuple[str, ...] = ()
    pi_calls: int = 0
    rewrites: int = 0
    path: tuple[str, ...] = ()
    # The class name of the rejected reply's exception, never its message.
    rejected_by: str | None = None


@dataclass(frozen=True)
class QuestionComparison:
    resolved: ResolvedQuestion
    chain: PipelineRun
    agent: PipelineRun

    @property
    def labeled(self) -> frozenset[str]:
        return frozenset[str]().union(*self.resolved.relevant)

    @property
    def answerable(self) -> bool:
        return bool(self.resolved.question.expected)


@dataclass(frozen=True)
class PipelineSummary:
    n: int
    answered: int
    unknown: int
    rejected: int
    answerable: int
    answerable_answered: int
    unanswerable: int
    correct_unknown: int
    answered_unanswerable: int
    labeled_citation: int
    citation_precision: float | None
    pi_calls_total: int
    pi_calls_mean: float | None
    rewritten: int
    # IDs of answerable questions this pipeline did not answer.
    not_answered: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CompareOutput:
    results: list[QuestionComparison]
    skipped: list[str]
    chain: PipelineSummary
    agent: PipelineSummary


class PiCallCounter(BaseCallbackHandler):
    """Counts chat model calls. Each one is one Pi call."""

    def __init__(self) -> None:
        self.calls = 0

    @override
    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **kwargs: Any
    ) -> None:
        self.calls += 1


def _from_answer(
    result: GroundedAnswer,
    pi_calls: int,
    *,
    rewrites: int = 0,
    path: tuple[str, ...] = (),
) -> PipelineRun:
    answer = result.answer.root
    if isinstance(answer, Answered):
        return PipelineRun("answered", tuple(answer.citations), pi_calls, rewrites, path)
    return PipelineRun("unknown", (), pi_calls, rewrites, path)


def run_chain(chain: Runnable[dict[str, str], GroundedAnswer], question: str) -> PipelineRun:
    counter = PiCallCounter()
    try:
        result = chain.invoke({"question": question}, {"callbacks": [counter]})
    except REJECTED_REPLIES as error:
        return PipelineRun("rejected", (), counter.calls, rejected_by=type(error).__name__)
    return _from_answer(result, counter.calls)


def run_graph(graph: AgentGraph, question: str) -> PipelineRun:
    counter = PiCallCounter()
    path: list[str] = []
    try:
        run = run_agent(
            graph,
            question,
            on_node=lambda node, _update: path.append(node),
            config={"callbacks": [counter]},
        )
    except REJECTED_REPLIES as error:
        return PipelineRun(
            "rejected",
            (),
            counter.calls,
            rewrites=path.count("rewrite"),
            path=tuple(path),
            rejected_by=type(error).__name__,
        )
    if run.pi_calls != counter.calls:
        raise AssertionError(
            f"the graph counted {run.pi_calls} Pi calls, the callback {counter.calls}"
        )
    return _from_answer(run.result, counter.calls, rewrites=run.rewrites, path=run.path)


def summarise(results: Sequence[QuestionComparison], pipeline: Pipeline) -> PipelineSummary:
    runs = [(r, r.chain if pipeline == "chain" else r.agent) for r in results]
    answerable = [(r, run) for r, run in runs if r.answerable]
    unanswerable = [(r, run) for r, run in runs if not r.answerable]
    answered = [(r, run) for r, run in answerable if run.outcome == "answered"]
    cited = [(chunk in r.labeled) for r, run in answered for chunk in run.citations]
    total = sum(run.pi_calls for _, run in runs)
    return PipelineSummary(
        n=len(runs),
        answered=sum(run.outcome == "answered" for _, run in runs),
        unknown=sum(run.outcome == "unknown" for _, run in runs),
        rejected=sum(run.outcome == "rejected" for _, run in runs),
        answerable=len(answerable),
        answerable_answered=len(answered),
        unanswerable=len(unanswerable),
        correct_unknown=sum(run.outcome == "unknown" for _, run in unanswerable),
        answered_unanswerable=sum(run.outcome == "answered" for _, run in unanswerable),
        labeled_citation=sum(any(c in r.labeled for c in run.citations) for r, run in answered),
        citation_precision=sum(cited) / len(cited) if cited else None,
        pi_calls_total=total,
        pi_calls_mean=total / len(runs) if runs else None,
        rewritten=sum(run.rewrites > 0 for _, run in runs),
        not_answered=[r.resolved.question.id for r, run in answerable if run.outcome != "answered"],
    )


def compare(
    questions: Sequence[ResolvedQuestion],
    chain: Runnable[dict[str, str], GroundedAnswer],
    graph: AgentGraph,
    *,
    max_pi_calls: int | None = None,
    on_question: Callable[[QuestionComparison], None] | None = None,
) -> CompareOutput:
    """Run every question through the chain, then the graph. See the module docstring."""
    results: list[QuestionComparison] = []
    skipped: list[str] = []
    spent = 0
    for resolved in questions:
        if max_pi_calls is not None and spent + WORST_CASE_PI_CALLS > max_pi_calls:
            skipped.append(resolved.question.id)
            continue
        text = resolved.question.question
        comparison = QuestionComparison(resolved, run_chain(chain, text), run_graph(graph, text))
        spent += comparison.chain.pi_calls + comparison.agent.pi_calls
        results.append(comparison)
        if on_question is not None:
            on_question(comparison)
    return CompareOutput(
        results=results,
        skipped=skipped,
        chain=summarise(results, "chain"),
        agent=summarise(results, "agent"),
    )


def _detail_line(result: QuestionComparison) -> str:
    question = result.resolved.question
    record = {
        "id": question.id,
        "tags": list(question.tags),
        "labeled": sorted(result.labeled),
        "chain": asdict(result.chain),
        "agent": asdict(result.agent),
    }
    return json.dumps(record, ensure_ascii=False)


def _number(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_report(config: CompareConfig, output: CompareOutput) -> str:
    """The Markdown report. Numbers, settings, and question IDs only."""
    chain, agent = output.chain, output.agent
    rows = [
        ("questions run", chain.n, agent.n),
        ("answered", chain.answered, agent.answered),
        ("unknown", chain.unknown, agent.unknown),
        ("rejected replies", chain.rejected, agent.rejected),
        (
            f"answerable answered (of {chain.answerable})",
            chain.answerable_answered,
            agent.answerable_answered,
        ),
        (
            f"unanswerable answered unknown (of {chain.unanswerable})",
            chain.correct_unknown,
            agent.correct_unknown,
        ),
        ("unanswerable answered anyway", chain.answered_unanswerable, agent.answered_unanswerable),
        ("answers citing a labeled chunk", chain.labeled_citation, agent.labeled_citation),
        (
            "citation precision vs labels",
            _number(chain.citation_precision),
            _number(agent.citation_precision),
        ),
        ("Pi calls, total", chain.pi_calls_total, agent.pi_calls_total),
        (
            "Pi calls, mean per question",
            _number(chain.pi_calls_mean, 2),
            _number(agent.pi_calls_mean, 2),
        ),
        ("questions rewritten", chain.rewritten, agent.rewritten),
        (
            "answerable not answered",
            ", ".join(chain.not_answered) or "none",
            ", ".join(agent.not_answered) or "none",
        ),
    ]
    cap = "none" if config.max_pi_calls is None else str(config.max_pi_calls)
    lines = [
        f"# Agent vs chain: {config.set_name}, {config.model}",
        "",
        "Generated by `profile-rag compare-agent`. The table is rebuilt on each run;",
        "notes below the marker line are kept.",
        "",
        f"Settings: set {config.set_name}, embedding model {config.model}, mode {config.mode},"
        f" k {config.k}, Pi model {config.pi_model}, Pi call cap {cap}.",
        "",
        "| measure | chain | agent |",
        "| --- | --- | --- |",
        *(f"| {name} | {c} | {a} |" for name, c, a in rows),
        "",
        f"Skipped to stay under the Pi call cap: {', '.join(output.skipped) or 'none'}",
        "",
        "Agent paths (question: nodes):",
        "",
        *(
            f"- {r.resolved.question.id}: {' > '.join(r.agent.path) or '(rejected)'}"
            f" ({r.agent.pi_calls} Pi calls)"
            for r in output.results
        ),
        "",
        "```json",
        json.dumps(
            {"config": asdict(config), "chain": asdict(chain), "agent": asdict(agent)}, indent=2
        ),
        "```",
    ]
    return "\n".join(lines)


def write_outputs(
    config: CompareConfig, output: CompareOutput, *, results_dir: Path, docs_dir: Path
) -> tuple[Path, Path]:
    """Write the private detail file and the committed report. Returns both paths."""
    results_dir.mkdir(parents=True, exist_ok=True)
    detail = results_dir / f"{config.name}.jsonl"
    detail.write_text("".join(_detail_line(r) + "\n" for r in output.results), encoding="utf-8")

    docs_dir.mkdir(parents=True, exist_ok=True)
    report = docs_dir / f"{config.name}.md"
    notes = ""
    if report.exists():
        _, marker, kept = report.read_text(encoding="utf-8").partition(NOTES_MARKER)
        if marker:
            notes = kept.lstrip("\n")
    report.write_text(
        render_report(config, output) + "\n\n" + NOTES_MARKER + "\n\n" + notes, encoding="utf-8"
    )
    return detail, report
