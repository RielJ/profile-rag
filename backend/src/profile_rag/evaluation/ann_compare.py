"""How far HNSW dense results drift from exact dense results.

For each question, the dense retriever runs twice: once exact, once through
the HNSW index at each `ef_search` value. overlap@k is the share of the
exact top k that the HNSW top k also holds:
`|hnsw ∩ exact| / |exact|`. 1.0 means the index found the same chunks.
Order inside the top k does not count; recall@k and MRR of each run show
the effect on the labelled questions.

How each run is forced (vectorstore/hnsw.py, `open_engine`):

- Exact: the engine's connections have `enable_indexscan = off`. HNSW
  supports no bitmap scan, so the dense query is a sequential scan that
  compares the question with every row.
- HNSW: the engine's connections have `enable_seqscan = off`, so Postgres
  takes the HNSW index whenever the query can use it. Without this, a
  small table (the 15 example chunks) is read by sequential scan and the
  index is never used. The store sets `hnsw.ef_search` with SET LOCAL
  before each dense query.

`plans` records the scan line of EXPLAIN for the store's dense query under
each setting, so a report shows whether the index was used at all.

The report (`docs/eval/hnsw-<set>-<model>.md`) holds numbers, settings, and
plan node names only. Never question or chunk text.

Tests: backend/tests/test_hnsw.py (overlap, the runs, and the CLI command).
"""

import re
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import override

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from profile_rag.evaluation.labels import ResolvedQuestion
from profile_rag.evaluation.metrics import Summary, summarise
from profile_rag.evaluation.runner import run_questions
from profile_rag.retrievers.dense import VISIBLE, VISIBLE_VALUES, dense_retriever
from profile_rag.vectorstore import hnsw
from profile_rag.vectorstore.models import EmbeddingModel
from profile_rag.vectorstore.store import open_search_store


def overlap_at_k(hnsw_ids: Sequence[str], exact_ids: Sequence[str]) -> float:
    """The share of `exact_ids` that `hnsw_ids` also holds. 1.0 if both are empty."""
    if not exact_ids:
        return 1.0 if not hnsw_ids else 0.0
    return len(set(hnsw_ids) & set(exact_ids)) / len(set(exact_ids))


@dataclass(frozen=True)
class EfResult:
    """The HNSW run at one `ef_search`."""

    ef_search: int
    overlaps: tuple[float, ...]  # one per question, in question order
    short: int  # questions where HNSW returned fewer chunks than exact
    summary: Summary  # recall@k and MRR of this run

    @property
    def mean(self) -> float:
        return sum(self.overlaps) / len(self.overlaps)

    @property
    def min(self) -> float:
        return min(self.overlaps)


def compare_runs(
    questions: Sequence[ResolvedQuestion],
    *,
    exact: BaseRetriever,
    by_ef: Mapping[int, BaseRetriever],
    k: int,
) -> tuple[Summary, list[EfResult]]:
    """Score the exact run, then each HNSW run against it. Needs one question or more."""
    if not questions:
        raise ValueError("no questions to compare")
    exact_results = run_questions(exact, questions, k)
    exact_ids = [result.chunk_ids for result in exact_results]
    found: list[EfResult] = []
    for ef_search, retriever in by_ef.items():
        results = run_questions(retriever, questions, k)
        ids = [result.chunk_ids for result in results]
        found.append(
            EfResult(
                ef_search=ef_search,
                overlaps=tuple(overlap_at_k(h, e) for h, e in zip(ids, exact_ids, strict=True)),
                short=sum(len(h) < len(e) for h, e in zip(ids, exact_ids, strict=True)),
                summary=summarise(result.score for result in results),
            )
        )
    return summarise(result.score for result in exact_results), found


@dataclass(frozen=True)
class AnnComparison:
    set_name: str
    model: str
    k: int
    chunks: int  # rows in the model's table
    questions: int
    index: hnsw.HnswStatus
    plans: dict[str, str]  # "default" / "index" / "exact" -> scan line of EXPLAIN
    exact: Summary
    by_ef: list[EfResult]


class _CachedQueries(Embeddings):
    """Embeds each question once, so every run searches with the same vector."""

    def __init__(self, inner: Embeddings) -> None:
        self._inner = inner
        self._queries: dict[str, list[float]] = {}

    @override
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._inner.embed_documents(texts)

    @override
    def embed_query(self, text: str) -> list[float]:
        if text not in self._queries:
            self._queries[text] = self._inner.embed_query(text)
        return self._queries[text]

    @override
    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# `Index Scan using x on y  (cost=...)` -> `Index Scan using x on y`
_COST = re.compile(r"\s*\(cost=.*$")


def scan_line(plan: Sequence[str]) -> str:
    """The first scan node of an EXPLAIN plan, without costs or arrows."""
    for line in plan:
        if "Scan" in line:
            return _COST.sub("", line).strip().removeprefix("->").strip()
    return "no scan node"


def run_ann_compare(
    url: str,
    model: EmbeddingModel,
    embeddings: Embeddings,
    questions: Sequence[ResolvedQuestion],
    *,
    set_name: str,
    chunks: int,
    k: int,
    ef_values: Sequence[int],
) -> AnnComparison:
    """Run the exact and HNSW dense searches on the model's table and compare them.

    Raises ValueError if the table has no valid HNSW index (run `profile-rag
    hnsw apply`) or there are no questions.
    """
    index = hnsw.hnsw_status(url, model)
    if index is None or not index.valid:
        raise ValueError(f"no valid HNSW index {hnsw.hnsw_index_name(model)}")
    if not questions:
        raise ValueError("no questions to compare")
    cached = _CachedQueries(embeddings)
    vector = cached.embed_query(questions[0].question.question)
    plans = {
        plan: scan_line(
            hnsw.explain_dense(url, model, vector, k=k, visibility=VISIBLE_VALUES, plan=plan)
        )
        for plan in ("default", "index", "exact")
    }

    # One engine per run; the stack closes every one of them, also on an error.
    with ExitStack() as engines:
        exact_engine = engines.enter_context(hnsw.open_engine(url, plan="exact"))
        exact = dense_retriever(open_search_store(exact_engine, model, cached), k=k, filter=VISIBLE)
        by_ef: dict[int, BaseRetriever] = {}
        for ef_search in ef_values:
            engine = engines.enter_context(hnsw.open_engine(url, plan="index"))
            store = open_search_store(engine, model, cached, ef_search=ef_search)
            by_ef[ef_search] = dense_retriever(store, k=k, filter=VISIBLE)
        exact_summary, results = compare_runs(questions, exact=exact, by_ef=by_ef, k=k)
    return AnnComparison(
        set_name=set_name,
        model=model.key,
        k=k,
        chunks=chunks,
        questions=len(questions),
        index=index,
        plans=plans,
        exact=exact_summary,
        by_ef=results,
    )


def number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def render_ann_report(comparison: AnnComparison) -> str:
    """The Markdown report. Numbers, settings, and plan node names only."""
    c = comparison
    lines = [
        f"# HNSW vs exact dense search: {c.set_name} set, {c.model}",
        "",
        "Generated by `profile-rag hnsw compare`. Do not edit by hand; rerun the command.",
        "",
        f"Index `{c.index.name}`: m {c.index.m}, ef_construction {c.index.ef_construction},"
        f" {c.index.size_bytes} bytes. Table rows: {c.chunks}. k = {c.k}."
        f" Questions: {c.questions}.",
        "",
        "Exact: connections with `enable_indexscan = off` (sequential scan).",
        "HNSW: connections with `enable_seqscan = off`, and `SET LOCAL hnsw.ef_search`",
        "before each dense query. overlap@k = |HNSW top k ∩ exact top k| / |exact top k|.",
        "",
        "## Query plan of the dense query",
        "",
        "| planner setting | scan |",
        "| --- | --- |",
        f"| default | {c.plans['default']} |",
        f"| enable_seqscan = off | {c.plans['index']} |",
        f"| enable_indexscan = off | {c.plans['exact']} |",
        "",
        "## Overlap and retrieval quality",
        "",
        "| run | ef_search | mean overlap@k | min overlap@k | short results"
        " | recall@k | MRR | n | unanswerable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        f"| exact | - | 1.000 | 1.000 | 0 | {number(c.exact.recall)} | {number(c.exact.mrr)}"
        f" | {c.exact.n} | {c.exact.unanswerable} |",
    ]
    for result in c.by_ef:
        s = result.summary
        lines.append(
            f"| hnsw | {result.ef_search} | {result.mean:.3f} | {result.min:.3f}"
            f" | {result.short} | {number(s.recall)} | {number(s.mrr)} | {s.n}"
            f" | {s.unanswerable} |"
        )
    lines += [
        "",
        "Short results: questions where HNSW returned fewer chunks than the exact run.",
        "pgvector's HNSW scan returns at most ef_search rows (iterative scans are off),",
        "so an ef_search below k always gives short results.",
        "",
    ]
    return "\n".join(lines)


def write_ann_report(comparison: AnnComparison, docs_dir: Path) -> Path:
    """Write `docs_dir/hnsw-<set>-<model>.md`."""
    docs_dir.mkdir(parents=True, exist_ok=True)
    path = docs_dir / f"hnsw-{comparison.set_name}-{comparison.model}.md"
    path.write_text(render_ann_report(comparison), encoding="utf-8")
    return path
