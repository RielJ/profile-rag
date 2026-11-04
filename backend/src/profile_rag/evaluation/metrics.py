"""Recall@k and mean reciprocal rank (MRR).

Each answerable question has one or more labels, and each label resolves to
a set of chunk IDs (labels.py). A label is found when any of its chunks is
in the top k results.

- Recall@k of a question: found labels / labels. A one-label question
  scores 1 or 0; a two-source question that finds one source scores 0.5.
- Reciprocal rank of a question: 1 / rank of the first top-k result that
  belongs to any label, or 0 when none does. Ranks start at 1.
- Recall@k and MRR of a run: the mean over answerable questions.
- A miss: an answerable question with recall@k below 1.

Unanswerable questions (no labels) have no relevant chunk, so they are left
out of both means and counted on their own. Every mean comes with its sample
size `n`; with n = 30, one question moves recall by 3.3 points.

Tests: backend/tests/test_metrics.py.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def _found(top: Sequence[str], label: frozenset[str]) -> bool:
    return any(chunk_id in label for chunk_id in top)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[frozenset[str]], k: int) -> float:
    """The share of labels with a chunk in the first `k` of `retrieved`.

    Raises ValueError when `relevant` is empty (an unanswerable question has
    no recall) or `k < 1`.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if not relevant:
        raise ValueError("recall is undefined for a question without labels")
    top = retrieved[:k]
    return sum(_found(top, label) for label in relevant) / len(relevant)


def first_relevant_rank(
    retrieved: Sequence[str], relevant: Sequence[frozenset[str]], k: int
) -> int | None:
    """The 1-based rank of the first of the top `k` results in any label, or None."""
    if k < 1:
        raise ValueError("k must be at least 1")
    wanted = frozenset[str]().union(*relevant)
    for rank, chunk_id in enumerate(retrieved[:k], start=1):
        if chunk_id in wanted:
            return rank
    return None


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[frozenset[str]], k: int) -> float:
    """1 / `first_relevant_rank`, or 0.0 when no top-`k` result is relevant.

    Raises ValueError when `relevant` is empty or `k < 1`.
    """
    if not relevant:
        raise ValueError("reciprocal rank is undefined for a question without labels")
    rank = first_relevant_rank(retrieved, relevant, k)
    return 0.0 if rank is None else 1 / rank


@dataclass(frozen=True)
class Score:
    """The metrics of one question. Both are None for an unanswerable question."""

    question_id: str
    recall: float | None
    reciprocal_rank: float | None

    @property
    def answerable(self) -> bool:
        return self.recall is not None


def score(
    question_id: str, retrieved: Sequence[str], relevant: Sequence[frozenset[str]], k: int
) -> Score:
    """Score one question. `relevant` is empty for an unanswerable question."""
    if not relevant:
        return Score(question_id, None, None)
    return Score(
        question_id,
        recall_at_k(retrieved, relevant, k),
        reciprocal_rank(retrieved, relevant, k),
    )


@dataclass(frozen=True)
class Summary:
    """The means of a run, with sample sizes and the IDs of misses.

    `recall` and `mrr` are None when `n` is 0.
    """

    n: int
    unanswerable: int
    recall: float | None
    mrr: float | None
    misses: tuple[str, ...]


def summarise(scores: Iterable[Score]) -> Summary:
    """Average the answerable scores; count the unanswerable ones."""
    scores = list(scores)
    answered = [s for s in scores if s.answerable]
    n = len(answered)
    recalls = [s.recall for s in answered if s.recall is not None]
    ranks = [s.reciprocal_rank for s in answered if s.reciprocal_rank is not None]
    return Summary(
        n=n,
        unanswerable=len(scores) - n,
        recall=sum(recalls) / n if n else None,
        mrr=sum(ranks) / n if n else None,
        misses=tuple(s.question_id for s in answered if s.recall is not None and s.recall < 1),
    )
