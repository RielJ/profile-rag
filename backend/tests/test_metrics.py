"""Tests for evaluation/metrics.py on hand-built rankings."""

import pytest

from profile_rag.evaluation.metrics import (
    Score,
    first_relevant_rank,
    recall_at_k,
    reciprocal_rank,
    score,
    summarise,
)

A = frozenset({"a"})
B = frozenset({"b"})
# A span inside the overlap of two chunks: either chunk counts.
OVERLAP = frozenset({"c1", "c2"})


def test_recall_is_one_when_the_relevant_chunk_is_in_the_top_k() -> None:
    assert recall_at_k(["x", "a", "y"], [A], k=2) == 1.0


def test_recall_ignores_hits_below_k() -> None:
    assert recall_at_k(["x", "y", "a"], [A], k=2) == 0.0


def test_no_relevant_chunk_retrieved_scores_zero() -> None:
    assert recall_at_k(["x", "y", "z"], [A], k=3) == 0.0
    assert reciprocal_rank(["x", "y", "z"], [A], k=3) == 0.0
    assert first_relevant_rank(["x", "y", "z"], [A], k=3) is None


def test_an_empty_result_scores_zero() -> None:
    assert recall_at_k([], [A], k=8) == 0.0
    assert reciprocal_rank([], [A], k=8) == 0.0


def test_two_labels_count_separately() -> None:
    assert recall_at_k(["a", "x"], [A, B], k=2) == 0.5
    assert recall_at_k(["b", "a"], [A, B], k=2) == 1.0


def test_either_chunk_of_an_overlap_finds_the_label() -> None:
    assert recall_at_k(["x", "c2"], [OVERLAP], k=2) == 1.0
    # Both chunks retrieved still count once.
    assert recall_at_k(["c1", "c2"], [OVERLAP, A], k=2) == 0.5


def test_reciprocal_rank_uses_the_first_relevant_hit_of_any_label() -> None:
    assert reciprocal_rank(["a"], [A], k=1) == 1.0
    assert reciprocal_rank(["x", "y", "b", "a"], [A, B], k=4) == pytest.approx(1 / 3)
    assert first_relevant_rank(["x", "y", "b", "a"], [A, B], k=4) == 3


def test_reciprocal_rank_ignores_hits_below_k() -> None:
    assert reciprocal_rank(["x", "y", "a"], [A], k=2) == 0.0


def test_an_unanswerable_question_has_no_recall_or_reciprocal_rank() -> None:
    with pytest.raises(ValueError, match="without labels"):
        recall_at_k(["a"], [], k=1)
    with pytest.raises(ValueError, match="without labels"):
        reciprocal_rank(["a"], [], k=1)
    assert score("q", ["a"], [], k=1) == Score("q", None, None)


def test_k_below_one_is_an_error() -> None:
    with pytest.raises(ValueError, match="k must be"):
        recall_at_k(["a"], [A], k=0)
    with pytest.raises(ValueError, match="k must be"):
        reciprocal_rank(["a"], [A], k=0)


def test_summary_excludes_unanswerable_questions_and_counts_them() -> None:
    scores = [
        score("hit-first", ["a", "x"], [A], k=2),  # recall 1, rr 1
        score("hit-second", ["x", "a"], [A], k=2),  # recall 1, rr 0.5
        score("half", ["a", "x"], [A, B], k=2),  # recall 0.5, rr 1
        score("none", ["x", "y"], [A], k=2),  # recall 0, rr 0
        score("unanswerable", ["a", "b"], [], k=2),
    ]
    summary = summarise(scores)
    assert summary.n == 4
    assert summary.unanswerable == 1
    assert summary.recall == pytest.approx((1 + 1 + 0.5 + 0) / 4)
    assert summary.mrr == pytest.approx((1 + 0.5 + 1 + 0) / 4)
    assert summary.misses == ("half", "none")


def test_summary_of_only_unanswerable_questions_has_no_means() -> None:
    summary = summarise([score("u1", ["a"], [], k=1), score("u2", [], [], k=1)])
    assert (summary.n, summary.unanswerable, summary.recall, summary.mrr) == (0, 2, None, None)
    assert summary.misses == ()


def test_summary_of_nothing_is_empty() -> None:
    summary = summarise([])
    assert (summary.n, summary.unanswerable, summary.recall, summary.mrr) == (0, 0, None, None)
