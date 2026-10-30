import numpy as np
import pytest

from hirefair.fairness.counterfactual import (
    flip_rate,
    mean_abs_score_delta,
    mean_rank_shift,
    ranks,
    summary,
    top_k_change_count,
)


def test_flip_rate():
    assert flip_rate(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 1])) == pytest.approx(0.5)
    assert flip_rate(np.array([1, 1]), np.array([1, 1])) == 0.0


def test_mean_abs_score_delta():
    a = np.array([0.2, 0.8, 0.5])
    b = np.array([0.4, 0.5, 0.5])
    assert mean_abs_score_delta(a, b) == pytest.approx((0.2 + 0.3 + 0.0) / 3)


def test_ranks_are_one_based_and_ordered():
    assert np.array_equal(ranks(np.array([0.9, 0.1, 0.5])), np.array([1.0, 3.0, 2.0]))


def test_mean_rank_shift():
    a = np.array([0.9, 0.8, 0.7])
    b = np.array([0.1, 0.8, 0.9])
    # ranks move 1->3 and 3->1, the middle candidate stays: mean 4/3
    assert mean_rank_shift(a, b) == pytest.approx(4 / 3)


def test_top_k_change_count():
    a = np.array([0.9, 0.8, 0.7, 0.6])
    b = np.array([0.9, 0.1, 0.7, 0.6])
    assert top_k_change_count(a, b, k=2) == 2


def test_summary_contains_flip_and_delta():
    scores_a = np.array([0.9, 0.2, 0.6])
    scores_b = np.array([0.4, 0.2, 0.6])
    decisions_a = (scores_a >= 0.5).astype(int)
    decisions_b = (scores_b >= 0.5).astype(int)
    out = summary(scores_a, scores_b, decisions_a, decisions_b, k=2)
    assert out["flip_rate"] == pytest.approx(1 / 3)
    assert out["mean_abs_score_delta"] == pytest.approx(0.5 / 3)
    assert out["n_pairs"] == 3.0
