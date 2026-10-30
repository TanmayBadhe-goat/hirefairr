"""Counterfactual metrics over swapped gender cues."""

from __future__ import annotations

import numpy as np


def flip_rate(decisions_a: np.ndarray, decisions_b: np.ndarray) -> float:
    """Share of pairs whose decision changes."""
    a = np.asarray(decisions_a).astype(int)
    b = np.asarray(decisions_b).astype(int)
    if len(a) == 0:
        return float("nan")
    return float((a != b).mean())


def mean_abs_score_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Mean of ``|s(x) - s(swap(x))|``."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float(np.abs(a - b).mean())


def mean_signed_score_delta(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Mean of ``s(swap(x)) - s(x)``."""
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if len(a) == 0:
        return float("nan")
    return float((b - a).mean())


def ranks(scores: np.ndarray) -> np.ndarray:
    """1-based ranks by descending score, ties broken by original order."""
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="stable")
    out = np.empty(len(scores), dtype=float)
    out[order] = np.arange(1, len(scores) + 1)
    return out


def mean_rank_shift(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Mean absolute rank change after swapping cues."""
    if len(np.asarray(scores_a)) == 0:
        return float("nan")
    return float(np.abs(ranks(scores_a) - ranks(scores_b)).mean())


def top_k_change_count(scores_a: np.ndarray, scores_b: np.ndarray, k: int = 10) -> int:
    """Number of candidates that leave or enter the top-k after swapping."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    k = min(k, len(scores_a))
    if k == 0:
        return 0
    top_a = set(np.argsort(-scores_a, kind="stable")[:k].tolist())
    top_b = set(np.argsort(-scores_b, kind="stable")[:k].tolist())
    return int(len(top_a.symmetric_difference(top_b)))


def summary(
    scores_orig: np.ndarray,
    scores_swapped: np.ndarray,
    decisions_orig: np.ndarray | None = None,
    decisions_swapped: np.ndarray | None = None,
    k: int = 10,
) -> dict[str, float]:
    """Flip rate, score delta and rank shift for one model."""
    out = {
        "n_pairs": float(len(np.asarray(scores_orig))),
        "mean_abs_score_delta": mean_abs_score_delta(scores_orig, scores_swapped),
        "mean_signed_score_delta": mean_signed_score_delta(scores_orig, scores_swapped),
        "mean_rank_shift": mean_rank_shift(scores_orig, scores_swapped),
        "top_k_changes": float(top_k_change_count(scores_orig, scores_swapped, k)),
    }
    if decisions_orig is not None and decisions_swapped is not None:
        out["flip_rate"] = flip_rate(decisions_orig, decisions_swapped)
    return out
