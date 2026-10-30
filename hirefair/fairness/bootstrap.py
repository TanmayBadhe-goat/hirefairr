"""Bootstrap confidence intervals for fairness gaps.

1,000 resamples of the test set by default. No p-values.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from hirefair.fairness.metrics import fairness_metrics


def bootstrap_ci(
    y: np.ndarray,
    yhat: np.ndarray,
    g: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap interval for one statistic.

    Returns ``point``, ``lo``, ``hi`` and ``n_boot``.
    """
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    g = np.asarray(g).astype(int)
    n = len(y)
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            values[i] = float(statistic(y[idx], yhat[idx], g[idx]))
        except Exception:
            values[i] = np.nan
    finite = values[np.isfinite(values)]
    point = float(statistic(y, yhat, g))
    if len(finite) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": n_boot}
    lo = float(np.quantile(finite, alpha / 2))
    hi = float(np.quantile(finite, 1 - alpha / 2))
    return {"point": point, "lo": lo, "hi": hi, "n_boot": n_boot}


def abs_eod(y: np.ndarray, yhat: np.ndarray, g: np.ndarray) -> float:
    """Absolute equal-opportunity difference."""
    return float(fairness_metrics(y, yhat, g)["eod_abs"])


def paired_bootstrap_diff(
    y: np.ndarray,
    yhat_a: np.ndarray,
    yhat_b: np.ndarray,
    g: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Bootstrap the difference ``|EOD_a| - |EOD_b|`` on shared resamples.

    ``a`` is the baseline (B0) and ``b`` the debiased model (B3). A positive
    difference means the baseline gap is larger. ``excludes_zero`` is True when
    the interval does not contain zero.
    """
    y = np.asarray(y).astype(int)
    yhat_a = np.asarray(yhat_a).astype(int)
    yhat_b = np.asarray(yhat_b).astype(int)
    g = np.asarray(g).astype(int)
    n = len(y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = abs_eod(y[idx], yhat_a[idx], g[idx]) - abs_eod(y[idx], yhat_b[idx], g[idx])
    finite = diffs[np.isfinite(diffs)]
    point = abs_eod(y, yhat_a, g) - abs_eod(y, yhat_b, g)
    if len(finite) == 0:
        return {
            "point": float(point),
            "lo": float("nan"),
            "hi": float("nan"),
            "excludes_zero": False,
            "n_boot": n_boot,
        }
    lo = float(np.quantile(finite, alpha / 2))
    hi = float(np.quantile(finite, 1 - alpha / 2))
    return {
        "point": float(point),
        "lo": lo,
        "hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": n_boot,
    }
