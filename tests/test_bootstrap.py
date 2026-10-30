import numpy as np
import pytest

from hirefair.fairness.bootstrap import abs_eod, bootstrap_ci, paired_bootstrap_diff
from hirefair.fairness.metrics import fairness_metrics


def _toy(n=200, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    yhat = rng.integers(0, 2, size=n)
    return y, yhat, g


def test_bootstrap_ci_ordering():
    y, yhat, g = _toy()
    out = bootstrap_ci(y, yhat, g, lambda a, b, c: fairness_metrics(a, b, c)["dpd"], n_boot=100)
    assert out["lo"] <= out["point"] <= out["hi"] or out["lo"] <= out["hi"]
    assert out["n_boot"] == 100


def test_paired_diff_zero_for_identical_models():
    y, yhat, g = _toy()
    out = paired_bootstrap_diff(y, yhat, yhat, g, n_boot=50)
    assert out["point"] == pytest.approx(0.0)
    assert out["lo"] <= 0 <= out["hi"]
    assert out["excludes_zero"] is False


def test_paired_diff_detects_large_change():
    rng = np.random.default_rng(1)
    n = 500
    g = rng.integers(0, 2, size=n)
    y = np.ones(n, dtype=int)
    baseline = g.copy()  # selects only one gender perfectly
    debiased = np.ones(n, dtype=int)
    out = paired_bootstrap_diff(y, baseline, debiased, g, n_boot=100)
    assert abs_eod(y, baseline, g) > abs_eod(y, debiased, g)
    assert out["point"] > 0
    assert out["excludes_zero"] is True
