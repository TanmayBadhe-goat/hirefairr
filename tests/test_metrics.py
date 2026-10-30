import math
import warnings

import numpy as np
import pytest

from hirefair.fairness.metrics import (
    classification_metrics,
    decision,
    fairness_metrics,
    threshold_for_best_f1,
    top_k_metrics,
)

# Hand-computed toy case.
# M rows (g=0): y=[1,1,0,0], yhat=[1,1,1,0]
#   SR = 3/4 = 0.75, TPR = 2/2 = 1.0, FPR = 1/2 = 0.5
# F rows (g=1): y=[1,1,0,0], yhat=[1,0,0,0]
#   SR = 1/4 = 0.25, TPR = 1/2 = 0.5, FPR = 0/2 = 0.0
Y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
YHAT = np.array([1, 1, 1, 0, 1, 0, 0, 0])
G = np.array([0, 0, 0, 0, 1, 1, 1, 1])


def test_group_rates_and_gaps():
    m = fairness_metrics(Y, YHAT, G)
    assert m["sr_m"] == pytest.approx(0.75)
    assert m["sr_f"] == pytest.approx(0.25)
    assert m["tpr_m"] == pytest.approx(1.0)
    assert m["tpr_f"] == pytest.approx(0.5)
    assert m["fpr_m"] == pytest.approx(0.5)
    assert m["fpr_f"] == pytest.approx(0.0)

    assert m["dpd"] == pytest.approx(-0.5)
    assert m["dpd_abs"] == pytest.approx(0.5)
    assert m["dir"] == pytest.approx(1 / 3)
    assert m["eod"] == pytest.approx(-0.5)
    assert m["eod_abs"] == pytest.approx(0.5)
    assert m["fpr_gap"] == pytest.approx(-0.5)


def test_task_metrics_hand_computed():
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.95, 0.4, 0.3, 0.2])
    c = classification_metrics(Y, YHAT, scores)
    assert c["accuracy"] == pytest.approx(6 / 8)
    assert c["precision"] == pytest.approx(3 / 4)
    assert c["recall"] == pytest.approx(3 / 4)
    assert c["f1"] == pytest.approx(3 / 4)
    assert 0.0 <= c["roc_auc"] <= 1.0


def test_zero_positive_group_returns_nan_and_warns():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0])
    yhat = np.array([1, 1, 0, 0, 0, 1, 0, 0])
    g = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    with pytest.warns(RuntimeWarning):
        m = fairness_metrics(y, yhat, g)
    assert math.isnan(m["tpr_f"])
    assert math.isnan(m["eod"])
    assert not math.isnan(m["sr_f"])


def test_small_subgroup_is_flagged():
    y = np.array([1, 1, 0, 0] + [1] * 2)
    yhat = np.array([1, 0, 1, 0] + [1] * 2)
    g = np.array([0, 0, 0, 0] + [1] * 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fairness_metrics(y, yhat, g)
    # Both groups here are smaller than 30, so both are flagged.
    assert m["flag_f"] == 1.0
    assert m["flag_m"] == 1.0


def test_decision_and_threshold():
    scores = np.array([0.2, 0.4, 0.6, 0.8])
    y = np.array([0, 1, 0, 1])
    threshold, f1 = threshold_for_best_f1(y, scores)
    # threshold 0.4 selects [0, 1, 0, 1]: TP=2, FP=1 -> F1 = 0.8
    assert threshold == pytest.approx(0.4)
    assert f1 == pytest.approx(0.8)
    # Selecting all scores >= 0.4 gives [0, 1, 1, 1]: TP=2, FP=1.
    assert np.array_equal(decision(scores, threshold), np.array([0, 1, 1, 1]))
    assert np.array_equal(decision(scores, 0.7), np.array([0, 0, 0, 1]))


def test_top_k_metrics():
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    y = np.array([1, 0, 1, 0, 0])
    g = np.array([0, 1, 1, 0, 1])
    m = top_k_metrics(scores, y, g, frac=0.2)
    assert m["top_k"] == 1.0
    assert m["top_share_m"] == 1.0
    assert m["top_sr_m"] == pytest.approx(0.5)
    assert m["top_sr_f"] == pytest.approx(0.0)
