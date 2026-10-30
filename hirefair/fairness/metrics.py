"""Fairness and task metrics.

Gender is encoded as ``g = 0`` (M) and ``g = 1`` (F). Groups with no positive
examples return ``NaN`` for TPR and emit a warning instead of raising.
Subgroups smaller than 30 are flagged and should not be interpreted.
"""

from __future__ import annotations

import warnings

import numpy as np

MIN_SUBGROUP = 30


def _rate(mask_positives: np.ndarray, decisions: np.ndarray) -> float:
    if mask_positives.sum() == 0:
        return float("nan")
    return float(decisions[mask_positives].mean())


def fairness_metrics(y: np.ndarray, yhat: np.ndarray, g: np.ndarray) -> dict[str, float]:
    """Selection rate, DP, DIR, TPR, EO, FPR gap and subgroup flags.

    Args:
        y: True labels, 0/1.
        yhat: Decisions at the model's threshold, 0/1.
        g: Gender, 0 = M, 1 = F.

    Returns:
        Dict of signed and absolute gaps. Undefined rates are ``NaN`` and emit
        a ``RuntimeWarning``.
    """
    y = np.asarray(y).astype(int).ravel()
    yhat = np.asarray(yhat).astype(int).ravel()
    g = np.asarray(g).astype(int).ravel()
    if not (len(y) == len(yhat) == len(g)):
        raise ValueError("y, yhat and g must have the same length")

    out: dict[str, float] = {}
    for name, group in (("m", 0), ("f", 1)):
        in_group = g == group
        positives = in_group & (y == 1)
        negatives = in_group & (y == 0)
        out[f"n_{name}"] = float(in_group.sum())
        out[f"n_pos_{name}"] = float(positives.sum())
        out[f"sr_{name}"] = float(yhat[in_group].mean()) if in_group.sum() else float("nan")
        out[f"tpr_{name}"] = _rate(positives, yhat)
        out[f"fpr_{name}"] = _rate(negatives, yhat)
        out[f"fnr_{name}"] = (
            float(1.0 - out[f"tpr_{name}"]) if not np.isnan(out[f"tpr_{name}"]) else float("nan")
        )
        out[f"flag_{name}"] = float(in_group.sum() < MIN_SUBGROUP)
        out[f"flag_pos_{name}"] = float(positives.sum() < MIN_SUBGROUP)

    for metric in ("tpr", "fpr", "sr"):
        male = out[f"{metric}_m"]
        female = out[f"{metric}_f"]
        value = female - male
        out[f"{metric}_gap"] = float(value) if not (np.isnan(male) or np.isnan(female)) else float("nan")

    out["dpd"] = out["sr_f"] - out["sr_m"]
    out["dpd_abs"] = abs(out["dpd"])
    out["eod"] = out["tpr_f"] - out["tpr_m"]
    out["eod_abs"] = abs(out["eod"])
    out["fpr_gap"] = out["fpr_f"] - out["fpr_m"]

    sr_m, sr_f = out["sr_m"], out["sr_f"]
    if np.isnan(sr_m) or np.isnan(sr_f):
        out["dir"] = float("nan")
    elif max(sr_m, sr_f) == 0:
        out["dir"] = float("nan")
    elif min(sr_m, sr_f) == 0:
        out["dir"] = 0.0
    else:
        out["dir"] = float(min(sr_m, sr_f) / max(sr_m, sr_f))

    if np.isnan(out["tpr_m"]) or np.isnan(out["tpr_f"]):
        warnings.warn(
            "TPR is undefined for at least one gender group (no positive examples); "
            "EOD returned as NaN",
            RuntimeWarning,
            stacklevel=2,
        )
    for name in ("m", "f"):
        if out[f"flag_{name}"]:
            warnings.warn(
                f"gender group {name.upper()} has n={int(out[f'n_{name}'])} < {MIN_SUBGROUP}; "
                "subgroup metrics are flagged, not interpreted",
                RuntimeWarning,
                stacklevel=2,
            )
    return out


def classification_metrics(
    y: np.ndarray, yhat: np.ndarray, scores: np.ndarray | None = None
) -> dict[str, float]:
    """Accuracy, precision, recall, F1, ROC-AUC and PR-AUC."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y = np.asarray(y).astype(int).ravel()
    yhat = np.asarray(yhat).astype(int).ravel()
    out = {
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
    }
    if scores is not None:
        scores = np.asarray(scores, dtype=float).ravel()
        if len(np.unique(y)) > 1:
            out["roc_auc"] = float(roc_auc_score(y, scores))
            out["pr_auc"] = float(average_precision_score(y, scores))
        else:
            out["roc_auc"] = float("nan")
            out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


def threshold_for_best_f1(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold maximising F1 on the given split. Fixed per model."""
    from sklearn.metrics import f1_score, precision_recall_curve

    y = np.asarray(y).astype(int).ravel()
    scores = np.asarray(scores, dtype=float).ravel()
    if len(np.unique(y)) < 2:
        return 0.5, float("nan")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if len(thresholds) == 0:
        return 0.5, float("nan")
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall + 1e-12),
            0.0,
        )
    f1 = f1[: len(thresholds)]
    best = int(np.nanargmax(f1))
    threshold = float(thresholds[best])
    return threshold, float(f1_score(y, (scores >= threshold).astype(int), zero_division=0))


def top_k_metrics(
    scores: np.ndarray, y: np.ndarray, g: np.ndarray, frac: float = 0.1
) -> dict[str, float]:
    """Gender composition and selection rate at the top-``frac`` cut."""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y).astype(int).ravel()
    g = np.asarray(g).astype(int).ravel()
    n = len(scores)
    k = max(1, int(np.ceil(frac * n)))
    order = np.argsort(-scores, kind="stable")
    top = order[:k]
    selected = np.zeros(n, dtype=int)
    selected[top] = 1
    out = {
        "top_k": float(k),
        "top_share_f": float(g[top].mean()) if k else float("nan"),
        "top_share_m": float(1.0 - g[top].mean()) if k else float("nan"),
        "top_precision": float(y[top].mean()) if k else float("nan"),
        "top_sr_m": _rate(g == 0, selected),
        "top_sr_f": _rate(g == 1, selected),
    }
    out["top_dpd"] = out["top_sr_f"] - out["top_sr_m"]
    return out


def full_metrics(
    y: np.ndarray,
    yhat: np.ndarray,
    scores: np.ndarray,
    g: np.ndarray,
    top_frac: float = 0.1,
) -> dict[str, float]:
    """Task, fairness and top-k metrics in one flat dict."""
    out = classification_metrics(y, yhat, scores)
    out.update(fairness_metrics(y, yhat, g))
    out.update(top_k_metrics(scores, y, g, frac=top_frac))
    return out


def decision(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Apply a stored threshold."""
    return (np.asarray(scores, dtype=float) >= float(threshold)).astype(int)
