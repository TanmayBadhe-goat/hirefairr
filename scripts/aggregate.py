#!/usr/bin/env python
"""Aggregate every finished run into ``results/summary.csv``.

One row per (role, variant, seed). Sweep runs live one directory deeper and are
excluded; they are summarised separately in ``results/sweep.csv``.

Examples:
    python scripts/aggregate.py
    python scripts/aggregate.py --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.utils import Config, get_logger, read_json

LOGGER = get_logger()

COLUMNS = [
    "role",
    "variant",
    "seed",
    "tag",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "n_m",
    "n_f",
    "n_pos_m",
    "n_pos_f",
    "sr_m",
    "sr_f",
    "dpd",
    "dpd_abs",
    "dir",
    "tpr_m",
    "tpr_f",
    "eod",
    "eod_abs",
    "fpr_gap",
    "top_share_f",
    "top_sr_m",
    "top_sr_f",
    "adv_acc",
    "adv_balanced_acc",
    "probe_accuracy",
    "probe_auc",
    "majority_baseline",
    "flip_rate",
    "mean_abs_score_delta",
    "mean_rank_shift",
    "top_k_changes",
    "track_b_flip_rate",
    "track_b_mean_abs_delta",
    "track_b_mean_rank_shift",
    "lambda_max",
    "adv_mode",
    "threshold",
    "selected_epoch",
    "train_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Aggregate results_quick instead")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    overrides: dict[str, dict] = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    root = cfg.results_root

    rows = []
    for path in sorted(root.glob("**/run.json")):
        payload = read_json(path)
        run = payload.get("run", {})
        if str(run.get("tag") or "").startswith("lambda_"):
            continue
        test = payload.get("test", {})
        track_a = payload.get("counterfactual", {}).get("track_a", {})
        track_b = payload.get("counterfactual", {}).get("track_b", {})
        training = payload.get("training", {})
        rows.append(
            {
                "role": run.get("role"),
                "variant": run.get("variant"),
                "seed": run.get("seed"),
                "tag": run.get("tag"),
                "f1": test.get("f1"),
                "accuracy": test.get("accuracy"),
                "precision": test.get("precision"),
                "recall": test.get("recall"),
                "roc_auc": test.get("roc_auc"),
                "pr_auc": test.get("pr_auc"),
                "n_m": test.get("n_m"),
                "n_f": test.get("n_f"),
                "n_pos_m": test.get("n_pos_m"),
                "n_pos_f": test.get("n_pos_f"),
                "sr_m": test.get("sr_m"),
                "sr_f": test.get("sr_f"),
                "dpd": test.get("dpd"),
                "dpd_abs": test.get("dpd_abs"),
                "dir": test.get("dir"),
                "tpr_m": test.get("tpr_m"),
                "tpr_f": test.get("tpr_f"),
                "eod": test.get("eod"),
                "eod_abs": test.get("eod_abs"),
                "fpr_gap": test.get("fpr_gap"),
                "top_share_f": test.get("top_share_f"),
                "top_sr_m": test.get("top_sr_m"),
                "top_sr_f": test.get("top_sr_f"),
                "adv_acc": test.get("adv_acc"),
                "adv_balanced_acc": test.get("adv_balanced_acc"),
                "probe_accuracy": test.get("probe_accuracy"),
                "probe_auc": test.get("probe_auc"),
                "majority_baseline": test.get("majority_baseline"),
                "flip_rate": track_a.get("flip_rate"),
                "mean_abs_score_delta": track_a.get("mean_abs_score_delta"),
                "mean_rank_shift": track_a.get("mean_rank_shift"),
                "top_k_changes": track_a.get("top_k_changes"),
                "track_b_flip_rate": track_b.get("flip_rate"),
                "track_b_mean_abs_delta": track_b.get("mean_abs_score_delta"),
                "track_b_mean_rank_shift": track_b.get("mean_rank_shift"),
                "lambda_max": training.get("lambda_max"),
                "adv_mode": training.get("adv_mode"),
                "threshold": training.get("threshold"),
                "selected_epoch": training.get("selected_epoch"),
                "train_seconds": run.get("train_seconds"),
            }
        )

    frame = pd.DataFrame(rows, columns=COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["role", "variant", "seed"]).reset_index(drop=True)
    out = root / "summary.csv"
    frame.to_csv(out, index=False)
    LOGGER.info("wrote %s with %d rows", out, len(frame))
    if frame.empty:
        LOGGER.warning("no runs found under %s", root)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
