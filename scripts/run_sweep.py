#!/usr/bin/env python
"""Lambda sweep for the adversarially debiased variant (B3 by default).

For each role, train with every ``lambda_max`` in the sweep and write one run
per value. The chosen lambda is logged in ``results/sweep_selection.json`` and
is then used by ``run_all.py`` for B2 and B3.

Selection rule: among values whose validation-selected test F1 is within 3
points of the best sweep F1, choose the smallest absolute TPR gap (EOD). Ties
go to the smaller lambda. The rule is applied to the sweep seed and is not
tuned per metric definition.

Examples:
    python scripts/run_sweep.py --role all
    python scripts/run_sweep.py --quick --role nurse --lambdas 0 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.experiment import run_experiment
from hirefair.utils import CONFIG_DIR, Config, get_logger, read_json, write_json
from hirefair.variants import resolve_variant

LOGGER = get_logger()
DEFAULT_LAMBDAS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0)
F1_TOLERANCE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", nargs="+", choices=[*DEFAULT_ROLES, "all"], default=["all"])
    parser.add_argument("--variant", choices=["B2", "B3"], default="B3")
    parser.add_argument("--seed", type=int, default=0, help="Sweep seed (development seed)")
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roles = tuple(DEFAULT_ROLES) if "all" in args.role else tuple(args.role)
    overrides: dict[str, dict] = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    variant = resolve_variant(args.variant)
    results_root = Config.load(
        None, overrides=overrides, quick=args.quick, profile=args.profile
    ).results_root

    rows = []
    for role in roles:
        for lambda_max in args.lambdas:
            config_path = CONFIG_DIR / f"{role}_{variant.id}.yaml"
            cfg = Config.load(
                config_path if config_path.exists() else None,
                overrides=overrides,
                quick=args.quick,
                profile=args.profile,
            )
            cfg.role = role
            cfg.variant = variant.id
            cfg.seed = args.seed
            cfg.adv.lambda_max = float(lambda_max)
            cfg.tag = f"lambda_{lambda_max:g}"
            payload = run_experiment(cfg)
            test = payload["test"]
            rows.append(
                {
                    "role": role,
                    "variant": variant.id,
                    "seed": args.seed,
                    "lambda_max": float(lambda_max),
                    "f1": test.get("f1"),
                    "eod": test.get("eod"),
                    "eod_abs": test.get("eod_abs"),
                    "dpd": test.get("dpd"),
                    "adv_acc": test.get("adv_acc"),
                    "probe_accuracy": test.get("probe_accuracy"),
                    "probe_auc": test.get("probe_auc"),
                    "flip_rate": payload["counterfactual"]["track_a"].get("flip_rate"),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(results_root / "sweep.csv", index=False)

    selection = {}
    for role, group in frame.groupby("role"):
        best_f1 = float(group["f1"].max())
        eligible = group[group["f1"] >= best_f1 - F1_TOLERANCE]
        chosen = eligible.sort_values(["eod_abs", "lambda_max"]).iloc[0]
        selection[role] = {
            "lambda_max": float(chosen["lambda_max"]),
            "f1": float(chosen["f1"]),
            "eod_abs": float(chosen["eod_abs"]),
            "best_sweep_f1": best_f1,
            "rule": (
                f"F1 within {F1_TOLERANCE:.2f} of best {best_f1:.4f}, "
                "then lowest |EOD|, ties to smaller lambda"
            ),
            "n_eligible": int(len(eligible)),
        }
        LOGGER.info(
            "chosen lambda for %s: %g (F1=%.4f, |EOD|=%.4f)",
            role,
            chosen["lambda_max"],
            chosen["f1"],
            chosen["eod_abs"],
        )
    write_json(results_root / "sweep_selection.json", selection)
    LOGGER.info("wrote %s", results_root / "sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
