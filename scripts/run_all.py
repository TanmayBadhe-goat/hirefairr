#!/usr/bin/env python
"""Run the experiment grid: roles x variants x seeds.

Examples:
    python scripts/run_all.py --role all --variant B0 B1 B2 B3 --seed 0 1 2 3 4
    python scripts/run_all.py --quick --role nurse --variant B0 B3 --seed 0
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.experiment import run_experiment
from hirefair.utils import CONFIG_DIR, Config, get_logger, read_json
from hirefair.variants import VARIANTS

LOGGER = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", nargs="+", choices=[*DEFAULT_ROLES, "all"], default=["all"])
    parser.add_argument("--variant", nargs="+", choices=sorted(VARIANTS), default=["B0", "B1", "B2", "B3"])
    parser.add_argument("--seed", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--profile", type=str, default=None, help="Compute profile YAML")
    parser.add_argument("--quick", action="store_true", help="Smoke test: 1,000 rows, 1 epoch")
    parser.add_argument("--skip-existing", action="store_true", help="Skip runs that already have run.json")
    parser.add_argument("--no-sweep-lambda", action="store_true", help="Ignore the selected sweep lambda")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    return parser.parse_args()


def selected_lambdas(results_dir: Path) -> dict[str, float]:
    """Read the chosen lambda per role from the sweep, if it has been run."""
    path = Path(results_dir) / "sweep_selection.json"
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return {role: float(item["lambda_max"]) for role, item in payload.items()}


def main() -> int:
    args = parse_args()
    roles = tuple(DEFAULT_ROLES) if "all" in args.role else tuple(args.role)
    variants = tuple(args.variant)
    overrides: dict[str, dict] = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir

    probe_cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    lambdas = {} if args.no_sweep_lambda else selected_lambdas(probe_cfg.results_root)
    if lambdas:
        LOGGER.info("using sweep-selected lambda per role: %s", lambdas)

    started = time.time()
    failures: list[str] = []
    completed = 0
    for role in roles:
        for variant in variants:
            for seed in args.seed:
                config_path = CONFIG_DIR / f"{role}_{variant}.yaml"
                cfg = Config.load(
                    config_path if config_path.exists() else None,
                    overrides=overrides,
                    quick=args.quick,
                    profile=args.profile,
                )
                cfg.role = role
                cfg.variant = variant
                cfg.seed = seed
                if lambdas and variant in {"B2", "B3"}:
                    cfg.adv.lambda_max = lambdas[role]
                run_json = cfg.run_dir / "run.json"
                if args.skip_existing and run_json.exists():
                    LOGGER.info("skip existing %s", run_json)
                    continue
                try:
                    run_experiment(cfg)
                    completed += 1
                except Exception as exc:  # noqa: BLE001 - report and continue
                    failures.append(f"{role}/{variant}/seed{seed}: {exc}")
                    LOGGER.error("run failed: %s", failures[-1])
                    traceback.print_exc()
    elapsed = time.time() - started
    LOGGER.info(
        "grid complete: %d runs, %d failures, %.1f minutes",
        completed,
        len(failures),
        elapsed / 60,
    )
    for failure in failures:
        LOGGER.error("FAILED %s", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
