#!/usr/bin/env python
"""Train and audit one model variant for one role and seed.

Examples:
    python scripts/run_experiment.py --role nurse --variant B3 --seed 0
    python scripts/run_experiment.py --config configs/nurse_B0.yaml --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.experiment import run_experiment
from hirefair.utils import CONFIG_DIR, Config, get_logger
from hirefair.variants import VARIANTS

LOGGER = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Run config YAML")
    parser.add_argument("--profile", type=str, default=None, help="Compute profile YAML")
    parser.add_argument("--role", choices=list(DEFAULT_ROLES), default=None)
    parser.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-max", type=float, default=None, help="Override adv.lambda_max")
    parser.add_argument("--tag", type=str, default=None, help="Extra directory level under the run dir")
    parser.add_argument("--quick", action="store_true", help="1,000 train rows and 1 epoch")
    parser.add_argument("--force-data", action="store_true", help="Rebuild the tokenised cache")
    parser.add_argument("--data-dir", type=str, default=None, help="Override paths.data_dir")
    parser.add_argument("--results-dir", type=str, default=None, help="Override paths.results_dir")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config
    if config_path is None and args.role and args.variant:
        candidate = CONFIG_DIR / f"{args.role}_{args.variant}.yaml"
        config_path = str(candidate) if candidate.exists() else None

    overrides: dict[str, dict] = {}
    if args.lambda_max is not None:
        overrides["adv"] = {"lambda_max": args.lambda_max}
    if args.data_dir:
        overrides["paths"] = {**overrides.get("paths", {}), "data_dir": args.data_dir}
    if args.results_dir:
        overrides["paths"] = {**overrides.get("paths", {}), "results_dir": args.results_dir}

    cfg = Config.load(config_path, overrides=overrides, quick=args.quick, profile=args.profile)
    if args.role:
        cfg.role = args.role
    if args.variant:
        cfg.variant = args.variant
    if args.seed is not None:
        cfg.seed = args.seed
    if args.tag:
        cfg.tag = args.tag
    cfg.validate()

    LOGGER.info("run dir: %s", cfg.run_dir)
    run_experiment(cfg, force_data=args.force_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
