#!/usr/bin/env python
"""Prepare Bias in Bios and Track B data.

Downloads the dataset once, verifies labels, builds the per-role tokenised
cache, writes Table T1 source data and the synthetic counterfactual set.

Examples:
    python scripts/prepare_data.py --role all
    python scripts/prepare_data.py --role nurse --quick
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES, save_dataset_statistics
from hirefair.data.preprocess import prepare_role
from hirefair.data.synthetic import write_synthetic
from hirefair.utils import Config, get_logger

LOGGER = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="YAML config for data settings")
    parser.add_argument("--profile", type=str, default=None, help="Compute profile YAML")
    parser.add_argument(
        "--role",
        action="append",
        choices=[*DEFAULT_ROLES, "all"],
        help="Role to prepare (repeatable, default: all)",
    )
    parser.add_argument("--quick", action="store_true", help="1,000 train rows for a smoke test")
    parser.add_argument("--force", action="store_true", help="Rebuild cached arrays")
    parser.add_argument("--skip-stats", action="store_true", help="Skip Table T1 statistics")
    parser.add_argument("--skip-synthetic", action="store_true", help="Skip Track B generation")
    parser.add_argument("--synthetic-count", type=int, default=1000, help="Base items per role")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic generator")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    roles = tuple(DEFAULT_ROLES) if not args.role or "all" in args.role else tuple(args.role)
    cfg = Config.load(args.config, quick=args.quick, profile=args.profile)

    for role in roles:
        cfg.role = role
        prepare_role(cfg, role, force=args.force)

    if not args.skip_stats:
        save_dataset_statistics(cfg, roles)

    if not args.skip_synthetic:
        paths = write_synthetic(
            cfg.data_dir / "synthetic", roles=roles, count=args.synthetic_count, seed=args.seed
        )
        for role, path in paths.items():
            LOGGER.info("wrote %s", path)

    LOGGER.info("data preparation complete for %s", ", ".join(roles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
