#!/usr/bin/env python
"""Gender-associated token analysis for trained checkpoints.

For each role and variant (seed 0 by default) this writes
``results/analysis/gender_tokens_{role}_{variant}.json`` containing:

* log-odds tokens by gender over the training split,
* the gender direction in the embedding table and the top projections,
* the embedding bias score,
* the overlap list,
* probe accuracy copied from the run's test metrics.

Examples:
    python scripts/analyze_gender_tokens.py --role all --variant B0 B1 B2 B3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.analysis.gender_tokens import (
    embedding_bias_score,
    gender_direction,
    log_odds_ratio,
    overlap_tokens,
    project_vocabulary,
)
from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.data.preprocess import load_split, load_vocab
from hirefair.models.hirefair_model import load_checkpoint
from hirefair.utils import Config, get_logger, read_json, write_json

LOGGER = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", nargs="+", choices=[*DEFAULT_ROLES, "all"], default=["all"])
    parser.add_argument("--variant", nargs="+", default=["B0", "B1", "B2", "B3"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-freq", type=int, default=20, help="Minimum token frequency for projections")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def counts_by_gender(cfg: Config, role: str) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """Token counts for male rows, female rows and both, from cached train ids."""
    arrays = load_split(cfg.processed_dir(role), "train", "raw")
    vocab = load_vocab(cfg.processed_dir(role) / "vocab.json")
    index_to_token = {index: token for token, index in vocab.items()}
    male: dict[str, int] = {}
    female: dict[str, int] = {}
    for row in range(len(arrays)):
        target = female if arrays.g[row] == 1 else male
        length = int(arrays.lengths[row])
        for token_id in arrays.ids[row, :length]:
            token = index_to_token.get(int(token_id))
            if token is None or token.startswith("<"):
                continue
            target[token] = target.get(token, 0) + 1
    total = dict(male)
    for token, count in female.items():
        total[token] = total.get(token, 0) + count
    return male, female, total


def main() -> int:
    args = parse_args()
    roles = tuple(DEFAULT_ROLES) if "all" in args.role else tuple(args.role)
    overrides: dict[str, dict] = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    cfg.seed = args.seed
    analysis_dir = cfg.results_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for role in roles:
        cfg.role = role
        male_counts, female_counts, total_counts = counts_by_gender(cfg, role)
        odds = log_odds_ratio(male_counts, female_counts, top_n=50)
        for variant in args.variant:
            run_dir = cfg.results_root / role / variant / f"seed{args.seed}"
            checkpoint_path = run_dir / "model.pt"
            run_path = run_dir / "run.json"
            if not checkpoint_path.exists():
                LOGGER.warning("missing checkpoint %s; skipping", checkpoint_path)
                continue
            checkpoint = load_checkpoint(checkpoint_path)
            embedding = checkpoint.model.encoder.embedding.weight.detach().cpu().numpy()
            direction = gender_direction(embedding, checkpoint.vocab)
            projection = project_vocabulary(
                embedding,
                direction,
                checkpoint.vocab,
                counts=total_counts,
                min_freq=args.min_freq,
                top_n=50,
            )
            bias = embedding_bias_score(embedding, direction, checkpoint.vocab)
            payload = {
                "role": role,
                "variant": variant,
                "seed": args.seed,
                "vocab_size": len(checkpoint.vocab),
                "n_tokens_male": int(sum(male_counts.values())),
                "n_tokens_female": int(sum(female_counts.values())),
                "log_odds": odds,
                "projection": projection,
                "embedding_bias": bias,
                "overlap": overlap_tokens(
                    odds["female"], odds["male"], projection["female"], projection["male"]
                ),
                "probe": {},
            }
            if run_path.exists():
                run = read_json(run_path)
                test = run.get("test", {})
                payload["probe"] = {
                    "accuracy": test.get("probe_accuracy"),
                    "auc": test.get("probe_auc"),
                    "majority_baseline": test.get("majority_baseline"),
                }
                payload["f1"] = test.get("f1")
                payload["eod_abs"] = test.get("eod_abs")
            write_json(analysis_dir / f"gender_tokens_{role}_{variant}.json", payload)
            summary_rows.append(
                {
                    "role": role,
                    "variant": variant,
                    "seed": args.seed,
                    "embedding_bias_mean_abs": bias.get("mean_abs_projection"),
                    "probe_accuracy": payload["probe"].get("accuracy"),
                    "probe_auc": payload["probe"].get("auc"),
                    "majority_baseline": payload["probe"].get("majority_baseline"),
                    "top_female_log_odds_token": odds["female"][0]["token"] if odds["female"] else None,
                    "top_male_log_odds_token": odds["male"][0]["token"] if odds["male"] else None,
                    "f1": payload.get("f1"),
                    "eod_abs": payload.get("eod_abs"),
                }
            )
            LOGGER.info(
                "%s/%s: bias score %.4f, probe acc %.3f",
                role,
                variant,
                float(bias.get("mean_abs_projection", float("nan"))),
                float(payload["probe"].get("accuracy") or float("nan")),
            )

    frame = pd.DataFrame(summary_rows)
    out = analysis_dir / "gender_tokens_summary.csv"
    frame.to_csv(out, index=False)
    LOGGER.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
