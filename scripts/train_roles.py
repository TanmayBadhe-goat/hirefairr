#!/usr/bin/env python3
"""Prepare and train the three HireFair 2.0 demo roles.

Builds the role-labelled resume corpus (see :mod:`hirefair.data.roles`), trains
a baseline (B0) and an adversarial equal-opportunity (B3) checkpoint per role
with the project's own :class:`hirefair.training.trainer.Trainer`, evaluates
them on a held-out split, and writes ``results/hirefair2/summary.csv`` for the
Fairness Audit tab.

Artefacts per run, mirroring ``scripts/run_experiment.py``:

    results/hirefair2/{role}/{variant}/seed{n}/model.pt
    results/hirefair2/{role}/{variant}/seed{n}/vocab.json
    results/hirefair2/{role}/{variant}/seed{n}/run.json
    results/hirefair2/{role}/{variant}/seed{n}/predictions.npz

Usage:
    python scripts/train_roles.py                       # all roles, B0 and B3
    python scripts/train_roles.py --roles data_analyst   # one role
    python scripts/train_roles.py --prepare-only         # data only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.preprocess import load_vocab
from hirefair.data.roles import (
    DEMO_ROLES,
    DEMO_VARIANTS,
    build_role_frames,
    load_role_split,
    prepare_role_arrays,
    processed_dir,
    read_corpus,
    synthetic_path,
)
from hirefair.data.swap import swap_many
from hirefair.experiment import score_texts
from hirefair.fairness.counterfactual import summary as counterfactual_summary
from hirefair.fairness.metrics import decision, full_metrics
from hirefair.models.hirefair_model import HireFairModel, save_checkpoint
from hirefair.training.trainer import Trainer
from hirefair.utils import (
    PROJECT_ROOT,
    Config,
    get_logger,
    read_jsonl,
    resolve_device,
    set_seed,
    write_json,
)
from hirefair.variants import resolve_variant

LOGGER = get_logger()

RESULTS_DIR = PROJECT_ROOT / "results" / "hirefair2"

SUMMARY_COLUMNS = [
    "role",
    "variant",
    "seed",
    "f1",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
    "sr_m",
    "sr_f",
    "dpd",
    "dpd_abs",
    "tpr_m",
    "tpr_f",
    "eod",
    "eod_abs",
    "flip_rate",
    "mean_abs_score_delta",
    "adv_acc",
    "adv_balanced_acc",
    "lambda_max",
    "adv_mode",
    "threshold",
    "selected_epoch",
    "train_seconds",
]


def _run_config(role: str, variant: str, epochs: int) -> Config:
    """Config for one resume-corpus run."""
    cfg = Config(role=role, variant=variant, seed=0)
    resolved = resolve_variant(variant)
    cfg.adv.mode = resolved.adv_mode
    cfg.train.epochs = epochs
    # No early stop: the demo keeps the final epoch so its probabilities are
    # calibrated on the role corpus (see Trainer.fit's ``select`` argument).
    cfg.train.patience = 0
    cfg.train.batch = 64
    cfg.train.device = "auto"
    cfg.data.max_len = 128
    cfg.validate()
    return cfg


def train_one(role: str, variant: str, epochs: int, results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    """Train, evaluate and save one role/variant checkpoint."""
    started = time.time()
    cfg = _run_config(role, variant, epochs)
    set_seed(cfg.seed)
    device = resolve_device(cfg.train.device)

    meta = prepare_role_arrays(role, max_len=cfg.data.max_len)
    directory = processed_dir(role)
    vocab = load_vocab(directory / "vocab.json")
    train_arrays = load_role_split(role, "train")
    val_arrays = load_role_split(role, "val")
    test_arrays = load_role_split(role, "test")

    model = HireFairModel(
        vocab_size=len(vocab),
        emb_dim=cfg.model.emb_dim,
        hidden=cfg.model.hidden,
        dropout=cfg.model.dropout,
        adv_mode=cfg.adv.mode,
        glove_path=cfg.model.glove_path,
        vocab=vocab,
    )
    trainer = Trainer(model, cfg, device)
    result = trainer.fit(train_arrays, val_arrays, select="final")

    prediction = trainer.predict(test_arrays)
    yhat = decision(prediction["scores"], result.threshold)
    metrics = full_metrics(prediction["y"], yhat, prediction["scores"], prediction["g"])
    metrics.update(trainer.adversary_health(prediction))

    # Counterfactual check on the held-out resume summaries: swap gender cues
    # and re-score with the same checkpoint.
    frames = build_role_frames(role)
    swapped_texts = swap_many([str(row["text"]) for row in frames["test"]], mode="swap", seed=0)
    swapped_scores = score_texts(
        model, swapped_texts, vocab, cfg.data.max_len, next(model.parameters()).device
    )
    swapped_yhat = decision(swapped_scores, result.threshold)
    track_a = counterfactual_summary(prediction["scores"], swapped_scores, yhat, swapped_yhat)

    run_dir = results_dir / role / variant / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        run_dir / "model.pt",
        model,
        result.threshold,
        vocab,
        cfg.to_dict(),
        {"test": metrics, "val": result.val_metrics},
    )
    write_json(run_dir / "vocab.json", vocab)
    np.savez_compressed(
        run_dir / "predictions.npz",
        scores=prediction["scores"],
        swapped_scores=swapped_scores,
        y=prediction["y"],
        g=prediction["g"],
        yhat=yhat,
        swapped_yhat=swapped_yhat,
    )

    train_seconds = time.time() - started
    payload = {
        "run": {
            "role": role,
            "variant": variant,
            "seed": cfg.seed,
            "device": str(device),
            "train_seconds": train_seconds,
            "corpus": [synthetic_path(name).name for name in DEMO_ROLES],
        },
        "data": meta,
        "training": {
            "history": result.history,
            "selected_epoch": result.selected_epoch,
            "selection_note": result.selection_note,
            "best_val_f1": result.best_val_f1,
            "threshold": result.threshold,
            "lambda_max": cfg.adv.lambda_max,
            "adv_mode": cfg.adv.mode,
        },
        "test": metrics,
        "counterfactual": track_a,
    }
    write_json(run_dir / "run.json", payload)
    LOGGER.info(
        "%s %s: F1=%.4f |EOD|=%.4f flip=%.3f (%.1fs)",
        role,
        variant,
        metrics.get("f1", float("nan")),
        metrics.get("eod_abs", float("nan")),
        track_a.get("flip_rate", float("nan")),
        train_seconds,
    )

    row = {
        "role": role,
        "variant": variant,
        "seed": cfg.seed,
        "lambda_max": cfg.adv.lambda_max,
        "adv_mode": cfg.adv.mode,
        "threshold": result.threshold,
        "selected_epoch": result.selected_epoch,
        "train_seconds": train_seconds,
    }
    for column in SUMMARY_COLUMNS:
        if column in row:
            continue
        row[column] = metrics.get(column, track_a.get(column, float("nan")))
    return row


def _merge_summary(rows: list[dict[str, Any]], results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    """Merge new rows into ``summary.csv``, replacing the same role/variant/seed."""
    path = results_dir / "summary.csv"
    if path.exists():
        existing = pd.read_csv(path)
        existing = existing[
            ~existing.apply(
                lambda row: any(
                    row["role"] == new["role"]
                    and row["variant"] == new["variant"]
                    and int(row["seed"]) == int(new["seed"])
                    for new in rows
                ),
                axis=1,
            )
        ]
        frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        frame = pd.DataFrame(rows)
    frame = frame.sort_values(["role", "variant", "seed"]).reset_index(drop=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--roles", nargs="+", default=list(DEMO_ROLES), choices=list(DEMO_ROLES))
    parser.add_argument("--variants", nargs="+", default=list(DEMO_VARIANTS), choices=["B0", "B3"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--force-prepare", action="store_true", help="Rebuild cached arrays")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the arrays")
    parser.add_argument(
        "--results-dir", type=Path, default=RESULTS_DIR, help="Where to write checkpoints"
    )
    args = parser.parse_args()

    for role in args.roles:
        prepare_role_arrays(role, force=args.force_prepare)
    if args.prepare_only:
        print(f"Prepared arrays for {', '.join(args.roles)}")
        return 0

    rows: list[dict[str, Any]] = []
    for role in args.roles:
        for variant in args.variants:
            print(f"\n=== {role} / {variant} ===")
            rows.append(train_one(role, variant, args.epochs, args.results_dir))

    frame = _merge_summary(rows, args.results_dir)
    print(f"\nWrote {args.results_dir / 'summary.csv'} ({len(frame)} rows)")
    for row in rows:
        print(
            f"  {row['role']:<26} {row['variant']}  F1={row['f1']:.3f}  "
            f"|EOD|={row['eod_abs']:.3f}  flip={row['flip_rate']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
