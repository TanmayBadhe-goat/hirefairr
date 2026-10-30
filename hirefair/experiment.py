"""One experiment: prepare data, train, audit, save artefacts.

A run is fully described by a :class:`~hirefair.utils.Config` (YAML + seed).
This module performs the whole pipeline so the scripts stay thin and the
notebook only calls the scripts.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hirefair.analysis.gender_tokens import probe_gender
from hirefair.data.preprocess import (
    SplitArrays,
    encode_texts,
    load_split,
    load_train_arrays,
    load_vocab,
    make_loader,
    prepare_role,
)
from hirefair.fairness.counterfactual import summary as counterfactual_summary
from hirefair.fairness.metrics import decision, full_metrics
from hirefair.models.hirefair_model import HireFairModel, save_checkpoint
from hirefair.training.trainer import Trainer
from hirefair.utils import (
    Config,
    get_logger,
    git_commit,
    read_jsonl,
    resolve_device,
    set_seed,
    write_json,
)
from hirefair.variants import resolve_variant

LOGGER = get_logger()


def score_texts(
    model: HireFairModel,
    texts: list[str],
    vocab: dict[str, int],
    max_len: int,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Score raw texts with a trained model."""
    ids, mask, lengths = encode_texts(texts, vocab, max_len)
    arrays = SplitArrays(
        ids=ids,
        mask=mask,
        lengths=lengths,
        y=np.zeros(len(texts), dtype=np.int8),
        g=np.zeros(len(texts), dtype=np.int8),
    )
    loader = make_loader(arrays, batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            out = model.score(batch["ids"].to(device), batch["mask"].to(device))
            scores.append(out.detach().cpu().numpy())
    return np.concatenate(scores) if scores else np.zeros(0)


def _track_b(
    cfg: Config,
    model: HireFairModel,
    vocab: dict[str, int],
    threshold: float,
    device: torch.device,
    limit: int = 1000,
) -> dict[str, float]:
    """Counterfactual results on the synthetic resume pairs."""
    path = cfg.data_dir / "synthetic" / f"{cfg.role}.jsonl"
    if not path.exists():
        LOGGER.warning("no synthetic data at %s; skipping Track B", path)
        return {}
    rows = read_jsonl(path)[:limit]
    if not rows:
        return {}
    male = [row["resume_M"] for row in rows]
    female = [row["resume_F"] for row in rows]
    neutral = [row["resume_neutral"] for row in rows]
    scores_m = score_texts(model, male, vocab, cfg.data.max_len, device)
    scores_f = score_texts(model, female, vocab, cfg.data.max_len, device)
    scores_n = score_texts(model, neutral, vocab, cfg.data.max_len, device)
    dec_m = decision(scores_m, threshold)
    dec_f = decision(scores_f, threshold)
    stats = counterfactual_summary(scores_m, scores_f, dec_m, dec_f)
    stats.update(
        {
            "mean_score_M": float(scores_m.mean()),
            "mean_score_F": float(scores_f.mean()),
            "mean_score_neutral": float(scores_n.mean()),
            "decision_rate_M": float(dec_m.mean()),
            "decision_rate_F": float(dec_f.mean()),
        }
    )
    return stats


def run_experiment(cfg: Config, force_data: bool = False) -> dict[str, Any]:
    """Execute one run and write all artefacts under the run directory.

    Returns the payload written to ``run.json``.
    """
    variant = resolve_variant(cfg.variant)
    if cfg.adv.mode != variant.adv_mode:
        LOGGER.info(
            "variant %s fixes adv.mode=%s (config said %s)",
            variant.id,
            variant.adv_mode,
            cfg.adv.mode,
        )
    cfg.adv.mode = variant.adv_mode
    set_seed(cfg.seed)
    started = time.time()
    device = resolve_device(cfg.train.device)

    meta = prepare_role(cfg, cfg.role, force=force_data)
    processed = cfg.processed_dir(cfg.role)
    vocab = load_vocab(processed / "vocab.json")

    train_arrays = load_train_arrays(processed, variant.text_mode, augment=variant.augment)
    val_arrays = load_split(processed, "val", variant.text_mode)
    test_arrays = load_split(processed, "test", variant.text_mode)
    if variant.text_mode == "raw":
        swapped_arrays = load_split(processed, "test", "swapped")
    else:
        # Masked input cannot change when cues are swapped: use the same arrays
        # so the measured delta is exactly zero rather than fabricated.
        swapped_arrays = test_arrays

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
    train_started = time.time()
    result = trainer.fit(train_arrays, val_arrays)
    train_seconds = time.time() - train_started

    prediction = trainer.predict(test_arrays)
    yhat = decision(prediction["scores"], result.threshold)
    test_metrics = full_metrics(prediction["y"], yhat, prediction["scores"], prediction["g"])
    test_metrics.update(trainer.adversary_health(prediction))

    probe = probe_gender(trainer.embed(test_arrays), prediction["g"], seed=cfg.seed)
    test_metrics.update(probe)

    swapped_prediction = trainer.predict(swapped_arrays)
    swapped_yhat = decision(swapped_prediction["scores"], result.threshold)
    track_a = counterfactual_summary(
        prediction["scores"],
        swapped_prediction["scores"],
        yhat,
        swapped_yhat,
    )
    for label, mask in (("M", prediction["g"] == 0), ("F", prediction["g"] == 1)):
        subset = counterfactual_summary(
            prediction["scores"][mask],
            swapped_prediction["scores"][mask],
            yhat[mask],
            swapped_yhat[mask],
        )
        track_a[f"flip_rate_{label}"] = subset["flip_rate"]
        track_a[f"mean_abs_score_delta_{label}"] = subset["mean_abs_score_delta"]

    track_b = _track_b(cfg, model, vocab, result.threshold, device)

    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metrics = {"test": test_metrics, "val": result.val_metrics}
    save_checkpoint(
        run_dir / "model.pt", model, result.threshold, vocab, cfg.to_dict(), checkpoint_metrics
    )
    write_json(run_dir / "vocab.json", vocab)
    cfg.save(run_dir / "config.yaml")
    np.savez_compressed(
        run_dir / "predictions.npz",
        scores=prediction["scores"],
        swapped_scores=swapped_prediction["scores"],
        y=prediction["y"],
        g=prediction["g"],
        yhat=yhat,
        swapped_yhat=swapped_yhat,
    )

    payload: dict[str, Any] = {
        "run": {
            "role": cfg.role,
            "variant": cfg.variant,
            "variant_name": variant.name,
            "seed": cfg.seed,
            "tag": cfg.tag,
            "quick": cfg.quick,
            "config": cfg.to_dict(),
            "git_commit": git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "train_seconds": train_seconds,
            "total_seconds": time.time() - started,
        },
        "data": {
            "counts": meta.get("counts", {}),
            "vocab_size": len(vocab),
            "train_rows_used": len(train_arrays),
            "test_rows": len(test_arrays),
        },
        "training": {
            "history": result.history,
            "step_trace": result.step_trace,
            "selected_epoch": result.selected_epoch,
            "selection_note": result.selection_note,
            "best_val_f1": result.best_val_f1,
            "threshold": result.threshold,
            "lambda_max": cfg.adv.lambda_max,
            "adv_mode": cfg.adv.mode,
        },
        "val": result.val_metrics,
        "test": test_metrics,
        "counterfactual": {"track_a": track_a, "track_b": track_b},
    }
    write_json(run_dir / "run.json", payload)
    LOGGER.info(
        "%s %s seed %d: F1=%.4f |EOD|=%.4f DIR=%.3f probe_acc=%.3f flip=%.3f (%.1fs)",
        cfg.role,
        cfg.variant,
        cfg.seed,
        test_metrics.get("f1", float("nan")),
        test_metrics.get("eod_abs", float("nan")),
        test_metrics.get("dir", float("nan")),
        test_metrics.get("probe_accuracy", float("nan")),
        track_a.get("flip_rate", float("nan")),
        payload["run"]["total_seconds"],
    )
    return payload
