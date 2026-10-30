"""Dual-objective training loop with the spec's checkpoint selection rule.

One Adam optimiser covers encoder, predictor and adversary. The reversal layer
creates the opposition, so the total loss is ``L_task + L_adv``.

Selection rule: among epochs whose validation F1 is within 2 points of that
run's best, take the epoch with the lowest validation TPR gap. The decision
threshold is the value that maximises F1 on validation and is then fixed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from torch import nn

from hirefair.data.preprocess import SplitArrays, make_loader
from hirefair.fairness.metrics import (
    decision,
    full_metrics,
    threshold_for_best_f1,
)
from hirefair.training.losses import adversary_loss, task_loss
from hirefair.training.schedule import ganin_lambda
from hirefair.utils import Config, get_logger, resolve_device

LOGGER = get_logger()

F1_TOLERANCE = 0.02


@dataclass
class TrainerResult:
    """Outcome of one training run."""

    history: list[dict[str, Any]] = field(default_factory=list)
    step_trace: list[dict[str, float]] = field(default_factory=list)
    selected_epoch: int = -1
    threshold: float = 0.5
    val_metrics: dict[str, float] = field(default_factory=dict)
    best_val_f1: float = float("nan")
    selection_note: str = ""


def pos_weight_from(labels: np.ndarray) -> float:
    """``n_neg / n_pos`` for the task loss. No resampling."""
    labels = np.asarray(labels).astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0:
        return 1.0
    return negatives / positives


class Trainer:
    """Train a :class:`HireFairModel` with the dual objective."""

    def __init__(self, model: nn.Module, cfg: Config, device: torch.device | None = None):
        self.model = model
        self.cfg = cfg
        self.device = device or resolve_device(cfg.train.device)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        if cfg.train.amp == "auto":
            self.use_amp = self.device.type == "cuda"
        else:
            self.use_amp = bool(cfg.train.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    # ------------------------------------------------------------- helpers

    def _lambda(self, step: int, total_steps: int) -> float:
        return ganin_lambda(
            step, total_steps, self.cfg.adv.lambda_max, self.cfg.adv.schedule
        )

    def _to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}

    # ----------------------------------------------------------- inference

    @torch.no_grad()
    def predict(self, arrays: SplitArrays, with_adversary: bool = True) -> dict[str, np.ndarray]:
        """Score a split. Returns scores, labels, gender and adversary outputs."""
        self.model.eval()
        loader = make_loader(arrays, self.cfg.train.batch, shuffle=False)
        scores: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        genders: list[np.ndarray] = []
        adv_preds: list[np.ndarray] = []
        adv_labels: list[np.ndarray] = []
        for batch in loader:
            batch = self._to_device(batch)
            output = self.model(batch["ids"], batch["mask"], batch["y"])
            scores.append(torch.sigmoid(output["logit"]).detach().cpu().numpy())
            labels.append(batch["y"].detach().cpu().numpy())
            genders.append(batch["g"].detach().cpu().numpy())
            if with_adversary and output["adv_logits"] is not None:
                preds = output["adv_logits"].argmax(dim=-1).detach().cpu().numpy()
                adv_preds.append(preds)
                mask = output["adv_mask"]
                if mask is None:
                    adv_labels.append(batch["g"].detach().cpu().numpy())
                else:
                    adv_labels.append(batch["g"][mask].detach().cpu().numpy())
        result = {
            "scores": np.concatenate(scores) if scores else np.zeros(0),
            "y": np.concatenate(labels).astype(int) if labels else np.zeros(0, dtype=int),
            "g": np.concatenate(genders).astype(int) if genders else np.zeros(0, dtype=int),
        }
        if adv_preds:
            result["adv_pred"] = np.concatenate(adv_preds).astype(int)
            result["adv_true"] = np.concatenate(adv_labels).astype(int)
        return result

    @torch.no_grad()
    def embed(self, arrays: SplitArrays) -> np.ndarray:
        """Return encoder representations for a split (used by the probe)."""
        self.model.eval()
        loader = make_loader(arrays, self.cfg.train.batch, shuffle=False)
        chunks: list[np.ndarray] = []
        for batch in loader:
            batch = self._to_device(batch)
            chunks.append(self.model.encode(batch["ids"], batch["mask"]).detach().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.zeros((0, self.model.encoder.output_dim))

    def adversary_health(self, prediction: dict[str, np.ndarray]) -> dict[str, float]:
        if "adv_pred" not in prediction or len(prediction["adv_pred"]) == 0:
            return {"adv_acc": float("nan"), "adv_balanced_acc": float("nan"), "adv_n": 0.0}
        true = prediction["adv_true"]
        pred = prediction["adv_pred"]
        acc = float((true == pred).mean())
        try:
            balanced = float(balanced_accuracy_score(true, pred))
        except Exception:
            balanced = float("nan")
        return {"adv_acc": acc, "adv_balanced_acc": balanced, "adv_n": float(len(true))}

    # -------------------------------------------------------------- training

    def fit(
        self,
        train_arrays: SplitArrays,
        val_arrays: SplitArrays,
        select: str = "fairness",
    ) -> TrainerResult:
        """Train, then restore the selected epoch's weights.

        ``select="fairness"`` applies the spec's audit rule (lowest val TPR gap
        among epochs within 2 F1 points of the best). ``select="final"`` keeps
        the last epoch instead, which the resume demo uses so its probabilities
        are calibrated rather than taken from the first epoch that already fits
        the training distribution.
        """
        if select not in {"fairness", "final"}:
            raise ValueError("select must be 'fairness' or 'final'")
        self._select = select
        cfg = self.cfg
        train_loader = make_loader(
            train_arrays, cfg.train.batch, shuffle=True, seed=cfg.seed, num_workers=cfg.train.num_workers
        )
        total_steps = max(1, cfg.train.epochs * len(train_loader))
        pos_weight = torch.tensor(
            pos_weight_from(train_arrays.y), dtype=torch.float32, device=self.device
        )
        LOGGER.info(
            "training %s/%s seed=%d on %d rows (pos_weight=%.2f, %d steps)",
            cfg.role,
            cfg.variant,
            cfg.seed,
            len(train_arrays),
            float(pos_weight),
            total_steps,
        )

        result = TrainerResult()
        step = 0
        best_f1 = float("-inf")
        epochs_since_best = 0
        states: dict[int, dict[str, torch.Tensor]] = {}

        for epoch in range(cfg.train.epochs):
            self.model.train()
            task_sum = adv_sum = 0.0
            batches = 0
            adv_correct = adv_total = 0
            for batch in train_loader:
                batch = self._to_device(batch)
                lambda_t = self._lambda(step, total_steps)
                self.model.set_lambda(lambda_t)
                self.optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
                    output = self.model(batch["ids"], batch["mask"], batch["y"])
                    loss_task = task_loss(output["logit"], batch["y"], pos_weight)
                    loss_adv = torch.zeros((), device=self.device)
                    if output["adv_logits"] is not None:
                        mask = output["adv_mask"]
                        adv_targets = batch["g"] if mask is None else batch["g"][mask]
                        loss_adv = adversary_loss(output["adv_logits"], adv_targets)
                        adv_correct += int(
                            (output["adv_logits"].argmax(dim=-1) == adv_targets).sum().item()
                        )
                        adv_total += int(adv_targets.numel())
                    loss = loss_task + loss_adv
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                task_sum += float(loss_task.detach().cpu())
                adv_sum += float(loss_adv.detach().cpu())
                batches += 1
                result.step_trace.append(
                    {
                        "step": float(step),
                        "lambda": float(lambda_t),
                        "task_loss": float(loss_task.detach().cpu()),
                        "adv_loss": float(loss_adv.detach().cpu()),
                    }
                )
                step += 1

            prediction = self.predict(val_arrays)
            threshold, _ = threshold_for_best_f1(prediction["y"], prediction["scores"])
            yhat = decision(prediction["scores"], threshold)
            metrics = full_metrics(prediction["y"], yhat, prediction["scores"], prediction["g"])
            health = self.adversary_health(prediction)
            record = {
                "epoch": epoch,
                "train_task_loss": task_sum / max(1, batches),
                "train_adv_loss": adv_sum / max(1, batches),
                "train_adv_acc": (adv_correct / adv_total) if adv_total else float("nan"),
                "val_threshold": threshold,
                "val_f1": metrics["f1"],
                "val_roc_auc": metrics["roc_auc"],
                "val_eod": metrics["eod"],
                "val_eod_abs": metrics["eod_abs"],
                "val_dpd": metrics["dpd"],
                "val_dpd_abs": metrics["dpd_abs"],
                "val_tpr_m": metrics["tpr_m"],
                "val_tpr_f": metrics["tpr_f"],
                "val_sr_m": metrics["sr_m"],
                "val_sr_f": metrics["sr_f"],
                "lambda_last": result.step_trace[-1]["lambda"] if result.step_trace else 0.0,
                **health,
            }
            result.history.append(record)
            states[epoch] = {
                key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()
            }
            LOGGER.info(
                "epoch %d: task_loss=%.4f adv_loss=%.4f val_f1=%.4f |EOD|=%.4f adv_acc=%.3f",
                epoch,
                record["train_task_loss"],
                record["train_adv_loss"],
                record["val_f1"],
                record["val_eod_abs"],
                record["adv_acc"],
            )

            if record["val_f1"] > best_f1 + 1e-6:
                best_f1 = record["val_f1"]
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if cfg.train.patience and epochs_since_best >= cfg.train.patience:
                    LOGGER.info("early stop after %d epochs without val F1 gain", epochs_since_best)
                    break

        result.best_val_f1 = best_f1
        eligible = [r for r in result.history if r["val_f1"] >= best_f1 - F1_TOLERANCE]
        if select == "final":
            selected = result.history[-1]
            result.selection_note = (
                f"selected final epoch {selected['epoch']} of {len(result.history)} "
                f"(val F1 {selected['val_f1']:.4f}, best {best_f1:.4f})"
            )
        else:
            ranked = sorted(
                eligible,
                key=lambda r: (
                    float(np.nan_to_num(r["val_eod_abs"], nan=float("inf"))),
                    r["epoch"],
                ),
            )
            selected = ranked[0]
            result.selection_note = (
                f"selected epoch {selected['epoch']} among {len(eligible)} within "
                f"{F1_TOLERANCE:.2f} F1 of best {best_f1:.4f}, lowest val |TPR gap| "
                f"{selected['val_eod_abs']:.4f}"
            )
        result.selected_epoch = int(selected["epoch"])
        self.model.load_state_dict(states[result.selected_epoch])
        self.model.set_lambda(0.0)

        final = self.predict(val_arrays)
        threshold, _ = threshold_for_best_f1(final["y"], final["scores"])
        yhat = decision(final["scores"], threshold)
        result.threshold = float(threshold)
        result.val_metrics = full_metrics(final["y"], yhat, final["scores"], final["g"])
        result.val_metrics.update(self.adversary_health(final))
        result.val_metrics["val_f1"] = result.val_metrics["f1"]
        LOGGER.info(result.selection_note)
        return result


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Deep copy of a model's weights (helper for callers)."""
    return copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
