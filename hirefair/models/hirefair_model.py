"""HireFair model: one encoder, a predictor head and a reversed adversary head.

The encoder minimises ``L_task - lambda * L_adv``. With the gradient reversal
layer between ``z`` and the adversary, the total loss can be written as
``L_task + L_adv`` while the encoder still receives the reversed gradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from hirefair.models.encoder import Encoder
from hirefair.models.grl import GradientReversal
from hirefair.models.heads import AdversaryHead, PredictorHead

ADV_MODES = {"none", "dp", "eo", "eodds"}


class HireFairModel(nn.Module):
    """Encoder with a job-fit predictor and an adversarial gender head."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 100,
        hidden: int = 128,
        dropout: float = 0.3,
        adv_mode: str = "none",
        glove_path: str | None = None,
        vocab: dict[str, int] | None = None,
    ):
        super().__init__()
        if adv_mode not in ADV_MODES:
            raise ValueError(f"adv_mode must be one of {sorted(ADV_MODES)}")
        self.adv_mode = adv_mode
        self.encoder = Encoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden=hidden,
            dropout=dropout,
            glove_path=glove_path,
            vocab=vocab,
        )
        z_dim = self.encoder.output_dim
        self.predictor = PredictorHead(z_dim, hidden=64, dropout=dropout)
        adv_in_dim = z_dim + 1 if adv_mode == "eodds" else z_dim
        self.adversary = AdversaryHead(adv_in_dim, hidden=64, n_classes=2) if adv_mode != "none" else None
        self.grl = GradientReversal(lambda_=0.0)

    def set_lambda(self, lambda_: float) -> None:
        """Update the reversal strength for the next backward pass."""
        self.grl.set_lambda(lambda_)

    def forward(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Run the encoder and both heads.

        The adversary is applied to all rows in ``dp``/``eodds`` mode and to
        positive rows only in ``eo`` mode. When ``y`` is omitted, ``eo`` falls
        back to all rows so that held-out diagnostics can still run.
        """
        z = self.encoder(ids, mask)
        logit = self.predictor(z)
        adv_logits: torch.Tensor | None = None
        adv_mask: torch.Tensor | None = None

        if self.adversary is not None:
            adv_features = z
            if self.adv_mode == "eo" and y is not None:
                adv_mask = y > 0.5
                adv_features = z[adv_mask]
            if self.adv_mode == "eodds" and y is not None:
                adv_features = torch.cat([z, y.unsqueeze(-1)], dim=-1)
            if adv_features.shape[0] > 0:
                adv_logits = self.adversary(self.grl(adv_features))
        return {"z": z, "logit": logit, "adv_logits": adv_logits, "adv_mask": adv_mask}

    @torch.no_grad()
    def score(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return job-fit probabilities for a batch."""
        self.eval()
        return torch.sigmoid(self.predictor(self.encoder(ids, mask)))

    @torch.no_grad()
    def encode(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return the frozen encoder representation."""
        self.eval()
        return self.encoder(ids, mask)


@dataclass
class Checkpoint:
    """Everything needed to reload a run for the demo or analysis."""

    model: HireFairModel
    threshold: float
    vocab: dict[str, int]
    config: dict[str, Any]
    metrics: dict[str, Any]


def save_checkpoint(
    path: str | Path,
    model: HireFairModel,
    threshold: float,
    vocab: dict[str, int],
    config: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Write a self-describing checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "threshold": float(threshold),
            "vocab": vocab,
            "config": config,
            "metrics": metrics,
            "adv_mode": model.adv_mode,
        },
        path,
    )
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    config = payload["config"]
    vocab = payload["vocab"]
    model_cfg = config.get("model", {})
    model = HireFairModel(
        vocab_size=len(vocab),
        emb_dim=int(model_cfg.get("emb_dim", 100)),
        hidden=int(model_cfg.get("hidden", 128)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        adv_mode=payload.get("adv_mode", "none"),
        vocab=vocab,
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return Checkpoint(
        model=model,
        threshold=float(payload["threshold"]),
        vocab=vocab,
        config=config,
        metrics=payload.get("metrics", {}),
    )
