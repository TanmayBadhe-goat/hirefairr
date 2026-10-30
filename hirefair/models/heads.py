"""Predictor and adversary heads."""

from __future__ import annotations

from torch import nn


class PredictorHead(nn.Module):
    """Linear(256,64)-ReLU-Dropout-Linear(64,1) -> job-fit logit."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


class AdversaryHead(nn.Module):
    """Linear(256,64)-ReLU-Dropout-Linear(64,2) -> gender logits."""

    def __init__(self, in_dim: int, hidden: int = 64, n_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, z):
        return self.net(z)
