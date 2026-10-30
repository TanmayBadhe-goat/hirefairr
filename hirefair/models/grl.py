"""Gradient reversal layer (Ganin and Lempitsky, 2015)."""

from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFn(torch.autograd.Function):
    """Identity forward, gradient multiplied by ``-lambda`` backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:  # type: ignore[override]
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.lambda_ * grad_output, None


class GradientReversal(nn.Module):
    """Module wrapper so the trainer can update lambda between steps."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = float(lambda_)

    def set_lambda(self, lambda_: float) -> None:
        """Set the reversal strength for the next backward pass."""
        self.lambda_ = float(lambda_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFn.apply(x, self.lambda_)
