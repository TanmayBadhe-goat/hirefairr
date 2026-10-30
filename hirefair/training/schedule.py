"""Lambda schedules for the gradient reversal layer."""

from __future__ import annotations

import math


def ganin_lambda(
    step: int, total_steps: int, lambda_max: float, schedule: str = "ganin"
) -> float:
    """Reversal strength at a training step.

    ``ganin`` follows ``lambda_max * (2 / (1 + exp(-10 p)) - 1)`` with
    ``p = step / total_steps``. ``constant`` returns ``lambda_max``.

    Args:
        step: Zero-based step index.
        total_steps: Total optimisation steps in the run.
        lambda_max: Peak reversal strength.
        schedule: ``"ganin"`` or ``"constant"``.

    Returns:
        The lambda value for this step.
    """
    if schedule not in {"ganin", "constant"}:
        raise ValueError("schedule must be 'ganin' or 'constant'")
    if schedule == "constant":
        return float(lambda_max)
    if total_steps <= 0:
        return 0.0
    p = min(max(step / total_steps, 0.0), 1.0)
    return float(lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0))
