"""Role inference for the HireFair 2.0 resume analyzer.

One trained checkpoint per demo role. Each checkpoint is a binary job-fit
classifier ("is this resume for this role?"), so scoring a text against all
three checkpoints gives a genuine three-way comparison: no score is derived
from another role's output and nothing is hardcoded.

The scoring itself is :func:`hirefair.experiment.score_texts`, the same helper
the training pipeline and :mod:`hirefair.demo.core` use. This module only adds
checkpoint loading, per-role truncation limits and the recommendation rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hirefair.data.preprocess import tokenize
from hirefair.demo.core import MissingCheckpoint, load_models
from hirefair.experiment import score_texts
from hirefair.utils import PROJECT_ROOT, get_logger

LOGGER = get_logger()

#: Roles exposed by the resume analyzer, in display order.
DEMO_ROLES: tuple[str, ...] = (
    "software_engineer",
    "data_analyst",
    "machine_learning_engineer",
)

#: Results directory holding the three-role resume-corpus checkpoints. Kept
#: apart from ``results/`` so the Bias in Bios audit runs are untouched.
RESULTS_ROOT: Path = PROJECT_ROOT / "results" / "hirefair2"

#: Role used for the bar chart so the three scores are directly comparable.
BASE_VARIANT = "B0"

TRAIN_COMMAND = "python scripts/train_roles.py"


@dataclass(frozen=True)
class RoleScore:
    """One role's job-fit score for a text."""

    role: str
    score: float
    threshold: float
    selected: bool


@dataclass(frozen=True)
class RoleAnalysis:
    """Three-role scores for one resume text."""

    text: str
    scores: dict[str, RoleScore]
    max_len: int
    n_tokens: int

    @property
    def recommended(self) -> str:
        """Role with the highest job-fit score."""
        return max(self.scores, key=lambda role: self.scores[role].score)

    @property
    def truncated(self) -> bool:
        """True when the text exceeds the model's context window."""
        return self.n_tokens > self.max_len

    def ordered(self) -> list[RoleScore]:
        """Scores sorted by descending score."""
        return sorted(self.scores.values(), key=lambda entry: -entry.score)


def _root(results_dir: str | Path | None) -> Path:
    return Path(results_dir) if results_dir is not None else RESULTS_ROOT


def _max_len(checkpoint) -> int:
    return int(checkpoint.config.get("data", {}).get("max_len", 128))


def analyse_text(text: str, results_dir: str | Path | None = None) -> RoleAnalysis:
    """Score ``text`` with every demo role's checkpoint.

    Raises:
        MissingCheckpoint: when a role has no trained checkpoint, with the
            command that produces one.
        ValueError: when ``text`` is empty.
    """
    if not text or not text.strip():
        raise ValueError("empty text")

    scores: dict[str, RoleScore] = {}
    max_len = 0
    for role in DEMO_ROLES:
        try:
            checkpoints = load_models(role, results_dir=_root(results_dir))
        except MissingCheckpoint as exc:
            raise MissingCheckpoint(
                f"{exc}\n\nThe resume-corpus checkpoints for the three demo roles are "
                f"built with:\n  {TRAIN_COMMAND}"
            ) from exc
        if BASE_VARIANT not in checkpoints:
            raise MissingCheckpoint(
                f"Missing {BASE_VARIANT} checkpoint for {role}. Produce it with:\n"
                f"  {TRAIN_COMMAND}"
            )
        checkpoint = checkpoints[BASE_VARIANT]
        role_max_len = _max_len(checkpoint)
        max_len = max(max_len, role_max_len)
        device = next(checkpoint.model.parameters()).device
        value = float(score_texts(checkpoint.model, [text], checkpoint.vocab, role_max_len, device)[0])
        scores[role] = RoleScore(
            role=role,
            score=value,
            threshold=float(checkpoint.threshold),
            selected=bool(value >= checkpoint.threshold),
        )

    return RoleAnalysis(
        text=text,
        scores=scores,
        max_len=max_len or 128,
        n_tokens=len(tokenize(text)),
    )


def recommend_role(analysis: RoleAnalysis) -> str:
    """Recommended role for an analysis (highest job-fit score)."""
    return analysis.recommended
