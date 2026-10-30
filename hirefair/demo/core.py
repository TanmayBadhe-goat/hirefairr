"""Shared demo logic. The Gradio app and the terminal fallback both call this.

No metric is computed here that is not read from ``results/`` or produced by
the loaded checkpoint. Missing artefacts raise :class:`MissingCheckpoint` with
the exact command that produces them.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from hirefair.data.preprocess import tokenize
from hirefair.data.swap import Change, find_gender_cues, swap_gender
from hirefair.experiment import score_texts
from hirefair.fairness.counterfactual import ranks
from hirefair.models.hirefair_model import Checkpoint, load_checkpoint
from hirefair.utils import PROJECT_ROOT, get_logger, read_json, read_jsonl

LOGGER = get_logger()

#: Roles with checkpoints in ``results/`` (Bias in Bios audit) plus the roles of
#: the resume analyzer, whose checkpoints live in ``results/hirefair2/``.
ROLES = (
    "software_engineer",
    "nurse",
    "surgeon",
    "data_analyst",
    "machine_learning_engineer",
)
DEMO_VARIANTS = ("B0", "B1", "B3")
TOP_K = 10


class MissingCheckpoint(RuntimeError):
    """Raised when a demo needs a checkpoint or results file that is absent."""


@dataclass
class ModelScores:
    """Scores and decisions for one model on one pair."""

    variant: str
    threshold: float
    score_original: float
    score_swapped: float
    decision_original: int
    decision_swapped: int

    @property
    def delta(self) -> float:
        """Signed score change after the swap."""
        return self.score_swapped - self.score_original

    @property
    def flipped(self) -> bool:
        """True when the decision changes after the swap."""
        return self.decision_original != self.decision_swapped


@dataclass
class PairResult:
    """Everything the demo shows for one candidate."""

    role: str
    original: str
    swapped: str
    cues: list[dict[str, str]] = field(default_factory=list)
    swapped_spans: list[tuple[int, int]] = field(default_factory=list)
    truncated: bool = False
    n_tokens: int = 0
    models: dict[str, ModelScores] = field(default_factory=dict)
    aggregate: dict[str, dict[str, float]] = field(default_factory=dict)


def _resolve(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def run_dir(role: str, variant: str, results_dir: str | Path | None = None, seed: int = 0) -> Path:
    """Path to a run directory."""
    root = _resolve(results_dir, PROJECT_ROOT / "results")
    return root / role / variant / f"seed{seed}"


def missing_checkpoint_message(role: str, variant: str) -> str:
    """Command that produces a missing checkpoint."""
    return (
        f"Missing checkpoint for {variant} / {role}. Produce it with:\n"
        f"  python scripts/run_experiment.py --role {role} --variant {variant} "
        f"--seed 0 --profile configs/fast_cpu.yaml"
    )


@lru_cache(maxsize=8)
def _load_models_cached(role: str, results_dir: str, seed: int) -> dict[str, Checkpoint]:
    checkpoints: dict[str, Checkpoint] = {}
    for variant in DEMO_VARIANTS:
        path = run_dir(role, variant, results_dir, seed) / "model.pt"
        if not path.exists():
            if variant == "B1":
                continue
            raise MissingCheckpoint(missing_checkpoint_message(role, variant))
        checkpoints[variant] = load_checkpoint(path)
    if not checkpoints:
        raise MissingCheckpoint(missing_checkpoint_message(role, "B0"))
    return checkpoints


def load_models(
    role: str, results_dir: str | Path | None = None, seed: int = 0
) -> dict[str, Checkpoint]:
    """Load B0 and B3, plus B1 when present. Raises with the fix command."""
    root = _resolve(results_dir, PROJECT_ROOT / "results")
    return dict(_load_models_cached(role, str(root), seed))


def available_models(role: str, results_dir: str | Path | None = None, seed: int = 0) -> list[str]:
    """Variants whose checkpoints exist for a role."""
    present = []
    for variant in DEMO_VARIANTS:
        if (run_dir(role, variant, results_dir, seed) / "model.pt").exists():
            present.append(variant)
    return present


def _max_len(checkpoints: dict[str, Checkpoint]) -> int:
    config = next(iter(checkpoints.values())).config
    return int(config.get("data", {}).get("max_len", 128))


def _to_html(text: str, spans: list[tuple[int, int]]) -> str:
    """Escape text and wrap the given spans in <mark>."""
    spans = sorted({(max(0, int(start)), int(end)) for start, end in spans if end > start})
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            start, end = cursor, max(cursor, end)
        pieces.append(html.escape(text[cursor:start]))
        pieces.append(f"<mark>{html.escape(text[start:end])}</mark>")
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces).replace("\n", "<br>")


def _mean(frame: pd.DataFrame, column: str) -> float:
    """Mean of a summary column, ``NaN`` when the column or values are absent."""
    if column not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[column], errors="coerce").mean())


def aggregate(role: str, results_dir: str | Path | None = None) -> dict[str, dict[str, float]]:
    """Per-model means read from summary.csv (task, fairness and counterfactual)."""
    root = _resolve(results_dir, PROJECT_ROOT / "results")
    path = root / "summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[(frame["role"] == role) & (frame["variant"].isin(DEMO_VARIANTS))]
    out: dict[str, dict[str, float]] = {}
    for variant, group in frame.groupby("variant"):
        values = {
            column: _mean(group, column)
            for column in (
                "flip_rate",
                "mean_abs_score_delta",
                "f1",
                "accuracy",
                "precision",
                "recall",
                "roc_auc",
                "sr_m",
                "sr_f",
                "dpd",
                "tpr_m",
                "tpr_f",
                "eod",
                "eod_abs",
            )
        }
        values["seeds"] = float(group["seed"].nunique())
        out[variant] = values
    return out


def score_pair(
    text: str,
    role: str,
    results_dir: str | Path | None = None,
    seed: int = 0,
) -> PairResult:
    """Score a text and its gender-swapped version with every loaded model."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    checkpoints = load_models(role, results_dir, seed)
    vocab = next(iter(checkpoints.values())).vocab
    max_len = _max_len(checkpoints)

    swap = swap_gender(text, direction="auto", mode="swap")
    cues = find_gender_cues(text)
    n_tokens = len(tokenize(text))
    truncated = n_tokens > max_len

    scores: dict[str, ModelScores] = {}
    for variant, checkpoint in checkpoints.items():
        values = score_texts(
            checkpoint.model,
            [text, swap.text],
            vocab,
            max_len,
            next(checkpoint.model.parameters()).device,
        )
        score_original, score_swapped = float(values[0]), float(values[1])
        scores[variant] = ModelScores(
            variant=variant,
            threshold=checkpoint.threshold,
            score_original=score_original,
            score_swapped=score_swapped,
            decision_original=int(score_original >= checkpoint.threshold),
            decision_swapped=int(score_swapped >= checkpoint.threshold),
        )

    return PairResult(
        role=role,
        original=text,
        swapped=swap.text,
        cues=[{"token": cue.original, "kind": cue.kind} for cue in cues],
        swapped_spans=[(change.start, change.end) for change in swap.changes],
        truncated=truncated,
        n_tokens=n_tokens,
        models=scores,
        aggregate=aggregate(role, results_dir),
    )


def highlight_original(result: PairResult) -> str:
    """HTML for the original text with cues marked."""
    spans = []
    cursor = 0
    for cue in result.cues:
        token = cue["token"]
        start = result.original.find(token, cursor)
        if start == -1:
            start = result.original.find(token)
        if start >= 0:
            spans.append((start, start + len(token)))
            cursor = start + len(token)
    return _to_html(result.original, spans)


def highlight_swapped(result: PairResult) -> str:
    """HTML for the swapped text with changed spans marked."""
    return _to_html(result.swapped, result.swapped_spans)


def samples_from_changes(changes: list[Change]) -> list[dict[str, str]]:
    """Convert swap changes to plain dicts (kept for the CLI)."""
    return [
        {"token": change.original, "replacement": change.replacement, "kind": change.kind}
        for change in changes
    ]


def sample_pool(rows: list[dict], size: int, seed: int = 0) -> list[dict]:
    """Deterministic candidate sample from the test set."""
    if size >= len(rows):
        return list(rows)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(rows), size=size, replace=False))
    return [rows[i] for i in index]


def _top(rows: list[dict], scores: np.ndarray, k: int = TOP_K) -> list[dict]:
    order = np.argsort(-scores, kind="stable")[:k]
    return [
        {
            "id": int(rows[i]["id"]),
            "gender": rows[i]["gender"],
            "score": float(scores[i]),
        }
        for i in order
    ]


def ranking(
    role: str,
    results_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    pool_size: int = 200,
    seed: int = 42,
    k: int = TOP_K,
) -> dict:
    """Rank a fixed pool of test candidates, then re-rank after swapping cues."""
    checkpoints = load_models(role, results_dir, seed=0)
    vocab = next(iter(checkpoints.values())).vocab
    max_len = _max_len(checkpoints)
    root = _resolve(data_dir, PROJECT_ROOT / "data")
    meta_path = root / "processed" / role / "test_meta.jsonl"
    if not meta_path.exists():
        raise MissingCheckpoint(
            f"Missing {meta_path}. Produce it with:\n"
            f"  python scripts/prepare_data.py --role {role} --profile configs/fast_cpu.yaml"
        )
    rows = read_jsonl(meta_path)
    pool = sample_pool(rows, pool_size, seed=seed)
    originals = [row["text"] for row in pool]
    swapped = [row["swapped_text"] for row in pool]

    payload: dict = {
        "role": role,
        "pool_size": len(pool),
        "seed": seed,
        "models": {},
    }
    for variant, checkpoint in checkpoints.items():
        device = next(checkpoint.model.parameters()).device
        scores = score_texts(checkpoint.model, originals, vocab, max_len, device)
        swapped_scores = score_texts(checkpoint.model, swapped, vocab, max_len, device)
        before_top = _top(pool, scores, k)
        after_top = _top(pool, swapped_scores, k)
        before_ids = {row["id"] for row in before_top}
        after_ids = {row["id"] for row in after_top}
        payload["models"][variant] = {
            "top_before": before_top,
            "top_after": after_top,
            "top_gender_before": {
                "M": sum(1 for row in before_top if row["gender"] == "M"),
                "F": sum(1 for row in before_top if row["gender"] == "F"),
            },
            "top_gender_after": {
                "M": sum(1 for row in after_top if row["gender"] == "M"),
                "F": sum(1 for row in after_top if row["gender"] == "F"),
            },
            "top_changes": len(before_ids.symmetric_difference(after_ids)),
            "mean_rank_shift": float(np.mean(np.abs(ranks(scores) - ranks(swapped_scores)))),
            "pool_flip_rate": float(
                np.mean(
                    (scores >= checkpoint.threshold).astype(int)
                    != (swapped_scores >= checkpoint.threshold).astype(int)
                )
            ),
        }
    return payload


def examples(role: str) -> dict:
    """Curated examples written by scripts/find_flip_examples.py."""
    path = PROJECT_ROOT / "demo" / "examples" / f"{role}.json"
    if not path.exists():
        raise MissingCheckpoint(
            f"Missing {path}. Produce it with:\n"
            f"  python scripts/find_flip_examples.py --role {role}"
        )
    return read_json(path)


def decision_label(value: int) -> str:
    """SELECT / REJECT label for a decision bit."""
    return "SELECT" if int(value) == 1 else "REJECT"


def result_table(result: PairResult) -> list[list[str]]:
    """Rows for the demo result table."""
    rows = []
    for variant in DEMO_VARIANTS:
        if variant not in result.models:
            continue
        entry = result.models[variant]
        rows.append(
            [
                variant,
                f"{entry.score_original:.3f}",
                decision_label(entry.decision_original),
                f"{entry.score_swapped:.3f}",
                decision_label(entry.decision_swapped),
                f"{entry.delta:+.3f}",
                "yes" if entry.flipped else "no",
            ]
        )
    return rows


def aggregate_markdown(role: str, results_dir: str | Path | None = None) -> str:
    """Aggregate flip rates for the demo notes."""
    stats = aggregate(role, results_dir)
    if not stats:
        return "Aggregate results are not available. Run the training grid first."
    lines = [f"Across all test pairs for {role} (mean over seeds):", ""]
    for variant, values in stats.items():
        lines.append(
            f"- {variant}: flip rate {values['flip_rate']:.3%}, "
            f"mean |score delta| {values['mean_abs_score_delta']:.4f}"
        )
    return "\n".join(lines)
