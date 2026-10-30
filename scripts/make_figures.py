#!/usr/bin/env python
"""Generate Figures F1 to F6 from ``results/``.

PNG at 200 dpi and PDF, written to ``report/figures``. Every panel reads from
summary.csv, sweep.csv, predictions.npz, run.json or the analysis JSON files.

Examples:
    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.utils import PROJECT_ROOT, Config, get_logger, read_json

LOGGER = get_logger()
COLORS = {"B0": "#b45f5f", "B1": "#c9a227", "B2": "#6c8ebf", "B3": "#2f7d5f", "B4": "#7a6fa8"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", nargs="+", choices=[*DEFAULT_ROLES, "all"], default=["all"])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    return parser.parse_args()


def save(fig: plt.Figure, name: str, out_dir: Path) -> None:
    """Save PNG (200 dpi) and PDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=200)
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    LOGGER.info("wrote %s", out_dir / name)


def load_summary(results: Path) -> pd.DataFrame:
    path = results / "summary.csv"
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/aggregate.py first")
    return pd.read_csv(path)


def f1_selection_rates(summary: pd.DataFrame, out_dir: Path) -> None:
    variants = [v for v in ("B0", "B1", "B3") if v in set(summary["variant"])]
    roles = [role for role in DEFAULT_ROLES if role in set(summary["role"])]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2))
    axes = axes.ravel()
    width = 0.8 / max(1, len(variants))
    for ax, role in zip(axes[:3], roles):
        part = summary[summary["role"] == role].groupby("variant")[["sr_m", "sr_f"]].mean()
        x = np.arange(2)
        for i, variant in enumerate(variants):
            if variant not in part.index:
                continue
            values = [part.loc[variant, "sr_m"], part.loc[variant, "sr_f"]]
            ax.bar(x + i * width, values, width=width, color=COLORS[variant], label=variant)
        ax.set_xticks(x + width * (len(variants) - 1) / 2, ["Male", "Female"])
        ax.set_title(role.replace("_", " "))
        ax.set_ylabel("Selection rate")
        ax.set_ylim(0, max(0.05, float(summary["sr_m"].max()) * 1.2, float(summary["sr_f"].max()) * 1.2))
        ax.legend(frameon=False, fontsize=8)
    ax = axes[3]
    dirs = summary[summary["variant"].isin(variants)].groupby(["role", "variant"])["dir"].mean().unstack()
    x = np.arange(len(dirs.index))
    for i, variant in enumerate(variants):
        ax.bar(x + i * width, dirs.get(variant, pd.Series(index=dirs.index, dtype=float)), width=width,
               color=COLORS[variant], label=variant)
    ax.axhline(0.8, color="black", linestyle="--", linewidth=1)
    ax.text(len(dirs.index) - 0.5, 0.81, "0.8 reference", fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x + width * (len(variants) - 1) / 2, [r.replace("_", "\n") for r in dirs.index])
    ax.set_title("Disparate impact ratio")
    ax.set_ylim(0, 1.05)
    save(fig, "F1_selection_rates", out_dir)


def f2_pareto(results: Path, summary: pd.DataFrame, out_dir: Path) -> None:
    sweep_path = results / "sweep.csv"
    if not sweep_path.exists():
        LOGGER.warning("missing %s; skipping F2", sweep_path)
        return
    sweep = pd.read_csv(sweep_path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, role in zip(axes, [r for r in DEFAULT_ROLES if r in set(sweep["role"])]):
        part = sweep[sweep["role"] == role].sort_values("lambda_max")
        ax.plot(part["f1"], part["eod_abs"], marker="o", color="#444444", linewidth=1)
        for _, row in part.iterrows():
            ax.annotate(f"{row['lambda_max']:g}", (row["f1"], row["eod_abs"]), fontsize=7,
                        textcoords="offset points", xytext=(3, 3))
        baseline = summary[(summary["role"] == role) & (summary["variant"] == "B0")]
        if not baseline.empty:
            ax.scatter(baseline["f1"].mean(), baseline["eod_abs"].mean(), marker="*", s=140,
                       color=COLORS["B0"], label="B0", zorder=3)
        ax.set_title(role.replace("_", " "))
        ax.set_xlabel("Test F1")
        ax.set_ylabel("|EOD|")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "F2_pareto_lambda", out_dir)


def f3_training_curves(results: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for role in DEFAULT_ROLES:
        path = results / role / "B3" / "seed0" / "run.json"
        if not path.exists():
            continue
        run = read_json(path)
        trace = run["training"]["step_trace"]
        steps = [row["step"] for row in trace]
        axes[0].plot(steps, [row["task_loss"] for row in trace], label=role.replace("_", " "))
        axes[1].plot(steps, [row["adv_loss"] for row in trace], label=role.replace("_", " "))
        axes[2].plot(steps, [row["lambda"] for row in trace], label=role.replace("_", " "))
    for ax, title in zip(axes, ("Task loss", "Adversary loss", "lambda_t")):
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(frameon=False, fontsize=8)
    save(fig, "F3_training_curves", out_dir)


def f4_score_distributions(results: Path, out_dir: Path) -> None:
    roles = [role for role in DEFAULT_ROLES
             if (results / role / "B0" / "seed0" / "predictions.npz").exists()
             and (results / role / "B3" / "seed0" / "predictions.npz").exists()]
    if not roles:
        LOGGER.warning("no predictions for F4")
        return
    fig, axes = plt.subplots(len(roles), 2, figsize=(9, 2.8 * len(roles)), squeeze=False)
    bins = np.linspace(0, 1, 30)
    for row, role in enumerate(roles):
        for col, variant in enumerate(("B0", "B3")):
            ax = axes[row][col]
            data = np.load(results / role / variant / "seed0" / "predictions.npz")
            for gender, color, label in ((0, "#3b6ea5", "Male"), (1, "#c0603d", "Female")):
                scores = data["scores"][data["g"] == gender]
                ax.hist(scores, bins=bins, density=True, alpha=0.5, color=color, label=label)
            ax.set_title(f"{role.replace('_', ' ')} — {variant}")
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=7)
    save(fig, "F4_score_distributions", out_dir)


def f5_gender_direction(results: Path, out_dir: Path) -> None:
    analysis = results / "analysis"
    roles = [role for role in DEFAULT_ROLES
             if (analysis / f"gender_tokens_{role}_B0.json").exists()]
    if not roles:
        LOGGER.warning("no analysis files for F5")
        return
    fig, axes = plt.subplots(1, len(roles), figsize=(4.2 * len(roles), 5.2), squeeze=False)
    for ax, role in zip(axes[0], roles):
        payloads = {}
        for variant, marker in (("B0", "o"), ("B3", "s")):
            path = analysis / f"gender_tokens_{role}_{variant}.json"
            if path.exists():
                payloads[variant] = json.loads(path.read_text(encoding="utf-8"))
        if not payloads:
            continue
        base = payloads.get("B0") or next(iter(payloads.values()))
        tokens = [row["token"] for row in base["projection"]["female"][:10]] + [
            row["token"] for row in base["projection"]["male"][:10]
        ]
        y = np.arange(len(tokens))
        for variant in payloads:
            lookup = {row["token"]: row["projection"] for side in ("female", "male")
                      for row in payloads[variant]["projection"][side]}
            values = [lookup.get(token, np.nan) for token in tokens]
            ax.scatter(values, y, s=22, label=variant, color=COLORS.get(variant, "#444"))
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y, tokens, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("Projection on gender direction")
        ax.set_title(role.replace("_", " "))
        ax.legend(frameon=False, fontsize=8)
    save(fig, "F5_gender_direction", out_dir)


def f6_counterfactual_scatter(results: Path, out_dir: Path) -> None:
    roles = [role for role in DEFAULT_ROLES
             if (results / role / "B0" / "seed0" / "predictions.npz").exists()]
    if not roles:
        LOGGER.warning("no predictions for F6")
        return
    fig, axes = plt.subplots(len(roles), 2, figsize=(8, 3.4 * len(roles)), squeeze=False)
    rng = np.random.default_rng(0)
    for row, role in enumerate(roles):
        for col, variant in enumerate(("B0", "B3")):
            ax = axes[row][col]
            path = results / role / variant / "seed0" / "predictions.npz"
            if not path.exists():
                ax.axis("off")
                continue
            data = np.load(path)
            x, y = data["scores"], data["swapped_scores"]
            if len(x) > 2500:
                idx = rng.choice(len(x), size=2500, replace=False)
                x, y = x[idx], y[idx]
            ax.scatter(x, y, s=4, alpha=0.25, color="#3b6ea5", linewidths=0)
            ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
            ax.set_title(f"{role.replace('_', ' ')} — {variant}")
            ax.set_xlabel("Score (original)")
            ax.set_ylabel("Score (swapped)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
    save(fig, "F6_counterfactual_scatter", out_dir)


def main() -> int:
    args = parse_args()
    overrides: dict[str, dict] = {}
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    results = cfg.results_root
    out_dir = PROJECT_ROOT / "report" / "figures"
    summary = load_summary(results)
    f1_selection_rates(summary, out_dir)
    f2_pareto(results, summary, out_dir)
    f3_training_curves(results, out_dir)
    f4_score_distributions(results, out_dir)
    f5_gender_direction(results, out_dir)
    f6_counterfactual_scatter(results, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
