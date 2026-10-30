#!/usr/bin/env python
"""Generate Tables T1 to T6 from ``results/`` only.

Every number is read from files written by training or analysis scripts. The
script never types a result by hand. CSVs land in ``results/tables`` and the
markdown versions in ``report/tables``.

Examples:
    python scripts/make_tables.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.utils import PROJECT_ROOT, Config, get_logger

LOGGER = get_logger()
VARIANTS = ["B0", "B1", "B2", "B3", "B4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    return parser.parse_args()


def fmt(value: float, nd: int = 3) -> str:
    """Format a number, or ``n/a`` when missing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:.{nd}f}"


def mean_sd_text(series: pd.Series, nd: int = 3) -> str:
    """``mean ± sd`` across seeds."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return "n/a"
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"


def t1(results: Path, out_csv: Path, out_md: Path) -> pd.DataFrame:
    path = results / "t1_dataset_stats.csv"
    if not path.exists():
        LOGGER.warning("missing %s", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.to_csv(out_csv, index=False)
    lines = ["# T1 — Dataset statistics", "", "| Role | n | Train | Val | Test | Positive rate | P(y=1\\|M) | P(y=1\\|F) | Female share of positives | Mean tokens |", "|---|---|---|---|---|---|---|---|---|---|"]
    for _, row in frame.iterrows():
        lines.append(
            f"| {row['role']} | {int(row['n_total'])} | {int(row['n_train'])} | {int(row['n_val'])} | "
            f"{int(row['n_test'])} | {fmt(row['positive_rate'])} | {fmt(row['positive_rate_M'])} | "
            f"{fmt(row['positive_rate_F'])} | {fmt(row['female_share_of_positives'])} | "
            f"{fmt(row['mean_tokens'], 1)} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def t2(summary: pd.DataFrame, out_csv: Path, out_md: Path) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    columns = [
        "f1", "roc_auc", "sr_m", "sr_f", "dpd", "dir",
        "tpr_m", "tpr_f", "eod", "fpr_gap", "probe_accuracy",
    ]
    rows = []
    lines = ["# T2 — Main results", ""]
    for role, group in summary.groupby("role"):
        lines.append(f"## {role}")
        lines.append("")
        lines.append(
            "| Model | F1 | AUC | SR_M | SR_F | DPD | DIR | TPR_M | TPR_F | EOD | FPR gap | Probe acc |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for variant in VARIANTS:
            part = group[group["variant"] == variant]
            if part.empty:
                continue
            record = {"role": role, "variant": variant, "seeds": int(part["seed"].nunique())}
            for column in columns:
                record[f"{column}_mean"] = float(pd.to_numeric(part[column], errors="coerce").mean())
                record[f"{column}_sd"] = float(
                    pd.to_numeric(part[column], errors="coerce").std(ddof=1)
                ) if len(part) > 1 else 0.0
            rows.append(record)
            cells = [mean_sd_text(part[column]) for column in columns]
            lines.append(f"| {variant} | " + " | ".join(cells) + " |")
        lines.append("")
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def t3(results: Path, out_csv: Path, out_md: Path) -> pd.DataFrame:
    path = results / "sweep.csv"
    if not path.exists():
        LOGGER.warning("missing %s (run scripts/run_sweep.py)", path)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame.to_csv(out_csv, index=False)
    lines = ["# T3 — Lambda sweep (B3)", "", "| Role | lambda_max | F1 | EOD | DPD | Adversary acc | Probe acc | Flip rate |", "|---|---|---|---|---|---|---|---|"]
    for _, row in frame.sort_values(["role", "lambda_max"]).iterrows():
        lines.append(
            f"| {row['role']} | {row['lambda_max']:g} | {fmt(row['f1'])} | {fmt(row['eod'])} | "
            f"{fmt(row['dpd'])} | {fmt(row['adv_acc'])} | {fmt(row['probe_accuracy'])} | "
            f"{fmt(row.get('flip_rate'))} |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def t4(summary: pd.DataFrame, out_csv: Path, out_md: Path) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    lines = ["# T4 — Counterfactual results", "", "| Role | Model | Track A flip | Track A mean abs delta | Track A rank shift | Track B flip | Track B mean abs delta | Track B rank shift |", "|---|---|---|---|---|---|---|---|"]
    for (role, variant), group in summary.groupby(["role", "variant"]):
        record = {
            "role": role,
            "variant": variant,
            "track_a_flip_rate": float(pd.to_numeric(group["flip_rate"], errors="coerce").mean()),
            "track_a_mean_abs_delta": float(pd.to_numeric(group["mean_abs_score_delta"], errors="coerce").mean()),
            "track_a_rank_shift": float(pd.to_numeric(group["mean_rank_shift"], errors="coerce").mean()),
            "track_b_flip_rate": float(pd.to_numeric(group["track_b_flip_rate"], errors="coerce").mean()),
            "track_b_mean_abs_delta": float(pd.to_numeric(group["track_b_mean_abs_delta"], errors="coerce").mean()),
            "track_b_rank_shift": float(pd.to_numeric(group["track_b_mean_rank_shift"], errors="coerce").mean()),
            "seeds": int(group["seed"].nunique()),
        }
        rows.append(record)
        lines.append(
            f"| {role} | {variant} | {fmt(record['track_a_flip_rate'])} | "
            f"{fmt(record['track_a_mean_abs_delta'])} | {fmt(record['track_a_rank_shift'], 1)} | "
            f"{fmt(record['track_b_flip_rate'])} | {fmt(record['track_b_mean_abs_delta'])} | "
            f"{fmt(record['track_b_rank_shift'], 1)} |"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def t5(results: Path, out_csv: Path, out_md: Path) -> pd.DataFrame:
    analysis_dir = results / "analysis"
    files = sorted(analysis_dir.glob("gender_tokens_*_*.json"))
    rows = []
    lines = ["# T5 — Gender-associated tokens", ""]
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        role = payload["role"]
        variant = payload["variant"]
        lines.append(f"## {role} / {variant}")
        lines.append("")
        lines.append("| Source | Side | Top 10 tokens |")
        lines.append("|---|---|---|")
        for source in ("log_odds", "projection"):
            for side in ("female", "male"):
                entries = payload.get(source, {}).get(side, [])[:10]
                tokens = ", ".join(str(item["token"]) for item in entries)
                value_key = "z" if source == "log_odds" else "projection"
                for rank, item in enumerate(entries, start=1):
                    rows.append(
                        {
                            "role": role,
                            "variant": variant,
                            "source": source,
                            "side": side,
                            "rank": rank,
                            "token": item["token"],
                            "value": item[value_key],
                        }
                    )
                lines.append(f"| {source} | {side} | {tokens} |")
        bias = payload.get("embedding_bias", {})
        lines.append("")
        lines.append(
            f"Embedding bias score (mean abs projection of occupation and skill words): "
            f"{fmt(bias.get('mean_abs_projection'))} over {int(bias.get('n_words', 0))} words."
        )
        lines.append("")
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def t6(summary: pd.DataFrame, out_csv: Path, out_md: Path) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    lines = ["# T6 — Subgroup table (gender x role)", "", "| Role | Model | Gender | n (mean) | Positives (mean) | TPR | Selection rate |", "|---|---|---|---|---|---|---|"]
    for (role, variant), group in summary.groupby(["role", "variant"]):
        for label, n_col, pos_col, tpr_col, sr_col in (
            ("M", "n_m", "n_pos_m", "tpr_m", "sr_m"),
            ("F", "n_f", "n_pos_f", "tpr_f", "sr_f"),
        ):
            record = {
                "role": role,
                "variant": variant,
                "gender": label,
                "n": float(pd.to_numeric(group.get(n_col), errors="coerce").mean()),
                "positives": float(pd.to_numeric(group.get(pos_col), errors="coerce").mean()),
                "tpr": float(pd.to_numeric(group[tpr_col], errors="coerce").mean()),
                "sr": float(pd.to_numeric(group[sr_col], errors="coerce").mean()),
            }
            rows.append(record)
            lines.append(
                f"| {role} | {variant} | {label} | {record['n']:.0f} | {record['positives']:.0f} | "
                f"{fmt(record['tpr'])} | {fmt(record['sr'])} |"
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(out_csv, index=False)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return frame


def main() -> int:
    args = parse_args()
    overrides: dict[str, dict] = {}
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    cfg = Config.load(None, overrides=overrides, quick=args.quick, profile=args.profile)
    results = cfg.results_root
    tables_dir = results / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    report_tables = PROJECT_ROOT / "report" / "tables"
    report_tables.mkdir(parents=True, exist_ok=True)

    summary_path = results / "summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if summary.empty:
        LOGGER.warning("summary.csv missing or empty; some tables will be skipped")

    t1(results, tables_dir / "T1.csv", report_tables / "T1.md")
    t2(summary, tables_dir / "T2.csv", report_tables / "T2.md")
    t3(results, tables_dir / "T3.csv", report_tables / "T3.md")
    t4(summary, tables_dir / "T4.csv", report_tables / "T4.md")
    t5(results, tables_dir / "T5.csv", report_tables / "T5.md")
    t6(summary, tables_dir / "T6.csv", report_tables / "T6.md")
    LOGGER.info("tables written to %s and %s", tables_dir, report_tables)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
