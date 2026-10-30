#!/usr/bin/env python
"""Mine demo examples from the test set and the synthetic pairs.

Criterion: the largest baseline (B0) score change under a gender-cue swap,
restricted to pairs where the debiased model (B3) stays stable
(``|delta_B3| <= --stability``). Decision flips are preferred, but are not
required. Eight examples per role are written to ``demo/examples/{role}.json``
with their measured scores and deltas, plus the aggregate flip rates so the
demo can show the typical case next to the selected ones.

Examples:
    python scripts/find_flip_examples.py --role all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.data.loaders import DEFAULT_ROLES
from hirefair.data.swap import find_gender_cues
from hirefair.experiment import score_texts
from hirefair.models.hirefair_model import load_checkpoint
from hirefair.utils import PROJECT_ROOT, Config, get_logger, read_jsonl, write_json

LOGGER = get_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", nargs="+", choices=[*DEFAULT_ROLES, "all"], default=["all"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-role", type=int, default=8)
    parser.add_argument("--stability", type=float, default=0.02, help="Max |delta_B3| for a stable pair")
    parser.add_argument("--test-limit", type=int, default=4000, help="Test rows to scan")
    parser.add_argument("--synthetic-limit", type=int, default=1000)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--quick", action="store_true", help="Smoke test against results_quick")
    return parser.parse_args()


def _aggregate(results: Path, role: str) -> dict[str, dict[str, float]]:
    path = results / "summary.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    frame = frame[(frame["role"] == role) & (frame["variant"].isin(["B0", "B1", "B3"]))]
    out = {}
    for variant, group in frame.groupby("variant"):
        out[variant] = {
            "flip_rate": float(group["flip_rate"].mean()),
            "mean_abs_score_delta": float(group["mean_abs_score_delta"].mean()),
            "seeds": float(group["seed"].nunique()),
        }
    return out


def main() -> int:
    args = parse_args()
    roles = tuple(DEFAULT_ROLES) if "all" in args.role else tuple(args.role)
    overrides: dict[str, dict] = {}
    if args.results_dir:
        overrides.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.data_dir:
        overrides.setdefault("paths", {})["data_dir"] = args.data_dir
    cfg = Config.load(None, overrides=overrides, profile=args.profile, quick=args.quick)
    device = torch.device("cpu")
    if args.quick:
        args.test_limit = min(args.test_limit, 300)
        args.synthetic_limit = min(args.synthetic_limit, 300)

    for role in roles:
        cfg.role = role
        checkpoints = {}
        for variant in ("B0", "B3"):
            path = cfg.results_root / role / variant / f"seed{args.seed}" / "model.pt"
            if not path.exists():
                raise SystemExit(
                    f"missing {path}. Produce it with:\n"
                    f"  python scripts/run_experiment.py --role {role} --variant {variant} "
                    f"--seed {args.seed} --profile configs/fast_cpu.yaml"
                )
            checkpoints[variant] = load_checkpoint(path)
        vocab = checkpoints["B0"].vocab

        test_rows = read_jsonl(cfg.processed_dir(role) / "test_meta.jsonl")[: args.test_limit]
        synthetic_path = cfg.data_dir / "synthetic" / f"{role}.jsonl"
        synthetic_rows = read_jsonl(synthetic_path)[: args.synthetic_limit] if synthetic_path.exists() else []

        candidates = []
        for kind, rows, text_key, swap_key in (
            ("track_a", test_rows, "text", "swapped_text"),
            ("track_b", synthetic_rows, "resume_F", None),
        ):
            if not rows:
                continue
            originals = [row[text_key] for row in rows]
            if swap_key is None:
                swapped = [row["resume_M"] for row in rows]
            else:
                swapped = [row[swap_key] for row in rows]
            scores = {}
            for variant, checkpoint in checkpoints.items():
                scores[f"{variant}_orig"] = score_texts(
                    checkpoint.model, originals, vocab, cfg.data.max_len, device
                )
                scores[f"{variant}_swap"] = score_texts(
                    checkpoint.model, swapped, vocab, cfg.data.max_len, device
                )
            for i, row in enumerate(rows):
                b0_orig = float(scores["B0_orig"][i])
                b0_swap = float(scores["B0_swap"][i])
                b3_orig = float(scores["B3_orig"][i])
                b3_swap = float(scores["B3_swap"][i])
                b0_decision = (b0_orig >= checkpoints["B0"].threshold, b0_swap >= checkpoints["B0"].threshold)
                b3_decision = (b3_orig >= checkpoints["B3"].threshold, b3_swap >= checkpoints["B3"].threshold)
                candidates.append(
                    {
                        "source": kind,
                        "text": originals[i],
                        "swapped_text": swapped[i],
                        "models": {
                            "B0": {
                                "score_original": b0_orig,
                                "score_swapped": b0_swap,
                                "decision_original": int(b0_decision[0]),
                                "decision_swapped": int(b0_decision[1]),
                                "delta": b0_swap - b0_orig,
                                "flipped": bool(b0_decision[0] != b0_decision[1]),
                            },
                            "B3": {
                                "score_original": b3_orig,
                                "score_swapped": b3_swap,
                                "decision_original": int(b3_decision[0]),
                                "decision_swapped": int(b3_decision[1]),
                                "delta": b3_swap - b3_orig,
                                "flipped": bool(b3_decision[0] != b3_decision[1]),
                            },
                        },
                    }
                )

        stable = [c for c in candidates if abs(c["models"]["B3"]["delta"]) <= args.stability]
        stable.sort(
            key=lambda c: (c["models"]["B0"]["flipped"], abs(c["models"]["B0"]["delta"])),
            reverse=True,
        )

        # Prefer a mix of real swaps and synthetic pairs.
        selected: list[dict] = []
        per_source = max(1, args.per_role // 2)
        for source in ("track_a", "track_b"):
            take = [c for c in stable if c["source"] == source][:per_source]
            selected.extend(take)
        for candidate in stable:
            if len(selected) >= args.per_role:
                break
            if candidate not in selected:
                selected.append(candidate)
        selected = selected[: args.per_role]

        for candidate in selected:
            cues = find_gender_cues(candidate["text"])
            candidate["cues"] = [
                {"token": cue.original, "kind": cue.kind} for cue in cues
            ]
            candidate["selection_note"] = (
                "selected to illustrate: largest B0 score change with |B3 delta| "
                f"<= {args.stability:.2f}"
            )

        aggregate = _aggregate(cfg.results_root, role)
        b0_flips = sum(1 for c in candidates if c["models"]["B0"]["flipped"])
        note = (
            "Examples are selected to illustrate the counterfactual test. "
            f"Across {len(candidates)} scanned pairs, B0 flipped on {b0_flips} "
            f"({b0_flips / max(1, len(candidates)):.1%}). Aggregate flip rates are "
            "reported next to them."
        )
        if b0_flips < args.per_role:
            note += " Few B0 flips were available for this role."
        payload = {
            "role": role,
            "seed": args.seed,
            "thresholds": {variant: checkpoints[variant].threshold for variant in checkpoints},
            "note": note,
            "aggregate": aggregate,
            "scanned_pairs": len(candidates),
            "b0_flips_scanned": b0_flips,
            "examples": selected,
        }
        out = PROJECT_ROOT / "demo" / "examples" / f"{role}.json"
        write_json(out, payload)
        LOGGER.info(
            "%s: wrote %d examples (B0 flips %d/%d), %s",
            role,
            len(selected),
            b0_flips,
            len(candidates),
            out,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
