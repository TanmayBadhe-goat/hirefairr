"""Terminal demo. Same core functions as the Gradio app.

Usage:
    python -m hirefair.demo.cli --role nurse
    python -m hirefair.demo.cli --role surgeon --text "She led the theatre team..."
    python -m hirefair.demo.cli --role nurse --example 3
"""

from __future__ import annotations

import argparse
import sys

from hirefair.demo import core


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", choices=list(core.ROLES), default="nurse")
    parser.add_argument("--text", type=str, default=None, help="Candidate text to score")
    parser.add_argument("--example", type=int, default=None, help="Index of a curated example")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def _default_text(role: str, index: int | None) -> tuple[str, dict]:
    payload = core.examples(role)
    entries = payload.get("examples", [])
    if not entries:
        raise SystemExit(f"no curated examples for {role}")
    position = 0 if index is None else max(0, min(index, len(entries) - 1))
    return entries[position]["text"], payload


def format_result(result: core.PairResult, payload: dict | None, index: int = 0) -> str:
    """Format one pair exactly as in the demo spec."""
    lines = [f"ROLE: {result.role}"]
    cue_tokens = [cue["token"] for cue in result.cues]
    lines.append(f"ORIGINAL (cues: {', '.join(cue_tokens) if cue_tokens else 'none'})")
    for variant in core.DEMO_VARIANTS:
        if variant not in result.models:
            continue
        entry = result.models[variant]
        lines.append(
            f"  {variant} score {entry.score_original:.2f}  "
            f"{core.decision_label(entry.decision_original)}"
        )
    swapped_tokens = [result.swapped[start:end] for start, end in result.swapped_spans]
    lines.append(f"SWAPPED (cues: {', '.join(swapped_tokens) if swapped_tokens else 'none'})")
    for variant in core.DEMO_VARIANTS:
        if variant not in result.models:
            continue
        entry = result.models[variant]
        suffix = f"  delta {entry.delta:+.2f}"
        if entry.flipped:
            suffix += "  FLIPPED"
        lines.append(
            f"  {variant} score {entry.score_swapped:.2f}  "
            f"{core.decision_label(entry.decision_swapped)}{suffix}"
        )
    if not result.cues:
        lines.append("")
        lines.append("no cues found, swap leaves text unchanged")
    if result.truncated:
        lines.append(
            f"input is {result.n_tokens} tokens; truncated to the model limit of 128"
        )
    if payload is not None:
        lines.append("")
        note = payload.get("note", "")
        if note:
            lines.append(note)
        if index < len(payload.get("examples", [])):
            lines.append(f"example {index + 1} of {len(payload['examples'])}: selected to illustrate")
    aggregates = core.aggregate(result.role)
    if aggregates:
        lines.append("")
        lines.append("Aggregate over all test pairs (mean over seeds):")
        for variant, values in aggregates.items():
            lines.append(
                f"  {variant}: flip rate {values['flip_rate']:.2%}, "
                f"mean |score delta| {values['mean_abs_score_delta']:.4f}"
            )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = None
    if args.text:
        text = args.text
        try:
            payload = core.examples(args.role)
        except core.MissingCheckpoint:
            payload = None
        index = 0
    else:
        text, payload = _default_text(args.role, args.example)
        index = args.example or 0
    try:
        result = core.score_pair(text, args.role, results_dir=args.results_dir, seed=args.seed)
    except core.MissingCheckpoint as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(format_result(result, payload, index))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
