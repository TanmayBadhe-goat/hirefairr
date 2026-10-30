#!/usr/bin/env python
"""Assemble the submission folder without model weights or raw data.

Creates ``submission/code`` (source, configs, tests, report text, tables and
figures), ``submission/demo`` (examples and the walkthrough script) and copies
the rendered report PDF when it exists.

Examples:
    python scripts/package_submission.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hirefair.utils import PROJECT_ROOT, get_logger

LOGGER = get_logger()

INCLUDE = [
    "hirefair",
    "scripts",
    "configs",
    "tests",
    "notebooks",
    "demo/examples",
    "demo/walkthrough.md",
    "README.md",
    "requirements.txt",
    "report/report.md",
    "report/tables",
    "report/figures",
]
IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", ".DS_Store", "*.npz", "*.pt", "*.parquet", "*.jsonl"
)


def package(submission: str | Path = PROJECT_ROOT / "submission") -> Path:
    """Copy the deliverable files into ``submission``."""
    submission = Path(submission)
    code_dir = submission / "code"
    demo_dir = submission / "demo"
    for directory in (code_dir, demo_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)

    for relative in INCLUDE:
        source = PROJECT_ROOT / relative
        if not source.exists():
            LOGGER.warning("missing %s; skipped", source)
            continue
        target = code_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, ignore=IGNORE)
        else:
            shutil.copy2(source, target)

    examples = PROJECT_ROOT / "demo" / "examples"
    if examples.exists():
        shutil.copytree(examples, demo_dir / "examples", dirs_exist_ok=True)
    walkthrough = PROJECT_ROOT / "demo" / "walkthrough.md"
    if walkthrough.exists():
        shutil.copy2(walkthrough, demo_dir / "walkthrough.md")
    pdf = submission / "HireFair_Fairness_Audit_Report.pdf"
    if not pdf.exists():
        LOGGER.warning("report PDF not found at %s", pdf)
    LOGGER.info("submission assembled at %s", submission)
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=str, default=None, help="Submission directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    package(args.out or PROJECT_ROOT / "submission")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
