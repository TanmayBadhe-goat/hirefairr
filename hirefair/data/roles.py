"""Role-labelled resume corpus for the three HireFair 2.0 demo roles.

The audit corpus (Bias in Bios, :mod:`hirefair.data.loaders`) has 28 occupations
and contains neither ``data_analyst`` nor ``machine_learning_engineer``. The
three demo roles are therefore trained on the project's own Track B resume
corpus, ``data/synthetic/{role}.jsonl``, where every base item is rendered with
male cues, female cues and neutral cues from identical content slots (see
:mod:`hirefair.data.synthetic`).

For a target role ``R`` the job-fit label is ``y = 1`` on ``R``'s renderings and
``y = 0`` on the other two demo roles' renderings, which is the spec's job-fit
definition (``y = 1`` when the occupation is the target role). Only the male and
female renderings are used: the gender label ``g`` that the adversary consumes
is only defined for them, and using them keeps ``g`` balanced inside both
classes.

Arrays are written to ``data/processed_roles/{role}/`` and are never mixed with
the Bias in Bios cache in ``data/processed/``.

Track B is synthetic by design (spec section 02). A model trained here answers
"which of the three demo roles does this resume describe?" on resume-style text;
it is not evidence about real hiring outcomes and is reported as such.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hirefair.data.preprocess import (
    SplitArrays,
    build_vocab,
    encode_texts,
    load_arrays,
    save_arrays,
    save_vocab,
    tokenize,
)
from hirefair.utils import PROJECT_ROOT, get_logger, write_json

LOGGER = get_logger()

#: The roles the HireFair 2.0 resume analyzer exposes.
DEMO_ROLES: tuple[str, ...] = (
    "software_engineer",
    "data_analyst",
    "machine_learning_engineer",
)

#: Variants the demo needs: baseline and the adversarial equal-opportunity model.
DEMO_VARIANTS: tuple[str, ...] = ("B0", "B3")

DEFAULT_MAX_LEN = 128
DEFAULT_VOCAB_SIZE = 30000
DEFAULT_MIN_FREQ = 1
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15


def synthetic_path(role: str) -> Path:
    """Path to the Track B corpus for a role."""
    return PROJECT_ROOT / "data" / "synthetic" / f"{role}.jsonl"


def processed_dir(role: str, data_dir: str | Path | None = None) -> Path:
    """Directory holding the prepared arrays for one role."""
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return root / "processed_roles" / role


def read_corpus(role: str) -> list[dict[str, Any]]:
    """Read ``data/synthetic/{role}.jsonl``."""
    path = synthetic_path(role)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Generate the Track B corpus with:\n"
            f"  python -c \"from hirefair.data.synthetic import write_synthetic; "
            f"write_synthetic('{role}', count=500)\""
        )
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"{path} is empty")
    return rows


def _split_indices(count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Shuffle one source role's items and split them 70/15/15."""
    order = rng.permutation(count)
    n_train = int(count * TRAIN_FRACTION)
    n_val = int(count * VAL_FRACTION)
    return {
        "train": order[:n_train],
        "val": order[n_train : n_train + n_val],
        "test": order[n_train + n_val :],
    }


def build_role_frames(
    role: str,
    roles: Sequence[str] = DEMO_ROLES,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Return ``{split: [{"text", "y", "g"}, ...]}`` for one target role.

    Items are split before rendering, so the male and female renderings of one
    base item always land in the same split and never leak across train/test.
    """
    if role not in roles:
        raise ValueError(f"unknown role {role!r}; expected one of {tuple(roles)}")
    rng = np.random.default_rng(seed)
    frames: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for source in roles:
        records = read_corpus(source)
        buckets = _split_indices(len(records), rng)
        label = 1 if source == role else 0
        for split, indices in buckets.items():
            for index in indices:
                record = records[int(index)]
                for key, gender in (("resume_M", 0), ("resume_F", 1)):
                    text = str(record.get(key, "")).strip()
                    if not text:
                        raise RuntimeError(f"record {record.get('id')!r} has an empty {key}")
                    frames[split].append({"text": text, "y": label, "g": gender})
    for split, rows in frames.items():
        if not rows:
            raise RuntimeError(f"role {role!r} produced no {split} rows")
        if len({row["y"] for row in rows}) < 2:
            raise RuntimeError(f"role {role!r} has a single class in {split}")
    return frames


def _arrays(rows: Sequence[Mapping[str, Any]], vocab: dict[str, int], max_len: int) -> SplitArrays:
    texts = [str(row["text"]) for row in rows]
    ids, mask, lengths = encode_texts(texts, vocab, max_len)
    return SplitArrays(
        ids=ids,
        mask=mask,
        lengths=lengths,
        y=np.asarray([int(row["y"]) for row in rows], dtype=np.int8),
        g=np.asarray([int(row["g"]) for row in rows], dtype=np.int8),
    )


def prepare_role_arrays(
    role: str,
    data_dir: str | Path | None = None,
    seed: int = 42,
    max_len: int = DEFAULT_MAX_LEN,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    min_freq: int = DEFAULT_MIN_FREQ,
    force: bool = False,
) -> dict[str, Any]:
    """Tokenise all splits for a role and cache them with a train-only vocab."""
    directory = processed_dir(role, data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path = directory / "meta.json"
    required = [directory / f"{split}_raw.npz" for split in ("train", "val", "test")]
    if not force and meta_path.exists() and all(path.exists() for path in required):
        LOGGER.info("using cached role data in %s", directory)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    frames = build_role_frames(role, seed=seed)
    vocab = build_vocab(
        [tokenize(str(row["text"])) for row in frames["train"]], vocab_size, min_freq
    )
    save_vocab(vocab, directory / "vocab.json")

    for split, rows in frames.items():
        save_arrays(directory / f"{split}_raw.npz", _arrays(rows, vocab, max_len))

    meta: dict[str, Any] = {
        "role": role,
        "source": [f"data/synthetic/{name}.jsonl" for name in DEMO_ROLES],
        "vocab_size": len(vocab),
        "max_len": max_len,
        "seed": seed,
        "counts": {split: len(rows) for split, rows in frames.items()},
        "positive_rates": {
            split: float(np.mean([row["y"] for row in rows])) for split, rows in frames.items()
        },
        "female_share": {
            split: float(np.mean([row["g"] for row in rows])) for split, rows in frames.items()
        },
        "note": (
            "Track B synthetic resume summaries; y=1 for the target role and 0 for "
            "the other two demo roles. Bias in Bios has no data_analyst or "
            "machine_learning_engineer label."
        ),
    }
    write_json(meta_path, meta)
    LOGGER.info("prepared %s: %s (vocab %d)", role, meta["counts"], len(vocab))
    return meta


def load_role_split(
    role: str, split: str, data_dir: str | Path | None = None
) -> SplitArrays:
    """Load cached arrays for a role split, failing with the prepare command."""
    path = processed_dir(role, data_dir) / f"{split}_raw.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Prepare the role arrays with:\n"
            f"  python scripts/train_roles.py --prepare-only"
        )
    return load_arrays(path)
