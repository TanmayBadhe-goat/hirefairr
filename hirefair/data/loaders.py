"""Bias in Bios loader and canonical schema.

Source: Hugging Face ``LabHC/bias_in_bios`` (De-Arteaga et al., 2019), the
Ravfogel et al. (2020) version. Expected columns are ``hard_text``,
``profession`` and ``gender``, with integer labels. The profession label map
and the gender map are documented on the dataset card and are re-verified at
load time by keyword and pronoun checks.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from hirefair.data.preprocess import tokenize
from hirefair.utils import Config, get_logger

LOGGER = get_logger()

PROFESSIONS: list[str] = [
    "accountant",
    "architect",
    "attorney",
    "chiropractor",
    "comedian",
    "composer",
    "dentist",
    "dietitian",
    "dj",
    "filmmaker",
    "interior_designer",
    "journalist",
    "model",
    "nurse",
    "painter",
    "paralegal",
    "pastor",
    "personal_trainer",
    "photographer",
    "physician",
    "poet",
    "professor",
    "psychologist",
    "rapper",
    "software_engineer",
    "surgeon",
    "teacher",
    "yoga_teacher",
]
PROFESSION_TO_ID = {name: i for i, name in enumerate(PROFESSIONS)}

# Documented on the dataset card: Male = 0, Female = 1. Verified below.
GENDER_MAP = {0: "M", 1: "F"}

ROLE_KEYWORDS = {
    "software_engineer": ("software", "engineer", "developer", "programmer"),
    "nurse": ("nurse", "nursing"),
    "surgeon": ("surgeon", "surgical"),
}

DEFAULT_ROLES = ("software_engineer", "nurse", "surgeon")

_PRONOMINAL_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SHE = re.compile(r"\b(she|her|hers)\b", re.IGNORECASE)
_HE = re.compile(r"\b(he|him|his)\b", re.IGNORECASE)


def _first_sentence(text: str) -> str:
    """Keep the first sentence of a biography, falling back to the whole text."""
    text = " ".join(str(text).split())
    if not text:
        return text
    parts = _PRONOMINAL_SPLIT.split(text, maxsplit=1)
    sentence = parts[0].strip()
    # Very short first fragments are usually initials or honorifics; keep more.
    if len(sentence.split()) >= 5:
        return sentence
    return text


def _load_from_hub(cfg: Config, raw_dir: Path) -> pd.DataFrame:
    """Download the dataset once and cache the canonical frame as parquet."""
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError(
            "the 'datasets' package is required; run: pip install -r requirements.txt"
        ) from exc

    LOGGER.info("downloading %s (first run only)", cfg.data.dataset)
    try:
        dataset = load_dataset(cfg.data.dataset, cache_dir=str(raw_dir / "hf_cache"))
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"could not load {cfg.data.dataset!r} from Hugging Face. "
            "Check the dataset name and network access, then rerun. "
            f"Original error: {exc}"
        ) from exc

    expected = {"hard_text", "profession", "gender"}
    frames = []
    for split_name, out_split in (("train", "train"), ("dev", "val"), ("test", "test")):
        if split_name not in dataset:
            LOGGER.warning("split %s missing from dataset", split_name)
            continue
        split = dataset[split_name]
        missing = expected - set(split.column_names)
        if missing:
            raise RuntimeError(
                f"dataset split {split_name!r} is missing columns {sorted(missing)}; "
                f"found {split.column_names}"
            )
        frame = split.to_pandas()
        frame["split"] = out_split
        frames.append(frame[["hard_text", "profession", "gender", "split"]])
    if not frames:
        raise RuntimeError("no usable splits in the dataset")
    raw = pd.concat(frames, ignore_index=True)
    raw = raw.rename(columns={"hard_text": "text"})
    raw["text"] = raw["text"].map(_first_sentence)
    raw["occupation"] = raw["profession"].map(
        lambda value: PROFESSIONS[int(value)] if 0 <= int(value) < len(PROFESSIONS) else None
    )
    if raw["occupation"].isna().any():
        raise RuntimeError("profession label outside the documented 28-class range")
    raw["gender"] = raw["gender"].map(lambda value: GENDER_MAP[int(value)])
    if raw["gender"].isna().any():
        raise RuntimeError("gender label outside the documented {0, 1} range")
    raw = raw.drop(columns=["profession"])
    raw = raw.reset_index(drop=True)
    raw.insert(0, "id", np.arange(len(raw), dtype=np.int64))
    return raw[["id", "text", "occupation", "gender", "split"]]


def _pronoun_counts(frame: pd.DataFrame, label: str) -> tuple[int, int]:
    subset = frame[frame["gender"] == label]["text"]
    she = sum(len(_SHE.findall(text)) for text in subset)
    he = sum(len(_HE.findall(text)) for text in subset)
    return she, he


def check_pronoun_mapping(frame: pd.DataFrame) -> dict[str, int]:
    """Fail loudly when gender labels disagree with third-person pronouns.

    Returns the she/her and he/his counts for the F group after verification.
    """
    she, he = _pronoun_counts(frame, "F")
    if she > he:
        return {"she_her": she, "he_him": he}
    # Try the flipped mapping once before failing.
    flipped = frame.copy()
    flipped["gender"] = flipped["gender"].map({"M": "F", "F": "M"})
    she_f, he_f = _pronoun_counts(flipped, "F")
    if she_f > he_f:
        raise RuntimeError(
            "gender labels are flipped: rows labelled F are dominated by he/him. "
            "Fix GENDER_MAP before continuing."
        )
    raise RuntimeError(
        f"pronoun sanity check failed: F rows have she/her={she_f} and he/his={he_f}"
    )


def check_profession_mapping(frame: pd.DataFrame, roles: tuple[str, ...] = DEFAULT_ROLES) -> None:
    """Verify that the documented profession ids match their keyword content.

    Biographies rarely name their own occupation, so the test is relative: the
    keyword must be several times more common inside the labelled group than
    across the whole corpus.
    """
    lowered = frame["text"].str.lower()
    for role in roles:
        keywords = ROLE_KEYWORDS[role]
        mask = frame["occupation"] == role
        if mask.sum() == 0:
            raise RuntimeError(f"role {role!r} not present in the dataset")
        in_role = float(lowered[mask].apply(lambda text: any(k in text for k in keywords)).mean())
        overall = float(lowered.apply(lambda text: any(k in text for k in keywords)).mean())
        if in_role < 0.05 or in_role < 3.0 * overall:
            raise RuntimeError(
                f"profession label check failed for {role!r}: keyword share {in_role:.1%} "
                f"inside the group vs {overall:.1%} overall. The label map may have changed."
            )


def load_canonical(cfg: Config, force_reload: bool = False) -> pd.DataFrame:
    """Load the canonical frame, caching the download as parquet.

    Columns: ``id``, ``text``, ``occupation``, ``gender`` (M/F), ``split``
    (train/val/test). The pronoun and profession checks run on every load.
    """
    raw_dir = cfg.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = raw_dir / "bios.parquet"
    if parquet_path.exists() and not force_reload:
        frame = pd.read_parquet(parquet_path)
    else:
        frame = _load_from_hub(cfg, raw_dir)
        frame.to_parquet(parquet_path, index=False)
        LOGGER.info("cached canonical frame to %s", parquet_path)

    required = {"id", "text", "occupation", "gender", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"canonical frame missing columns {sorted(missing)}")
    if frame["text"].isna().any() or (frame["text"].str.len() == 0).any():
        raise RuntimeError("canonical frame contains empty biographies")
    if not set(frame["gender"].unique()) <= {"M", "F"}:
        raise RuntimeError("gender column must contain only 'M' and 'F'")
    if not set(frame["occupation"].unique()) <= set(PROFESSIONS):
        raise RuntimeError("occupation column contains unknown labels")

    stats = check_pronoun_mapping(frame)
    LOGGER.info("pronoun check passed: F rows she/her=%d he/him=%d", stats["she_her"], stats["he_him"])
    check_profession_mapping(frame)
    return frame.reset_index(drop=True)


def official_splits(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test, falling back to a stratified split if needed."""
    splits = set(frame["split"].unique())
    if {"train", "val", "test"} <= splits:
        train = frame[frame["split"] == "train"]
        val = frame[frame["split"] == "val"]
        test = frame[frame["split"] == "test"]
        return train.copy(), val.copy(), test.copy()

    LOGGER.warning("official splits missing; creating 70/10/20 stratified splits")
    from sklearn.model_selection import train_test_split

    strat = frame["occupation"].astype(str) + "|" + frame["gender"]
    train, rest = train_test_split(frame, test_size=0.3, random_state=42, stratify=strat)
    val, test = train_test_split(
        rest, test_size=2 / 3, random_state=42, stratify=strat.loc[rest.index]
    )
    return train.copy(), val.copy(), test.copy()


def _stratified_cap(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Take at most ``cap`` rows, proportionally within (occupation, gender)."""
    if len(frame) <= cap:
        return frame.copy()
    frac = cap / len(frame)
    frame = frame.assign(_key=frame["occupation"].astype(str) + "|" + frame["gender"].astype(str))
    sizes = frame.groupby("_key").size()
    take = np.maximum(1, np.floor(sizes * frac).astype(int))
    rng = np.random.default_rng(seed)
    parts = []
    for key, group in frame.groupby("_key", sort=True):
        n = min(int(take.loc[key]), len(group))
        idx = rng.choice(len(group), size=n, replace=False)
        parts.append(group.iloc[np.sort(idx)])
    capped = pd.concat(parts).drop(columns=["_key"])
    return capped.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def build_role_frames(cfg: Config, role: str) -> dict[str, pd.DataFrame]:
    """Return train/val/test frames for one target role with the ``y`` label."""
    if role not in PROFESSIONS:
        raise ValueError(f"unknown role {role!r}")
    frame = load_canonical(cfg)
    train, val, test = official_splits(frame)
    train = _stratified_cap(train, cfg.data.max_train, cfg.data.split_seed)
    val = _stratified_cap(val, cfg.data.max_eval, cfg.data.split_seed)
    test = _stratified_cap(test, cfg.data.max_eval, cfg.data.split_seed)

    out = {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        part = part.copy()
        part["y"] = (part["occupation"] == role).astype(np.int8)
        part["g"] = (part["gender"] == "F").astype(np.int8)
        if part["y"].sum() == 0:
            raise RuntimeError(f"role {role!r} has no positives in the {name} split")
        out[name] = part.reset_index(drop=True)
    return out


def dataset_statistics(frames: dict[str, pd.DataFrame]) -> dict[str, float | int]:
    """Per-role statistics used by Table T1."""
    train, val, test = frames["train"], frames["val"], frames["test"]
    all_rows = pd.concat([train, val, test], ignore_index=True)
    lengths = [len(tokenize(text)) for text in all_rows["text"]]
    positives = all_rows[all_rows["y"] == 1]
    return {
        "n_total": int(len(all_rows)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "positive_rate": float(all_rows["y"].mean()),
        "positive_rate_M": float(all_rows.loc[all_rows["gender"] == "M", "y"].mean()),
        "positive_rate_F": float(all_rows.loc[all_rows["gender"] == "F", "y"].mean()),
        "female_share_of_positives": float((positives["gender"] == "F").mean()),
        "male_share_of_positives": float((positives["gender"] == "M").mean()),
        "mean_tokens": float(np.mean(lengths)),
        "median_tokens": float(np.median(lengths)),
    }


def save_dataset_statistics(
    cfg: Config, roles: tuple[str, ...] = DEFAULT_ROLES
) -> pd.DataFrame:
    """Compute and persist T1 source data for every role."""
    frame = load_canonical(cfg)
    pronoun_stats = check_pronoun_mapping(frame)
    checks = {
        "n_rows": int(len(frame)),
        "n_occupations": len(PROFESSIONS),
        "she_her": int(pronoun_stats["she_her"]),
        "he_him": int(pronoun_stats["he_him"]),
        "gender_map": "Male=0, Female=1 (verified by pronoun counts)",
    }
    (cfg.results_root / "dataset_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    rows = []
    for role in roles:
        frames = build_role_frames(cfg, role)
        stats = dataset_statistics(frames)
        stats["role"] = role
        rows.append(stats)
        LOGGER.info("stats %s: n=%d positive rate=%.3f", role, stats["n_total"], stats["positive_rate"])
    frame = pd.DataFrame(rows)[
        [
            "role",
            "n_total",
            "n_train",
            "n_val",
            "n_test",
            "positive_rate",
            "positive_rate_M",
            "positive_rate_F",
            "female_share_of_positives",
            "male_share_of_positives",
            "mean_tokens",
            "median_tokens",
        ]
    ]
    out = cfg.results_root / "t1_dataset_stats.csv"
    frame.to_csv(out, index=False)
    (cfg.results_root / "dataset_stats.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    LOGGER.info("wrote %s", out)
    return frame
