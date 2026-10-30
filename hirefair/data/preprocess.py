"""Tokenisation, vocabulary, tensors and cached per-role data preparation.

Arrays are written once per role and reused by every variant and seed:

``data/processed/{role}/``
    ``vocab.json``
    ``{split}_{mode}.npz`` for split in train/val/test and mode in
    raw/masked/swapped (train_swapped only when augmentation is prepared)
    ``test_meta.jsonl`` original and swapped test text for the demo
    ``meta.json`` counts and preparation settings
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from hirefair.data.swap import swap_many
from hirefair.utils import Config, get_logger

LOGGER = get_logger()

TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9']")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1


def tokenize(text: str) -> list[str]:
    """Lowercase and split with the regex from the data spec."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(
    token_lists: Iterable[Sequence[str]],
    vocab_size: int = 30000,
    min_freq: int = 3,
) -> dict[str, int]:
    """Build a vocabulary from training tokens only."""
    counts: Counter[str] = Counter()
    for tokens in token_lists:
        counts.update(tokens)
    vocab: dict[str, int] = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, count in counts.most_common():
        if count < min_freq or len(vocab) >= vocab_size:
            break
        vocab[token] = len(vocab)
    return vocab


def save_vocab(vocab: dict[str, int], path: str | Path) -> Path:
    """Write a vocabulary as JSON with ids in order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, indent=0), encoding="utf-8")
    return path


def load_vocab(path: str | Path) -> dict[str, int]:
    """Read a vocabulary written by :func:`save_vocab`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def encode_tokens(tokens: Sequence[str], vocab: dict[str, int], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Encode one token sequence into padded ids and an attention mask."""
    ids = np.zeros(max_len, dtype=np.int32)
    mask = np.zeros(max_len, dtype=np.int8)
    for i, token in enumerate(tokens[:max_len]):
        ids[i] = vocab.get(token, UNK_ID)
        mask[i] = 1
    return ids, mask


def encode_texts(
    texts: Sequence[str], vocab: dict[str, int], max_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode many texts. Returns ``ids``, ``mask`` and true lengths."""
    ids = np.zeros((len(texts), max_len), dtype=np.int32)
    mask = np.zeros((len(texts), max_len), dtype=np.int8)
    lengths = np.zeros(len(texts), dtype=np.int16)
    for row, text in enumerate(texts):
        tokens = tokenize(text)
        row_ids, row_mask = encode_tokens(tokens, vocab, max_len)
        ids[row] = row_ids
        mask[row] = row_mask
        lengths[row] = int(row_mask.sum())
    return ids, mask, lengths


@dataclass
class SplitArrays:
    """Tokenised arrays for one split."""

    ids: np.ndarray
    mask: np.ndarray
    lengths: np.ndarray
    y: np.ndarray
    g: np.ndarray

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def subset(self, index: np.ndarray) -> "SplitArrays":
        """Index into every array."""
        return SplitArrays(
            ids=self.ids[index],
            mask=self.mask[index],
            lengths=self.lengths[index],
            y=self.y[index],
            g=self.g[index],
        )


class TokenDataset(Dataset):
    """Dataset over padded id arrays."""

    def __init__(self, arrays: SplitArrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "ids": torch.from_numpy(self.arrays.ids[index].astype(np.int64)),
            "mask": torch.from_numpy(self.arrays.mask[index].astype(np.float32)),
            "y": torch.tensor(float(self.arrays.y[index]), dtype=torch.float32),
            "g": torch.tensor(int(self.arrays.g[index]), dtype=torch.long),
        }


def collate_trim(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence actually present in the batch."""
    max_len = int(max(int(item["mask"].sum()) for item in batch))
    max_len = max(max_len, 1)
    return {
        "ids": torch.stack([item["ids"][:max_len] for item in batch]),
        "mask": torch.stack([item["mask"][:max_len] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "g": torch.stack([item["g"] for item in batch]),
    }


class BucketBatchSampler(Sampler[list[int]]):
    """Sort by length with jitter, batch, then shuffle the batches.

    Keeps padding low without giving up shuffled batches.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        jitter: float = 4.0,
    ):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        rng = np.random.default_rng(seed)
        noise = rng.random(len(lengths)) * jitter
        self._order = np.argsort(lengths.astype(np.float64) + noise, kind="stable")
        self._batches = [
            self._order[start : start + batch_size].tolist()
            for start in range(0, len(self._order), batch_size)
        ]

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self._batches)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return len(self._batches)


def make_loader(
    arrays: SplitArrays,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader with length-bucketed batches."""
    dataset = TokenDataset(arrays)
    sampler = BucketBatchSampler(arrays.lengths, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_trim,
        num_workers=num_workers,
    )


# ------------------------------------------------------------ preparation


def _npz_path(directory: Path, split: str, mode: str) -> Path:
    return directory / f"{split}_{mode}.npz"


def save_arrays(path: Path, arrays: SplitArrays) -> None:
    """Save a split's arrays compressed."""
    np.savez_compressed(
        path,
        ids=arrays.ids,
        mask=arrays.mask,
        lengths=arrays.lengths,
        y=arrays.y,
        g=arrays.g,
    )


def load_arrays(path: Path) -> SplitArrays:
    """Load a split's arrays."""
    with np.load(path) as data:
        return SplitArrays(
            ids=data["ids"],
            mask=data["mask"],
            lengths=data["lengths"],
            y=data["y"],
            g=data["g"],
        )


def load_split(processed_dir: str | Path, split: str, mode: str = "raw") -> SplitArrays:
    """Load cached arrays for a split and input mode."""
    path = _npz_path(Path(processed_dir), split, mode)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run: python scripts/prepare_data.py --role <role>"
        )
    return load_arrays(path)


def load_train_arrays(
    processed_dir: str | Path, mode: str = "raw", augment: bool = False
) -> SplitArrays:
    """Load training arrays, optionally concatenating swapped copies (B4)."""
    base = load_split(processed_dir, "train", mode)
    if not augment:
        return base
    swapped = load_split(processed_dir, "train", "swapped")
    return SplitArrays(
        ids=np.concatenate([base.ids, swapped.ids]),
        mask=np.concatenate([base.mask, swapped.mask]),
        lengths=np.concatenate([base.lengths, swapped.lengths]),
        y=np.concatenate([base.y, swapped.y]),
        g=np.concatenate([base.g, swapped.g]),
    )


def prepare_role(
    cfg: Config,
    role: str,
    force: bool = False,
    with_augmentation: bool = True,
) -> dict[str, Any]:
    """Tokenise all splits for a role and cache them.

    Builds raw, masked (B1) and swapped (B4, counterfactuals) inputs. The
    vocabulary is built from raw training tokens only.

    Returns a metadata dict describing the prepared files.
    """
    # Imported here to avoid a circular import at module load time.
    from hirefair.data.loaders import build_role_frames

    directory = cfg.processed_dir(role)
    meta_path = directory / "meta.json"
    required = [
        _npz_path(directory, split, mode)
        for split in ("train", "val", "test")
        for mode in ("raw", "masked")
    ] + [_npz_path(directory, "test", "swapped")]
    if with_augmentation:
        required.append(_npz_path(directory, "train", "swapped"))
    if not force and meta_path.exists() and all(path.exists() for path in required):
        LOGGER.info("using cached data in %s", directory)
        return json.loads(meta_path.read_text(encoding="utf-8"))

    frames = build_role_frames(cfg, role)
    texts = {split: frames[split]["text"].tolist() for split in ("train", "val", "test")}

    LOGGER.info("building vocabulary for %s", role)
    token_lists = [tokenize(text) for text in texts["train"]]
    vocab = build_vocab(token_lists, cfg.data.vocab_size, cfg.data.min_freq)
    save_vocab(vocab, directory / "vocab.json")

    meta: dict[str, Any] = {
        "role": role,
        "vocab_size": len(vocab),
        "max_len": cfg.data.max_len,
        "counts": {split: len(values) for split, values in texts.items()},
        "positive_rates": {
            split: float(frames[split]["y"].mean()) for split in ("train", "val", "test")
        },
        "augmentation": bool(with_augmentation),
    }

    for split in ("train", "val", "test"):
        ids, mask, lengths = encode_texts(texts[split], vocab, cfg.data.max_len)
        save_arrays(
            _npz_path(directory, split, "raw"),
            SplitArrays(
                ids=ids,
                mask=mask,
                lengths=lengths,
                y=frames[split]["y"].to_numpy(np.int8),
                g=frames[split]["g"].to_numpy(np.int8),
            ),
        )

        LOGGER.info("masking gender cues for %s/%s", role, split)
        masked_texts = swap_many(texts[split], mode="mask", seed=0)
        ids, mask, lengths = encode_texts(masked_texts, vocab, cfg.data.max_len)
        save_arrays(
            _npz_path(directory, split, "masked"),
            SplitArrays(
                ids=ids,
                mask=mask,
                lengths=lengths,
                y=frames[split]["y"].to_numpy(np.int8),
                g=frames[split]["g"].to_numpy(np.int8),
            ),
        )

    LOGGER.info("swapping and tokenising test rows for %s", role)
    swapped_test = swap_many(texts["test"], mode="swap", seed=0)
    ids, mask, lengths = encode_texts(swapped_test, vocab, cfg.data.max_len)
    save_arrays(
        _npz_path(directory, "test", "swapped"),
        SplitArrays(
            ids=ids,
            mask=mask,
            lengths=lengths,
            y=frames["test"]["y"].to_numpy(np.int8),
            g=frames["test"]["g"].to_numpy(np.int8),
        ),
    )

    if with_augmentation:
        LOGGER.info("swapping and tokenising train rows for %s (B4)", role)
        swapped_train = swap_many(texts["train"], mode="swap", seed=0)
        ids, mask, lengths = encode_texts(swapped_train, vocab, cfg.data.max_len)
        save_arrays(
            _npz_path(directory, "train", "swapped"),
            SplitArrays(
                ids=ids,
                mask=mask,
                lengths=lengths,
                y=frames["train"]["y"].to_numpy(np.int8),
                g=frames["train"]["g"].to_numpy(np.int8),
            ),
        )

    test_frame = frames["test"].reset_index(drop=True)
    with (directory / "test_meta.jsonl").open("w", encoding="utf-8") as handle:
        for i, row in test_frame.iterrows():
            handle.write(
                json.dumps(
                    {
                        "index": int(i),
                        "id": int(row["id"]),
                        "text": row["text"],
                        "swapped_text": swapped_test[int(i)],
                        "gender": row["gender"],
                        "y": int(row["y"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    LOGGER.info("prepared %s: %s", role, meta["counts"])
    return meta
