"""Shared utilities: seeding, configuration, paths and small IO helpers.

All scripts import from here so that a run is fully described by a YAML config
plus a seed. Nothing in this module hardcodes a data or results path.
"""

from __future__ import annotations

import json
import logging
import random
import subprocess
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"

LOGGER_NAME = "hirefair"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return the project logger, configured once with a plain formatter."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (including CUDA) in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device string. ``auto`` prefers CUDA, then CPU.

    MPS is opt-in. The BiLSTM path is numerically checked on CPU, so ``auto``
    stays on CPU rather than silently switching backends.
    """
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return device


@dataclass
class DataConfig:
    """Data and preprocessing settings."""

    dataset: str = "LabHC/bias_in_bios"
    max_train: int = 80000
    max_eval: int = 20000
    max_len: int = 128
    vocab_size: int = 30000
    min_freq: int = 3
    split_seed: int = 42


@dataclass
class ModelConfig:
    """Encoder and head settings."""

    emb_dim: int = 100
    hidden: int = 128
    dropout: float = 0.3
    glove_path: str | None = None


@dataclass
class AdvConfig:
    """Adversary settings. ``mode`` is one of ``none``, ``dp``, ``eo``, ``eodds``."""

    mode: str = "dp"
    lambda_max: float = 1.0
    schedule: str = "ganin"


@dataclass
class TrainConfig:
    """Optimisation settings."""

    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: int = 128
    epochs: int = 8
    patience: int = 2
    clip: float = 1.0
    num_workers: int = 0
    device: str = "auto"
    amp: str = "auto"


@dataclass
class PathsConfig:
    """Paths, relative to the project root unless absolute."""

    data_dir: str = "data"
    results_dir: str = "results"


@dataclass
class Config:
    """Full configuration for one run."""

    role: str
    variant: str
    seed: int = 0
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    adv: AdvConfig = field(default_factory=AdvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    quick: bool = False
    tag: str | None = None

    # ---------------------------------------------------------------- paths

    @property
    def data_dir(self) -> Path:
        """Absolute data directory, created on access."""
        path = Path(self.paths.data_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def results_root(self) -> Path:
        """Absolute results directory, created on access."""
        path = Path(self.paths.results_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def run_dir(self) -> Path:
        """Directory for this run's artefacts."""
        parts = [self.role, self.variant]
        if self.tag:
            parts.append(self.tag)
        parts.append(f"seed{self.seed}")
        return self.results_root.joinpath(*parts)

    def processed_dir(self, role: str | None = None) -> Path:
        """Directory holding tokenised arrays for a role.

        Quick runs use ``processed_quick`` so a smoke test never replaces the
        arrays used for the reported results.
        """
        base = "processed_quick" if self.quick else "processed"
        path = self.data_dir / base / (role or self.role)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # ---------------------------------------------------------------- config

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly dict of the full config."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Config":
        """Build a validated config from a nested dict."""
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        payload = dict(raw)
        for key, dc_type in (
            ("data", DataConfig),
            ("model", ModelConfig),
            ("adv", AdvConfig),
            ("train", TrainConfig),
            ("paths", PathsConfig),
        ):
            value = payload.get(key) or {}
            allowed = {f.name for f in fields(dc_type)}
            extra = set(value) - allowed
            if extra:
                raise ValueError(f"unknown {key} keys: {sorted(extra)}")
            payload[key] = dc_type(**value)
        cfg = cls(**payload)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Check enum-like fields early so failures are loud."""
        allowed_modes = {"none", "dp", "eo", "eodds"}
        if self.adv.mode not in allowed_modes:
            raise ValueError(f"adv.mode must be one of {sorted(allowed_modes)}")
        if self.adv.schedule not in {"ganin", "constant"}:
            raise ValueError("adv.schedule must be 'ganin' or 'constant'")
        if self.data.max_len < 8:
            raise ValueError("data.max_len is too small")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def apply_quick(self) -> "Config":
        """Shrink the run to a smoke test: 1,000 train rows and one epoch.

        Quick runs write to ``data/quick`` and ``results_quick`` so a smoke
        test can never overwrite the reported results.
        """
        self.quick = True
        self.data.max_train = 1000
        self.data.max_eval = 2000
        self.train.epochs = 1
        self.train.patience = 0
        self.train.batch = min(self.train.batch, 64)
        self.paths.results_dir = "results_quick"
        self.tag = self.tag or "quick"
        return self

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, Mapping[str, Any]] | None = None,
        quick: bool = False,
        profile: str | Path | None = None,
    ) -> "Config":
        """Load ``configs/base.yaml``, an optional compute profile, ``path``,
        then explicit overrides. Later layers win.
        """
        raw: dict[str, Any] = {}
        base_path = CONFIG_DIR / "base.yaml"
        if base_path.exists():
            raw = _deep_merge(raw, _read_yaml(base_path))
        if profile is not None:
            raw = _deep_merge(raw, _read_yaml(Path(profile)))
        if path is not None:
            raw = _deep_merge(raw, _read_yaml(Path(path)))
        for section, values in (overrides or {}).items():
            if section in {"role", "variant", "seed", "tag"}:
                raw[section] = values
            else:
                raw[section] = _deep_merge(raw.get(section, {}), values)
        config = cls.from_dict(raw)
        return config.apply_quick() if quick else config

    def save(self, path: str | Path) -> Path:
        """Write the resolved config next to a run for reproducibility."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# --------------------------------------------------------------------- IO


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    """Write JSON with stable indentation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    """Read JSON from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def git_commit() -> str:
    """Return the current commit hash, or ``"unknown"`` outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def dataclass_to_dict(value: Any) -> Any:
    """Recursively convert dataclasses to plain dicts (for logging)."""
    if is_dataclass(value):
        return {f.name: dataclass_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {k: dataclass_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dataclass_to_dict(v) for v in value]
    return value
