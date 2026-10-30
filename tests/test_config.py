import pytest

from hirefair.utils import Config


def test_load_base_config():
    cfg = Config.load()
    assert cfg.role == "software_engineer"
    assert cfg.data.max_train == 80000
    assert cfg.train.epochs == 8


def test_role_variant_config_overrides_mode():
    cfg = Config.load("configs/nurse_B3.yaml")
    assert cfg.role == "nurse"
    assert cfg.variant == "B3"
    assert cfg.adv.mode == "eo"


def test_quick_mode_is_isolated():
    cfg = Config.load("configs/nurse_B0.yaml", quick=True)
    assert cfg.data.max_train == 1000
    assert cfg.train.epochs == 1
    assert cfg.paths.results_dir == "results_quick"
    assert cfg.processed_dir("nurse").name == "nurse"
    assert cfg.processed_dir("nurse").parent.name == "processed_quick"


def test_unknown_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"role": "nurse", "variant": "B0", "unknown": 1})


def test_unknown_nested_key_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"role": "nurse", "variant": "B0", "train": {"epochz": 2}})


def test_invalid_adv_mode_raises():
    with pytest.raises(ValueError):
        Config.from_dict({"role": "nurse", "variant": "B3", "adv": {"mode": "parity"}})


def test_cli_overrides_merge():
    cfg = Config.load(
        "configs/nurse_B3.yaml", overrides={"adv": {"lambda_max": 0.3}, "train": {"epochs": 3}}
    )
    assert cfg.adv.lambda_max == 0.3
    assert cfg.train.epochs == 3
    assert cfg.adv.mode == "eo"
