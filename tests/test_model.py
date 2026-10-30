import numpy as np
import torch

from hirefair.data.preprocess import SplitArrays
from hirefair.models.hirefair_model import HireFairModel
from hirefair.training.losses import adversary_loss, task_loss
from hirefair.training.trainer import Trainer
from hirefair.utils import Config, set_seed


def _cfg(adv_mode="none", epochs=1):
    return Config.from_dict(
        {
            "role": "nurse",
            "variant": "B0" if adv_mode == "none" else "B3",
            "seed": 0,
            "data": {"max_len": 16},
            "model": {"emb_dim": 8, "hidden": 8, "dropout": 0.0},
            "adv": {"mode": adv_mode, "lambda_max": 1.0},
            "train": {"batch": 8, "epochs": epochs, "patience": 0, "lr": 0.01, "device": "cpu"},
        }
    )


def _arrays(n=64, length=12, vocab=40, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(2, vocab, size=(n, length)).astype(np.int32)
    mask = np.ones((n, length), dtype=np.int8)
    y = rng.integers(0, 2, size=n).astype(np.int8)
    g = rng.integers(0, 2, size=n).astype(np.int8)
    lengths = np.full(n, length, dtype=np.int16)
    return SplitArrays(ids=ids, mask=mask, lengths=lengths, y=y, g=g)


def test_adv_logits_exist_only_for_enabled_modes():
    ids = torch.randint(2, 40, (4, 10))
    mask = torch.ones(4, 10)
    y = torch.tensor([1.0, 0.0, 1.0, 1.0])
    baseline = HireFairModel(40, 8, 8, 0.0, adv_mode="none")
    assert baseline(ids, mask, y)["adv_logits"] is None
    dp_model = HireFairModel(40, 8, 8, 0.0, adv_mode="dp")
    assert dp_model(ids, mask, y)["adv_logits"].shape[0] == 4


def test_eo_mode_adversary_receives_only_positives():
    ids = torch.randint(2, 40, (4, 10))
    mask = torch.ones(4, 10)
    y = torch.tensor([1.0, 0.0, 1.0, 1.0])
    model = HireFairModel(40, 8, 8, 0.0, adv_mode="eo")
    out = model(ids, mask, y)
    assert out["adv_logits"].shape[0] == 3
    assert int(out["adv_mask"].sum()) == 3
    assert torch.equal(out["adv_mask"], y > 0.5)


def test_eo_mode_with_no_positives_skips_adversary():
    ids = torch.randint(2, 40, (3, 10))
    mask = torch.ones(3, 10)
    y = torch.zeros(3)
    model = HireFairModel(40, 8, 8, 0.0, adv_mode="eo")
    out = model(ids, mask, y)
    assert out["adv_logits"] is None


def test_lambda_zero_matches_baseline():
    ids = torch.randint(2, 40, (6, 10))
    mask = torch.ones(6, 10)
    y = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    g = torch.tensor([0, 1, 0, 1, 1, 0])

    set_seed(3)
    baseline = HireFairModel(40, 8, 8, 0.0, adv_mode="none")
    set_seed(3)
    debiased = HireFairModel(40, 8, 8, 0.0, adv_mode="eo")
    debiased.encoder.load_state_dict(baseline.encoder.state_dict())
    debiased.predictor.load_state_dict(baseline.predictor.state_dict())
    debiased.set_lambda(0.0)

    out_base = baseline(ids, mask, y)
    out_debiased = debiased(ids, mask, y)
    assert torch.allclose(out_base["logit"], out_debiased["logit"], atol=1e-6)

    loss_base = task_loss(out_base["logit"], y)
    loss_debiased = task_loss(out_debiased["logit"], y)
    assert torch.allclose(loss_base, loss_debiased, atol=1e-6)

    loss_base.backward()
    eo_gender = g[y > 0.5]
    total = loss_debiased + adversary_loss(out_debiased["adv_logits"], eo_gender)
    total.backward()
    assert torch.allclose(
        baseline.encoder.embedding.weight.grad,
        debiased.encoder.embedding.weight.grad,
        atol=1e-6,
    )


def test_seeded_run_reproduces_val_f1():
    arrays = _arrays()
    val = _arrays(n=32, seed=1)

    set_seed(11)
    cfg = _cfg("none", epochs=2)
    model_a = HireFairModel(40, 8, 8, 0.0, adv_mode="none")
    result_a = Trainer(model_a, cfg).fit(arrays, val)

    set_seed(11)
    model_b = HireFairModel(40, 8, 8, 0.0, adv_mode="none")
    result_b = Trainer(model_b, cfg).fit(arrays, val)

    assert abs(result_a.val_metrics["f1"] - result_b.val_metrics["f1"]) < 1e-4
    assert abs(result_a.threshold - result_b.threshold) < 1e-6


def test_overfits_two_hundred_rows():
    rng = np.random.default_rng(5)
    n, length = 200, 12
    ids = rng.integers(4, 40, size=(n, length)).astype(np.int32)
    # Make the label easy by planting a token in every positive row.
    labels = rng.integers(0, 2, size=n).astype(np.int8)
    ids[labels == 1, 0] = 3
    mask = np.ones((n, length), dtype=np.int8)
    genders = rng.integers(0, 2, size=n).astype(np.int8)
    arrays = SplitArrays(
        ids=ids,
        mask=mask,
        lengths=np.full(n, length, dtype=np.int16),
        y=labels,
        g=genders,
    )
    cfg = _cfg("none", epochs=30)
    cfg.train.patience = 0
    set_seed(7)
    model = HireFairModel(40, 8, 8, 0.0, adv_mode="none")
    trainer = Trainer(model, cfg)
    trainer.fit(arrays, arrays)
    prediction = trainer.predict(arrays, with_adversary=False)
    accuracy = float(((prediction["scores"] >= 0.5).astype(int) == prediction["y"]).mean())
    assert accuracy > 0.95, accuracy
