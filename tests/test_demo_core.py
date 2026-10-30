import numpy as np
import pytest

from hirefair.demo import core
from hirefair.models.hirefair_model import HireFairModel, save_checkpoint


def _write_checkpoint(root, role, variant, threshold, vocab):
    run = root / role / variant / "seed0"
    model = HireFairModel(
        vocab_size=len(vocab),
        emb_dim=8,
        hidden=8,
        dropout=0.0,
        adv_mode="none",
        vocab=vocab,
    )
    config = {"model": {"emb_dim": 8, "hidden": 8, "dropout": 0.0}, "data": {"max_len": 16}}
    save_checkpoint(run / "model.pt", model, threshold, vocab, config, {})


@pytest.fixture()
def demo_results(tmp_path):
    vocab = {"<pad>": 0, "<unk>": 1, "she": 2, "he": 3, "nurse": 4, "priya": 5, "rahul": 6}
    _write_checkpoint(tmp_path, "nurse", "B0", 0.5, vocab)
    _write_checkpoint(tmp_path, "nurse", "B3", 0.7, vocab)
    return tmp_path


def test_score_pair_returns_both_models(demo_results):
    result = core.score_pair(
        "She is a nurse. Her name is Priya.", "nurse", results_dir=demo_results
    )
    assert set(result.models) == {"B0", "B3"}
    for entry in result.models.values():
        assert 0.0 <= entry.score_original <= 1.0
        assert 0.0 <= entry.score_swapped <= 1.0
        assert entry.decision_original in (0, 1)
        assert entry.flipped == (entry.decision_original != entry.decision_swapped)
        assert entry.delta == pytest.approx(entry.score_swapped - entry.score_original)


def test_score_pair_swaps_cues(demo_results):
    result = core.score_pair("She is a nurse.", "nurse", results_dir=demo_results)
    assert "He is a nurse" in result.swapped
    assert result.cues
    assert not result.truncated


def test_score_pair_flags_truncation(demo_results):
    text = "She is a nurse. " + "word " * 40
    result = core.score_pair(text, "nurse", results_dir=demo_results)
    assert result.truncated
    assert result.n_tokens > 16


def test_missing_checkpoint_raises_with_command(tmp_path):
    with pytest.raises(core.MissingCheckpoint) as exc:
        core.score_pair("She is a nurse.", "nurse", results_dir=tmp_path)
    assert "scripts/run_experiment.py" in str(exc.value)


def test_sample_pool_is_deterministic():
    rows = [{"id": i} for i in range(500)]
    first = core.sample_pool(rows, 50, seed=7)
    second = core.sample_pool(rows, 50, seed=7)
    other = core.sample_pool(rows, 50, seed=8)
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert [row["id"] for row in first] != [row["id"] for row in other]
    assert len(first) == 50
    assert first[0]["id"] <= first[-1]["id"]


def test_result_table_uses_stored_thresholds(demo_results):
    result = core.score_pair("She is a nurse.", "nurse", results_dir=demo_results)
    table = core.result_table(result)
    assert len(table) == 2
    assert table[0][0] == "B0"
    assert table[0][2] in {"SELECT", "REJECT"}


def test_cli_format_contains_expected_lines(demo_results):
    from hirefair.demo.cli import format_result

    result = core.score_pair("She is a nurse.", "nurse", results_dir=demo_results)
    text = format_result(result, None)
    assert text.startswith("ROLE: nurse")
    assert "ORIGINAL" in text
    assert "SWAPPED" in text
    assert "B0 score" in text and "B3 score" in text


def test_highlight_marks_cues(demo_results):
    result = core.score_pair("She is a nurse.", "nurse", results_dir=demo_results)
    html = core.highlight_original(result)
    assert "<mark>" in html
    assert "She" in html
