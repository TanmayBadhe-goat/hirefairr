import math

import pytest

from hirefair.training.schedule import ganin_lambda


def test_endpoints():
    assert ganin_lambda(0, 100, 1.0) == pytest.approx(0.0)
    # The sigmoid does not reach exactly 1 at p = 1.
    assert ganin_lambda(100, 100, 1.0) == pytest.approx(1.0, abs=1e-3)
    assert ganin_lambda(100, 100, 3.0) == pytest.approx(3.0, abs=3e-3)


def test_midpoint_matches_formula():
    expected = 1.0 * (2 / (1 + math.exp(-5)) - 1)
    assert ganin_lambda(50, 100, 1.0) == pytest.approx(expected)


def test_monotone_increasing():
    values = [ganin_lambda(step, 100, 2.0) for step in range(0, 101, 10)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_constant_schedule():
    assert ganin_lambda(10, 100, 1.5, schedule="constant") == 1.5


def test_unknown_schedule_raises():
    with pytest.raises(ValueError):
        ganin_lambda(10, 100, 1.0, schedule="cosine")
