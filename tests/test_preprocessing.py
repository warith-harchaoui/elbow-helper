"""Tests for Phase 1 preprocessing and its abstention signals."""

from __future__ import annotations

import numpy as np
import pytest

from elbow_helper import RobustKneeConfig
from elbow_helper.preprocessing import Abstain, prepare_curve

CFG = RobustKneeConfig()


def test_cleans_sorts_and_deduplicates():
    x = np.array([3.0, 1.0, 2.0, 2.0, np.nan, 5.0, 4.0] + list(range(6, 25)), float)
    y = np.array([0.3, 0.1, 0.2, 0.25, 9.9, 0.5, 0.4] + [0.5] * 19, float)
    prepared = prepare_curve(x, y, "concave", "increasing", CFG)
    assert prepared.n == 24  # nan pair dropped, one dup x aggregated
    assert np.all(np.diff(prepared.x_norm) > 0)  # strictly increasing
    assert prepared.x_norm[0] == 0.0 and prepared.x_norm[-1] == 1.0


def test_insufficient_data_abstains():
    x = np.arange(5, dtype=float)
    y = np.sqrt(x)
    with pytest.raises(Abstain) as e:
        prepare_curve(x, y, "concave", "increasing", CFG)
    assert e.value.reason == "INSUFFICIENT_DATA"


def test_zero_range_abstains():
    x = np.arange(30, dtype=float)
    y = np.ones(30)
    with pytest.raises(Abstain) as e:
        prepare_curve(x, y, "concave", "increasing", CFG)
    assert e.value.reason == "ZERO_RANGE"


def test_incompatible_shape_abstains():
    rng = np.random.default_rng(0)
    x = np.arange(60, dtype=float)
    y = rng.normal(0, 1, 60)  # no monotone trend
    with pytest.raises(Abstain) as e:
        prepare_curve(x, y, "concave", "increasing", CFG)
    assert e.value.reason == "INCOMPATIBLE_GLOBAL_SHAPE"


def test_length_mismatch_abstains():
    with pytest.raises(Abstain) as e:
        prepare_curve(np.arange(10.0), np.arange(9.0), "concave", "increasing", CFG)
    assert e.value.reason == "INVALID_INPUT"
