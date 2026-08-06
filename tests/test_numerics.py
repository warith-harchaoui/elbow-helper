"""Tests for the numpy-only numerical primitives.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper.numerics import (
    bic,
    peak_prominence,
    rankdata,
    robust_sigma_from_diffs,
    spearman,
    theil_sen_slope,
)


def test_rankdata_averages_ties():
    r = rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    assert list(r) == [1.0, 2.5, 2.5, 4.0]


def test_spearman_monotone_is_one():
    x = np.arange(50, dtype=float)
    y = np.exp(x / 10.0)  # strictly increasing, nonlinear
    assert spearman(x, y) == 1.0
    assert spearman(x, -y) == -1.0


def test_peak_prominence_of_triangle():
    signal = np.array([0.0, 1.0, 2.0, 5.0, 2.0, 1.0, 0.0])
    assert peak_prominence(signal, 3) == 5.0


def test_theil_sen_recovers_slope():
    x = np.linspace(0, 1, 40)
    y = 3.0 * x + 1.0
    assert abs(theil_sen_slope(x, y) - 3.0) < 1e-6


def test_robust_sigma_scales_with_noise():
    rng = np.random.default_rng(0)
    base = np.linspace(0, 1, 200)
    s_lo = robust_sigma_from_diffs(base + rng.normal(0, 0.01, 200))
    s_hi = robust_sigma_from_diffs(base + rng.normal(0, 0.10, 200))
    assert s_hi > s_lo > 0


def test_bic_prefers_lower_rss():
    assert bic(1.0, 100, 2) < bic(10.0, 100, 2)
