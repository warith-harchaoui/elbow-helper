"""Tests for curve/direction auto-detection and single-list input.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import ClearKnee, robust_elbow, robust_knee
from elbow_helper.preprocessing import infer_curve_direction

from conftest import clear_knee_curve, elbow_curve


def test_infer_concave_increasing():
    x, y = clear_knee_curve(seed=1, noise=0.0)
    curve, direction = infer_curve_direction(x, y)
    assert (curve, direction) == ("concave", "increasing")


def test_infer_concave_decreasing():
    x, y = clear_knee_curve(seed=1, noise=0.0)
    curve, direction = infer_curve_direction(x[::-1], y)
    assert (curve, direction) == ("concave", "decreasing")


def test_infer_convex_decreasing():
    x, y = elbow_curve(seed=1, noise=0.0)
    curve, direction = infer_curve_direction(x, y)
    assert (curve, direction) == ("convex", "decreasing")


def test_infer_convex_increasing():
    x, y = elbow_curve(seed=1, noise=0.0)
    curve, direction = infer_curve_direction(x[::-1], y)
    assert (curve, direction) == ("convex", "increasing")


def test_robust_knee_auto_matches_explicit(fast_config):
    x, y = clear_knee_curve(seed=1, knee_frac=0.3, noise=0.02)
    auto = robust_knee(x, y, config=fast_config)
    explicit = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(auto, ClearKnee)
    assert isinstance(explicit, ClearKnee)
    assert auto.knee_x == explicit.knee_x
    assert auto.diagnostics["curve"] == "concave"
    assert auto.diagnostics["direction"] == "increasing"


def test_robust_knee_single_list_matches_explicit_x(fast_config):
    _, y = clear_knee_curve(seed=1, knee_frac=0.3, n=80, noise=0.02)
    x = np.arange(80, dtype=float)
    from_pair = robust_knee(x, y, "concave", "increasing", fast_config)
    from_single = robust_knee(y, config=fast_config)
    assert isinstance(from_pair, ClearKnee)
    assert isinstance(from_single, ClearKnee)
    assert from_pair.knee_x == from_single.knee_x
    assert from_pair.diagnostics["curve"] == from_single.diagnostics["curve"]
    assert from_pair.diagnostics["direction"] == from_single.diagnostics["direction"]


def test_robust_elbow_single_list(fast_config):
    _, y = elbow_curve(seed=1, knee_frac=0.25, n=60, noise=0.02)
    x = np.arange(60, dtype=float)
    from_pair = robust_elbow(x, y, fast_config)
    from_single = robust_elbow(y, config=fast_config)
    assert isinstance(from_pair, ClearKnee)
    assert isinstance(from_single, ClearKnee)
    assert from_pair.knee_x == from_single.knee_x
