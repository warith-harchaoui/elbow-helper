"""Tests that abstentions carry stable reason codes and diagnostics.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import NoClearKnee, Reason, robust_knee

CFG_REASONS = {
    Reason.INSUFFICIENT_DATA,
    Reason.INVALID_INPUT,
    Reason.ZERO_RANGE,
    Reason.INCOMPATIBLE_GLOBAL_SHAPE,
    Reason.NO_KNEE_CANDIDATES,
    Reason.ALL_CANDIDATES_WEAK,
    Reason.NO_PERSISTENT_CLUSTER,
    Reason.MULTIPLE_PLAUSIBLE_KNEES,
    Reason.BOUNDARY_KNEE,
    Reason.WEAK_SLOPE_CHANGE,
    Reason.SEGMENTED_MODEL_NOT_BETTER,
    Reason.BOOTSTRAP_UNSTABLE,
    Reason.BOOTSTRAP_MULTIMODAL,
    Reason.NULL_NOT_REJECTED,
    Reason.INTERNAL_NUMERICAL_FAILURE,
}


def test_insufficient_data(fast_config):
    x = np.arange(5, dtype=float)
    r = robust_knee(x, np.sqrt(x), "concave", "increasing", fast_config)
    assert isinstance(r, NoClearKnee)
    assert r.reason == Reason.INSUFFICIENT_DATA
    assert isinstance(r.diagnostics, dict)


def test_length_mismatch(fast_config):
    r = robust_knee(np.arange(30.0), np.arange(29.0), "concave", "increasing", fast_config)
    assert r.reason == Reason.INVALID_INPUT


def test_every_abstention_uses_a_known_code(fast_config):
    rng = np.random.default_rng(0)
    curves = [
        (np.arange(30.0), np.ones(30)),                      # zero range
        (np.arange(60.0), rng.normal(0, 1, 60)),             # incompatible shape
        (np.linspace(0, 1, 80), 0.2 + 0.5 * np.linspace(0, 1, 80)
         + rng.normal(0, 0.01, 80)),                          # clean line
    ]
    for x, y in curves:
        r = robust_knee(x, y, "concave", "increasing", fast_config)
        if isinstance(r, NoClearKnee):
            assert r.reason in CFG_REASONS


def test_never_raises_on_degenerate_input(fast_config):
    # Malformed / degenerate inputs must return a result, never raise.
    for x, y in [
        (np.array([]), np.array([])),
        (np.full(30, 3.0), np.arange(30.0)),
        (np.arange(30.0), np.full(30, np.nan)),
    ]:
        r = robust_knee(x, y, "concave", "increasing", fast_config)
        assert isinstance(r, NoClearKnee)
