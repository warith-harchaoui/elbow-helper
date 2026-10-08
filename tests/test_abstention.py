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
    r = robust_knee(
        np.arange(30.0), np.arange(29.0), "concave", "increasing", fast_config
    )
    assert r.reason == Reason.INVALID_INPUT


def test_every_abstention_uses_a_known_code(fast_config):
    rng = np.random.default_rng(0)
    curves = [
        (np.arange(30.0), np.ones(30)),  # zero range
        (np.arange(60.0), rng.normal(0, 1, 60)),  # incompatible shape
        (
            np.linspace(0, 1, 80),
            0.2 + 0.5 * np.linspace(0, 1, 80) + rng.normal(0, 0.01, 80),
        ),  # clean line
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


def test_every_abstention_says_what_closed_the_gate():
    """A reason code names the gate; ``detail`` says how far off the curve was.

    Without it a user cannot tell a hopeless curve from an under-measured
    one, and those call for opposite responses -- stop looking, or measure
    more finely. Each detail must therefore carry at least one number.
    """
    rng = np.random.default_rng(0)
    x = np.arange(1, 121.0)
    curves = [
        ("pure noise", rng.normal(size=120)),
        ("straight line", 50 - 0.3 * x),
        ("straight + noise", 50 - 0.3 * x + 0.05 * rng.normal(size=120)),
        ("sigmoid", 10 / (1 + np.exp((x - 60) / 4))),
        ("short", np.exp(-np.arange(16.0) / 3)),
    ]
    seen = set()
    for name, y in curves:
        xs = x if len(y) == 120 else np.arange(len(y), dtype=float)
        r = robust_knee(xs, y, "convex", "decreasing")
        if not isinstance(r, NoClearKnee):
            continue
        seen.add(r.reason)
        assert r.detail, (name, r.reason)
        assert any(ch.isdigit() for ch in r.detail), (name, r.detail)
        # The same sentence has to reach callers that only read diagnostics.
        assert r.diagnostics["detail"] == r.detail, name
    assert len(seen) >= 3, seen


def test_an_early_knee_on_a_long_curve_is_not_a_boundary_artefact():
    """A fixed *fraction* is the wrong margin once the curve is long.

    At n=200 a 0.10 margin discards the first 20 samples, which is where a
    rank-10 factor structure lives. Worse than missing the knee, it used to
    reject the knee at its true position while keeping the copy the coarse
    smoothing scales had dragged further in: candidates at 11, 13, 14, 15 and
    18 were all thrown out and the answer came back as 21.
    """
    x = np.arange(1, 201.0)
    y = np.where(x <= 10, 100 - (x - 1) * 9.0, 10.0)
    r = robust_knee(x, y, "convex", "decreasing")
    assert not isinstance(r, NoClearKnee), r
    # The line runs on to x=11 before flattening, so 11 is the breakpoint.
    assert abs(r.knee_x - 11.0) <= 1.0, r.knee_x
