"""End-to-end tests for robust_knees (plural): the DP + subtractive-mBIC +
FWER-gated multi-knee pipeline. See ELBOW-en.tex sections 5-20 and
research/multiknee/RESULTS.md for the design this ships.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from elbow_helper import InvalidKnees, Knees, Reason, RobustKneesConfig, robust_knees


@pytest.fixture
def fast_multi_config():
    return RobustKneesConfig(random_seed=0, fwer_permutations=100)


def _one_knee_curve(seed, n=100, noise=0.03):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.4, 3 * x, 1.2 + 0.1 * (x - 0.4)) + rng.normal(0, noise, n)
    return x, y


def _two_knee_curve(seed, n=100, noise=0.02):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.piecewise(
        x,
        [x < 0.3, (x >= 0.3) & (x < 0.65), x >= 0.65],
        [
            lambda t: 3 * t,
            lambda t: 0.9 + 0.2 * (t - 0.3),
            lambda t: 0.97 + 2.2 * (t - 0.65),
        ],
    ) + rng.normal(0, noise, n)
    return x, y


def _no_knee_curve(seed, n=100, noise=0.03):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = 0.5 * x + 0.1 + rng.normal(0, noise, n)
    return x, y


def sigmoid_staircase_curve(seed, n=200, n_steps=3, steepness=40.0, noise=0.02):
    """A sum of ``n_steps`` logistic sigmoids: a smooth staircase.

    Each sigmoid rise has no literal breakpoint (it is infinitely
    differentiable), but a piecewise-linear search approximates each rise
    with a short segment bracketed by two breakpoints (start and end of the
    rise), so a clean ``n_steps``-step staircase is expected to yield
    ``2 * n_steps`` breakpoints once the rises are steep enough to look
    locally linear against the flats on either side. Shared with
    ``ELBOW-en.tex``'s multi-knee figure, which runs
    ``robust_knees`` on this exact curve.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    centers = np.linspace(0.15, 0.85, n_steps)
    y = np.zeros(n)
    for c in centers:
        y += 1.0 / (1.0 + np.exp(-steepness * (x - c)))
    y = y / y.max()
    y = y + rng.normal(0, noise, n)
    return x, y, centers


def test_recovers_zero_knees_on_a_straight_line(fast_multi_config):
    for seed in range(5):
        x, y = _no_knee_curve(seed)
        r = robust_knees(x, y, config=fast_multi_config)
        assert isinstance(r, Knees)
        assert r.is_valid
        assert r.k == 0
        assert r.knees == []
        assert r.reason == Reason.KNEES_FOUND


def test_recovers_one_knee_near_the_true_location(fast_multi_config):
    for seed in range(5):
        x, y = _one_knee_curve(seed)
        r = robust_knees(x, y, config=fast_multi_config)
        assert isinstance(r, Knees)
        assert r.k == 1
        assert abs(r.knees[0].x - 0.4) < 0.05


def test_recovers_two_knees_near_the_true_locations(fast_multi_config):
    for seed in range(5):
        x, y = _two_knee_curve(seed)
        r = robust_knees(x, y, config=fast_multi_config)
        assert isinstance(r, Knees)
        assert r.k == 2
        locs = sorted(kn.x for kn in r.knees)
        assert abs(locs[0] - 0.3) < 0.05
        assert abs(locs[1] - 0.65) < 0.05


def alternating_slope_curve(seed, n=150, noise=0.02):
    """A "mountain": steep up, flat, steep down, flat.

    Unlike :func:`robust_knee`, which needs an explicit ``curve``/
    ``direction`` naming one of four fixed shapes, :func:`robust_knees`
    fits each segment its own independent slope, so a sign change (up then
    down) needs no special handling. Shared with ``ELBOW-en.tex``'s
    alternating-slope figure.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.piecewise(
        x,
        [x < 0.2, (x >= 0.2) & (x < 0.45), (x >= 0.45) & (x < 0.65), x >= 0.65],
        [
            lambda t: 4 * t,
            lambda t: 0.8 + 0.1 * (t - 0.2),
            lambda t: 0.825 - 4 * (t - 0.45),
            lambda t: 0.025 - 0.05 * (t - 0.65),
        ],
    )
    y = y + rng.normal(0, noise, n)
    return x, y


def subtle_knee_curve(seed, n=150, noise=0.05):
    """A real but small slope change: easy to confuse with noise.

    Used to demonstrate the conservative philosophy directly: at low noise
    the same shape is reliably detected, at high noise ``robust_knees``
    mostly abstains rather than force an answer. Shared with the
    detection-boundary figure in ``ELBOW-en.tex``.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.5, 0.8 * x, 0.4 + 0.35 * (x - 0.5))
    y = y + rng.normal(0, noise, n)
    return x, y


def test_alternating_slope_mountain_recovers_three_sign_changing_breakpoints():
    cfg = RobustKneesConfig(random_seed=0, k_max=4, fwer_permutations=300)
    for seed in range(5):
        x, y = alternating_slope_curve(seed)
        r = robust_knees(x, y, config=cfg)
        assert isinstance(r, Knees)
        assert r.k == 3
        locs = sorted(kn.x for kn in r.knees)
        assert abs(locs[0] - 0.2) < 0.05
        assert abs(locs[1] - 0.45) < 0.05
        assert abs(locs[2] - 0.65) < 0.05
        slopes = [kn.slope_left for kn in r.knees] + [r.knees[-1].slope_right]
        # up, (flat), down, (flat): signs must alternate up/down/... correctly,
        # not just be nonzero, since this is the whole point of the example.
        assert slopes[0] > 0.5  # steep up into breakpoint 1
        assert slopes[-1] < -0.01  # ends heading down out of breakpoint 3


def test_subtle_knee_detected_at_low_noise_mostly_abstains_at_high_noise():
    cfg = RobustKneesConfig(random_seed=0, k_max=3, fwer_permutations=300)
    low_detections = sum(
        robust_knees(*subtle_knee_curve(seed, noise=0.03), config=cfg).k >= 1
        for seed in range(4)
    )
    assert low_detections == 4  # reliable at low noise

    high_detections = sum(
        robust_knees(*subtle_knee_curve(seed, noise=0.12), config=cfg).k >= 1
        for seed in range(8)
    )
    # The conservative design means most high-noise draws should abstain
    # (k=0) rather than force a guess; this is not "always right", it is
    # "wrong in the honest direction" (see ELBOW-en.tex section 16 / RESULTS.md).
    assert high_detections <= 3


def test_sigmoid_staircase_brackets_each_rise_with_a_breakpoint_pair():
    # k_max=6 with only 100 permutations would make the Bonferroni-corrected
    # threshold (alpha/6) smaller than the best achievable p-value
    # (1/(100+1)), so the FWER gate could never pass regardless of the true
    # effect; fwer_permutations must scale with k_max. See ELBOW-en.tex
    # section 20 and this test's own history for why 300 is used here.
    cfg = RobustKneesConfig(random_seed=0, k_max=6, fwer_permutations=300)
    for seed in range(5):
        x, y, centers = sigmoid_staircase_curve(seed, n=200, n_steps=3, steepness=40.0)
        r = robust_knees(x, y, config=cfg)
        assert isinstance(r, Knees)
        assert r.k == 6
        locs = sorted(kn.x for kn in r.knees)
        # Each sigmoid's rise should be bracketed by exactly one pair of
        # breakpoints, straddling its true center.
        for i, center in enumerate(centers):
            lo, hi = locs[2 * i], locs[2 * i + 1]
            assert lo < center < hi, (seed, i, center, lo, hi)


def test_knee_estimate_carries_distinct_slopes(fast_multi_config):
    x, y = _one_knee_curve(seed=0)
    r = robust_knees(x, y, config=fast_multi_config)
    assert isinstance(r, Knees)
    kn = r.knees[0]
    assert kn.slope_left > kn.slope_right > 0  # steep then shallow, both positive


def test_single_list_input_matches_explicit_x(fast_multi_config):
    _, y = _one_knee_curve(seed=0)
    x = np.arange(len(y), dtype=float)
    from_pair = robust_knees(x, y, config=fast_multi_config)
    from_single = robust_knees(y, config=fast_multi_config)
    assert isinstance(from_pair, Knees) and isinstance(from_single, Knees)
    assert from_pair.k == from_single.k
    assert from_pair.knees[0].x == from_single.knees[0].x


def test_invalid_input_returns_invalid_knees_not_a_crash(fast_multi_config):
    r = robust_knees(np.arange(5.0), np.arange(5.0), config=fast_multi_config)
    assert isinstance(r, InvalidKnees)
    assert not r.is_valid
    assert r.reason == Reason.INSUFFICIENT_DATA


def test_length_mismatch_is_invalid_input(fast_multi_config):
    r = robust_knees(np.arange(30.0), np.arange(29.0), config=fast_multi_config)
    assert isinstance(r, InvalidKnees)
    assert r.reason == Reason.INVALID_INPUT


def test_fwer_confirmation_can_be_disabled(fast_multi_config):
    x, y = _one_knee_curve(seed=0)
    cfg_off = fast_multi_config.with_(require_fwer_confirmation=False)
    r = robust_knees(x, y, config=cfg_off)
    assert isinstance(r, Knees)
    assert "fwer_k" not in r.diagnostics
    assert r.diagnostics["final_k"] == r.diagnostics["mbic_k"]


def test_diagnostics_are_populated(fast_multi_config):
    x, y = _one_knee_curve(seed=0)
    r = robust_knees(x, y, config=fast_multi_config)
    assert isinstance(r, Knees)
    for key in (
        "n",
        "min_seg",
        "k_max_requested",
        "k_max_effective",
        "mbic_scores",
        "mbic_k",
        "fwer_k",
        "fwer_p_values",
        "final_k",
    ):
        assert key in r.diagnostics


def test_result_is_always_tagged_union(fast_multi_config):
    for seed in range(3):
        x, y = _one_knee_curve(seed)
        r = robust_knees(x, y, config=fast_multi_config)
        assert isinstance(r, (Knees, InvalidKnees))
        assert r.is_valid == isinstance(r, Knees)
