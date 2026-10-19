"""Tests for Phase 1 preprocessing and its abstention signals.

Author
------
Warith Harchaoui, <warithmetics@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from elbow_helper import RobustKneeConfig
from elbow_helper.preprocessing import (
    Abstain,
    _plateau_aware_spearman,
    _plateau_starts,
    prepare_curve,
)

CFG = RobustKneeConfig()


def test_cleans_sorts_and_deduplicates():
    x = np.array([3.0, 1.0, 2.0, 2.0, np.nan, 5.0, 4.0, *range(6, 25)], float)
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


# ---------------------------------------------------------------------------
# Plateau collapsing has to survive noise, because real plateaus are noisy
# ---------------------------------------------------------------------------
# _plateau_aware_spearman exists because a saturating curve's flat tail is a
# block of tied ranks, so the better the knee the worse its plain Spearman --
# "the screen ends up anti-correlated with the thing it is screening for", as
# the docstring puts it. It collapsed runs of *exactly* equal values, and real
# measurements are never exactly equal, so the protection evaporated at the
# first hint of noise and the gate it feeds rejected the curves it was built to
# admit.


def _knee_curve(sigma: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A hard knee at x=10: an 8-unit drop, then a flat tail."""
    y = np.concatenate([np.linspace(10, 2, 10), np.full(40, 1.8)])
    noise = np.random.default_rng(seed).normal(0, 1, 50)
    return np.arange(50.0), y + noise * sigma


def test_a_noiseless_plateau_still_collapses() -> None:
    """The original behaviour, unchanged: exact ties give an exact -1."""
    x, y = _knee_curve(0.0)
    assert _plateau_aware_spearman(x, (y - y.min()) / np.ptp(y)) == pytest.approx(-1.0)


@pytest.mark.parametrize("sigma", [0.005, 0.01, 0.05])
def test_a_noisy_plateau_collapses_too(sigma: float) -> None:
    """Before the fix this fell to about -0.22 and the curve was rejected.

    An 8-unit drop against sigma=0.005 is a signal-to-noise ratio of 1600. No
    gate screening for knees should refuse that one.
    """
    x, y = _knee_curve(sigma)
    rho = _plateau_aware_spearman(x, (y - y.min()) / np.ptp(y))
    assert rho < -0.6, f"rho={rho:.3f} would fail the default min_spearman_abs"


def test_collapsing_preserves_a_real_trend() -> None:
    """Grouping neighbours is fine; losing the trend is not.

    A steady ramp has no plateau, but a tolerance of two median steps still
    groups points in small batches -- which is harmless, because a monotonic
    curve stays monotonic once collapsed. The property that matters is not how
    many runs there are, it is that the correlation the gate reads is
    unchanged.
    """
    x = y = np.linspace(0.0, 1.0, 200)
    assert _plateau_starts(y).size > 10, "a ramp must not collapse to a handful of runs"
    assert _plateau_aware_spearman(x, y) == pytest.approx(1.0)


def test_collapsing_does_not_rescue_noise() -> None:
    """The gate must keep reading pure noise as uncorrelated."""
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 1, 200)
    noise = (noise - noise.min()) / np.ptp(noise)
    rho = _plateau_aware_spearman(np.linspace(0.0, 1.0, 200), noise)
    assert abs(rho) < 0.2, f"rho={rho:.3f} — noise must not pass the shape gate"


def test_each_run_is_anchored_not_chained() -> None:
    """A slow drift must not collapse however many tiny steps it takes.

    Extending a run against its predecessor rather than its first value would
    let 200 steps of 0.004 become a single plateau spanning the whole range.
    """
    y = np.arange(200) * 0.004  # every neighbour within any plausible tolerance
    starts = _plateau_starts(y)
    assert starts.size > 1, "an anchored run must break when the drift accumulates"


@pytest.mark.parametrize(
    "label, y",
    [
        ("pure noise", np.random.default_rng(1).normal(0, 1, 50)),
        ("straight line", np.arange(50.0)),
        (
            "noisy ramp",
            np.linspace(10, 1, 50) + np.random.default_rng(2).normal(0, 0.3, 50),
        ),
    ],
)
def test_curves_with_no_knee_are_still_refused(label: str, y: np.ndarray) -> None:
    """The fix must loosen nothing: these all have to keep abstaining."""
    from elbow_helper import robust_knee

    result = robust_knee(np.arange(50.0), y)
    assert type(result).__name__ == "NoClearKnee", f"{label} should not yield a knee"
