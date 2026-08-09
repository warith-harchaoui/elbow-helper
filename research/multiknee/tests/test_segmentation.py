"""Correctness tests for segment costs, DP optimal partitioning, and greedy search.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from segmentation import (
    SegmentCostTable,
    dp_optimal_partition,
    greedy_binary_segmentation,
)


def _direct_ols_rss(x, y):
    if len(x) < 2 or np.ptp(x) < 1e-12:
        return float(np.sum((y - np.mean(y)) ** 2))
    design = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def test_segment_cost_matches_direct_ols():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 20)
    y = 2.0 * x + 0.3 + rng.normal(0, 0.1, x.size)
    table = SegmentCostTable(x, y)
    for i, j in itertools.combinations(range(21), 2):
        if j - i < 2:
            continue
        got = table.cost(i, j)
        want = _direct_ols_rss(x[i:j], y[i:j])
        assert abs(got - want) < 1e-8, (i, j, got, want)


def _brute_force_best_partition(x, y, k, min_seg):
    """Exhaustively search all valid k-breakpoint partitions; return min SSE."""
    n = x.size
    table = SegmentCostTable(x, y)
    if k == 0:
        return table.cost(0, n)
    candidates = range(min_seg, n - min_seg + 1)
    best = np.inf
    for cuts in itertools.combinations(candidates, k):
        if any(cuts[i + 1] - cuts[i] < min_seg for i in range(len(cuts) - 1)):
            continue
        boundaries = (0, *cuts, n)
        sse = sum(
            table.cost(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        )
        best = min(best, sse)
    return best


def test_dp_matches_brute_force_small_n():
    rng = np.random.default_rng(1)
    n = 18
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.5, x, 0.5 + 2 * (x - 0.5)) + rng.normal(0, 0.05, n)
    min_seg = 3

    results = dp_optimal_partition(x, y, k_max=3, min_seg=min_seg)
    for k in range(len(results)):
        brute = _brute_force_best_partition(x, y, k, min_seg)
        assert abs(results[k].sse - brute) < 1e-6, (k, results[k].sse, brute)


def test_dp_sse_is_nonincreasing_in_k():
    rng = np.random.default_rng(2)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(3 * x) + rng.normal(0, 0.1, n)
    results = dp_optimal_partition(x, y, k_max=5, min_seg=3)
    sses = [r.sse for r in results]
    assert all(sses[i] >= sses[i + 1] - 1e-9 for i in range(len(sses) - 1))


def test_dp_breakpoints_respect_min_seg_and_are_sorted():
    rng = np.random.default_rng(3)
    n = 50
    x = np.linspace(0, 1, n)
    y = rng.normal(0, 1, n)
    results = dp_optimal_partition(x, y, k_max=4, min_seg=5)
    for r in results:
        bp = list(r.breakpoints)
        assert bp == sorted(bp)
        full = (0, *bp, n)
        assert all(full[i + 1] - full[i] >= 5 for i in range(len(full) - 1))


def test_greedy_is_never_better_than_dp():
    rng = np.random.default_rng(4)
    n = 70
    x = np.linspace(0, 1, n)
    y = np.piecewise(
        x,
        [x < 0.3, (x >= 0.3) & (x < 0.6), x >= 0.6],
        [
            lambda t: 3 * t,
            lambda t: 0.9 + 0.2 * (t - 0.3),
            lambda t: 0.96 - 1.5 * (t - 0.6),
        ],
    ) + rng.normal(0, 0.05, n)

    dp = dp_optimal_partition(x, y, k_max=4, min_seg=3)
    greedy = greedy_binary_segmentation(x, y, k_max=4, min_seg=3)
    for k in range(min(len(dp), len(greedy))):
        assert greedy[k].sse >= dp[k].sse - 1e-8, (k, greedy[k].sse, dp[k].sse)


def test_greedy_can_strictly_underperform_dp_even_when_noiseless():
    # This is a documented finding, not a bug: greedy binary segmentation
    # commits to its first split (the single best split of the *whole*
    # series) before any later breakpoint exists to compete with it, so on
    # an asymmetric multi-breakpoint curve it can lock in a suboptimal first
    # cut and never reach the DP optimum, even with zero noise. This is the
    # textbook motivation for Wild Binary Segmentation (Fryzlewicz 2014).
    n = 90
    x = np.linspace(0, 1, n)
    y = np.piecewise(
        x,
        [x < 0.33, (x >= 0.33) & (x < 0.66), x >= 0.66],
        [
            lambda t: 5 * t,
            lambda t: 1.65 - 0.1 * (t - 0.33),
            lambda t: 1.617 + 4 * (t - 0.66),
        ],
    )
    dp = dp_optimal_partition(x, y, k_max=2, min_seg=3)
    greedy = greedy_binary_segmentation(x, y, k_max=2, min_seg=3)
    assert dp[2].sse < 1e-6  # DP recovers the exact noiseless fit
    assert greedy[2].sse > dp[2].sse  # greedy provably does not, here
    # Still monotonically non-increasing as greedy adds breakpoints.
    assert all(
        greedy[i].sse >= greedy[i + 1].sse - 1e-9 for i in range(len(greedy) - 1)
    )
