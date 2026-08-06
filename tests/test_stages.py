"""Tests for smoothing, candidate generation, clustering and segmentation.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig
from elbow_helper.candidates import generate_candidates
from elbow_helper.clustering import cluster_candidates, select_unique_cluster
from elbow_helper.metrics import passes_basic_filters
from elbow_helper.preprocessing import prepare_curve
from elbow_helper.segmented import confirm_segmented_model
from elbow_helper.smoothing import smooth_curve, smoothing_grid

from conftest import clear_knee_curve, noisy_line

CFG = RobustKneeConfig(random_seed=0)


def test_smoothing_grid_includes_no_smoothing_and_is_odd():
    grid = smoothing_grid(100, CFG)
    assert grid[0] == 1
    assert all(w % 2 == 1 for w in grid)
    assert grid == sorted(set(grid))


def test_smoothing_preserves_length_and_reduces_variance():
    rng = np.random.default_rng(0)
    y = np.linspace(0, 1, 100) + rng.normal(0, 0.1, 100)
    ys = smooth_curve(y, 11)
    assert ys.shape == y.shape
    assert np.var(np.diff(ys)) < np.var(np.diff(y))


def test_candidates_cluster_near_true_knee():
    x, y = clear_knee_curve(seed=1, knee_frac=0.3)
    prepared = prepare_curve(x, y, "concave", "increasing", CFG)
    cands = generate_candidates(prepared, CFG)
    kept = [c for c in cands if passes_basic_filters(c, prepared.n, CFG)]
    assert kept, "expected surviving candidates on a clear knee"
    clusters = cluster_candidates(kept, prepared, CFG)
    selected, reason = select_unique_cluster(clusters, CFG)
    assert selected is not None and reason is None
    assert abs(selected.median_knee - 0.3) < 0.1


def test_segmented_confirms_kink_not_line():
    x, y = clear_knee_curve(seed=2, knee_frac=0.35, noise=0.01)
    prepared = prepare_curve(x, y, "concave", "increasing", CFG)
    good = confirm_segmented_model(prepared, 0.35, CFG)
    assert good.passes
    assert good.bic_improvement >= CFG.min_bic_improvement

    xl, yl = noisy_line(seed=3)
    prep_line = prepare_curve(xl, yl, "concave", "increasing", CFG)
    bad = confirm_segmented_model(prep_line, 0.5, CFG)
    assert not bad.passes
