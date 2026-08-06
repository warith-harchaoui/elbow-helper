"""Two applied worked examples for :func:`robust_elbow`, shared with the
figures and prose in MATH-en.md / MATH-fr.md: the k-means inertia elbow and
the PCA scree plot. Both are the "convex-decreasing" case the docstring of
:func:`robust_elbow` names directly.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import ClearKnee, RobustKneeConfig, robust_elbow

# Real k-means/PCA curves are short, one point per candidate k or per
# component, and their genuine elbow usually sits close to the left edge
# (few clusters, few signal dimensions), not in the middle of the curve.
# elbow-helper's defaults are calibrated for longer, noisier, elbow-
# anywhere measurement curves (min_samples=20, min_side_points=5, and
# persistence/bootstrap/null thresholds tuned for that regime), and reject
# this shape outright, so a profile tuned for short applied curves is
# needed instead; see MATH-en.md / MATH-fr.md for this trade-off.
_SHORT_CURVE_CONFIG = RobustKneeConfig(
    min_samples=6,
    min_side_points=2,
    min_consecutive_scales=2,
    min_sensitivity_support=0.5,
    min_slope_contrast=0.15,
    min_bic_improvement=3.0,
    min_cv_improvement=0.02,
    bootstrap_replicates=200,
    min_bootstrap_detection_rate=0.7,
    max_ci90_width=0.2,
    min_primary_cluster_rate=0.6,
    max_bootstrap_median_shift=0.08,
    null_replicates=200,
    max_null_p_value=0.05,
    random_seed=0,
)


def _kmeans_inertia(x, k, seed, n_init=15, max_iter=100):
    rng = np.random.default_rng(seed)
    best_inertia = None
    for _ in range(n_init):
        centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(max_iter):
            dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            labels = dists.argmin(axis=1)
            new_centroids = np.array(
                [
                    x[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                    for j in range(k)
                ]
            )
            if np.allclose(new_centroids, centroids):
                centroids = new_centroids
                break
            centroids = new_centroids
        dists = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        labels = dists.argmin(axis=1)
        inertia = dists[np.arange(len(x)), labels].sum()
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
    return best_inertia


def kmeans_inertia_curve(seed, true_k=8, k_max=24, n_per_cluster=50, spread=0.8):
    """Inertia(k) for k-means run on ``true_k`` well-separated 2D blobs.

    A grid of fixed, widely spaced centers (not random ones) keeps the true
    elbow at ``true_k`` sharp and reproducible across seeds; only the points
    within each blob are randomized. Shared with MATH-en.md / MATH-fr.md's
    k-means figure.
    """
    rng = np.random.default_rng(seed)
    grid = np.array(
        [[0, 0], [12, 0], [0, 12], [12, 12], [24, 0], [0, 24], [24, 24], [24, 12]]
    )[:true_k]
    points = np.vstack(
        [grid[i] + rng.normal(0, spread, size=(n_per_cluster, 2)) for i in range(true_k)]
    )
    ks = np.arange(1, k_max + 1)
    inertias = np.array([_kmeans_inertia(points, k, seed=seed * 97 + k) for k in ks])
    return ks.astype(float), inertias


def pca_scree_curve(seed, n_samples=300, true_d=6, n_noise_dims=19, signal_scale=8.0, noise_scale=0.5):
    """Eigenvalues of the sample covariance for data with ``true_d`` signal
    dimensions (large variance) and ``n_noise_dims`` noise dimensions (small
    variance): a scree plot with a genuine elbow at ``true_d``. Shared with
    MATH-en.md / MATH-fr.md's PCA figure.
    """
    rng = np.random.default_rng(seed)
    d = true_d + n_noise_dims
    variances = np.concatenate(
        [
            np.full(true_d, signal_scale) * np.linspace(1.3, 0.9, true_d),
            np.full(n_noise_dims, noise_scale),
        ]
    )
    latent = rng.normal(0, 1, size=(n_samples, d)) * np.sqrt(variances)
    rotation, _ = np.linalg.qr(rng.normal(0, 1, size=(d, d)))
    x = latent @ rotation.T
    x = x - x.mean(axis=0)
    cov = (x.T @ x) / (n_samples - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    components = np.arange(1, d + 1)
    return components.astype(float), eigvals


def test_kmeans_inertia_elbow_detects_true_cluster_count():
    for seed in range(5):
        x, y = kmeans_inertia_curve(seed, true_k=8, k_max=24)
        r = robust_elbow(x, y, config=_SHORT_CURVE_CONFIG)
        assert isinstance(r, ClearKnee), (seed, r)
        assert abs(r.knee_x - 8.0) < 1.5


def test_pca_scree_elbow_detects_true_signal_dimension():
    true_d = 6
    for seed in range(5):
        x, y = pca_scree_curve(seed, true_d=true_d, n_noise_dims=19)
        r = robust_elbow(x, y, config=_SHORT_CURVE_CONFIG)
        assert isinstance(r, ClearKnee), (seed, r)
        # Kneedle's "last point before the bend" convention consistently
        # lands about 1-2 components past the last true signal component,
        # not exactly on it; see MATH-en.md / MATH-fr.md for this reading.
        assert true_d <= r.knee_x <= true_d + 2
