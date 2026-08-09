"""Phase 7 — bootstrap robustness of the detected knee.

Refits the accepted broken-line model, then resamples its residuals (IID
residual bootstrap) and reruns the *full inner search* on each synthetic curve.
A knee is only robust if it is redetected almost every time, its location has a
tight interval, and the replicate knees are unimodal.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

import numpy as np

from .config import RobustKneeConfig
from .numerics import ols_rss
from .search import run_search
from .types import BootstrapEvidence, PreparedCurve, Reason


def _greedy_1d_clusters(values: np.ndarray, tol: float) -> List[int]:
    """Sizes of greedy 1-D clusters (sorted values, running-median centers)."""
    if values.size == 0:
        return []
    vs = np.sort(values)
    sizes = []
    current = [vs[0]]
    center = vs[0]
    for v in vs[1:]:
        if abs(v - center) <= tol:
            current.append(v)
            center = float(np.median(current))
        else:
            sizes.append(len(current))
            current = [v]
            center = v
    sizes.append(len(current))
    return sorted(sizes, reverse=True)


def bootstrap_knee(
    prepared: PreparedCurve,
    knee_x_norm: float,
    config: RobustKneeConfig,
) -> BootstrapEvidence:
    """Assess knee stability under an IID residual bootstrap.

    Parameters
    ----------
    prepared : PreparedCurve
        The observed normalized curve.
    knee_x_norm : float
        The accepted knee location, used to fit the residual model and to
        measure the bootstrap median shift against.
    config : RobustKneeConfig
        Bootstrap thresholds and ``random_seed``.

    Returns
    -------
    BootstrapEvidence
        Detection rate, 90% interval, uni/multimodality rates, median shift,
        and a pass flag with a reason code on failure.
    """
    x = prepared.x_norm
    y = prepared.y_scaled
    k = float(knee_x_norm)

    design = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - k)])
    coef, _ = ols_rss(design, y)
    yhat = design @ coef
    residuals = y - yhat

    rng = np.random.default_rng(config.random_seed)
    knees: List[float] = []
    b = config.bootstrap_replicates
    for _ in range(b):
        resampled = rng.choice(residuals, size=residuals.size, replace=True)
        y_star = yhat + resampled
        prepared_star = replace(prepared, y_scaled=y_star)
        res = run_search(prepared_star, config, confirm=False)
        if res.detected and res.knee_x_norm is not None:
            knees.append(res.knee_x_norm)

    detection_rate = len(knees) / b if b else 0.0
    if not knees:
        return BootstrapEvidence(
            passes=False,
            detection_rate=detection_rate,
            ci90=(0.0, 0.0),
            ci90_width=1.0,
            primary_cluster_rate=0.0,
            secondary_cluster_rate=0.0,
            median_shift=1.0,
            knees=[],
            reason=Reason.BOOTSTRAP_UNSTABLE,
        )

    knees_arr = np.array(knees)
    lo, hi = np.percentile(knees_arr, [5, 95])
    ci90 = (float(lo), float(hi))
    ci90_width = float(hi - lo)
    median_shift = float(abs(np.median(knees_arr) - k))

    sizes = _greedy_1d_clusters(knees_arr, config.cluster_tolerance)
    primary_rate = sizes[0] / len(knees) if sizes else 0.0
    secondary_rate = sizes[1] / len(knees) if len(sizes) > 1 else 0.0

    detect_ok = detection_rate >= config.min_bootstrap_detection_rate
    width_ok = ci90_width <= config.max_ci90_width
    primary_ok = primary_rate >= config.min_primary_cluster_rate
    secondary_ok = secondary_rate <= config.max_secondary_cluster_rate
    shift_ok = median_shift <= config.max_bootstrap_median_shift

    reason = None
    if not secondary_ok or not primary_ok:
        reason = Reason.BOOTSTRAP_MULTIMODAL
    elif not (detect_ok and width_ok and shift_ok):
        reason = Reason.BOOTSTRAP_UNSTABLE

    return BootstrapEvidence(
        passes=detect_ok and width_ok and primary_ok and secondary_ok and shift_ok,
        detection_rate=detection_rate,
        ci90=ci90,
        ci90_width=ci90_width,
        primary_cluster_rate=primary_rate,
        secondary_cluster_rate=secondary_rate,
        median_shift=median_shift,
        knees=knees,
        reason=reason,
    )
