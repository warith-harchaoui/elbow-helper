"""Phase 9 — the public ``robust_knee`` orchestration.

Chains preprocessing -> inner search -> segmented confirmation -> bootstrap ->
null test, short-circuiting to an explicit :class:`NoClearKnee` at the first
gate that fails, and accumulating diagnostics at every stage. Only a candidate
that survives *all* gates is returned as a :class:`ClearKnee`.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import os_helper as oh

from .bootstrap import bootstrap_knee
from .config import RobustKneeConfig
from .null_test import no_knee_null_test
from .preprocessing import Abstain, prepare_curve
from .search import run_search
from .types import ClearKnee, KneeResult, NoClearKnee, Reason


def robust_knee(
    x,
    y,
    curve: str = "concave",
    direction: str = "increasing",
    config: Optional[RobustKneeConfig] = None,
) -> KneeResult:
    """Detect a knee conservatively, or abstain with a reason.

    Parameters
    ----------
    x, y : array-like
        The curve. ``x`` need not be sorted or unique; preprocessing handles
        cleaning, sorting, deduplication and normalization.
    curve : str, optional
        ``"concave"`` (knees, default) or ``"convex"`` (elbows).
    direction : str, optional
        ``"increasing"`` (default) or ``"decreasing"``.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts. Defaults to :class:`RobustKneeConfig`.

    Returns
    -------
    KneeResult
        A :class:`ClearKnee` (with location, 90% interval and diagnostics) or a
        :class:`NoClearKnee` (with a reason code and diagnostics).
    """
    config = config or RobustKneeConfig()
    diagnostics: dict = {"curve": curve, "direction": direction}

    try:
        prepared = prepare_curve(x, y, curve, direction, config)
    except Abstain as a:
        diagnostics.update(a.diagnostics)
        return NoClearKnee(reason=a.reason, diagnostics=diagnostics)

    diagnostics.update(
        n=prepared.n, spearman=round(prepared.spearman, 4),
        violation_rate=round(prepared.violation_rate, 4),
    )

    try:
        result = run_search(prepared, config, confirm=True)
        diagnostics.update(
            n_candidates=result.n_candidates, n_filtered=result.n_filtered
        )
        if not result.detected:
            oh.info(f"[elbow-helper] abstain: {result.reason}")
            return NoClearKnee(reason=result.reason, diagnostics=diagnostics)

        cluster = result.cluster
        seg = result.segment
        diagnostics["knee_x_norm"] = round(result.knee_x_norm, 5)
        diagnostics["cluster"] = {
            "support": cluster.support,
            "support_frac": round(cluster.support_frac, 3),
            "consecutive_scales": cluster.consecutive_scales,
            "sensitivity_support": round(cluster.sensitivity_support, 3),
            "mad": round(cluster.mad, 4),
            "neighbor_shift": round(cluster.neighbor_shift, 4),
        }
        diagnostics["segment"] = {
            "slope_contrast": round(seg.slope_contrast, 3),
            "bic_improvement": round(seg.bic_improvement, 2),
            "cv_improvement": round(seg.cv_improvement, 3),
            "passes": seg.passes,
        }
        if not seg.passes:
            oh.info(f"[elbow-helper] abstain: {seg.reason}")
            return NoClearKnee(reason=seg.reason, diagnostics=diagnostics)

        boot = bootstrap_knee(prepared, result.knee_x_norm, config)
        diagnostics["bootstrap"] = {
            "detection_rate": round(boot.detection_rate, 3),
            "ci90_norm": [round(boot.ci90[0], 4), round(boot.ci90[1], 4)],
            "ci90_width": round(boot.ci90_width, 4),
            "primary_cluster_rate": round(boot.primary_cluster_rate, 3),
            "secondary_cluster_rate": round(boot.secondary_cluster_rate, 3),
            "median_shift": round(boot.median_shift, 4),
            "passes": boot.passes,
        }
        if not boot.passes:
            oh.info(f"[elbow-helper] abstain: {boot.reason}")
            return NoClearKnee(reason=boot.reason, diagnostics=diagnostics)

        null = no_knee_null_test(prepared, result.statistic, result.knee_x_norm, config)
        diagnostics["null"] = {
            "p_value": round(null.p_value, 4),
            "replicates": null.null_replicates,
            "passes": null.passes,
        }
        if not null.passes:
            oh.info(f"[elbow-helper] abstain: {null.reason}")
            return NoClearKnee(reason=null.reason, diagnostics=diagnostics)

        # Accepted: map back to data units.
        knee_x = prepared.denormalize_x(result.knee_x_norm)
        ci90 = (
            prepared.denormalize_x(boot.ci90[0]),
            prepared.denormalize_x(boot.ci90[1]),
        )
        knee_index = int(np.argmin(np.abs(prepared.x_norm - result.knee_x_norm)))
        sensitivity = float(
            np.median([m.sensitivity for m in cluster.members]) if cluster.members else 1.0
        )

        oh.info(
            f"[elbow-helper] CLEAR_KNEE at x={knee_x:.4g} "
            f"(detection_rate={boot.detection_rate:.2f}, null_p={null.p_value:.3g})"
        )
        return ClearKnee(
            reason=Reason.CLEAR_KNEE,
            diagnostics=diagnostics,
            knee_x=knee_x,
            knee_x_norm=result.knee_x_norm,
            knee_index=knee_index,
            ci90=ci90,
            detection_rate=boot.detection_rate,
            smoothing_window=result.window or 1,
            sensitivity=sensitivity,
            prominence=cluster.median_prominence,
            slope_contrast=seg.slope_contrast,
            bic_improvement=seg.bic_improvement,
            null_p_value=null.p_value,
        )

    except Abstain as a:
        diagnostics.update(a.diagnostics)
        return NoClearKnee(reason=a.reason, diagnostics=diagnostics)
    except Exception as exc:  # numerical safety net — never crash the caller
        oh.warning(f"[elbow-helper] internal failure: {exc}")
        diagnostics["error"] = str(exc)
        return NoClearKnee(reason=Reason.INTERNAL_NUMERICAL_FAILURE, diagnostics=diagnostics)


def robust_elbow(
    x, y, config: Optional[RobustKneeConfig] = None
) -> KneeResult:
    """Convenience wrapper for the classic convex-decreasing *elbow*.

    Equivalent to :func:`robust_knee` with ``curve="convex"`` and
    ``direction="decreasing"`` — the k-means inertia / scree-plot case.
    """
    return robust_knee(x, y, curve="convex", direction="decreasing", config=config)
