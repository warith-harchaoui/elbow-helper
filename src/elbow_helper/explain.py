"""Human-readable detail lines for every abstention.

A :class:`~elbow_helper.types.Reason` code says *which* gate closed. It does
not say whether the curve is hopeless or merely noisy, and those call for
opposite responses: measure more finely, or stop looking. A user who reads
``NO_PERSISTENT_CLUSTER`` and nothing else has no way to tell the two apart.

Every gate in the pipeline is a number against a threshold, and both are in
hand the moment it closes. These renderers put them in the sentence: what was
measured, what it had to clear, and -- where the quantity is a position -- in
the caller's own x units rather than the normalized ones the gate works in.
The result rides along as ``NoClearKnee.detail`` and as
``diagnostics["detail"]``; the :class:`~elbow_helper.types.Reason` code itself
is untouched, so anything matching on it keeps working.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List, Optional

from .config import RobustKneeConfig
from .types import (
    BootstrapEvidence,
    CandidateCluster,
    KneeCandidate,
    NullEvidence,
    PreparedCurve,
    Reason,
    SegmentEvidence,
)


def _g(value: float) -> str:
    """Format a number for prose: short, but never misleadingly rounded."""
    return f"{value:.4g}"


def _span(prepared: PreparedCurve, width_norm: float) -> str:
    """Render a normalized *width* as a span in the caller's x units."""
    return _g(width_norm * (prepared.x_hi - prepared.x_lo))


def _at(prepared: PreparedCurve, x_norm: float) -> str:
    """Render a normalized *location* in the caller's x units."""
    return _g(prepared.denormalize_x(x_norm))


def explain_candidates(
    candidates: List[KneeCandidate],
    prepared: PreparedCurve,
    config: RobustKneeConfig,
) -> str:
    """Say why no candidate survived the cheap structural filters.

    Parameters
    ----------
    candidates : list of KneeCandidate
        Every candidate generated, each carrying the ``rejected`` code the
        filter that killed it left behind.
    prepared : PreparedCurve
        For reporting distances in the caller's x units.
    config : RobustKneeConfig
        The thresholds the candidates were measured against.

    Returns
    -------
    str
        One sentence naming the dominant rejection and its margin.
    """
    if not candidates:
        return (
            "the difference curve has no local maximum clearing its "
            "sensitivity threshold at any smoothing scale: nothing on this "
            "curve bends sharply enough to be a candidate"
        )

    boundary = [c for c in candidates if c.rejected == Reason.BOUNDARY_KNEE]
    weak = [c for c in candidates if c.rejected == Reason.ALL_CANDIDATES_WEAK]
    parts = []

    if boundary:
        margin = max(
            config.boundary_margin,
            config.boundary_min_samples / max(prepared.n - 1, 1),
        )
        parts.append(
            f"{len(boundary)} sat within {_span(prepared, margin)} of an end "
            f"of the curve, where a bend cannot be told from the curve simply "
            f"starting or ending steeply"
        )
    if weak:
        best_prom = max(c.prominence for c in weak)
        best_ratio = max(c.noise_prominence_ratio for c in weak)
        parts.append(
            f"{len(weak)} were too faint (best prominence {_g(best_prom)} "
            f"against {_g(config.min_prominence)} required, best "
            f"prominence-to-noise {_g(best_ratio)} against "
            f"{_g(config.min_noise_prominence_ratio)})"
        )
    body = ", and ".join(parts) or "none cleared the structural filters"
    return f"all {len(candidates)} candidates were rejected: {body}"


def explain_clusters(
    clusters: List[CandidateCluster],
    reason: str,
    prepared: PreparedCurve,
    config: RobustKneeConfig,
) -> str:
    """Say why no single cluster of candidates could be accepted.

    Parameters
    ----------
    clusters : list of CandidateCluster
        Every cluster found, persistent or not.
    reason : str
        The :class:`~elbow_helper.types.Reason` code being explained.
    prepared : PreparedCurve
        For reporting locations in the caller's x units.
    config : RobustKneeConfig
        The persistence and uniqueness thresholds.

    Returns
    -------
    str
        One sentence naming the failed gates and their margins.
    """
    if not clusters:
        return "the surviving candidates formed no cluster at all"

    if reason == Reason.MULTIPLE_PLAUSIBLE_KNEES:
        top = sorted(
            (c for c in clusters if c.persistent), key=lambda c: c.support, reverse=True
        )[:2]
        if len(top) < 2:
            return "more than one location is equally plausible"
        a, b = top
        return (
            f"two locations are both credible -- x={_at(prepared, a.median_knee)} "
            f"(support {a.support_frac:.0%}) and "
            f"x={_at(prepared, b.median_knee)} (support {b.support_frac:.0%}) -- "
            f"and neither leads the other by the {_g(config.min_dominance_ratio)}x "
            f"required to call it the knee"
        )

    best = max(clusters, key=lambda c: c.support)
    failed = []
    if best.consecutive_scales < config.min_consecutive_scales:
        failed.append(
            f"it survives {best.consecutive_scales} consecutive smoothing "
            f"scales, not {config.min_consecutive_scales}"
        )
    if best.sensitivity_support < config.min_sensitivity_support:
        failed.append(
            f"it appears at {best.sensitivity_support:.0%} of the sensitivity "
            f"settings, not {config.min_sensitivity_support:.0%}"
        )
    mad_max = config.positional(config.max_cluster_mad, prepared.n)
    if best.mad > mad_max:
        failed.append(
            f"its location scatters by {_span(prepared, best.mad)} across "
            f"scales, over the {_span(prepared, mad_max)} allowed"
        )
    shift_max = config.positional(config.max_neighbor_shift, prepared.n)
    if best.neighbor_shift > shift_max:
        failed.append(
            f"it jumps {_span(prepared, best.neighbor_shift)} between "
            f"neighbouring scales, over the {_span(prepared, shift_max)} allowed"
        )

    where = f"the strongest candidate location is x={_at(prepared, best.median_knee)}"
    if not failed:
        return where
    return f"{where}, but " + ", and ".join(failed)


def explain_shape(spearman: float, violation: float, config: RobustKneeConfig) -> str:
    """Say which global-shape screen the curve failed."""
    parts = []
    if abs(spearman) < config.min_spearman_abs:
        parts.append(
            f"x and y are only {abs(spearman):.2f} rank-correlated, under the "
            f"{config.min_spearman_abs:.2f} a single monotone bend needs -- the "
            f"curve has no consistent overall trend for a knee to interrupt"
        )
    if violation > config.max_direction_violation_rate:
        parts.append(
            f"{violation:.0%} of the curve's movement runs against its own "
            f"direction, over the {config.max_direction_violation_rate:.0%} allowed"
        )
    return "; ".join(parts) or "the curve's global shape is incompatible"


def explain_segment(
    seg: SegmentEvidence, prepared: PreparedCurve, config: RobustKneeConfig
) -> str:
    """Say why the two-line model did not beat the one-line model."""
    if seg.reason == Reason.WEAK_SLOPE_CHANGE:
        return (
            f"the slope barely changes across the candidate: contrast "
            f"{_g(seg.slope_contrast)} against {_g(config.min_slope_contrast)} "
            f"required"
        )
    parts = []
    if seg.bic_improvement < config.min_bic_improvement:
        parts.append(
            f"BIC improves by {_g(seg.bic_improvement)}, not the "
            f"{_g(config.min_bic_improvement)} required"
        )
    if seg.cv_improvement < config.min_cv_improvement:
        if seg.cv_improvement < 0:
            parts.append(
                f"held-out error actually rises by {-seg.cv_improvement:.1%} "
                f"when the bend is added"
            )
        else:
            parts.append(
                f"held-out error falls by {seg.cv_improvement:.1%}, not the "
                f"{config.min_cv_improvement:.1%} required"
            )
    joined = ", and ".join(parts) if parts else "neither criterion improved enough"
    return f"a bent line fits no better than a straight one here: {joined}"


def explain_bootstrap(
    boot: BootstrapEvidence, prepared: PreparedCurve, config: RobustKneeConfig
) -> str:
    """Say which bootstrap stability check the knee failed."""
    n = prepared.n
    if boot.reason == Reason.BOOTSTRAP_MULTIMODAL:
        return (
            f"resampling the noise sends the knee to two different places: "
            f"the largest group holds {boot.primary_cluster_rate:.0%} of the "
            f"resamples (needs {config.min_primary_cluster_rate:.0%}) and the "
            f"next holds {boot.secondary_cluster_rate:.0%} (allowed "
            f"{config.max_secondary_cluster_rate:.0%}). The curve may genuinely "
            f"have more than one bend"
        )
    parts = []
    if boot.detection_rate < config.min_bootstrap_detection_rate:
        parts.append(
            f"it survives only {boot.detection_rate:.0%} of resamples, not "
            f"{config.min_bootstrap_detection_rate:.0%}"
        )
    width_max = config.positional(config.max_ci90_width, n, samples=3.0)
    if boot.ci90_width > width_max:
        parts.append(
            f"its 90% interval spans {_span(prepared, boot.ci90_width)} in x, "
            f"over the {_span(prepared, width_max)} allowed"
        )
    shift_max = config.positional(config.max_bootstrap_median_shift, n)
    if boot.median_shift > shift_max:
        parts.append(
            f"resampling moves it {_span(prepared, boot.median_shift)}, over "
            f"the {_span(prepared, shift_max)} allowed"
        )
    joined = ", and ".join(parts) if parts else "it did not resample stably"
    return f"the knee does not hold up under resampling: {joined}"


def explain_null(null: NullEvidence, config: RobustKneeConfig) -> str:
    """Say how often a knee-free curve would have looked this convincing."""
    return (
        f"a straight line carrying this curve's own noise produces evidence "
        f"this strong {null.p_value:.1%} of the time, over the "
        f"{config.max_null_p_value:.1%} that would make it surprising "
        f"({null.null_replicates} simulations)"
    )


def explain_insufficient(n: int, min_samples: int) -> str:
    """Say how far short of ``min_samples`` the curve fell.

    Parameters
    ----------
    n : int
        Usable points found after cleaning.
    min_samples : int
        The configured minimum.

    Returns
    -------
    str
        A detail sentence naming the shortfall and the way past it.
    """
    return (
        f"{n} usable points, under the {min_samples} this detector is "
        f"calibrated for. Its persistence and bootstrap gates need a scale "
        f"space to work across; below that they measure noise. Pass a config "
        f"with a lower min_samples to try anyway, and read the result as "
        f"indicative rather than confirmed"
    )


def detail_for(
    reason: str,
    prepared: Optional[PreparedCurve],
    config: RobustKneeConfig,
    *,
    seg: Optional[SegmentEvidence] = None,
    boot: Optional[BootstrapEvidence] = None,
    null: Optional[NullEvidence] = None,
) -> str:
    """Render the detail for a pipeline-stage abstention.

    Parameters
    ----------
    reason : str
        The :class:`~elbow_helper.types.Reason` code being explained.
    prepared : PreparedCurve or None
        The prepared curve, when the stage had one.
    config : RobustKneeConfig
        The thresholds in force.
    seg, boot, null : optional
        Whichever stage's evidence produced ``reason``.

    Returns
    -------
    str
        A detail sentence, or ``""`` when the reason needs no elaboration.
    """
    if reason == Reason.INCOMPATIBLE_GLOBAL_SHAPE and prepared is not None:
        return explain_shape(prepared.spearman, prepared.violation_rate, config)
    if seg is not None and reason in (
        Reason.WEAK_SLOPE_CHANGE,
        Reason.SEGMENTED_MODEL_NOT_BETTER,
    ):
        return explain_segment(seg, prepared, config)
    if boot is not None and reason in (
        Reason.BOOTSTRAP_UNSTABLE,
        Reason.BOOTSTRAP_MULTIMODAL,
    ):
        return explain_bootstrap(boot, prepared, config)
    if null is not None and reason == Reason.NULL_NOT_REJECTED:
        return explain_null(null, config)
    return ""
