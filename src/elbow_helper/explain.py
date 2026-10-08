"""Human-readable detail lines for every abstention, in English or French.

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

The sentences are assembled from :data:`_STRINGS`, the same shape
:mod:`elbow_helper.plotting` uses for its chrome text, and the language comes
from ``RobustKneeConfig.language``. Phrasing lives here rather than in each
consumer because the logic that decides *which* clauses apply -- which of four
persistence gates failed, whether held-out error fell short or rose -- is the
part worth writing once. A presentation layer that only had the reason code
would have to re-derive all of it to say anything more specific than the code
itself.

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

_STRINGS = {
    "en": {
        "join": ", and ",
        "cand_none": (
            "the difference curve has no local maximum clearing its sensitivity "
            "threshold at any smoothing scale: nothing on this curve bends "
            "sharply enough to be a candidate"
        ),
        "cand_rejected": "all {n} candidates were rejected: {body}",
        "cand_nothing_cleared": "none cleared the structural filters",
        "cand_boundary": (
            "{n} sat within {margin} of an end of the curve, where a bend "
            "cannot be told from the curve simply starting or ending steeply"
        ),
        "cand_weak": (
            "{n} were too faint (best prominence {prominence} against "
            "{min_prominence} required, best prominence-to-noise {ratio} "
            "against {min_ratio})"
        ),
        "cluster_none": "the surviving candidates formed no cluster at all",
        "cluster_two": (
            "two locations are both credible -- x={first} (support "
            "{first_support}) and x={second} (support {second_support}) -- and "
            "neither leads the other by the {ratio}x required to call it the knee"
        ),
        "cluster_many": "more than one location is equally plausible",
        "cluster_where": "the strongest candidate location is x={x}",
        "cluster_but": "{where}, but {failures}",
        "cluster_scales": (
            "it survives {scales} consecutive smoothing scales, not {required}"
        ),
        "cluster_sensitivity": (
            "it appears at {support} of the sensitivity settings, not {required}"
        ),
        "cluster_mad": (
            "its location scatters by {spread} across scales, over the "
            "{allowed} allowed"
        ),
        "cluster_neighbor": (
            "it jumps {shift} between neighbouring scales, over the {allowed} allowed"
        ),
        "shape_generic": "the curve's global shape is incompatible",
        "shape_join": "; ",
        "shape_spearman": (
            "x and y are only {rho} rank-correlated, under the {required} a "
            "single monotone bend needs -- the curve has no consistent overall "
            "trend for a knee to interrupt"
        ),
        "shape_violation": (
            "{rate} of the curve's movement runs against its own direction, "
            "over the {allowed} allowed"
        ),
        "segment_slope": (
            "the slope barely changes across the candidate: contrast {contrast} "
            "against {required} required"
        ),
        "segment_wrap": "a bent line fits no better than a straight one here: {body}",
        "segment_neither": "neither criterion improved enough",
        "segment_bic": "BIC improves by {bic}, not the {required} required",
        "segment_cv_short": (
            "held-out error falls by {improvement}, not the {required} required"
        ),
        "segment_cv_worse": (
            "held-out error actually rises by {worsening} when the bend is added"
        ),
        "bootstrap_multimodal": (
            "resampling the noise sends the knee to two different places: the "
            "largest group holds {primary} of the resamples (needs "
            "{min_primary}) and the next holds {secondary} (allowed "
            "{max_secondary}). The curve may genuinely have more than one bend"
        ),
        "bootstrap_wrap": "the knee does not hold up under resampling: {body}",
        "bootstrap_generic": "it did not resample stably",
        "bootstrap_detection": "it survives only {rate} of resamples, not {required}",
        "bootstrap_width": (
            "its 90% interval spans {width} in x, over the {allowed} allowed"
        ),
        "bootstrap_shift": "resampling moves it {shift}, over the {allowed} allowed",
        "null": (
            "a straight line carrying this curve's own noise produces evidence "
            "this strong {p} of the time, over the {allowed} that would make it "
            "surprising ({replicates} simulations)"
        ),
        "insufficient": (
            "{n} usable points, under the {required} this detector is "
            "calibrated for. Its persistence and bootstrap gates need a scale "
            "space to work across; below that they measure noise. Pass a config "
            "with a lower min_samples to try anyway, and read the result as "
            "indicative rather than confirmed"
        ),
    },
    "fr": {
        # Clauses are joined with a semicolon rather than a comma: two
        # independent clauses welded by ", et" is precisely what this
        # project's French style rules out.
        "join": " ; ",
        "cand_none": (
            "la courbe des différences n'a, à aucune échelle de lissage, de "
            "maximum local qui franchisse son seuil de sensibilité : rien sur "
            "cette courbe ne plie assez franchement pour faire un candidat"
        ),
        "cand_rejected": "les {n} candidats ont tous été rejetés : {body}",
        "cand_nothing_cleared": "aucun n'a franchi les filtres structurels",
        "cand_boundary": (
            "{n} se tenaient à moins de {margin} d'une extrémité de la courbe, "
            "là où un pli ne se distingue pas d'une courbe qui démarre ou "
            "s'achève simplement en pente raide"
        ),
        "cand_weak": (
            "{n} étaient trop ténus (meilleure proéminence {prominence} pour "
            "{min_prominence} exigée ; meilleur rapport proéminence sur bruit "
            "{ratio} pour {min_ratio})"
        ),
        "cluster_none": "les candidats survivants n'ont formé aucun groupe",
        "cluster_two": (
            "deux emplacements sont également crédibles — x={first} (soutien "
            "{first_support}) et x={second} (soutien {second_support}) — et "
            "aucun ne devance l'autre du facteur {ratio} qu'il faudrait pour "
            "le désigner comme le coude"
        ),
        "cluster_many": "plus d'un emplacement est également plausible",
        "cluster_where": "le meilleur emplacement candidat est x={x}",
        "cluster_but": "{where}, mais {failures}",
        "cluster_scales": (
            "il ne survit qu'à {scales} échelles de lissage consécutives au "
            "lieu de {required}"
        ),
        "cluster_sensitivity": (
            "il n'apparaît qu'à {support} des réglages de sensibilité au lieu "
            "de {required}"
        ),
        "cluster_mad": (
            "son emplacement se disperse de {spread} d'une échelle à l'autre, "
            "au-delà des {allowed} tolérés"
        ),
        "cluster_neighbor": (
            "il saute de {shift} entre échelles voisines, au-delà des {allowed} tolérés"
        ),
        "shape_generic": "la forme globale de la courbe est incompatible",
        "shape_join": " ; ",
        "shape_spearman": (
            "x et y ne sont corrélés en rang qu'à hauteur de {rho}, sous le "
            "{required} qu'exige un pli monotone unique — la courbe n'a aucune "
            "tendance d'ensemble qu'un coude viendrait interrompre"
        ),
        "shape_violation": (
            "{rate} du mouvement de la courbe va à l'encontre de sa propre "
            "direction, au-delà des {allowed} tolérés"
        ),
        "segment_slope": (
            "la pente change à peine de part et d'autre du candidat : "
            "contraste de {contrast} pour {required} exigé"
        ),
        "segment_wrap": (
            "une ligne brisée n'ajuste pas mieux qu'une droite ici : {body}"
        ),
        "segment_neither": "aucun des deux critères ne progresse assez",
        "segment_bic": "le BIC ne gagne que {bic} au lieu des {required} exigés",
        "segment_cv_short": (
            "l'erreur hors échantillon ne baisse que de {improvement} au lieu "
            "des {required} exigés"
        ),
        "segment_cv_worse": (
            "l'erreur hors échantillon augmente même de {worsening} lorsqu'on "
            "ajoute le pli"
        ),
        "bootstrap_multimodal": (
            "le rééchantillonnage du bruit envoie le coude à deux endroits "
            "différents : le plus gros groupe réunit {primary} des "
            "rééchantillons (il en faut {min_primary}) et le suivant en réunit "
            "{secondary} (plafond {max_secondary}). La courbe a peut-être "
            "réellement plus d'un pli"
        ),
        "bootstrap_wrap": ("le coude ne résiste pas au rééchantillonnage : {body}"),
        "bootstrap_generic": "il ne s'est pas rééchantillonné de façon stable",
        "bootstrap_detection": (
            "il ne survit qu'à {rate} des rééchantillons au lieu de {required}"
        ),
        "bootstrap_width": (
            "son intervalle à 90 % couvre {width} en x, au-delà des {allowed} tolérés"
        ),
        "bootstrap_shift": (
            "le rééchantillonnage le déplace de {shift}, au-delà des {allowed} tolérés"
        ),
        "null": (
            "une droite portant le bruit propre de cette courbe produit une "
            "évidence aussi forte {p} du temps, au-delà des {allowed} qui la "
            "rendraient surprenante ({replicates} simulations)"
        ),
        "insufficient": (
            "{n} points exploitables, sous les {required} pour lesquels ce "
            "détecteur est calibré. Ses portes de persistance et de bootstrap "
            "ont besoin d'un espace d'échelles où travailler ; en dessous, "
            "elles mesurent du bruit. Passez une configuration avec un "
            "min_samples plus bas pour tenter quand même, et lisez le résultat "
            "comme une indication plutôt que comme une confirmation"
        ),
    },
}


def _t(language: str) -> dict:
    """Return the phrase table for ``language``, falling back to English.

    Parameters
    ----------
    language : str
        ``"en"`` or ``"fr"``.

    Returns
    -------
    dict
        The phrase templates for that language.
    """
    return _STRINGS.get(language, _STRINGS["en"])


def _g(value: float) -> str:
    """Format a number for prose: short, but never misleadingly rounded."""
    return f"{value:.4g}"


def _pct(value: float, language: str, decimals: int = 0) -> str:
    """Format a fraction as a percentage, spaced the way the language wants.

    French puts a non-breaking space before the sign; English does not. Only
    the spacing is localized -- the decimal separator stays a point, matching
    :mod:`elbow_helper.plotting`, so a number never has to be read twice to
    tell a separator from a thousands mark.

    Parameters
    ----------
    value : float
        A fraction, where ``1.0`` is 100%.
    language : str
        ``"en"`` or ``"fr"``.
    decimals : int, optional
        Digits after the point. Default ``0``.

    Returns
    -------
    str
        The formatted percentage.
    """
    text = f"{value:.{decimals}%}"
    if language == "fr":
        return text[:-1] + "\u00a0%"
    return text


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
        The thresholds the candidates were measured against, and the language.

    Returns
    -------
    str
        One sentence naming the dominant rejection and its margin.
    """
    t = _t(config.language)
    if not candidates:
        return t["cand_none"]

    boundary = [c for c in candidates if c.rejected == Reason.BOUNDARY_KNEE]
    weak = [c for c in candidates if c.rejected == Reason.ALL_CANDIDATES_WEAK]
    parts = []

    if boundary:
        margin = max(
            config.boundary_margin,
            config.boundary_min_samples / max(prepared.n - 1, 1),
        )
        parts.append(
            t["cand_boundary"].format(n=len(boundary), margin=_span(prepared, margin))
        )
    if weak:
        parts.append(
            t["cand_weak"].format(
                n=len(weak),
                prominence=_g(max(c.prominence for c in weak)),
                min_prominence=_g(config.min_prominence),
                ratio=_g(max(c.noise_prominence_ratio for c in weak)),
                min_ratio=_g(config.min_noise_prominence_ratio),
            )
        )
    body = t["join"].join(parts) or t["cand_nothing_cleared"]
    return t["cand_rejected"].format(n=len(candidates), body=body)


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
        The persistence and uniqueness thresholds, and the language.

    Returns
    -------
    str
        One sentence naming the failed gates and their margins.
    """
    t = _t(config.language)
    if not clusters:
        return t["cluster_none"]

    if reason == Reason.MULTIPLE_PLAUSIBLE_KNEES:
        top = sorted(
            (c for c in clusters if c.persistent),
            key=lambda c: c.support,
            reverse=True,
        )[:2]
        if len(top) < 2:
            return t["cluster_many"]
        first, second = top
        return t["cluster_two"].format(
            first=_at(prepared, first.median_knee),
            first_support=_pct(first.support_frac, config.language),
            second=_at(prepared, second.median_knee),
            second_support=_pct(second.support_frac, config.language),
            ratio=_g(config.min_dominance_ratio),
        )

    best = max(clusters, key=lambda c: c.support)
    failures = []
    if best.consecutive_scales < config.min_consecutive_scales:
        failures.append(
            t["cluster_scales"].format(
                scales=best.consecutive_scales,
                required=config.min_consecutive_scales,
            )
        )
    if best.sensitivity_support < config.min_sensitivity_support:
        failures.append(
            t["cluster_sensitivity"].format(
                support=_pct(best.sensitivity_support, config.language),
                required=_pct(config.min_sensitivity_support, config.language),
            )
        )
    mad_max = config.positional(config.max_cluster_mad, prepared.n)
    if best.mad > mad_max:
        failures.append(
            t["cluster_mad"].format(
                spread=_span(prepared, best.mad),
                allowed=_span(prepared, mad_max),
            )
        )
    shift_max = config.positional(config.max_neighbor_shift, prepared.n)
    if best.neighbor_shift > shift_max:
        failures.append(
            t["cluster_neighbor"].format(
                shift=_span(prepared, best.neighbor_shift),
                allowed=_span(prepared, shift_max),
            )
        )

    where = t["cluster_where"].format(x=_at(prepared, best.median_knee))
    if not failures:
        return where
    return t["cluster_but"].format(where=where, failures=t["join"].join(failures))


def explain_shape(spearman: float, violation: float, config: RobustKneeConfig) -> str:
    """Say which global-shape screen the curve failed.

    Parameters
    ----------
    spearman : float
        The measured rank correlation.
    violation : float
        The measured direction-violation rate.
    config : RobustKneeConfig
        The screen's thresholds, and the language.

    Returns
    -------
    str
        One sentence naming the screens that failed.
    """
    t = _t(config.language)
    parts = []
    if abs(spearman) < config.min_spearman_abs:
        parts.append(
            t["shape_spearman"].format(
                rho=f"{abs(spearman):.2f}",
                required=f"{config.min_spearman_abs:.2f}",
            )
        )
    if violation > config.max_direction_violation_rate:
        parts.append(
            t["shape_violation"].format(
                rate=_pct(violation, config.language),
                allowed=_pct(config.max_direction_violation_rate, config.language),
            )
        )
    return t["shape_join"].join(parts) or t["shape_generic"]


def explain_segment(
    seg: SegmentEvidence, prepared: PreparedCurve, config: RobustKneeConfig
) -> str:
    """Say why the two-line model did not beat the one-line model.

    Parameters
    ----------
    seg : SegmentEvidence
        The confirmation stage's measurements.
    prepared : PreparedCurve
        Unused here; kept so every renderer takes the same arguments.
    config : RobustKneeConfig
        The confirmation thresholds, and the language.

    Returns
    -------
    str
        One sentence naming the criteria that fell short.
    """
    t = _t(config.language)
    if seg.reason == Reason.WEAK_SLOPE_CHANGE:
        return t["segment_slope"].format(
            contrast=_g(seg.slope_contrast),
            required=_g(config.min_slope_contrast),
        )
    parts = []
    if seg.bic_improvement < config.min_bic_improvement:
        parts.append(
            t["segment_bic"].format(
                bic=_g(seg.bic_improvement),
                required=_g(config.min_bic_improvement),
            )
        )
    if seg.cv_improvement < config.min_cv_improvement:
        if seg.cv_improvement < 0:
            parts.append(
                t["segment_cv_worse"].format(
                    worsening=_pct(-seg.cv_improvement, config.language, 1)
                )
            )
        else:
            parts.append(
                t["segment_cv_short"].format(
                    improvement=_pct(seg.cv_improvement, config.language, 1),
                    required=_pct(config.min_cv_improvement, config.language, 1),
                )
            )
    body = t["join"].join(parts) or t["segment_neither"]
    return t["segment_wrap"].format(body=body)


def explain_bootstrap(
    boot: BootstrapEvidence, prepared: PreparedCurve, config: RobustKneeConfig
) -> str:
    """Say which bootstrap stability check the knee failed.

    Parameters
    ----------
    boot : BootstrapEvidence
        The bootstrap stage's measurements.
    prepared : PreparedCurve
        For reporting spans in the caller's x units.
    config : RobustKneeConfig
        The bootstrap thresholds, and the language.

    Returns
    -------
    str
        One sentence naming the checks that failed.
    """
    t = _t(config.language)
    n = prepared.n
    if boot.reason == Reason.BOOTSTRAP_MULTIMODAL:
        return t["bootstrap_multimodal"].format(
            primary=_pct(boot.primary_cluster_rate, config.language),
            min_primary=_pct(config.min_primary_cluster_rate, config.language),
            secondary=_pct(boot.secondary_cluster_rate, config.language),
            max_secondary=_pct(config.max_secondary_cluster_rate, config.language),
        )
    parts = []
    if boot.detection_rate < config.min_bootstrap_detection_rate:
        parts.append(
            t["bootstrap_detection"].format(
                rate=_pct(boot.detection_rate, config.language),
                required=_pct(config.min_bootstrap_detection_rate, config.language),
            )
        )
    width_max = config.positional(config.max_ci90_width, n, samples=3.0)
    if boot.ci90_width > width_max:
        parts.append(
            t["bootstrap_width"].format(
                width=_span(prepared, boot.ci90_width),
                allowed=_span(prepared, width_max),
            )
        )
    shift_max = config.positional(config.max_bootstrap_median_shift, n)
    if boot.median_shift > shift_max:
        parts.append(
            t["bootstrap_shift"].format(
                shift=_span(prepared, boot.median_shift),
                allowed=_span(prepared, shift_max),
            )
        )
    body = t["join"].join(parts) or t["bootstrap_generic"]
    return t["bootstrap_wrap"].format(body=body)


def explain_null(null: NullEvidence, config: RobustKneeConfig) -> str:
    """Say how often a knee-free curve would have looked this convincing.

    Parameters
    ----------
    null : NullEvidence
        The null test's p-value and replicate count.
    config : RobustKneeConfig
        ``max_null_p_value``, and the language.

    Returns
    -------
    str
        One sentence putting the p-value against its threshold.
    """
    t = _t(config.language)
    return t["null"].format(
        p=_pct(null.p_value, config.language, 1),
        allowed=_pct(config.max_null_p_value, config.language, 1),
        replicates=null.null_replicates,
    )


def explain_insufficient(n: int, min_samples: int, language: str = "en") -> str:
    """Say how far short of ``min_samples`` the curve fell.

    Takes ``min_samples`` and ``language`` rather than a config because the
    cleaning stage that raises this runs before any config-wide state is in
    scope.

    Parameters
    ----------
    n : int
        Usable points found after cleaning.
    min_samples : int
        The configured minimum.
    language : str, optional
        ``"en"`` (default) or ``"fr"``.

    Returns
    -------
    str
        A detail sentence naming the shortfall and the way past it.
    """
    return _t(language)["insufficient"].format(n=n, required=min_samples)


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
        The thresholds in force, and the language.
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
