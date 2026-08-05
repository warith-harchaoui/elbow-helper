"""Phase 4 — per-candidate metrics and basic rejection filters.

Given a raw :class:`~elbow_helper.types.KneeCandidate` (location + the Kneedle
difference curve it came from), attach its prominence, local-noise estimate,
prominence-to-noise ratio and boundary distance, then decide whether it clears
the cheap structural filters before any expensive stability analysis.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

import numpy as np

from .config import RobustKneeConfig
from .numerics import peak_prominence, robust_sigma_from_diffs
from .types import KneeCandidate, Reason


def evaluate_candidate(
    candidate: KneeCandidate,
    y_difference: np.ndarray,
    threshold_index: int,
    y_scaled: np.ndarray,
) -> KneeCandidate:
    """Fill in a candidate's prominence, noise, ratio and boundary distance.

    Parameters
    ----------
    candidate : KneeCandidate
        The candidate to annotate (mutated in place and returned).
    y_difference : numpy.ndarray
        The Kneedle difference curve the candidate was found on.
    threshold_index : int
        Index of the generating peak on ``y_difference``.
    y_scaled : numpy.ndarray
        The (unsmoothed) scaled signal, for the robust noise estimate.

    Returns
    -------
    KneeCandidate
        The same object, with metric fields populated.
    """
    candidate.prominence = peak_prominence(y_difference, threshold_index)
    candidate.local_noise = robust_sigma_from_diffs(y_scaled)
    denom = candidate.local_noise if candidate.local_noise > 1e-9 else 1e-9
    candidate.noise_prominence_ratio = candidate.prominence / denom
    candidate.boundary_distance = min(
        candidate.knee_x_norm, 1.0 - candidate.knee_x_norm
    )
    return candidate


def passes_basic_filters(
    candidate: KneeCandidate, n: int, config: RobustKneeConfig
) -> bool:
    """Return ``True`` iff a candidate survives the cheap structural filters.

    Sets ``candidate.rejected`` to a reason code when it fails. Rejections:
    within the boundary margin, too few points on one side, within half a
    smoothing window of an end, weak prominence, or weak prominence-to-noise.

    Parameters
    ----------
    candidate : KneeCandidate
        The annotated candidate.
    n : int
        Number of samples in the curve.
    config : RobustKneeConfig
        Filter thresholds.

    Returns
    -------
    bool
        Whether the candidate passes.
    """
    if candidate.boundary_distance < config.boundary_margin:
        candidate.rejected = Reason.BOUNDARY_KNEE
        return False

    left = candidate.knee_index
    right = n - 1 - candidate.knee_index
    if left < config.min_side_points or right < config.min_side_points:
        candidate.rejected = Reason.BOUNDARY_KNEE
        return False

    half_window = candidate.window // 2
    if candidate.knee_index < half_window or candidate.knee_index > n - 1 - half_window:
        candidate.rejected = Reason.BOUNDARY_KNEE
        return False

    if candidate.prominence < config.min_prominence:
        candidate.rejected = Reason.ALL_CANDIDATES_WEAK
        return False

    if candidate.noise_prominence_ratio < config.min_noise_prominence_ratio:
        candidate.rejected = Reason.ALL_CANDIDATES_WEAK
        return False

    return True
