"""Phase 6 — slope contrast and broken-line model confirmation.

Confirms a candidate knee three ways: a robust (Theil-Sen) slope change across
it, a continuous broken-line fit that beats a single line on blocked
cross-validation, and a decisive BIC improvement. All NumPy; the broken-line
term ``c * max(0, x - k)`` guarantees continuity at the knee.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from .config import RobustKneeConfig
from .numerics import bic, ols_rss, theil_sen_slope
from .types import PreparedCurve, Reason, SegmentEvidence

_EPS = 1e-9

# Smallest number of samples a Theil-Sen slope is read from.
_MIN_SLOPE_POINTS = 3


def _design_single(x: np.ndarray) -> np.ndarray:
    """Design matrix for ``y = a + b x``.

    Parameters
    ----------
    x : numpy.ndarray
        Normalized ``x`` values.

    Returns
    -------
    numpy.ndarray
        Design matrix with an intercept column and an ``x`` column.
    """
    return np.column_stack([np.ones_like(x), x])


def _design_broken(x: np.ndarray, k: float) -> np.ndarray:
    """Design matrix for the continuous broken line ``y = a + b x + c*relu(x-k)``.

    Parameters
    ----------
    x : numpy.ndarray
        Normalized ``x`` values.
    k : float
        Candidate knee location.

    Returns
    -------
    numpy.ndarray
        Design matrix with an intercept, ``x`` and ``max(0, x - k)`` column.
    """
    return np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - k)])


def _blocked_cv_sse(x: np.ndarray, y: np.ndarray, k: float, folds: int) -> tuple:
    """Blocked (contiguous) cross-validated SSE for single vs broken models.

    Folds whose *training* half lies entirely on one side of ``k`` are skipped.
    There ``max(0, x - k)`` is either all zero or an exact affine function of
    ``x``, so the broken-line design is rank-deficient and its fit coincides
    with the single line's: the fold cannot tell the two models apart and
    contributes an identical term to both sums. Counting it anyway reads as
    evidence against the knee, and it is precisely the fold that holds the
    knee -- hence the steepest part of the curve and the bulk of the error --
    that gets held out that way. On a 784-point PCA scree spectrum the four
    informative folds each showed a 99% error reduction while the one
    degenerate fold, carrying 96% of the total squared error, showed exactly
    none; summed naively that came out as a 3.6% improvement and the knee was
    rejected as ``SEGMENTED_MODEL_NOT_BETTER``.

    Parameters
    ----------
    x, y : numpy.ndarray
        The normalized curve.
    k : float
        Candidate knee location, held fixed across folds.
    folds : int
        Requested number of contiguous folds (clamped to the data size).

    Returns
    -------
    tuple of float
        ``(sse_single, sse_broken)``, each the held-out sum of squared errors
        summed over the folds on which the two models are distinguishable.
        ``(0.0, 0.0)`` when no fold is.
    """
    n = x.size
    folds = max(2, min(folds, n // 4)) if n >= 8 else 2
    bounds = np.linspace(0, n, folds + 1).astype(int)
    sse_single = sse_broken = 0.0
    for f in range(folds):
        lo, hi = bounds[f], bounds[f + 1]
        if hi <= lo:
            continue
        test = np.zeros(n, dtype=bool)
        test[lo:hi] = True
        train = ~test
        if train.sum() < 4:
            continue
        # Both pieces of the broken line must be identifiable from the
        # training half, otherwise the two models are the same model here.
        if (x[train] < k).sum() < 2 or (x[train] > k).sum() < 2:
            continue

        cs, _ = ols_rss(_design_single(x[train]), y[train])
        pred_s = _design_single(x[test]) @ cs
        sse_single += float(np.sum((y[test] - pred_s) ** 2))

        cb, _ = ols_rss(_design_broken(x[train], k), y[train])
        pred_b = _design_broken(x[test], k) @ cb
        sse_broken += float(np.sum((y[test] - pred_b) ** 2))

    return sse_single, sse_broken


def _side_mask(
    x: np.ndarray, k: float, near: float, far: float, side: str, min_points: int
) -> np.ndarray:
    """Boolean mask for the slope-fitting window on one side of ``k``.

    The configured window is a pair of offsets in normalized ``x``: skip the
    ``near`` band hugging the knee (where the bend itself lives), then fit over
    the next ``far - near`` of the range. Near either end of the curve that
    window runs off the data, and a knee at 2% of the range -- where a scree
    plot's elbow routinely sits -- gets an *empty* left window. Reporting that
    as ``WEAK_SLOPE_CHANGE`` says the slope did not change when in fact it was
    never measured, so when the configured window holds fewer than
    ``min_points`` samples this falls back to the ``min_points`` samples
    nearest the knee on that side, still skipping the single sample adjacent
    to it.

    Parameters
    ----------
    x : numpy.ndarray
        Normalized ``x`` values, ascending.
    k : float
        Candidate knee location.
    near, far : float
        Offsets from the knee bounding the window, ``near < far``.
    side : str
        ``"left"`` or ``"right"``.
    min_points : int
        Smallest usable window, in samples.

    Returns
    -------
    numpy.ndarray
        Boolean mask selecting the window's samples.
    """
    if side == "left":
        mask = (x >= k - far) & (x <= k - near)
        available = np.nonzero(x < k)[0]
        fallback = available[max(0, available.size - 1 - min_points) : -1]
    else:
        mask = (x >= k + near) & (x <= k + far)
        available = np.nonzero(x > k)[0]
        fallback = available[1 : 1 + min_points]
    if int(mask.sum()) >= min_points or fallback.size < min_points:
        return mask
    out = np.zeros_like(mask)
    out[fallback] = True
    return out


def confirm_segmented_model(
    prepared: PreparedCurve, knee_x_norm: float, config: RobustKneeConfig
) -> SegmentEvidence:
    """Test whether a broken line at ``knee_x_norm`` is genuinely better.

    Parameters
    ----------
    prepared : PreparedCurve
        The normalized curve.
    knee_x_norm : float
        Candidate knee location in ``[0, 1]``.
    config : RobustKneeConfig
        Slope/CV/BIC thresholds.

    Returns
    -------
    SegmentEvidence
        Slope contrast, both slopes, BIC and CV improvements, and a pass flag
        (with a reason code when it fails).
    """
    x = prepared.x_norm
    y = prepared.y_scaled
    n = x.size
    k = float(knee_x_norm)

    # --- robust local slopes on either side of the knee ---
    far_l, near_l = config.slope_left_window
    near_r, far_r = config.slope_right_window
    left = _side_mask(x, k, near_l, far_l, "left", _MIN_SLOPE_POINTS)
    right = _side_mask(x, k, near_r, far_r, "right", _MIN_SLOPE_POINTS)

    if left.sum() < _MIN_SLOPE_POINTS or right.sum() < _MIN_SLOPE_POINTS:
        return SegmentEvidence(
            passes=False,
            slope_contrast=0.0,
            m_left=float("nan"),
            m_right=float("nan"),
            bic_improvement=0.0,
            cv_improvement=0.0,
            reason=Reason.WEAK_SLOPE_CHANGE,
        )

    m_left = theil_sen_slope(x[left], y[left])
    m_right = theil_sen_slope(x[right], y[right])
    contrast = abs(m_left - m_right) / (abs(m_left) + abs(m_right) + _EPS)

    # --- single vs broken line: BIC ---
    _, rss_single = ols_rss(_design_single(x), y)
    _, rss_broken = ols_rss(_design_broken(x, k), y)
    bic_single = bic(rss_single, n, n_params=2)
    bic_broken = bic(rss_broken, n, n_params=3)
    bic_improvement = bic_single - bic_broken

    # --- single vs broken line: blocked cross-validation ---
    sse_single, sse_broken = _blocked_cv_sse(x, y, k, config.cv_folds)
    cv_improvement = (sse_single - sse_broken) / (sse_single + _EPS)

    slope_ok = contrast >= config.min_slope_contrast
    cv_ok = cv_improvement >= config.min_cv_improvement
    bic_ok = bic_improvement >= config.min_bic_improvement

    reason = None
    if not slope_ok:
        reason = Reason.WEAK_SLOPE_CHANGE
    elif not (cv_ok and bic_ok):
        reason = Reason.SEGMENTED_MODEL_NOT_BETTER

    return SegmentEvidence(
        passes=slope_ok and cv_ok and bic_ok,
        slope_contrast=float(contrast),
        m_left=float(m_left),
        m_right=float(m_right),
        bic_improvement=float(bic_improvement),
        cv_improvement=float(cv_improvement),
        reason=reason,
    )
