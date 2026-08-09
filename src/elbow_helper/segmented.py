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
        ``(sse_single, sse_broken)``, each the held-out sum of squared
        errors summed over all folds.
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

        cs, _ = ols_rss(_design_single(x[train]), y[train])
        pred_s = _design_single(x[test]) @ cs
        sse_single += float(np.sum((y[test] - pred_s) ** 2))

        cb, _ = ols_rss(_design_broken(x[train], k), y[train])
        pred_b = _design_broken(x[test], k) @ cb
        sse_broken += float(np.sum((y[test] - pred_b) ** 2))

    return sse_single, sse_broken


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
    left = (x >= k - far_l) & (x <= k - near_l)
    right = (x >= k + near_r) & (x <= k + far_r)

    if left.sum() < 3 or right.sum() < 3:
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
