"""Small NumPy-only numerical primitives shared across the pipeline.

These replace the handful of ``scipy`` / ``scikit-learn`` calls the plan
reached for (Spearman correlation, robust noise, peak prominence, Theil-Sen
slope, OLS + BIC) so the package keeps to a numpy-only core.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

_EPS = 1e-12


def rankdata(a: np.ndarray) -> np.ndarray:
    """Rank values with ties averaged (``scipy.stats.rankdata`` semantics).

    Parameters
    ----------
    a : numpy.ndarray
        One-dimensional array of values to rank.

    Returns
    -------
    numpy.ndarray
        Ranks, 1-indexed, ties resolved to the average rank of their group.
    """
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # Average ranks within tied groups.
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i + 1
        while j < a.size and sorted_a[j] == sorted_a[i]:
            j += 1
        if j - i > 1:
            avg = (i + 1 + j) / 2.0  # mean of ranks (i+1 .. j)
            ranks[order[i:j]] = avg
        i = j
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation coefficient between ``x`` and ``y``.

    Parameters
    ----------
    x, y : numpy.ndarray
        Equal-length arrays to correlate.

    Returns
    -------
    float
        Spearman's ``rho``, in ``[-1, 1]``; ``0.0`` if either array is
        constant.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if denom < _EPS:
        return 0.0
    return float((rx * ry).sum() / denom)


def robust_sigma_from_diffs(y: np.ndarray) -> float:
    """Robust noise estimate from first differences.

    Uses the MAD of successive differences, scaled by ``1.4826`` (consistency
    with the normal) and by ``1/sqrt(2)`` because differencing two independent
    samples inflates the variance twofold.

    Parameters
    ----------
    y : numpy.ndarray
        The (scaled) signal.

    Returns
    -------
    float
        Estimated per-sample standard deviation.
    """
    d = np.diff(y)
    if d.size == 0:
        return 0.0
    mad = np.median(np.abs(d - np.median(d)))
    return float(1.4826 * mad / np.sqrt(2.0))


def peak_prominence(signal: np.ndarray, index: int) -> float:
    """Topographic prominence of the peak at ``index``.

    Implements the standard definition (as in ``scipy.signal.peak_prominences``)
    directly: descend from the peak on both sides until the signal rises above
    the peak (or an array end is reached), take the highest valley on each side,
    and return the peak height above the higher of the two valleys.

    Parameters
    ----------
    signal : numpy.ndarray
        The curve the peak lives on (here, the difference curve).
    index : int
        Index of the peak.

    Returns
    -------
    float
        The peak's prominence (``>= 0``).
    """
    n = signal.size
    if n == 0 or index < 0 or index >= n:
        return 0.0
    peak = signal[index]

    # Left base: walk left, stop when we exceed the peak height.
    i = index
    left_min = peak
    while i > 0:
        i -= 1
        if signal[i] > peak:
            break
        left_min = min(left_min, signal[i])

    # Right base: walk right, stop when we exceed the peak height.
    i = index
    right_min = peak
    while i < n - 1:
        i += 1
        if signal[i] > peak:
            break
        right_min = min(right_min, signal[i])

    base = max(left_min, right_min)
    return float(max(0.0, peak - base))


def theil_sen_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen robust slope: the median of all pairwise slopes.

    Parameters
    ----------
    x, y : numpy.ndarray
        Equal-length arrays.

    Returns
    -------
    float
        The median of ``(y_j - y_i) / (x_j - x_i)`` over all pairs ``i < j``
        with distinct ``x``; ``nan`` when fewer than two distinct x values
        are available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan")
    i, j = np.triu_indices(x.size, k=1)
    dx = x[j] - x[i]
    mask = np.abs(dx) > _EPS
    if not mask.any():
        return float("nan")
    slopes = (y[j] - y[i])[mask] / dx[mask]
    return float(np.median(slopes))


def ols_rss(design: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Ordinary least squares fit; return ``(coefficients, residual_sum_sq)``.

    Parameters
    ----------
    design : numpy.ndarray
        Design matrix, one row per sample.
    y : numpy.ndarray
        Target values, one per row of ``design``.

    Returns
    -------
    tuple of (numpy.ndarray, float)
        The fitted coefficients and the residual sum of squares.
    """
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return coef, float(resid @ resid)


def bic(rss: float, n: int, n_params: int) -> float:
    """Bayesian information criterion for a Gaussian OLS fit.

    ``BIC = n * ln(RSS / n) + k * ln(n)`` with ``k = n_params + 1`` (the extra
    parameter is the noise variance). Lower is better.

    Parameters
    ----------
    rss : float
        Residual sum of squares.
    n : int
        Number of data points.
    n_params : int
        Number of regression parameters (not counting the noise variance).

    Returns
    -------
    float
        The BIC score; lower is better.
    """
    rss = max(rss, _EPS)
    k = n_params + 1
    return float(n * np.log(rss / n) + k * np.log(n))
