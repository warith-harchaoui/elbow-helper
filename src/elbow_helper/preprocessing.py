"""Phase 1 — cleaning, robust normalization, and global-shape screening.

Turns raw ``(x, y)`` into a :class:`~elbow_helper.types.PreparedCurve` on the
unit square or raises :class:`Abstain` with a reason code when the data are
unusable or globally incompatible with the requested curve/direction.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import os_helper as oh

from .config import RobustKneeConfig
from .numerics import spearman
from .smoothing import smooth_curve
from .types import PreparedCurve, Reason


class Abstain(Exception):
    """Internal control-flow signal carrying a reason code and diagnostics.

    Parameters
    ----------
    reason : str
        A :class:`~elbow_helper.types.Reason` code explaining the abstention.
    **diagnostics
        Arbitrary supporting values, exposed to callers via
        :attr:`diagnostics`.
    """

    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def infer_curve_direction(x_norm: np.ndarray, y_scaled: np.ndarray) -> tuple:
    """Infer ``(curve, direction)`` from a cleaned, normalized curve.

    ``direction`` is the sign of the Spearman correlation between ``x_norm``
    and ``y_scaled``: positive means increasing, negative means decreasing.
    ``curve`` is read off the sign of the (lightly smoothed) curve's average
    signed deviation from the chord connecting its first and last point: a
    curve lying above its chord is concave (the mathematical definition, e.g.
    a square-root-shaped knee); a curve lying below its chord is convex (e.g.
    a k-means inertia elbow). Both reads are direction-agnostic, so they
    combine independently into all four concave/convex x increasing/decreasing
    cases.

    Parameters
    ----------
    x_norm, y_scaled : numpy.ndarray
        A cleaned, normalized curve on the unit square.

    Returns
    -------
    tuple of (str, str)
        ``(curve, direction)``, each one of the values accepted by
        :func:`prepare_curve`.
    """
    rho = spearman(x_norm, y_scaled)
    direction = "increasing" if rho >= 0 else "decreasing"

    window = max(5, y_scaled.size // 8)
    ys = smooth_curve(y_scaled, window, method="moving_average")
    chord = np.interp(x_norm, [x_norm[0], x_norm[-1]], [ys[0], ys[-1]])
    deviation = ys - chord
    # Manual trapezoidal rule: avoids numpy.trapz (deprecated in NumPy >= 2.0)
    # vs. numpy.trapezoid (added in NumPy 2.0), keeping this compatible with
    # the numpy>=1.24 floor declared in pyproject.toml.
    area = float(np.sum((deviation[:-1] + deviation[1:]) / 2.0 * np.diff(x_norm)))
    curve = "concave" if area >= 0 else "convex"
    return curve, direction


def _clean_and_normalize(x, y, min_samples: int, robust_y_scaling: bool = True):
    """Clean, sort, deduplicate, and normalize a raw curve to the unit square.

    Shared by :func:`prepare_curve` (single-knee, adds shape screening on
    top) and :func:`prepare_curve_unconstrained` (multi-knee, no shape
    assumption). Raises :class:`Abstain` with ``INVALID_INPUT``,
    ``INSUFFICIENT_DATA`` or ``ZERO_RANGE`` when the data cannot be
    processed.

    Parameters
    ----------
    robust_y_scaling : bool, optional
        If ``True`` (default, used by :func:`prepare_curve`), scale ``y`` on
        its 5th/95th percentiles, clipped to ``[0, 1]``, for outlier
        robustness ahead of the single-knee difference-curve search. If ``False``
        (used by :func:`prepare_curve_unconstrained`), scale on the plain
        min/max instead: percentile clipping flattens roughly the bottom
        and top 5% of points to exactly 0 or 1, which the single-knee
        pipeline's boundary-margin filter discards as a
        ``BOUNDARY_KNEE``, but which a multi-breakpoint search has no such
        filter for and will happily read as a genuine extra breakpoint at
        the clipped tail (found by testing: a clean single-breakpoint curve
        was reported as two breakpoints, one of them sitting exactly at the
        edge of the percentile-clipped flat region).

    Returns
    -------
    tuple
        ``(x_norm, y_scaled, n, x_lo, x_hi, y_lo, y_hi)``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size != y.size:
        raise Abstain(Reason.INVALID_INPUT, detail="x and y length mismatch")

    # Drop non-finite pairs.
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size == 0:
        raise Abstain(Reason.INVALID_INPUT, detail="no finite samples")

    # Sort by x.
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]

    # Aggregate duplicate x with the median y.
    ux, inv = np.unique(x, return_inverse=True)
    if ux.size != x.size:
        uy = np.array([np.median(y[inv == k]) for k in range(ux.size)])
        x, y = ux, uy

    n = x.size
    if n < min_samples:
        raise Abstain(Reason.INSUFFICIENT_DATA, n=int(n), min_samples=min_samples)

    x_lo, x_hi = float(x.min()), float(x.max())
    if x_hi - x_lo < 1e-12 or float(y.max() - y.min()) < 1e-12:
        raise Abstain(Reason.ZERO_RANGE, x_range=x_hi - x_lo)

    # Normalize x to [0, 1].
    x_norm = (x - x_lo) / (x_hi - x_lo)

    if robust_y_scaling:
        # Robust y scaling on the 5th/95th percentiles, clipped to [0, 1].
        y_lo, y_hi = np.quantile(y, [0.05, 0.95])
        if y_hi - y_lo < 1e-12:
            y_lo, y_hi = float(y.min()), float(y.max())
        y_scaled = np.clip((y - y_lo) / (y_hi - y_lo), 0.0, 1.0)
    else:
        y_lo, y_hi = float(y.min()), float(y.max())
        y_scaled = (y - y_lo) / (y_hi - y_lo)

    return x_norm, y_scaled, n, x_lo, x_hi, float(y_lo), float(y_hi)


def prepare_curve(
    x, y, curve: Optional[str], direction: Optional[str], config: RobustKneeConfig
) -> PreparedCurve:
    """Clean, sort, deduplicate, normalize, and screen a curve.

    Parameters
    ----------
    x, y : array-like
        Raw input coordinates of equal length.
    curve : str or None
        ``"concave"`` or ``"convex"`` (passed through to the locator). If
        ``None``, inferred from the data via :func:`infer_curve_direction`.
    direction : str or None
        ``"increasing"`` or ``"decreasing"``. If ``None``, inferred from the
        data via :func:`infer_curve_direction`.
    config : RobustKneeConfig
        Thresholds; ``min_samples``, ``min_spearman_abs`` and
        ``max_direction_violation_rate`` are consulted here.

    Returns
    -------
    PreparedCurve
        The normalized curve plus inverse-transform metadata, with ``curve``
        and ``direction`` resolved to the actual values used (never ``None``).

    Raises
    ------
    Abstain
        With ``INVALID_INPUT``, ``INSUFFICIENT_DATA``, ``ZERO_RANGE`` or
        ``INCOMPATIBLE_GLOBAL_SHAPE`` when the data cannot be processed.
    """
    if curve is not None and curve not in ("concave", "convex"):
        raise Abstain(Reason.INVALID_INPUT, detail="curve invalid")
    if direction is not None and direction not in ("increasing", "decreasing"):
        raise Abstain(Reason.INVALID_INPUT, detail="direction invalid")

    x_norm, y_scaled, n, x_lo, x_hi, y_lo, y_hi = _clean_and_normalize(
        x, y, config.min_samples
    )

    # Auto-detect whatever the caller left unspecified, from the cleaned data.
    if curve is None or direction is None:
        inferred_curve, inferred_direction = infer_curve_direction(x_norm, y_scaled)
        curve = curve or inferred_curve
        direction = direction or inferred_direction

    # Global shape compatibility.
    rho = spearman(x_norm, y_scaled)
    viol = _direction_violation_rate(y_scaled, direction)

    if abs(rho) < config.min_spearman_abs or viol > config.max_direction_violation_rate:
        raise Abstain(
            Reason.INCOMPATIBLE_GLOBAL_SHAPE,
            spearman=round(rho, 4),
            violation_rate=round(viol, 4),
        )

    oh.info(
        f"[elbow-helper] prepared curve: n={n}, spearman={rho:.3f}, "
        f"violation_rate={viol:.3f}"
    )

    return PreparedCurve(
        x_norm=x_norm,
        y_scaled=y_scaled,
        n=n,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        curve=curve,
        direction=direction,
        spearman=rho,
        violation_rate=viol,
    )


def prepare_curve_unconstrained(x, y, min_samples: int) -> PreparedCurve:
    """Clean, sort, deduplicate and normalize a curve with no shape assumption.

    Used by :func:`elbow_helper.robust_knees`: a multi-breakpoint search has
    no single global ``curve``/``direction`` to check against (segments may
    alternate slope sign freely), so this skips the shape-compatibility gate
    in :func:`prepare_curve` entirely. ``curve``/``direction`` on the
    returned :class:`PreparedCurve` are set to ``"n/a"`` and unused.

    Parameters
    ----------
    x, y : array-like
        The raw curve.
    min_samples : int
        Minimum number of points required after cleaning.

    Returns
    -------
    PreparedCurve
        The cleaned, normalized curve, ready for candidate search.

    Raises
    ------
    Abstain
        With ``INVALID_INPUT``, ``INSUFFICIENT_DATA`` or ``ZERO_RANGE``.
    """
    x_norm, y_scaled, n, x_lo, x_hi, y_lo, y_hi = _clean_and_normalize(
        x, y, min_samples, robust_y_scaling=False
    )
    return PreparedCurve(
        x_norm=x_norm,
        y_scaled=y_scaled,
        n=n,
        x_lo=x_lo,
        x_hi=x_hi,
        y_lo=y_lo,
        y_hi=y_hi,
        curve="n/a",
        direction="n/a",
        spearman=float(spearman(x_norm, y_scaled)),
        violation_rate=0.0,
    )


def _direction_violation_rate(y: np.ndarray, direction: str) -> float:
    """Magnitude-weighted fraction of movement *against* ``direction``.

    A plain sign-count is unusable for knee/elbow curves: their flat tail has a
    near-zero slope, so noise flips roughly half of the local first differences
    even though the curve is globally monotone. Weighting each step by its
    magnitude lets the (large) steep-region steps dominate, so a genuine
    saturating curve scores a low violation rate while true non-monotonicity
    (a real reversal or pure noise) still scores high.

    Parameters
    ----------
    y : numpy.ndarray
        The (scaled) signal.
    direction : str
        ``"increasing"`` or ``"decreasing"``, the claimed global direction.

    Returns
    -------
    float
        Fraction, in ``[0, 1]``, of magnitude-weighted movement against
        ``direction``.
    """
    if y.size < 3:
        return 0.0
    # Smooth at the *trend* scale (~n/8) so the flat tail's noise does not
    # masquerade as non-monotonicity; a genuine reversal survives this.
    window = max(5, y.size // 8)
    ys = smooth_curve(y, window, method="moving_average")
    d = np.diff(ys)
    total = np.abs(d).sum()
    if total < 1e-12:
        return 0.0
    against = d < 0 if direction == "increasing" else d > 0
    return float(np.abs(d[against]).sum() / total)
