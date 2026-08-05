"""Phase 1 — cleaning, robust normalization, and global-shape screening.

Turns raw ``(x, y)`` into a :class:`~elbow_helper.types.PreparedCurve` on the
unit square, or raises :class:`Abstain` with a reason code when the data are
unusable or globally incompatible with the requested curve/direction.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

import numpy as np
import os_helper as oh

from .config import RobustKneeConfig
from .numerics import spearman
from .smoothing import smooth_curve
from .types import PreparedCurve, Reason


class Abstain(Exception):
    """Internal control-flow signal carrying a reason code and diagnostics."""

    def __init__(self, reason: str, **diagnostics):
        super().__init__(reason)
        self.reason = reason
        self.diagnostics = diagnostics


def prepare_curve(x, y, curve: str, direction: str, config: RobustKneeConfig) -> PreparedCurve:
    """Clean, sort, deduplicate, normalize, and screen a curve.

    Parameters
    ----------
    x, y : array-like
        Raw input coordinates of equal length.
    curve : str
        ``"concave"`` or ``"convex"`` (passed through to Kneedle).
    direction : str
        ``"increasing"`` or ``"decreasing"``.
    config : RobustKneeConfig
        Thresholds; ``min_samples``, ``min_spearman_abs`` and
        ``max_direction_violation_rate`` are consulted here.

    Returns
    -------
    PreparedCurve
        The normalized curve plus inverse-transform metadata.

    Raises
    ------
    Abstain
        With ``INVALID_INPUT``, ``INSUFFICIENT_DATA``, ``ZERO_RANGE`` or
        ``INCOMPATIBLE_GLOBAL_SHAPE`` when the data cannot be processed.
    """
    if curve not in ("concave", "convex") or direction not in ("increasing", "decreasing"):
        raise Abstain(Reason.INVALID_INPUT, detail="curve/direction invalid")

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
    if n < config.min_samples:
        raise Abstain(Reason.INSUFFICIENT_DATA, n=int(n), min_samples=config.min_samples)

    x_lo, x_hi = float(x.min()), float(x.max())
    if x_hi - x_lo < 1e-12 or float(y.max() - y.min()) < 1e-12:
        raise Abstain(Reason.ZERO_RANGE, x_range=x_hi - x_lo)

    # Normalize x to [0, 1].
    x_norm = (x - x_lo) / (x_hi - x_lo)

    # Robust y scaling on the 5th/95th percentiles, clipped to [0, 1].
    y_lo, y_hi = np.quantile(y, [0.05, 0.95])
    if y_hi - y_lo < 1e-12:
        y_lo, y_hi = float(y.min()), float(y.max())
    y_scaled = np.clip((y - y_lo) / (y_hi - y_lo), 0.0, 1.0)

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
        y_lo=float(y_lo),
        y_hi=float(y_hi),
        curve=curve,
        direction=direction,
        spearman=rho,
        violation_rate=viol,
    )


def _direction_violation_rate(y: np.ndarray, direction: str) -> float:
    """Magnitude-weighted fraction of movement *against* ``direction``.

    A plain sign-count is unusable for knee/elbow curves: their flat tail has a
    near-zero slope, so noise flips roughly half of the local first differences
    even though the curve is globally monotone. Weighting each step by its
    magnitude lets the (large) steep-region steps dominate, so a genuine
    saturating curve scores a low violation rate while true non-monotonicity
    (a real reversal, or pure noise) still scores high.
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
