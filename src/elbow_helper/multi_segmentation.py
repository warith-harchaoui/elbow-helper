"""Phase M1: piecewise-linear segmentation for :func:`elbow_helper.robust_knees`.

Ported from the validated ``research/multiknee/segmentation.py`` (see
``research/multiknee/RESULTS.md`` and ``ELBOW-en.tex`` for the derivation and the
empirical comparison against greedy binary segmentation). Segments are
independent OLS lines, not the continuous broken line
:mod:`elbow_helper.segmented` uses for the single-knee pipeline; see
``ELBOW-en.tex`` section 5 for why that difference is deliberate.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

_EPS = 1e-12


@dataclass
class Segmentation:
    """A concrete k-breakpoint segmentation of ``x, y``.

    Parameters
    ----------
    breakpoints : tuple of int
        Interior cut indices, one per breakpoint.
    boundaries : tuple of int
        Segment boundaries, ``(0, *breakpoints, n)``.
    sse : float
        Total residual sum of squares summed over all segments.
    n : int
        Number of points in the curve this segmentation was fit to.
    """

    breakpoints: Tuple[int, ...]
    boundaries: Tuple[int, ...]
    sse: float
    n: int

    @property
    def k(self) -> int:
        """Number of breakpoints (segments = k + 1)."""
        return len(self.breakpoints)

    @property
    def segment_lengths(self) -> Tuple[int, ...]:
        """Length, in points, of each segment in ``boundaries`` order."""
        b = self.boundaries
        return tuple(b[i + 1] - b[i] for i in range(len(b) - 1))


class SegmentCostTable:
    """O(1)-per-query OLS segment cost, via prefix sums of sufficient statistics.

    Parameters
    ----------
    x, y : numpy.ndarray
        The curve, already cleaned and sorted by ``x``.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        z = np.zeros(1)
        self.n = n
        self._c = np.concatenate([z, np.arange(1, n + 1, dtype=float)])
        self._sx = np.concatenate([z, np.cumsum(x)])
        self._sy = np.concatenate([z, np.cumsum(y)])
        self._sxx = np.concatenate([z, np.cumsum(x * x)])
        self._sxy = np.concatenate([z, np.cumsum(x * y)])
        self._syy = np.concatenate([z, np.cumsum(y * y)])

    def cost(self, i: int, j: int) -> float:
        """RSS of the best OLS line on points ``i..j-1``.

        Parameters
        ----------
        i, j : int
            Half-open segment bounds into the original ``x, y`` arrays.

        Returns
        -------
        float
            The residual sum of squares of the best-fit line on that segment.
        """
        m = j - i
        if m <= 0:
            return 0.0
        c = self._c[j] - self._c[i]
        sx = self._sx[j] - self._sx[i]
        sy = self._sy[j] - self._sy[i]
        sxx = self._sxx[j] - self._sxx[i]
        sxy = self._sxy[j] - self._sxy[i]
        syy = self._syy[j] - self._syy[i]

        denom = c * sxx - sx * sx
        if m < 2 or denom < _EPS:
            mean_y = sy / c
            rss = syy - c * mean_y * mean_y
            return float(max(rss, 0.0))

        b = (c * sxy - sx * sy) / denom
        a = (sy - b * sx) / c
        rss = syy - a * sy - b * sxy
        return float(max(rss, 0.0))

    def fit(self, i: int, j: int) -> Tuple[float, float]:
        """OLS ``(intercept, slope)`` on points ``i..j-1``.

        Parameters
        ----------
        i, j : int
            Half-open segment bounds into the original ``x, y`` arrays.

        Returns
        -------
        tuple of float
            ``(intercept, slope)`` of the best-fit line on that segment.
        """
        m = j - i
        c = self._c[j] - self._c[i]
        sx = self._sx[j] - self._sx[i]
        sy = self._sy[j] - self._sy[i]
        sxx = self._sxx[j] - self._sxx[i]
        sxy = self._sxy[j] - self._sxy[i]
        denom = c * sxx - sx * sx
        if m < 2 or denom < _EPS:
            return float(sy / c), 0.0
        b = (c * sxy - sx * sy) / denom
        a = (sy - b * sx) / c
        return float(a), float(b)


def dp_optimal_partition(
    x: np.ndarray, y: np.ndarray, k_max: int, min_seg: int = 3
) -> List[Segmentation]:
    """Exact optimal-partitioning DP: the best segmentation for every k = 0..k_max.

    See ``ELBOW-en.tex`` section 7 for the recursion and complexity, and
    ``research/multiknee/RESULTS.md`` for why this is used instead of greedy
    binary segmentation.

    Parameters
    ----------
    x, y : numpy.ndarray
        The curve, already cleaned and sorted by ``x``.
    k_max : int
        Largest number of breakpoints to solve for.
    min_seg : int, optional
        Minimum number of points per segment. Defaults to ``3``.

    Returns
    -------
    list of Segmentation
        One optimal :class:`Segmentation` per achievable ``k`` from ``0`` up
        to ``min(k_max, (n // min_seg) - 1)``, in increasing order of ``k``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    table = SegmentCostTable(x, y)

    max_k = min(k_max, max(0, (n // min_seg) - 1))

    C = [np.full(n + 1, np.inf) for _ in range(max_k + 1)]
    back = [np.full(n + 1, -1, dtype=int) for _ in range(max_k + 1)]

    for t in range(min_seg, n + 1):
        C[0][t] = table.cost(0, t)

    for k in range(1, max_k + 1):
        lo_t = min_seg * (k + 1)
        for t in range(lo_t, n + 1):
            best_cost = np.inf
            best_s = -1
            s_lo = min_seg * k
            s_hi = t - min_seg
            for s in range(s_lo, s_hi + 1):
                if not np.isfinite(C[k - 1][s]):
                    continue
                cand = C[k - 1][s] + table.cost(s, t)
                if cand < best_cost:
                    best_cost = cand
                    best_s = s
            C[k][t] = best_cost
            back[k][t] = best_s

    results: List[Segmentation] = []
    for k in range(max_k + 1):
        if not np.isfinite(C[k][n]):
            break
        cuts = []
        t = n
        for level in range(k, 0, -1):
            s = back[level][t]
            cuts.append(s)
            t = s
        cuts.reverse()
        boundaries = (0, *cuts, n)
        results.append(
            Segmentation(breakpoints=tuple(cuts), boundaries=boundaries,
                         sse=float(C[k][n]), n=n)
        )
    return results
