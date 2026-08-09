"""Core piecewise-linear segmentation primitives for the multi-knee study.

Model choice, stated explicitly: segments are **independent** ordinary-least-
squares lines (intercept + slope per segment, discontinuous at breakpoints).
This is a deliberate departure from elbow-helper's shipped single-knee model
(a *continuous* broken line via a relu basis, see ``elbow_helper.segmented``).
The continuous model does not decompose into independent per-segment costs,
so it is not directly amenable to the additive-cost dynamic programming and
binary-segmentation algorithms this study is comparing (Yao 1988,
Zhang-Siegmund 2007, and standard binary segmentation are all stated for
independent segments). Using the same independent-segment model as that
literature is what makes the comparison faithful to it. Whether the shipped
API should eventually use continuous or independent segments is a separate
design question, deferred.

Segment cost is the OLS residual sum of squares for a line fit to that
segment, computed in O(1) from prefix sums of 1, x, y, x^2, xy (the standard
"sufficient statistics" trick used in real changepoint implementations), so
that the O(n^2) segment-cost table costs O(n) to build.

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
    """A concrete k-breakpoint segmentation of ``x, y``."""

    breakpoints: Tuple[int, ...]  # sorted interior cut indices, 0 < b < n
    boundaries: Tuple[int, ...]  # 0, *breakpoints, n  (len = k + 2)
    sse: float  # total residual sum of squares across all segments
    n: int

    @property
    def k(self) -> int:
        """Number of breakpoints (segments = k + 1)."""
        return len(self.breakpoints)

    @property
    def segment_lengths(self) -> Tuple[int, ...]:
        b = self.boundaries
        return tuple(b[i + 1] - b[i] for i in range(len(b) - 1))


class SegmentCostTable:
    """O(1)-per-query OLS segment cost, via prefix sums of sufficient statistics.

    ``cost(i, j)`` is the residual sum of squares of the best-fit line
    ``y = a + b*x`` on points ``x[i:j], y[i:j]`` (half-open, 0-indexed).
    Segments shorter than 2 points fall back to a constant fit (cost = 0 for
    a single point, since one point is fit exactly by *some* line, but the
    caller should enforce a minimum segment length; see ``min_seg``).
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = x.size
        z = np.zeros(1)
        self.n = n
        self._c = np.concatenate([z, np.arange(1, n + 1, dtype=float)])  # count
        self._sx = np.concatenate([z, np.cumsum(x)])
        self._sy = np.concatenate([z, np.cumsum(y)])
        self._sxx = np.concatenate([z, np.cumsum(x * x)])
        self._sxy = np.concatenate([z, np.cumsum(x * y)])
        self._syy = np.concatenate([z, np.cumsum(y * y)])

    def cost(self, i: int, j: int) -> float:
        """RSS of the best OLS line on points ``i..j-1``."""
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
            # Degenerate (single point or all-equal x): best constant fit.
            mean_y = sy / c
            rss = syy - c * mean_y * mean_y
            return float(max(rss, 0.0))

        b = (c * sxy - sx * sy) / denom
        a = (sy - b * sx) / c
        rss = syy - a * sy - b * sxy
        return float(max(rss, 0.0))


def dp_optimal_partition(
    x: np.ndarray, y: np.ndarray, k_max: int, min_seg: int = 3
) -> List[Segmentation]:
    """Exact optimal-partitioning DP: the best segmentation for every k = 0..k_max.

    Standard "segment neighbourhood" dynamic program. ``C[k][t]`` = minimal
    total RSS of the first ``t`` points cut into ``k+1`` segments; recursion
    ``C[k][t] = min_s C[k-1][s] + cost(s, t)`` over valid cut points ``s``.
    Complexity ``O(k_max * n^2)`` with O(1) segment costs.

    Parameters
    ----------
    x, y : array-like
        The curve, ``x`` sorted.
    k_max : int
        Largest number of breakpoints to solve for.
    min_seg : int, optional
        Minimum points per segment. Default 3 (enough to fit a line with one
        residual degree of freedom).

    Returns
    -------
    list of Segmentation
        ``result[k]`` is the optimal k-breakpoint segmentation, for
        ``k = 0, ..., k_max`` (or fewer, if ``n`` cannot support ``k_max``
        breakpoints at ``min_seg`` each).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    table = SegmentCostTable(x, y)

    max_k = min(k_max, max(0, (n // min_seg) - 1))

    # C[k] and back[k] indexed by t = 0..n; C[k][t] = best cost of t points in
    # k+1 segments. back[k][t] = the last cut point s achieving that optimum.
    C = [np.full(n + 1, np.inf) for _ in range(max_k + 1)]
    back = [np.full(n + 1, -1, dtype=int) for _ in range(max_k + 1)]

    for t in range(min_seg, n + 1):
        C[0][t] = table.cost(0, t)

    for k in range(1, max_k + 1):
        lo_t = min_seg * (k + 1)
        for t in range(lo_t, n + 1):
            best_cost = np.inf
            best_s = -1
            # s is the end of the first k segments / start of the last one.
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
            Segmentation(
                breakpoints=tuple(cuts), boundaries=boundaries, sse=float(C[k][n]), n=n
            )
        )
    return results


def greedy_binary_segmentation(
    x: np.ndarray, y: np.ndarray, k_max: int, min_seg: int = 3
) -> List[Segmentation]:
    """Greedy binary segmentation: repeatedly split the segment that most
    reduces total RSS, up to ``k_max`` breakpoints.

    At each step, every current segment is scanned for its single best
    internal split (largest RSS reduction); the best split across all
    segments is committed and the process repeats. This is the standard
    binary-segmentation heuristic adapted to RSS reduction as the split
    criterion (rather than a CUSUM statistic), suboptimal relative to
    :func:`dp_optimal_partition` in general, but sub-quadratic per split and
    a fair proxy for "greedy" search in this comparison.

    Returns
    -------
    list of Segmentation
        ``result[k]`` for ``k = 0, ..., k_max`` (or fewer if no further split
        can respect ``min_seg``), each carrying the *cumulative* SSE after
        ``k`` greedy splits (segments are re-fit exactly at each level, so
        this is the true SSE of that specific nested segmentation, not an
        approximation).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    table = SegmentCostTable(x, y)

    boundaries = [0, n]
    results = [
        Segmentation(breakpoints=(), boundaries=(0, n), sse=table.cost(0, n), n=n)
    ]

    for _ in range(k_max):
        best_gain = -np.inf
        best_seg_idx = -1
        best_cut = -1
        for i in range(len(boundaries) - 1):
            lo, hi = boundaries[i], boundaries[i + 1]
            if hi - lo < 2 * min_seg:
                continue
            base_cost = table.cost(lo, hi)
            for cut in range(lo + min_seg, hi - min_seg + 1):
                gain = base_cost - (table.cost(lo, cut) + table.cost(cut, hi))
                if gain > best_gain:
                    best_gain = gain
                    best_seg_idx = i
                    best_cut = cut
        if best_seg_idx < 0 or best_gain <= 0:
            break
        boundaries.insert(best_seg_idx + 1, best_cut)
        total_sse = sum(
            table.cost(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        )
        cuts = tuple(boundaries[1:-1])
        results.append(
            Segmentation(
                breakpoints=cuts,
                boundaries=tuple(boundaries),
                sse=float(total_sse),
                n=n,
            )
        )
    return results
