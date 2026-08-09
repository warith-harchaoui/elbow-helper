"""Phase M3: the Bonferroni-gated sequential test for :func:`elbow_helper.robust_knees`.

Ported from the validated ``research/multiknee/fwer.py``; see ``ELBOW-en.tex``
section 20 for the derivation. Independent of :mod:`multi_criteria`: walks
the nested DP segmentation sequence and, at each step, tests whether adding
one more breakpoint reduces error by more than an IID-residual permutation
null would produce, at a Bonferroni-corrected significance level.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .multi_segmentation import Segmentation, SegmentCostTable


def _fitted_values(
    x: np.ndarray, y: np.ndarray, boundaries: Sequence[int]
) -> np.ndarray:
    """OLS-fitted y for a given (discontinuous) segmentation.

    Parameters
    ----------
    x, y : numpy.ndarray
        The curve.
    boundaries : sequence of int
        Segment boundaries, ``(0, ..., n)``.

    Returns
    -------
    numpy.ndarray
        Per-point fitted values, one independent OLS line per segment.
    """
    fitted = np.empty_like(y)
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        xs, ys = x[lo:hi], y[lo:hi]
        if hi - lo < 2 or np.ptp(xs) < 1e-12:
            fitted[lo:hi] = np.mean(ys)
            continue
        design = np.column_stack([np.ones_like(xs), xs])
        coef, _, _, _ = np.linalg.lstsq(design, ys, rcond=None)
        fitted[lo:hi] = design @ coef
    return fitted


def _best_single_split_reduction(
    table: SegmentCostTable, boundaries: Sequence[int], min_seg: int
) -> float:
    """Largest RSS reduction from adding exactly one more split, anywhere.

    Parameters
    ----------
    table : SegmentCostTable
        O(1) segment-cost lookup for the current (possibly resampled) curve.
    boundaries : sequence of int
        The accepted ``(k-1)``-breakpoint segmentation's boundaries; each
        segment is searched independently for its single best extra cut.
    min_seg : int
        Minimum number of points a candidate sub-segment must keep.

    Returns
    -------
    float
        The largest achievable RSS reduction from one additional split,
        or ``0.0`` if no segment is long enough to split.
    """
    best = 0.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        if hi - lo < 2 * min_seg:
            continue
        base_cost = table.cost(lo, hi)
        for cut in range(lo + min_seg, hi - min_seg + 1):
            reduction = base_cost - (table.cost(lo, cut) + table.cost(cut, hi))
            if reduction > best:
                best = reduction
    return best


def sequential_fwer_gate(
    x: np.ndarray,
    y: np.ndarray,
    segmentations: Sequence[Segmentation],
    alpha: float,
    n_permutations: int,
    min_seg: int,
    seed: int,
) -> Tuple[int, List[float]]:
    """Bonferroni-gated sequential test over a nested segmentation sequence.

    Parameters
    ----------
    x, y : numpy.ndarray
        The curve.
    segmentations : sequence of Segmentation
        One :class:`~elbow_helper.multi_segmentation.Segmentation` per
        ``k = 0, 1, ..., k_max``, as returned by
        :func:`~elbow_helper.multi_segmentation.dp_optimal_partition`.
    alpha : float
        Nominal family-wise significance level.
    n_permutations : int
        Number of IID-residual permutation replicates per tested step.
    min_seg : int
        Minimum number of points per segment.
    seed : int
        Seed for the permutation random number generator.

    Returns
    -------
    (int, list of float)
        The accepted k (the largest k whose sequential test, together with
        every test before it, passed the Bonferroni-corrected threshold)
        and the p-value computed at each tested step (stopping at the first
        failure).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    k_max = len(segmentations) - 1
    if k_max <= 0:
        return 0, []
    alpha_corrected = alpha / k_max

    accepted_k = 0
    p_values: List[float] = []
    for k in range(1, k_max + 1):
        prev = segmentations[k - 1]
        cur = segmentations[k]
        observed_reduction = prev.sse - cur.sse

        fitted = _fitted_values(x, y, prev.boundaries)
        resid = y - fitted

        count_ge = 0
        for _ in range(n_permutations):
            r_star = rng.choice(resid, size=resid.size, replace=True)
            y_star = fitted + r_star
            table_star = SegmentCostTable(x, y_star)
            best_reduction = _best_single_split_reduction(
                table_star, prev.boundaries, min_seg
            )
            if best_reduction >= observed_reduction - 1e-12:
                count_ge += 1
        p_value = (count_ge + 1) / (n_permutations + 1)
        p_values.append(p_value)

        if p_value <= alpha_corrected:
            accepted_k = k
        else:
            break

    return accepted_k, p_values
