"""Sequential, Bonferroni-corrected permutation test for the number of breakpoints.

Independent of BIC/mBIC/ICL: rather than scoring every k and picking a
minimizer, this walks the nested sequence of segmentations k = 1, 2, ...
(from :func:`segmentation.dp_optimal_partition` or
:func:`segmentation.greedy_binary_segmentation`) and, at each step, tests
whether the SSE reduction from adding breakpoint k over the (k-1)-breakpoint
model is larger than chance would produce, via an IID-residual permutation
test against a Bonferroni-corrected significance level ``alpha / k_max``.
Stops (accepts no further breakpoints) at the first non-significant step,
so the family-wise false-positive rate across up to ``k_max`` sequential
tests is controlled at (approximately) ``alpha``: approximately because
Bonferroni is conservative for correlated tests. These sequential tests are
themselves correlated, each conditioning on the model accepted at the
previous step.

This directly implements the "sequential gate" design the earlier
literature-research forks (BIC/mBIC/PELT/alternatives and ICL) converged
on recommending as elbow-helper's existing gate architecture already works
this way (each additional claim independently has to survive its own
evidence), rather than a global joint test over all k at once.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from segmentation import Segmentation, SegmentCostTable


def _fitted_values(
    x: np.ndarray, y: np.ndarray, boundaries: Sequence[int]
) -> np.ndarray:
    """OLS-fitted y for a given (possibly discontinuous) segmentation."""
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
    """Largest SSE reduction from adding exactly one more split, anywhere."""
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
    alpha: float = 0.05,
    n_permutations: int = 200,
    min_seg: int = 3,
    seed: int = 0,
) -> Tuple[int, List[float]]:
    """Bonferroni-gated sequential test over a nested segmentation sequence.

    Parameters
    ----------
    x, y : array-like
        The curve.
    segmentations : list of Segmentation
        ``segmentations[k]`` for k = 0, ..., k_max, e.g. from
        :func:`segmentation.dp_optimal_partition`. Must be nested in the
        sense that k increases by exactly one breakpoint at a time (true for
        both the DP and greedy searches here).
    alpha : float, optional
        Family-wise error rate target. Default 0.05.
    n_permutations : int, optional
        IID-residual permutation draws per sequential test.
    min_seg : int, optional
        Minimum segment length, must match what produced ``segmentations``.

    Returns
    -------
    (int, list of float)
        The accepted k (the largest k whose sequential test, together with
        every test before it, passed the Bonferroni-corrected threshold)
        and the p-value computed at each tested step (length = number of
        steps actually tested, stopping at the first failure).
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
