"""Phase M4: the public ``robust_knees`` (plural) orchestration.

Ships the exact combination validated in ``research/multiknee/RESULTS.md``
and derived in ``MATH-en.tex`` (sections 5-20): dynamic-program search, the
subtractive-sign modified BIC as the selection criterion, and a
Bonferroni-gated sequential permutation test layered on top by default,
matching this package's design priority of minimising false-positive knees
even at the cost of more abstentions.

Segments are independent (discontinuous) OLS lines, not the continuous
broken-line model :func:`elbow_helper.robust_knee` uses; see ``MATH-en.tex``
section 5 for why that difference is deliberate.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import os_helper as oh

from .config import RobustKneesConfig
from .multi_criteria import modified_bic
from .multi_fwer import sequential_fwer_gate
from .multi_segmentation import SegmentCostTable, dp_optimal_partition
from .preprocessing import Abstain, prepare_curve_unconstrained
from .types import InvalidKnees, KneeEstimate, Knees, MultiKneeResult, Reason


def robust_knees(
    x, y=None, config: Optional[RobustKneesConfig] = None
) -> MultiKneeResult:
    """Detect zero or more knees, with the same abstain-rather-than-guess discipline.

    Unlike :func:`robust_knee`, an empty result is not an abstention: it is
    the pipeline's confident conclusion that the data has no real
    breakpoint, having survived the same search and false-positive gates a
    nonempty result would have to survive. Only preprocessing failures
    (bad input, too little data, zero range) return :class:`InvalidKnees`.

    Parameters
    ----------
    x, y : array-like
        The curve: ``x[i]`` maps to ``y[i]``. ``y`` may be omitted, in which
        case ``x`` is taken to be the y-values alone against an implicit
        ``0, 1, ..., n-1``, as in :func:`robust_knee`. No ``curve`` or
        ``direction`` is needed: segments may alternate slope sign freely.
    config : RobustKneesConfig, optional
        Search size, false-positive-control settings. Defaults to
        :class:`RobustKneesConfig`.

    Returns
    -------
    MultiKneeResult
        A :class:`Knees` (with zero or more :class:`KneeEstimate`, and
        diagnostics from every stage) or an :class:`InvalidKnees` (with a
        reason code, for unusable input only).
    """
    if y is None:
        y = x
        x = np.arange(len(np.asarray(y).ravel()), dtype=float)

    config = config or RobustKneesConfig()
    diagnostics: dict = {}

    try:
        prepared = prepare_curve_unconstrained(x, y, config.min_samples)
    except Abstain as a:
        diagnostics.update(a.diagnostics)
        return InvalidKnees(reason=a.reason, diagnostics=diagnostics)

    try:
        x_norm, y_scaled, n = prepared.x_norm, prepared.y_scaled, prepared.n
        min_seg = max(5, round(config.min_seg_fraction * n))
        seed = config.random_seed if config.random_seed is not None else 0

        segs = dp_optimal_partition(x_norm, y_scaled, config.k_max, min_seg)
        diagnostics.update(
            n=n, min_seg=min_seg, k_max_requested=config.k_max,
            k_max_effective=len(segs) - 1,
        )

        scores = [modified_bic(s) for s in segs]
        mbic_k = int(np.argmin(scores))
        diagnostics["mbic_scores"] = [round(s, 3) for s in scores]
        diagnostics["mbic_k"] = mbic_k

        final_k = mbic_k
        p_values: list = []
        if config.require_fwer_confirmation:
            fwer_k, p_values = sequential_fwer_gate(
                x_norm, y_scaled, segs, config.fwer_alpha,
                config.fwer_permutations, min_seg, seed,
            )
            diagnostics["fwer_k"] = fwer_k
            diagnostics["fwer_p_values"] = [round(p, 4) for p in p_values]
            final_k = min(mbic_k, fwer_k)
        diagnostics["final_k"] = final_k

        chosen = segs[final_k]
        table = SegmentCostTable(x_norm, y_scaled)
        y_scale = prepared.y_hi - prepared.y_lo
        x_scale = prepared.x_hi - prepared.x_lo

        knees = []
        for idx, cut in enumerate(chosen.breakpoints):
            lo, hi = chosen.boundaries[idx], chosen.boundaries[idx + 2]
            _, slope_left_norm = table.fit(lo, cut)
            _, slope_right_norm = table.fit(cut, hi)
            x_norm_at_cut = float(x_norm[cut])
            knees.append(
                KneeEstimate(
                    x=prepared.denormalize_x(x_norm_at_cut),
                    x_norm=x_norm_at_cut,
                    index=cut,
                    slope_left=float(slope_left_norm * y_scale / x_scale),
                    slope_right=float(slope_right_norm * y_scale / x_scale),
                    fwer_p_value=p_values[idx] if idx < len(p_values) else None,
                )
            )

        oh.info(f"[elbow-helper] robust_knees: k={len(knees)}")
        return Knees(reason=Reason.KNEES_FOUND, diagnostics=diagnostics, knees=knees)

    except Exception as exc:  # numerical safety net -- never crash the caller
        oh.warning(f"[elbow-helper] internal failure: {exc}")
        diagnostics["error"] = str(exc)
        return InvalidKnees(reason=Reason.INTERNAL_NUMERICAL_FAILURE, diagnostics=diagnostics)
