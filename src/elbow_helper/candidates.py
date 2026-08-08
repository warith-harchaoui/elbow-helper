"""Phase 3 — candidate generation with the from-scratch locator.

Sweeps every (smoothing window, sensitivity) setting, runs
:class:`~elbow_helper.locator.KneeLocator` in online mode, and collects every
returned knee as a :class:`~elbow_helper.types.KneeCandidate` — annotated with
metrics but *not* yet accepted.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import RobustKneeConfig
from .locator import KneeLocator
from .metrics import evaluate_candidate
from .smoothing import smooth_curve, smoothing_grid
from .types import KneeCandidate, PreparedCurve


def sensitivity_grid(n: int, config: RobustKneeConfig) -> List[float]:
    """Distinct sensitivities ``S`` from ``sensitivity_fractions``."""
    values = set()
    for frac in config.sensitivity_fractions:
        values.add(float(max(1, round(frac * n))))
    return sorted(values)


def generate_candidates(
    prepared: PreparedCurve, config: RobustKneeConfig
) -> List[KneeCandidate]:
    """Run the full scale-space × sensitivity locator sweep.

    Parameters
    ----------
    prepared : PreparedCurve
        The normalized curve.
    config : RobustKneeConfig
        Supplies the smoothing and sensitivity grids.

    Returns
    -------
    list of KneeCandidate
        Every knee found, annotated with metrics. No acceptance yet.
    """
    x = prepared.x_norm
    y = prepared.y_scaled
    n = prepared.n
    windows = smoothing_grid(n, config)
    sensitivities = sensitivity_grid(n, config)

    candidates: List[KneeCandidate] = []
    for window in windows:
        y_smooth = smooth_curve(y, window, method="gaussian")
        for s in sensitivities:
            try:
                kl = KneeLocator(
                    x,
                    y_smooth,
                    S=s,
                    curve=prepared.curve,
                    direction=prepared.direction,
                    interp_method="interp1d",
                    online=True,
                )
            except Exception:
                continue

            for rec in kl.all_knee_records:
                knee_x = float(rec["knee"])
                idx = int(np.argmin(np.abs(x - knee_x)))
                cand = KneeCandidate(
                    knee_x_norm=knee_x,
                    knee_index=idx,
                    window=window,
                    sensitivity=s,
                )
                evaluate_candidate(
                    cand, kl.y_difference, int(rec["threshold_index"]), y
                )
                candidates.append(cand)

    return candidates
