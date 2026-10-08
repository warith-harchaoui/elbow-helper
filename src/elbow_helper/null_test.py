"""Phase 8 — the no-knee null test.

Asks: how often does the *entire* search procedure find a knee at least as
strong as the observed one when the data really have no knee? The null model is
a monotonic straight line carrying the observed residual structure. The test
statistic is the search-adjusted lexicographic tuple from :mod:`search`, and
the p-value is the usual ``(1 + #{null >= observed}) / (B + 1)``.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import RobustKneeConfig
from .numerics import ols_rss
from .search import run_search
from .smoothing import signal_and_noise
from .types import NullEvidence, PreparedCurve, Reason


def no_knee_null_test(
    prepared: PreparedCurve,
    observed_statistic: tuple,
    knee_x_norm: float,
    config: RobustKneeConfig,
) -> NullEvidence:
    """Monte-Carlo test of the observed knee against a straight-line null.

    The null model is a straight line (the *shape* under "no knee") carrying
    noise of the magnitude the data actually show — not the straight-line
    fit's own residuals, which on a genuinely kinked curve are the knee signal
    itself and would inflate the null distribution. The noise scale comes from
    :func:`~elbow_helper.smoothing.signal_and_noise`, i.e. the residuals of a
    light smoother rescaled to the successive-difference noise estimate. It
    used to come from the accepted broken line's residuals, which carry that
    model's misfit on any smoothly bending curve and so made the null
    replicates far noisier than the data, miscalibrating the test in the
    permissive direction.

    Parameters
    ----------
    prepared : PreparedCurve
        The observed normalized curve.
    observed_statistic : tuple
        The search statistic of the accepted knee (from :func:`run_search`).
    knee_x_norm : float
        The accepted knee. Unused since the noise scale stopped coming from
        the broken-line fit; kept so the stage signatures stay uniform.
    config : RobustKneeConfig
        ``null_replicates``, ``max_null_p_value`` and ``random_seed``.

    Returns
    -------
    NullEvidence
        The Monte-Carlo p-value and a pass flag (with reason on failure).
    """
    x = prepared.x_norm
    y = prepared.y_scaled

    # No-knee mean shape: the straight line.
    line_design = np.column_stack([np.ones_like(x), x])
    line_coef, _ = ols_rss(line_design, y)
    yhat = line_design @ line_coef

    # True noise scale: the curve's own high-frequency component.
    _, residuals = signal_and_noise(y, config)

    # Offset the seed so the null draws differ from the bootstrap draws.
    seed = None if config.random_seed is None else config.random_seed + 10_007
    rng = np.random.default_rng(seed)

    b = config.null_replicates
    at_least = 0
    for _ in range(b):
        resampled = rng.choice(residuals, size=residuals.size, replace=True)
        y_star = yhat + resampled
        prepared_star = replace(prepared, y_scaled=y_star)
        res = run_search(prepared_star, config, confirm=True)
        null_stat = res.statistic if res.detected else (0, 0.0, 0.0)
        if null_stat >= observed_statistic:
            at_least += 1

    p_value = (1 + at_least) / (b + 1)
    passes = p_value <= config.max_null_p_value

    return NullEvidence(
        passes=passes,
        p_value=p_value,
        observed_statistic=observed_statistic,
        null_replicates=b,
        reason=None if passes else Reason.NULL_NOT_REJECTED,
    )
