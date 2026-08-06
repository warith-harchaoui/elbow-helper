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
from .types import NullEvidence, PreparedCurve, Reason


def no_knee_null_test(
    prepared: PreparedCurve,
    observed_statistic: tuple,
    knee_x_norm: float,
    config: RobustKneeConfig,
) -> NullEvidence:
    """Monte-Carlo test of the observed knee against a straight-line null.

    The null model is a straight line (the *shape* under "no knee") carrying
    noise of the magnitude estimated from the **accepted broken-line fit** — not
    from the straight-line fit, whose residuals on a genuinely kinked curve are
    the knee signal itself and would inflate the null distribution.

    Parameters
    ----------
    prepared : PreparedCurve
        The observed normalized curve.
    observed_statistic : tuple
        The search statistic of the accepted knee (from :func:`run_search`).
    knee_x_norm : float
        The accepted knee, used to estimate the true noise scale.
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

    # True noise scale: residuals of the accepted broken-line model.
    k = float(knee_x_norm)
    broken_design = np.column_stack([np.ones_like(x), x, np.maximum(0.0, x - k)])
    broken_coef, _ = ols_rss(broken_design, y)
    residuals = y - broken_design @ broken_coef

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
