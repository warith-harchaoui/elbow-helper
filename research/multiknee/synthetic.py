"""Synthetic piecewise-linear curves with a known true number of breakpoints.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# Breakpoint fractions and per-segment slopes for each true k, chosen to
# alternate steep/shallow/steep so each breakpoint is a genuine, visually
# real slope change rather than a borderline case, but not exaggerated to
# the point where the comparison would be uninformative (every method
# should be *able* to succeed here; the noise level is what makes it hard).
_PROFILES = {
    0: {"fracs": [], "slopes": [1.0]},
    1: {"fracs": [0.40], "slopes": [3.0, 0.3]},
    2: {"fracs": [0.30, 0.65], "slopes": [3.0, 0.3, 2.2]},
    3: {"fracs": [0.22, 0.48, 0.74], "slopes": [3.0, 0.3, 2.2, 0.2]},
}


def make_piecewise_curve(
    true_k: int, n: int = 100, noise_sigma: float = 0.05, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """A piecewise-linear curve with exactly ``true_k`` real breakpoints.

    Parameters
    ----------
    true_k : int
        Number of true breakpoints, one of ``0, 1, 2, 3``.
    n : int, optional
        Number of samples.
    noise_sigma : float, optional
        Gaussian noise standard deviation added to y (y itself is scaled to
        roughly unit range before noise is added, so this is comparable
        across noise levels).
    seed : int, optional
        RNG seed.

    Returns
    -------
    (x, y, true_breakpoint_indices)
        ``x`` and ``y`` as float arrays and the true breakpoint sample
        indices (0-indexed positions in the sorted, regular-grid ``x``).
    """
    if true_k not in _PROFILES:
        raise ValueError(f"true_k must be one of {sorted(_PROFILES)}")
    profile = _PROFILES[true_k]
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, n)
    fracs = profile["fracs"]
    slopes = profile["slopes"]
    bp_x = [f * (x[-1] - x[0]) + x[0] for f in fracs]

    y = np.empty(n)
    knots_x = [x[0], *bp_x, x[-1] + 1.0]
    level = 0.0
    for seg_i in range(len(slopes)):
        lo_x, hi_x = knots_x[seg_i], knots_x[seg_i + 1]
        mask = (x >= lo_x) & (x < hi_x)
        y[mask] = level + slopes[seg_i] * (x[mask] - lo_x)
        seg_pts = x[mask]
        if seg_pts.size:
            level = level + slopes[seg_i] * (seg_pts[-1] - lo_x)

    y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    y = y + rng.normal(0, noise_sigma, n)

    true_indices = [int(np.searchsorted(x, bx)) for bx in bp_x]
    return x, y, true_indices
