"""Phase 2 — the smoothing scale-space.

Generates a grid of odd smoothing windows (always including ``1`` = no
smoothing) and applies a centered Gaussian smoother with reflected boundaries,
implemented in pure NumPy (no ``scipy.ndimage``).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import RobustKneeConfig


def _nearest_odd(v: float) -> int:
    """Round ``v`` to the nearest odd integer ``>= 1``.

    Parameters
    ----------
    v : float
        Value to round.

    Returns
    -------
    int
        The nearest odd integer, at least ``1``.
    """
    w = int(round(v))
    if w < 1:
        w = 1
    if w % 2 == 0:
        w += 1
    return w


def smoothing_grid(n: int, config: RobustKneeConfig) -> List[int]:
    """Odd smoothing windows for ``n`` samples, sorted ascending, deduplicated.

    Parameters
    ----------
    n : int
        Number of samples in the curve.
    config : RobustKneeConfig
        ``smoothing_fractions`` are multiplied by ``n`` to size the windows.

    Returns
    -------
    list of int
        Distinct odd window widths, always starting with ``1``.
    """
    windows = {1}
    for frac in config.smoothing_fractions:
        w = _nearest_odd(max(1.0, frac * n))
        # Keep windows small enough to leave a curve to analyse.
        if w <= max(3, n // 2):
            windows.add(w)
    return sorted(windows)


def _gaussian_kernel(window: int) -> np.ndarray:
    """A normalized Gaussian kernel whose support approximates ``window``.

    Parameters
    ----------
    window : int
        Approximate support width, in samples.

    Returns
    -------
    numpy.ndarray
        A 1-D kernel summing to ``1.0``.
    """
    # Treat the window as ~ +/- 2 sigma of support.
    sigma = max(window / 4.0, 0.5)
    radius = max(int(window // 2), 1)
    t = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(t * t) / (2.0 * sigma * sigma))
    return k / k.sum()


def smooth_curve(y: np.ndarray, window: int, method: str = "gaussian") -> np.ndarray:
    """Smooth ``y`` with a centered kernel and reflected boundaries.

    Parameters
    ----------
    y : numpy.ndarray
        The signal to smooth.
    window : int
        Approximate support width; ``window <= 1`` returns ``y`` unchanged.
    method : str, optional
        ``"gaussian"`` (default) or ``"moving_average"`` (a baseline).

    Returns
    -------
    numpy.ndarray
        The smoothed signal, same length as ``y``.
    """
    if window <= 1 or y.size < 3:
        return np.asarray(y, dtype=float).copy()

    if method == "moving_average":
        # Force an odd kernel size so the convolution below is centered on
        # each sample. An even-sized kernel breaks the "radius = size // 2,
        # padded length = n + 2*radius, valid convolution yields n samples"
        # arithmetic that makes the reflect-pad + valid-convolve trick below
        # centered: with an even kernel, valid convolution over the padded
        # signal actually yields n+1 samples, and slicing back down to n
        # silently keeps the first n instead of the centered n, shifting the
        # whole smoothed curve by half a sample (found by testing: a linear
        # ramp's moving average, which should reproduce the ramp exactly at
        # every interior point, came out shifted by +0.5 for even windows).
        size = min(window, y.size)
        if size % 2 == 0:
            size = max(size - 1, 1)
        kernel = np.ones(size) / float(size)
    else:
        kernel = _gaussian_kernel(window)

    radius = kernel.size // 2
    # Reflect (mode="reflect") the edges to avoid boundary bias.
    padded = np.pad(y, radius, mode="reflect")
    smoothed = np.convolve(padded, kernel, mode="valid")
    # np.convolve 'valid' over a length n+2*radius signal with a (2r+1) kernel
    # yields exactly n samples.
    return smoothed[: y.size]
