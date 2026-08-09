"""Shared fixtures and synthetic-curve generators for the test suite.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np
import pytest

from elbow_helper import RobustKneeConfig


@pytest.fixture
def fast_config():
    """A seeded config with modest replicate counts for fast, reliable tests.

    ``null_replicates=150`` keeps the smallest attainable p-value
    (``1/151 ~ 0.0066``) below ``max_null_p_value=0.01`` so a genuinely clear
    knee can actually be accepted.
    """
    return RobustKneeConfig(
        random_seed=0,
        bootstrap_replicates=80,
        null_replicates=150,
    )


def clear_knee_curve(seed=0, knee_frac=0.3, n=80, noise=0.02):
    """Concave-increasing curve with a sharp knee (steep then flat)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    kx = knee_frac
    steep, flat = 3.0, 0.2
    y = np.where(x <= kx, steep * x, steep * kx + flat * (x - kx))
    y = y / y.max()
    return x, y + rng.normal(0, noise, n)


def elbow_curve(seed=0, knee_frac=0.25, n=60, noise=0.02):
    """Convex-decreasing curve with an elbow (fast drop then flat)."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    kx = knee_frac
    y = np.where(x <= kx, 1.0 - 3.0 * x, 1.0 - 3.0 * kx - 0.15 * (x - kx))
    y = (y - y.min()) / (y.max() - y.min())
    return x, y + rng.normal(0, noise, n)


def noisy_line(seed=0, n=80, noise=0.01):
    """A monotone straight line with small noise: high correlation, no knee."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = 0.2 + 0.5 * x
    return x, y + rng.normal(0, noise, n)


def two_knee_curve(seed=0, n=90, noise=0.015):
    """Two comparable knees: the ambiguous case."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.piecewise(
        x,
        [x <= 0.33, (x > 0.33) & (x <= 0.66), x > 0.66],
        [
            lambda t: 2.5 * t,
            lambda t: 2.5 * 0.33 + 0.3 * (t - 0.33),
            lambda t: 2.5 * 0.33 + 0.3 * 0.33 + 2.5 * (t - 0.66),
        ],
    )
    y = (y - y.min()) / (y.max() - y.min())
    return x, y + rng.normal(0, noise, n)
