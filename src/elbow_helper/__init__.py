"""elbow-helper — noise-robust knee/elbow detection.

A conservative wrapper around a from-scratch, NumPy-only port of the Kneedle
algorithm. ``kneed`` proposes *where* plausible knees are; this package decides
whether any candidate is strong, unique, persistent, reproducible and unlikely
under a no-knee model — and otherwise abstains explicitly.

Public API
----------
``robust_knee(x, y, curve, direction, config)`` -> :class:`ClearKnee` |
:class:`NoClearKnee`. ``robust_elbow(x, y, config)`` is the convex-decreasing
convenience. :class:`RobustKneeConfig` holds every threshold. :class:`KneeLocator`
is the from-scratch Kneedle, usable standalone.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

from .config import RobustKneeConfig
from .kneedle import KneeLocator
from .pipeline import robust_elbow, robust_knee
from .types import (
    ClearKnee,
    KneeResult,
    NoClearKnee,
    Reason,
)

__version__ = "0.1.0"

__all__ = [
    "robust_knee",
    "robust_elbow",
    "RobustKneeConfig",
    "KneeLocator",
    "KneeResult",
    "ClearKnee",
    "NoClearKnee",
    "Reason",
    "__version__",
]
