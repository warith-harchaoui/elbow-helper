"""elbow-helper — noise-robust knee/elbow detection.

A conservative wrapper around a from-scratch, NumPy-only difference-curve
locator. Point-estimate knee detectors propose *where* plausible
knees are; this package decides whether any candidate is strong, unique,
persistent, reproducible and unlikely under a no-knee model and otherwise
abstains explicitly.

Public API
----------
``robust_knee(x, y=None, curve=None, direction=None, config=None)`` ->
:class:`ClearKnee` | :class:`NoClearKnee`. ``y`` may be omitted (``x`` is then
the y-values alone, against an implicit ``0, 1, ..., n-1``); ``curve`` and
``direction`` are inferred from the data when omitted. ``robust_elbow(x,
y=None, config=None)`` is the convex-decreasing convenience.
:class:`RobustKneeConfig` holds every threshold. :class:`KneeLocator` is the
from-scratch locator, usable standalone.

``robust_knees(x, y=None, config=None)`` (plural) -> :class:`Knees` |
:class:`InvalidKnees`: how many knees, if any, a curve genuinely has, via
dynamic-program search and a Bonferroni-gated modified-BIC criterion (see
``ELBOW-en.tex`` and ``research/multiknee/RESULTS.md``). :class:`RobustKneesConfig`
holds its settings.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from .config import RobustKneeConfig, RobustKneesConfig
from .locator import KneeLocator
from .multi_pipeline import robust_knees
from .pipeline import robust_elbow, robust_knee
from .types import (
    ClearKnee,
    InvalidKnees,
    KneeEstimate,
    KneeResult,
    Knees,
    MultiKneeResult,
    NoClearKnee,
    Reason,
)

__version__ = "0.1.4"

__all__ = [
    "robust_knee",
    "robust_elbow",
    "robust_knees",
    "RobustKneeConfig",
    "RobustKneesConfig",
    "KneeLocator",
    "KneeResult",
    "ClearKnee",
    "NoClearKnee",
    "MultiKneeResult",
    "Knees",
    "InvalidKnees",
    "KneeEstimate",
    "Reason",
    "__version__",
]
