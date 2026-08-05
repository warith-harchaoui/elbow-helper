"""Result and intermediate data types for the robust knee detector.

The public contract is a tagged union: :func:`elbow_helper.robust_knee` always
returns either a :class:`ClearKnee` or a :class:`NoClearKnee`, both subclasses
of :class:`KneeResult`, so callers must handle abstention explicitly. Every
:class:`NoClearKnee` carries a machine-readable :class:`Reason` code plus a
``diagnostics`` dict; every :class:`ClearKnee` carries the full evidence chain.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


class Reason:
    """Stable, machine-readable abstention (and status) reason codes."""

    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    ZERO_RANGE = "ZERO_RANGE"
    INCOMPATIBLE_GLOBAL_SHAPE = "INCOMPATIBLE_GLOBAL_SHAPE"
    NO_KNEED_CANDIDATES = "NO_KNEED_CANDIDATES"
    ALL_CANDIDATES_WEAK = "ALL_CANDIDATES_WEAK"
    NO_PERSISTENT_CLUSTER = "NO_PERSISTENT_CLUSTER"
    MULTIPLE_PLAUSIBLE_KNEES = "MULTIPLE_PLAUSIBLE_KNEES"
    BOUNDARY_KNEE = "BOUNDARY_KNEE"
    WEAK_SLOPE_CHANGE = "WEAK_SLOPE_CHANGE"
    SEGMENTED_MODEL_NOT_BETTER = "SEGMENTED_MODEL_NOT_BETTER"
    BOOTSTRAP_UNSTABLE = "BOOTSTRAP_UNSTABLE"
    BOOTSTRAP_MULTIMODAL = "BOOTSTRAP_MULTIMODAL"
    NULL_NOT_REJECTED = "NULL_NOT_REJECTED"
    INTERNAL_NUMERICAL_FAILURE = "INTERNAL_NUMERICAL_FAILURE"
    CLEAR_KNEE = "CLEAR_KNEE"


@dataclass
class PreparedCurve:
    """A cleaned, sorted, normalized curve plus inverse-transform metadata.

    Attributes
    ----------
    x_norm, y_scaled : numpy.ndarray
        The curve on the unit square: ``x`` linearly scaled to ``[0, 1]`` and
        ``y`` robustly scaled (5th/95th percentile) and clipped to ``[0, 1]``.
    n : int
        Number of retained samples.
    x_lo, x_hi : float
        Original x-range, used to map a normalized knee back to data units.
    y_lo, y_hi : float
        Robust y-limits used for scaling (inverse transform for y).
    curve, direction : str
        The caller-supplied Kneedle orientation.
    spearman : float
        Spearman rank correlation between x and y.
    violation_rate : float
        Fraction of lightly-smoothed first differences that move against
        ``direction``.
    """

    x_norm: np.ndarray
    y_scaled: np.ndarray
    n: int
    x_lo: float
    x_hi: float
    y_lo: float
    y_hi: float
    curve: str
    direction: str
    spearman: float
    violation_rate: float

    def denormalize_x(self, x_norm: float) -> float:
        """Map a normalized x back to original data units."""
        return float(self.x_lo + x_norm * (self.x_hi - self.x_lo))


@dataclass
class KneeCandidate:
    """A single Kneedle hit at one (smoothing window, sensitivity) setting."""

    knee_x_norm: float
    knee_index: int
    window: int
    sensitivity: float
    prominence: float = 0.0
    local_noise: float = 0.0
    noise_prominence_ratio: float = 0.0
    boundary_distance: float = 0.0
    rejected: Optional[str] = None


@dataclass
class CandidateCluster:
    """A group of candidates at nearby knee locations across the scale space."""

    median_knee: float
    mad: float
    members: List[KneeCandidate]
    n_windows: int
    consecutive_scales: int
    sensitivity_support: float
    neighbor_shift: float
    support: int
    support_frac: float
    median_prominence: float
    median_noise_prominence_ratio: float
    persistent: bool = False
    stable_window: Optional[int] = None


@dataclass
class SegmentEvidence:
    """Slope-contrast and broken-line vs single-line model comparison."""

    passes: bool
    slope_contrast: float
    m_left: float
    m_right: float
    bic_improvement: float
    cv_improvement: float
    reason: Optional[str] = None


@dataclass
class BootstrapEvidence:
    """Stability of the detected knee across residual-bootstrap replicates."""

    passes: bool
    detection_rate: float
    ci90: Tuple[float, float]
    ci90_width: float
    primary_cluster_rate: float
    secondary_cluster_rate: float
    median_shift: float
    knees: List[float]
    reason: Optional[str] = None


@dataclass
class NullEvidence:
    """Search-adjusted Monte-Carlo test against a no-knee null model."""

    passes: bool
    p_value: float
    observed_statistic: tuple
    null_replicates: int
    reason: Optional[str] = None


@dataclass
class KneeResult:
    """Base class for the tagged union returned by :func:`robust_knee`."""

    reason: str
    diagnostics: Dict = field(default_factory=dict)

    @property
    def is_clear(self) -> bool:
        """``True`` for :class:`ClearKnee`, ``False`` for :class:`NoClearKnee`."""
        return isinstance(self, ClearKnee)


@dataclass
class ClearKnee(KneeResult):
    """A knee accepted by every stage of the pipeline, with uncertainty."""

    knee_x: float = 0.0
    knee_x_norm: float = 0.0
    knee_index: int = 0
    ci90: Tuple[float, float] = (0.0, 0.0)
    detection_rate: float = 0.0
    smoothing_window: int = 1
    sensitivity: float = 1.0
    prominence: float = 0.0
    slope_contrast: float = 0.0
    bic_improvement: float = 0.0
    null_p_value: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lo, hi = self.ci90
        return (
            f"ClearKnee(knee_x={self.knee_x:.4g}, "
            f"ci90=({lo:.4g}, {hi:.4g}), "
            f"detection_rate={self.detection_rate:.2f}, "
            f"null_p={self.null_p_value:.3g})"
        )


@dataclass
class NoClearKnee(KneeResult):
    """An explicit abstention: no knee is strong enough to report."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"NoClearKnee(reason={self.reason!r})"
