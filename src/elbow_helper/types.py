"""Result and intermediate data types for the robust knee detector.

The public contract is a tagged union: :func:`elbow_helper.robust_knee` always
returns either a :class:`ClearKnee` or a :class:`NoClearKnee`, both subclasses
of :class:`KneeResult`, so callers must handle abstention explicitly. Every
:class:`NoClearKnee` carries a machine-readable :class:`Reason` code plus a
``diagnostics`` dict; every :class:`ClearKnee` carries the full evidence chain.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
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
    NO_KNEE_CANDIDATES = "NO_KNEE_CANDIDATES"
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
    KNEES_FOUND = "KNEES_FOUND"


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
        """Map a normalized x back to original data units.

        Parameters
        ----------
        x_norm : float
            A location in ``[0, 1]``.

        Returns
        -------
        float
            The corresponding location in the caller's original data units.
        """
        return float(self.x_lo + x_norm * (self.x_hi - self.x_lo))


@dataclass
class KneeCandidate:
    """A single Kneedle hit at one (smoothing window, sensitivity) setting.

    Attributes
    ----------
    knee_x_norm : float
        Candidate location, normalized ``x``.
    knee_index : int
        Index of the candidate on the (smoothed) curve.
    window : int
        Smoothing window that produced this candidate.
    sensitivity : float
        Kneedle sensitivity ``S`` that produced this candidate.
    prominence : float
        Topographic prominence of the peak (see
        :func:`~elbow_helper.numerics.peak_prominence`).
    local_noise : float
        Robust noise estimate of the (unsmoothed) curve.
    noise_prominence_ratio : float
        ``prominence / local_noise``.
    boundary_distance : float
        Distance from the candidate to the nearer curve endpoint.
    rejected : str, optional
        A :class:`Reason` code if this candidate failed a basic filter,
        ``None`` if it passed.
    """

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
    """A group of candidates at nearby knee locations across the scale space.

    Attributes
    ----------
    median_knee : float
        Median normalized location of the cluster's members.
    mad : float
        Median absolute deviation of the members' locations.
    members : list of KneeCandidate
        The candidates in this cluster.
    n_windows : int
        Number of distinct smoothing windows represented.
    consecutive_scales : int
        Longest run of consecutive smoothing scales the cluster spans.
    sensitivity_support : float
        Fraction of distinct sensitivities represented.
    neighbor_shift : float
        Largest jump in per-window median location between adjacent windows.
    support : int
        Number of member candidates.
    support_frac : float
        ``support`` divided by the total candidate count.
    median_prominence : float
        Median prominence across members.
    median_noise_prominence_ratio : float
        Median noise-prominence ratio across members.
    persistent : bool
        Whether this cluster cleared the persistence gates.
    stable_window : int, optional
        Smallest smoothing window meeting the sensitivity-support threshold,
        set only when ``persistent`` is ``True``.
    """

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
    """Slope-contrast and broken-line vs single-line model comparison.

    Attributes
    ----------
    passes : bool
        Whether the slope, CV and BIC checks all passed.
    slope_contrast : float
        Normalized contrast between the left and right Theil-Sen slopes.
    m_left, m_right : float
        The robust slopes on either side of the candidate knee.
    bic_improvement : float
        BIC of the single line minus BIC of the broken line (higher favors
        the broken line).
    cv_improvement : float
        Fractional reduction in blocked cross-validated SSE from using the
        broken line instead of the single line.
    reason : str, optional
        A :class:`Reason` code if a check failed, ``None`` otherwise.
    """

    passes: bool
    slope_contrast: float
    m_left: float
    m_right: float
    bic_improvement: float
    cv_improvement: float
    reason: Optional[str] = None


@dataclass
class BootstrapEvidence:
    """Stability of the detected knee across residual-bootstrap replicates.

    Attributes
    ----------
    passes : bool
        Whether the detection rate, CI width and cluster-rate checks passed.
    detection_rate : float
        Fraction of bootstrap replicates in which a knee was redetected.
    ci90 : tuple of float
        The 5th-95th percentile interval of redetected locations, data units.
    ci90_width : float
        Width of ``ci90``, in normalized ``x`` units.
    primary_cluster_rate : float
        Fraction of redetections in the dominant location cluster.
    secondary_cluster_rate : float
        Fraction of redetections in the next-largest location cluster.
    median_shift : float
        Shift between the observed knee and the median redetected location.
    knees : list of float
        The redetected knee locations, one per successful replicate.
    reason : str, optional
        A :class:`Reason` code if a check failed, ``None`` otherwise.
    """

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
    """Search-adjusted Monte-Carlo test against a no-knee null model.

    Attributes
    ----------
    passes : bool
        Whether ``p_value`` cleared ``config.max_null_p_value``.
    p_value : float
        The Monte-Carlo p-value, finite-sample corrected.
    observed_statistic : tuple
        The observed test statistic, as computed on the real data.
    null_replicates : int
        Number of null replicates the p-value was computed from.
    reason : str, optional
        A :class:`Reason` code if the check failed, ``None`` otherwise.
    """

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
    """A knee accepted by every stage of the pipeline, with uncertainty.

    Attributes
    ----------
    knee_x : float
        The detected knee, in the caller's original data units.
    knee_x_norm : float
        The detected knee, normalized to ``[0, 1]``.
    knee_index : int
        Index of the knee on the cleaned, sorted curve.
    ci90 : tuple of float
        90% bootstrap interval for the knee location, data units.
    detection_rate : float
        Fraction of bootstrap replicates in which the knee was redetected.
    smoothing_window, sensitivity : int, float
        The scale-space setting that produced the winning candidate.
    prominence : float
        Topographic prominence of the winning candidate's peak.
    slope_contrast : float
        Normalized Theil-Sen slope contrast across the knee.
    bic_improvement : float
        BIC improvement of the broken line over a single line.
    null_p_value : float
        Monte-Carlo p-value from the no-knee null test.
    """

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


@dataclass
class KneeEstimate:
    """One accepted breakpoint from :func:`elbow_helper.robust_knees`.

    Segments are independent (discontinuous) OLS lines, not the continuous
    broken-line model :class:`ClearKnee` uses; ``slope_left``/``slope_right``
    are each segment's own fitted slope, in data units and need not agree at
    the breakpoint.

    Attributes
    ----------
    x : float
        The breakpoint, in the caller's original data units.
    x_norm : float
        The breakpoint, normalized to ``[0, 1]``.
    index : int
        Index of the breakpoint on the cleaned, sorted curve.
    slope_left, slope_right : float
        Each neighbouring segment's own fitted slope, data units.
    fwer_p_value : float, optional
        The sequential FWER test's p-value for this breakpoint, ``None``
        when the FWER gate was not run (``require_fwer_confirmation=False``).
    """

    x: float
    x_norm: float
    index: int
    slope_left: float
    slope_right: float
    fwer_p_value: Optional[float] = None

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"KneeEstimate(x={self.x:.4g}, "
            f"slope_left={self.slope_left:.4g}, slope_right={self.slope_right:.4g})"
        )


@dataclass
class MultiKneeResult:
    """Base class for the tagged union returned by :func:`elbow_helper.robust_knees`."""

    reason: str
    diagnostics: Dict = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """``True`` for :class:`Knees`, ``False`` for :class:`InvalidKnees`."""
        return isinstance(self, Knees)


@dataclass
class Knees(MultiKneeResult):
    """A valid multi-knee result: zero or more accepted breakpoints.

    Unlike :class:`NoClearKnee`, an empty ``knees`` list here is not an
    abstention: it is the pipeline's confident conclusion that the data has
    no real breakpoint, having survived the same search and false-positive
    gates a nonempty result would have to survive.
    """

    knees: List[KneeEstimate] = field(default_factory=list)

    @property
    def k(self) -> int:
        """Number of accepted breakpoints."""
        return len(self.knees)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Knees(k={self.k}, x=[{', '.join(f'{e.x:.4g}' for e in self.knees)}])"


@dataclass
class InvalidKnees(MultiKneeResult):
    """Preprocessing failed: the input could not be searched at all."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"InvalidKnees(reason={self.reason!r})"
