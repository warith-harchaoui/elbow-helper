"""Configuration for the robust knee detector.

All positional thresholds are expressed in **normalized x-range units** (the
x-axis is scaled to ``[0, 1]`` during preprocessing), so they are independent
of the absolute scale of the caller's data.

The defaults follow the "first practical prototype" scope: modest replicate
counts so the full pipeline runs in seconds. For validation-grade runs raise
``bootstrap_replicates`` to 500 and ``null_replicates`` to 1000.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class RobustKneeConfig:
    """Immutable bundle of thresholds and search settings.

    Use :meth:`with_` to derive a tweaked copy (the dataclass is frozen).
    """

    # --- data adequacy ---
    min_samples: int = 20

    # --- scale-space search ---
    smoothing_fractions: Tuple[float, ...] = (
        0.0,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
        0.18,
        0.25,
    )
    sensitivity_fractions: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)

    # --- global shape compatibility ---
    min_spearman_abs: float = 0.60
    max_direction_violation_rate: float = 0.25

    # --- candidate basic filters ---
    boundary_margin: float = 0.10
    min_side_points: int = 5
    min_prominence: float = 0.05
    min_noise_prominence_ratio: float = 4.0

    # --- persistence clustering ---
    # cluster_tolerance and max_neighbor_shift are calibrated (§19) slightly
    # above the plan's 0.05 to absorb the one-to-two-sample locator
    # discretization jitter seen at modest sample sizes (n ~ 60-100).
    cluster_tolerance: float = 0.06
    min_consecutive_scales: int = 3
    min_sensitivity_support: float = 0.70
    max_cluster_mad: float = 0.03
    max_neighbor_shift: float = 0.07

    # --- uniqueness ---
    secondary_support_frac: float = 0.30
    min_dominance_ratio: float = 2.0

    # --- slope / model confirmation ---
    slope_left_window: Tuple[float, float] = (0.15, 0.03)  # (far, near) offsets
    slope_right_window: Tuple[float, float] = (0.03, 0.15)  # (near, far) offsets
    min_slope_contrast: float = 0.30
    min_cv_improvement: float = 0.10
    min_bic_improvement: float = 10.0
    cv_folds: int = 5

    # --- bootstrap robustness ---
    bootstrap_replicates: int = 100
    min_bootstrap_detection_rate: float = 0.90
    max_ci90_width: float = 0.10
    min_primary_cluster_rate: float = 0.80
    max_secondary_cluster_rate: float = 0.15
    max_bootstrap_median_shift: float = 0.03

    # --- no-knee null test ---
    null_replicates: int = 200
    max_null_p_value: float = 0.01

    # --- reproducibility ---
    random_seed: Optional[int] = None

    def with_(self, **changes) -> "RobustKneeConfig":
        """Return a copy of this config with ``changes`` applied.

        Parameters
        ----------
        **changes
            Field overrides, e.g. ``config.with_(bootstrap_replicates=500)``.

        Returns
        -------
        RobustKneeConfig
            A new, independent configuration.
        """
        return replace(self, **changes)


@dataclass(frozen=True)
class RobustKneesConfig:
    """Immutable settings for :func:`elbow_helper.robust_knees` (plural).

    The multi-knee search ships the combination validated in
    ``research/multiknee/RESULTS.md``: dynamic-program search, the
    subtractive-sign modified BIC as the selection criterion, and a
    Bonferroni-gated sequential permutation test layered on top by default,
    matching this package's design priority of minimising false-positive
    knees. Use :meth:`with_` to derive a tweaked copy.
    """

    # --- data adequacy ---
    min_samples: int = 20

    # --- search ---
    k_max: int = 4
    min_seg_fraction: float = 0.08

    # --- false-positive control ---
    fwer_alpha: float = 0.05
    fwer_permutations: int = 200
    require_fwer_confirmation: bool = True

    # --- reproducibility ---
    random_seed: Optional[int] = None

    def with_(self, **changes) -> "RobustKneesConfig":
        """Return a copy of this config with ``changes`` applied."""
        return replace(self, **changes)
