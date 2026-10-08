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

    # --- y normalization ---
    # Width of the median despiker applied to y before min/max scaling. A
    # median filter is the identity on locally monotone data, so it removes
    # isolated outliers without touching the head or tail of a genuinely
    # steep curve; set to 1 to disable it.
    despike_window: int = 3

    # --- positional resolution ---
    # No precision requirement can be meaningfully finer than the x sampling
    # grid: a knee located on n samples is only ever known to within
    # 1 / (n - 1) of the normalized range. The gates that demand positional
    # precision -- cluster MAD, neighbor shift, bootstrap median shift and CI
    # width -- are floored at this many sample spacings. ``cluster_tolerance``
    # deliberately is *not*: it decides which hits count as the same knee, so
    # widening it changes which knee gets reported, not how precisely.
    # See :meth:`RobustKneeConfig.positional`.
    positional_floor_samples: float = 2.0

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

    # How many smoothing scales beyond the finest one a cluster appears at are
    # read for its *location*. Persistence is judged across the whole scale
    # space, but a smoothed corner's difference-curve peak slides toward the
    # shallower side, so the coarse scales sit systematically late; see
    # :func:`~elbow_helper.clustering._fine_scale_location`. ``0`` reads the
    # finest scale alone, which is the only one carrying no displacement at
    # all: on curves whose breakpoint is unambiguous it is accurate to 0.18
    # samples on average, against 0.50 for the two finest scales pooled,
    # because pooling two scales splits the difference with a drift rather
    # than averaging away a jitter. Jitter is still averaged, over the
    # several sensitivities the cluster has at that one scale.
    fine_scale_span: int = 0

    # --- global shape compatibility ---
    min_spearman_abs: float = 0.60
    max_direction_violation_rate: float = 0.25

    # --- candidate basic filters ---
    # A knee is rejected when it sits within ``boundary_margin`` of either end
    # of the normalized range, or within ``boundary_min_samples`` samples of
    # it, whichever band is wider. Two parameters rather than one because the
    # two reasons an end knee is untrustworthy scale differently: too few
    # points on one side to read a slope from is a count, and it does not get
    # better on a longer curve; a feature indistinguishable from the curve
    # simply starting steeply is a fraction of the range. Expressing the
    # count as a fraction, as a single ``boundary_margin`` had to, overstates
    # it badly once the curve is long -- at n=200 a 0.10 margin discards the
    # first 20 samples, and a rank-10 factor structure lives there. See
    # :func:`~elbow_helper.metrics.passes_basic_filters`.
    #
    # 0.03 is calibrated, not guessed. Swept against 160 no-knee control
    # curves (straight, flat, flat plus noise, pure noise, line plus noise at
    # four lengths and eight seeds each) and the detection suites:
    #
    #   margin   plateau  factor-rank   false positives on controls
    #   0.10        4/5       2/4            1/160
    #   0.06        5/5       3/4            1/160
    #   0.04        5/5       4/4            1/160
    #   0.03        5/5       4/4            1/160
    #   0.01        5/5       4/4            1/160
    #
    # The false-positive count does not move anywhere in that range, so the
    # margin was not buying conservatism; detection saturates at 0.04, and
    # 0.03 takes the full gain one step inside the plateau rather than
    # perched on its edge.
    boundary_margin: float = 0.03
    boundary_min_samples: int = 5
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
    # Width of the smoother that supplies the bootstrap's signal. Small on
    # purpose: it has to average noise away without rounding off the knee the
    # replicates are meant to re-find.
    bootstrap_signal_window: int = 5
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

    def positional(
        self, threshold: float, n: int, samples: Optional[float] = None
    ) -> float:
        """Widen a positional threshold to at least the x sampling resolution.

        Thresholds on normalized ``x`` are written as curve fractions, but the
        quantity they bound is quantized at the sample spacing
        ``h = 1 / (n - 1)``: on a 49-point curve a single sample is already
        0.021 of the range, so a 0.03 tolerance leaves no room for the
        one-to-two-sample jitter the locator inevitably has. This returns the
        threshold widened to ``samples * h`` whenever the grid is coarser than
        the constant, and leaves it untouched on finely sampled curves.

        Parameters
        ----------
        threshold : float
            The configured tolerance, in normalized ``x`` units.
        n : int
            Number of samples on the curve.
        samples : float, optional
            Number of sample spacings to floor at. Defaults to
            ``positional_floor_samples``.

        Returns
        -------
        float
            ``max(threshold, samples / (n - 1))``.
        """
        if samples is None:
            samples = self.positional_floor_samples
        return max(float(threshold), float(samples) / max(int(n) - 1, 1))

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
