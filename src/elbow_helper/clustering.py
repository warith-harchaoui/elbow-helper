"""Phase 5 — persistence clustering, uniqueness, and scale selection.

Groups candidate knees that recur at nearby locations across the smoothing
scales and sensitivities, measures how *persistent* each group is, rejects the
ambiguous multi-knee case, and picks the smallest stable smoothing scale of the
winning cluster.

Author
------
`Warith Harchaoui, Ph.D. <https://www.linkedin.com/in/warith-harchaoui/>`_
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .config import RobustKneeConfig
from .smoothing import smoothing_grid
from .types import CandidateCluster, KneeCandidate, PreparedCurve, Reason


def _longest_consecutive_run(ranks: List[int]) -> int:
    """Longest run of consecutive integers present in ``ranks``."""
    if not ranks:
        return 0
    s = sorted(set(ranks))
    best = run = 1
    for a, b in zip(s, s[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def cluster_candidates(
    candidates: List[KneeCandidate],
    prepared: PreparedCurve,
    config: RobustKneeConfig,
) -> List[CandidateCluster]:
    """Cluster candidates by location and score each cluster's persistence.

    Parameters
    ----------
    candidates : list of KneeCandidate
        Candidates that already passed the basic filters.
    prepared : PreparedCurve
        The curve (for the total candidate count and window grid).
    config : RobustKneeConfig
        ``cluster_tolerance`` and the persistence thresholds.

    Returns
    -------
    list of CandidateCluster
        Clusters sorted by support (descending), each flagged ``persistent``.
    """
    if not candidates:
        return []

    windows = smoothing_grid(prepared.n, config)
    window_rank = {w: i for i, w in enumerate(windows)}
    n_sensitivities = len({c.sensitivity for c in candidates}) or 1

    ordered = sorted(candidates, key=lambda c: c.knee_x_norm)
    clusters_members: List[List[KneeCandidate]] = []
    current: List[KneeCandidate] = [ordered[0]]
    center = ordered[0].knee_x_norm
    for cand in ordered[1:]:
        if abs(cand.knee_x_norm - center) <= config.cluster_tolerance:
            current.append(cand)
            center = float(np.median([m.knee_x_norm for m in current]))
        else:
            clusters_members.append(current)
            current = [cand]
            center = cand.knee_x_norm
    clusters_members.append(current)

    total = len(candidates)
    all_sensitivities = float(len(_sensitivity_values(config, prepared.n)))
    clusters: List[CandidateCluster] = []
    for members in clusters_members:
        locs = np.array([m.knee_x_norm for m in members])
        median_knee = float(np.median(locs))
        mad = float(np.median(np.abs(locs - median_knee)))

        member_windows = [m.window for m in members]
        ranks = [window_rank[w] for w in member_windows if w in window_rank]
        consecutive = _longest_consecutive_run(ranks)

        distinct_sens = {m.sensitivity for m in members}
        sens_support = len(distinct_sens) / (all_sensitivities or 1.0)

        neighbor_shift = _neighbor_shift(members, window_rank)

        cluster = CandidateCluster(
            median_knee=median_knee,
            mad=mad,
            members=members,
            n_windows=len(set(member_windows)),
            consecutive_scales=consecutive,
            sensitivity_support=sens_support,
            neighbor_shift=neighbor_shift,
            support=len(members),
            support_frac=len(members) / total,
            median_prominence=float(np.median([m.prominence for m in members])),
            median_noise_prominence_ratio=float(
                np.median([m.noise_prominence_ratio for m in members])
            ),
        )
        cluster.persistent = _is_persistent(cluster, config)
        if cluster.persistent:
            cluster.stable_window = _smallest_stable_window(cluster, windows, config)
        clusters.append(cluster)

    clusters.sort(key=lambda c: c.support, reverse=True)
    return clusters


def _sensitivity_values(config: RobustKneeConfig, n: int) -> List[float]:
    """The sensitivity grid (kept in sync with :mod:`candidates`)."""
    values = set()
    for frac in config.sensitivity_fractions:
        values.add(float(max(1, round(frac * n))))
    return sorted(values)


def _neighbor_shift(members: List[KneeCandidate], window_rank: dict) -> float:
    """Max shift in per-window median location between adjacent windows."""
    by_window: dict = {}
    for m in members:
        by_window.setdefault(m.window, []).append(m.knee_x_norm)
    if len(by_window) < 2:
        return 0.0
    ordered = sorted(by_window.items(), key=lambda kv: window_rank.get(kv[0], 0))
    medians = [float(np.median(v)) for _, v in ordered]
    return float(np.max(np.abs(np.diff(medians))))


def _is_persistent(cluster: CandidateCluster, config: RobustKneeConfig) -> bool:
    """Apply the four persistence gates from the plan."""
    return (
        cluster.consecutive_scales >= config.min_consecutive_scales
        and cluster.sensitivity_support >= config.min_sensitivity_support
        and cluster.mad <= config.max_cluster_mad
        and cluster.neighbor_shift <= config.max_neighbor_shift
    )


def _smallest_stable_window(
    cluster: CandidateCluster, windows: List[int], config: RobustKneeConfig
) -> int:
    """Smallest window in the cluster's stable run with enough sensitivity support."""
    all_sens = len({m.sensitivity for m in cluster.members}) or 1
    by_window: dict = {}
    for m in cluster.members:
        by_window.setdefault(m.window, set()).add(m.sensitivity)
    for w in windows:  # ascending
        if w in by_window and len(by_window[w]) / all_sens >= config.min_sensitivity_support:
            return w
    # Fall back to the smallest window that appears at all.
    present = sorted(by_window)
    return present[0] if present else windows[0]


def select_unique_cluster(
    clusters: List[CandidateCluster], config: RobustKneeConfig
) -> Tuple[Optional[CandidateCluster], Optional[str]]:
    """Pick the single dominant persistent cluster, or explain the abstention.

    Parameters
    ----------
    clusters : list of CandidateCluster
        All clusters (sorted by support).
    config : RobustKneeConfig
        ``secondary_support_frac`` and ``min_dominance_ratio``.

    Returns
    -------
    tuple
        ``(cluster, None)`` on success, or ``(None, reason_code)`` when there is
        no persistent cluster or the winner is not clearly dominant.
    """
    persistent = [c for c in clusters if c.persistent]
    if not persistent:
        return None, Reason.NO_PERSISTENT_CLUSTER

    persistent.sort(key=lambda c: c.support, reverse=True)
    primary = persistent[0]
    if len(persistent) >= 2:
        second = persistent[1]
        both_strong = second.support_frac > config.secondary_support_frac
        not_dominant = primary.support < config.min_dominance_ratio * second.support
        well_separated = abs(primary.median_knee - second.median_knee) > 2 * config.cluster_tolerance
        if well_separated and (both_strong or not_dominant):
            return None, Reason.MULTIPLE_PLAUSIBLE_KNEES

    return primary, None
