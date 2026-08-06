"""The shared inner search: candidates -> filter -> cluster -> confirm.

Factored out so the main pipeline, the bootstrap replicates, and the null-test
replicates all run *the same* detection logic. The search-adjusted test
statistic (used by the null test) is a lexicographic tuple, in decreasing
priority: passed model confirmation, cluster dominance, prominence-to-noise.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .candidates import generate_candidates
from .clustering import cluster_candidates, select_unique_cluster
from .config import RobustKneeConfig
from .metrics import passes_basic_filters
from .segmented import confirm_segmented_model
from .types import CandidateCluster, PreparedCurve, Reason, SegmentEvidence


@dataclass
class SearchResult:
    """Outcome of one inner detection pass over a (prepared) curve."""

    detected: bool
    reason: Optional[str] = None
    knee_x_norm: Optional[float] = None
    window: Optional[int] = None
    cluster: Optional[CandidateCluster] = None
    segment: Optional[SegmentEvidence] = None
    statistic: Tuple[int, float, float] = (0, 0.0, 0.0)
    n_candidates: int = 0
    n_filtered: int = 0


def run_search(
    prepared: PreparedCurve, config: RobustKneeConfig, confirm: bool = True
) -> SearchResult:
    """Run the full inner detection pass (no bootstrap, no null test).

    Parameters
    ----------
    prepared : PreparedCurve
        The normalized curve.
    config : RobustKneeConfig
        Search thresholds.
    confirm : bool, optional
        Whether to run the segmented-model confirmation (Phase 6). The main
        pipeline and the null test set this ``True``; the bootstrap sets it
        ``False`` for speed (it re-tests stability, not model fit).

    Returns
    -------
    SearchResult
        ``detected`` with the winning cluster and statistic or an abstention
        with a reason code.
    """
    candidates = generate_candidates(prepared, config)
    if not candidates:
        return SearchResult(False, reason=Reason.NO_KNEE_CANDIDATES)

    filtered = [c for c in candidates if passes_basic_filters(c, prepared.n, config)]
    if not filtered:
        return SearchResult(
            False, reason=Reason.ALL_CANDIDATES_WEAK, n_candidates=len(candidates)
        )

    clusters = cluster_candidates(filtered, prepared, config)
    selected, reason = select_unique_cluster(clusters, config)
    if selected is None:
        return SearchResult(
            False, reason=reason, n_candidates=len(candidates), n_filtered=len(filtered)
        )

    knee_x_norm = selected.median_knee
    window = selected.stable_window

    segment = None
    passed_model = 1
    if confirm:
        segment = confirm_segmented_model(prepared, knee_x_norm, config)
        passed_model = int(segment.passes)

    statistic = (
        passed_model,
        float(selected.support_frac),
        float(selected.median_noise_prominence_ratio),
    )

    return SearchResult(
        detected=True,
        knee_x_norm=knee_x_norm,
        window=window,
        cluster=selected,
        segment=segment,
        statistic=statistic,
        n_candidates=len(candidates),
        n_filtered=len(filtered),
    )
