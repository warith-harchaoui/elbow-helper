"""Correctness/sanity tests for the BIC, mBIC, and ICL criteria.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from criteria import (
    estimate_noise_variance_first_diff,
    estimate_noise_variance_richest_fit,
    icl_scores,
    modified_bic_additive,
    modified_bic_subtractive,
    plain_bic,
)
from segmentation import Segmentation, dp_optimal_partition


def test_plain_bic_matches_hand_formula():
    seg = Segmentation(breakpoints=(10, 20), boundaries=(0, 10, 20, 30), sse=1.5, n=30)
    got = plain_bic(seg)
    n_params = 2 * 3 + 1
    want = 30 * np.log(1.5 / 30) + n_params * np.log(30)
    assert abs(got - want) < 1e-9


def test_modified_bic_additive_matches_hand_formula():
    seg = Segmentation(breakpoints=(10, 20), boundaries=(0, 10, 20, 30), sse=1.5, n=30)
    got = modified_bic_additive(seg)
    seg_len_term = sum(np.log(ell / 30) for ell in (10, 10, 10))
    want = 30 * np.log(1.5 / 30) + 3 * 2 * np.log(30) + seg_len_term
    assert abs(got - want) < 1e-9


def test_modified_bic_subtractive_matches_hand_formula():
    seg = Segmentation(breakpoints=(10, 20), boundaries=(0, 10, 20, 30), sse=1.5, n=30)
    got = modified_bic_subtractive(seg)
    seg_len_term = sum(np.log(ell / 30) for ell in (10, 10, 10))
    want = 30 * np.log(1.5 / 30) + 3 * 2 * np.log(30) - seg_len_term
    assert abs(got - want) < 1e-9


def test_modified_bic_additive_rewards_uneven_segments():
    # Documented finding, not the literature's stated intent: the additive
    # combination makes uneven segments score *better* (lower), the
    # opposite of "penalizes short/uneven segments". See criteria.py.
    even = Segmentation(breakpoints=(15,), boundaries=(0, 15, 30), sse=1.0, n=30)
    uneven = Segmentation(breakpoints=(3,), boundaries=(0, 3, 30), sse=1.0, n=30)
    assert plain_bic(even) == plain_bic(uneven)  # plain BIC is blind to this
    assert modified_bic_additive(uneven) < modified_bic_additive(even)


def test_modified_bic_subtractive_penalizes_uneven_segments():
    # This is the sign convention that matches the literature's stated
    # "penalizes short/uneven segments" behaviour.
    even = Segmentation(breakpoints=(15,), boundaries=(0, 15, 30), sse=1.0, n=30)
    uneven = Segmentation(breakpoints=(3,), boundaries=(0, 3, 30), sse=1.0, n=30)
    assert modified_bic_subtractive(uneven) > modified_bic_subtractive(even)


def test_icl_entropy_nonnegative_within_monte_carlo_tolerance():
    # H(K) = logZ_K - E[ll] must be >= 0 exactly (Gibbs entropy is
    # nonnegative); the Monte Carlo estimate can dip slightly negative from
    # sampling noise, but not by much with n_samples=300.
    rng = np.random.default_rng(1)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.5, x, 0.5 + 3 * (x - 0.5)) + rng.normal(0, 0.15, n)
    richest = dp_optimal_partition(x, y, k_max=3, min_seg=3)[-1]
    sigma2 = estimate_noise_variance_richest_fit(x, y, richest)

    from criteria import SegmentCostTable, _ForwardTables

    table = SegmentCostTable(x, y)
    fwd = _ForwardTables(table, n, k_max_segments=4, sigma2=sigma2, min_seg=3)
    rng2 = np.random.default_rng(2)
    for K in (2, 3, 4):
        logZ_K = fwd.logZ[K][n]
        samples_ll = [
            fwd.total_loglik(fwd.sample_segmentation(K, rng2)) for _ in range(300)
        ]
        entropy = logZ_K - float(np.mean(samples_ll))
        assert entropy > -0.5, (K, entropy)  # allow small MC slack, not a real deficit


def test_bare_integrated_likelihood_plus_entropy_overselects_on_a_straight_line():
    # Documented finding that motivated icl_scores' current BIC + 2H form:
    # a *bare* -logZ_K(n) + H(K) construction (no explicit parameter-count
    # penalty) systematically overselects k, because logZ_K(n) grows with K
    # from sheer combinatorial search-space growth faster than entropy can
    # cancel it, even on pure noise. Reproduced directly here (not via
    # icl_scores, which no longer implements this form).
    from criteria import SegmentCostTable, _ForwardTables

    rng = np.random.default_rng(3)
    n = 60
    x = np.linspace(0, 1, n)
    y = 0.5 * x + 0.1 + rng.normal(0, 0.05, n)  # no real structure beyond k=0
    sigma2 = estimate_noise_variance_first_diff(y)
    table = SegmentCostTable(x, y)
    fwd = _ForwardTables(table, n, k_max_segments=4, sigma2=sigma2, min_seg=5)
    rng2 = np.random.default_rng(1)

    bare_scores = []
    for K in (1, 2, 3, 4):
        logZ_K = fwd.logZ[K][n]
        if K == 1:
            bare_scores.append(-logZ_K)
            continue
        samples_ll = [
            fwd.total_loglik(fwd.sample_segmentation(K, rng2)) for _ in range(300)
        ]
        entropy = logZ_K - float(np.mean(samples_ll))
        bare_scores.append(-logZ_K + entropy)
    assert bare_scores[0] > bare_scores[-1]  # the overselection this test documents


def test_icl_bic_plus_entropy_prefers_fewer_breakpoints_on_a_straight_line():
    rng = np.random.default_rng(3)
    n = 60
    x = np.linspace(0, 1, n)
    y = 0.5 * x + 0.1 + rng.normal(0, 0.05, n)  # no real structure beyond k=0
    dp_results = dp_optimal_partition(x, y, k_max=3, min_seg=5)
    sigma2 = estimate_noise_variance_first_diff(y)
    scores = icl_scores(dp_results, x, y, sigma2=sigma2, min_seg=5, n_samples=200, seed=0)
    assert int(np.argmin(scores)) == 0


def test_icl_recovers_a_real_breakpoint():
    rng = np.random.default_rng(7)
    n = 80
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.4, 3 * x, 1.2 + 0.1 * (x - 0.4)) + rng.normal(0, 0.05, n)
    dp_results = dp_optimal_partition(x, y, k_max=3, min_seg=5)
    sigma2 = estimate_noise_variance_first_diff(y)
    scores = icl_scores(dp_results, x, y, sigma2=sigma2, min_seg=5, n_samples=200, seed=0)
    assert int(np.argmin(scores)) == 1
