"""Tests for the sequential Bonferroni-corrected FWER gate.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fwer import sequential_fwer_gate
from segmentation import dp_optimal_partition


def test_gate_rejects_all_breakpoints_on_pure_noise_most_of_the_time():
    rng = np.random.default_rng(0)
    n = 60
    x = np.linspace(0, 1, n)
    false_positive_count = 0
    trials = 20
    for trial in range(trials):
        y = rng.normal(0, 1, n)
        segs = dp_optimal_partition(x, y, k_max=3, min_seg=5)
        k_hat, _ = sequential_fwer_gate(
            x, y, segs, alpha=0.05, n_permutations=100, min_seg=5, seed=trial
        )
        if k_hat > 0:
            false_positive_count += 1
    # With alpha=0.05 Bonferroni-corrected over up to 3 tests, false
    # acceptance should be rare, allow generous slack for a 20-trial sample.
    assert false_positive_count <= 4, false_positive_count


def test_gate_accepts_a_real_breakpoint_with_low_noise():
    rng = np.random.default_rng(1)
    n = 80
    x = np.linspace(0, 1, n)
    y = np.where(x < 0.4, 3 * x, 1.2 + 0.1 * (x - 0.4)) + rng.normal(0, 0.03, n)
    segs = dp_optimal_partition(x, y, k_max=3, min_seg=5)
    k_hat, p_values = sequential_fwer_gate(
        x, y, segs, alpha=0.05, n_permutations=300, min_seg=5, seed=0
    )
    assert k_hat >= 1
    assert p_values[0] <= 0.05 / 3


def test_gate_is_monotone_stopping_not_skipping():
    # The gate must stop at the first failure, never "skip" a rejected k and
    # accept a later one: every p-value except possibly the last must have
    # passed; if the last one failed, k_hat must be one less than the
    # number of steps tested.
    rng = np.random.default_rng(2)
    n = 70
    x = np.linspace(0, 1, n)
    y = rng.normal(0, 1, n)
    segs = dp_optimal_partition(x, y, k_max=4, min_seg=5)
    k_hat, p_values = sequential_fwer_gate(
        x, y, segs, alpha=0.1, n_permutations=100, min_seg=5, seed=3
    )
    alpha_corrected = 0.1 / 4
    assert all(p <= alpha_corrected for p in p_values[:-1])
    if p_values and p_values[-1] > alpha_corrected:
        assert k_hat == len(p_values) - 1
    else:
        assert k_hat == len(p_values)


def test_gate_returns_zero_for_k_max_zero():
    rng = np.random.default_rng(4)
    n = 20
    x = np.linspace(0, 1, n)
    y = rng.normal(0, 1, n)
    segs = dp_optimal_partition(x, y, k_max=0, min_seg=5)
    k_hat, p_values = sequential_fwer_gate(x, y, segs, alpha=0.05, n_permutations=50, min_seg=5)
    assert k_hat == 0
    assert p_values == []
