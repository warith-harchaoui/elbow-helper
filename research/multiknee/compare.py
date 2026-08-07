"""Empirical comparison: DP vs. greedy search, crossed with plain BIC, both
mBIC sign conventions, ICL, and the FWER-gated sequential test, plus one
FWER-confirmed combination, on synthetic curves with a *known* true number
of breakpoints.

Run as a script: ``python compare.py`` (from this directory or anywhere
with ``elbow-helper``'s environment active: it only needs numpy). Writes
``RESULTS.md`` next to this file and also prints a summary to stdout.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from criteria import (
    estimate_noise_variance_first_diff,
    icl_scores,
    modified_bic_additive,
    modified_bic_subtractive,
    plain_bic,
)
from fwer import sequential_fwer_gate
from segmentation import Segmentation, dp_optimal_partition, greedy_binary_segmentation
from synthetic import make_piecewise_curve

K_MAX = 4
MIN_SEG = 8
N = 100
TRUE_KS = (0, 1, 2, 3)
NOISE_LEVELS = {"low": 0.02, "medium": 0.05, "high": 0.10}
N_REPLICATES = 25
ALPHA = 0.05


def _argmin_k(scores: List[float]) -> int:
    return int(np.argmin(scores))


def _criterion_methods(
    segs: List[Segmentation], x, y, label_prefix: str
) -> Dict[str, int]:
    out = {}
    out[f"{label_prefix}+BIC"] = _argmin_k([plain_bic(s) for s in segs])
    out[f"{label_prefix}+mBIC_add"] = _argmin_k([modified_bic_additive(s) for s in segs])
    out[f"{label_prefix}+mBIC_sub"] = _argmin_k([modified_bic_subtractive(s) for s in segs])
    return out


@dataclass
class Trial:
    method: str
    true_k: int
    noise: str
    k_hat: int
    seconds: float


def run_one_curve(true_k: int, noise: float, noise_label: str, seed: int) -> List[Trial]:
    x, y, _true_idx = make_piecewise_curve(true_k, n=N, noise_sigma=noise, seed=seed)
    trials: List[Trial] = []

    t0 = time.perf_counter()
    dp_segs = dp_optimal_partition(x, y, k_max=K_MAX, min_seg=MIN_SEG)
    dp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    greedy_segs = greedy_binary_segmentation(x, y, k_max=K_MAX, min_seg=MIN_SEG)
    greedy_time = time.perf_counter() - t0

    for name, k_hat in _criterion_methods(dp_segs, x, y, "DP").items():
        trials.append(Trial(name, true_k, noise_label, k_hat, dp_time))
    for name, k_hat in _criterion_methods(greedy_segs, x, y, "Greedy").items():
        trials.append(Trial(name, true_k, noise_label, k_hat, greedy_time))

    t0 = time.perf_counter()
    sigma2 = estimate_noise_variance_first_diff(y)
    icl_sc = icl_scores(dp_segs, x, y, sigma2=sigma2, min_seg=MIN_SEG, n_samples=150, seed=seed)
    icl_time = time.perf_counter() - t0
    icl_k = _argmin_k(icl_sc)
    trials.append(Trial("DP+ICL", true_k, noise_label, icl_k, dp_time + icl_time))

    t0 = time.perf_counter()
    dp_fwer_k, _ = sequential_fwer_gate(
        x, y, dp_segs, alpha=ALPHA, n_permutations=150, min_seg=MIN_SEG, seed=seed
    )
    dp_fwer_time = time.perf_counter() - t0
    trials.append(Trial("DP+FWER", true_k, noise_label, dp_fwer_k, dp_time + dp_fwer_time))

    t0 = time.perf_counter()
    greedy_fwer_k, _ = sequential_fwer_gate(
        x, y, greedy_segs, alpha=ALPHA, n_permutations=150, min_seg=MIN_SEG, seed=seed
    )
    greedy_fwer_time = time.perf_counter() - t0
    trials.append(Trial("Greedy+FWER", true_k, noise_label, greedy_fwer_k,
                         greedy_time + greedy_fwer_time))

    # Combination: mBIC (subtractive) proposes k, but capped at whatever the
    # DP+FWER gate independently accepts -- "explore with a criterion,
    # confirm with a family-wise-controlled test", the sibling forks'
    # recommended composition.
    mbic_k = _argmin_k([modified_bic_subtractive(s) for s in dp_segs])
    combined_k = min(mbic_k, dp_fwer_k)
    trials.append(Trial("DP+mBIC_sub+FWERconfirm", true_k, noise_label, combined_k,
                         dp_time + dp_fwer_time))

    return trials


def run_all() -> List[Trial]:
    all_trials: List[Trial] = []
    seed = 0
    total = len(TRUE_KS) * len(NOISE_LEVELS) * N_REPLICATES
    done = 0
    for true_k in TRUE_KS:
        for noise_label, noise in NOISE_LEVELS.items():
            for rep in range(N_REPLICATES):
                all_trials.extend(run_one_curve(true_k, noise, noise_label, seed))
                seed += 1
                done += 1
                if done % 20 == 0:
                    print(f"  ... {done}/{total} curves done")
    return all_trials


def summarize(trials: List[Trial]) -> str:
    by_method: Dict[str, List[Trial]] = defaultdict(list)
    for t in trials:
        by_method[t.method].append(t)

    lines = []
    lines.append("# Multi-knee method comparison — results\n")
    lines.append(
        f"Synthetic piecewise-linear curves, n={N}, min_seg={MIN_SEG}, "
        f"k_max={K_MAX}, {N_REPLICATES} replicates per (true_k, noise) cell, "
        f"true_k in {TRUE_KS}, noise levels {NOISE_LEVELS}.\n"
    )

    lines.append("## Overall accuracy by method\n")
    lines.append("| method | P(k_hat = true_k) | P(over) | P(under) | mean |bias| | mean seconds/curve |")
    lines.append("|---|---|---|---|---|---|")
    overall_rank = []
    for method in sorted(by_method):
        ts = by_method[method]
        n = len(ts)
        exact = sum(t.k_hat == t.true_k for t in ts) / n
        over = sum(t.k_hat > t.true_k for t in ts) / n
        under = sum(t.k_hat < t.true_k for t in ts) / n
        bias = np.mean([abs(t.k_hat - t.true_k) for t in ts])
        secs = np.mean([t.seconds for t in ts])
        overall_rank.append((exact, method))
        lines.append(
            f"| {method} | {exact:.2f} | {over:.2f} | {under:.2f} | {bias:.2f} | {secs:.4f} |"
        )

    lines.append("\n## Accuracy by true_k and noise level\n")
    for method in sorted(by_method):
        lines.append(f"\n### {method}\n")
        lines.append("| true_k | noise | P(k_hat=true_k) | mean k_hat |")
        lines.append("|---|---|---|---|")
        ts = by_method[method]
        for true_k in TRUE_KS:
            for noise_label in NOISE_LEVELS:
                cell = [t for t in ts if t.true_k == true_k and t.noise == noise_label]
                if not cell:
                    continue
                exact = sum(t.k_hat == true_k for t in cell) / len(cell)
                mean_k = np.mean([t.k_hat for t in cell])
                lines.append(f"| {true_k} | {noise_label} | {exact:.2f} | {mean_k:.2f} |")

    lines.append("\n## False-positive rate on true_k = 0 specifically\n")
    lines.append("This is the headline number the original research question was about:")
    lines.append("does the method overselect breakpoints on data with no real structure?\n")
    lines.append("| method | P(k_hat > 0 | true_k = 0) |")
    lines.append("|---|---|")
    for method in sorted(by_method):
        cell = [t for t in by_method[method] if t.true_k == 0]
        fp = sum(t.k_hat > 0 for t in cell) / len(cell)
        lines.append(f"| {method} | {fp:.2f} |")

    overall_rank.sort(reverse=True)
    lines.append("\n## Ranking by overall exact-k accuracy\n")
    for rank, (exact, method) in enumerate(overall_rank, 1):
        lines.append(f"{rank}. **{method}**: {exact:.2f}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(f"Running {len(TRUE_KS) * len(NOISE_LEVELS) * N_REPLICATES} synthetic curves "
          f"x ~9 methods each...")
    trials = run_all()
    report = summarize(trials)
    out_path = __file__.replace("compare.py", "RESULTS.md")
    with open(out_path, "w") as f:
        f.write(report)
    print(f"\nWrote {out_path}\n")
    print(report)
