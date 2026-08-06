# Multi-knee method comparison: results

Synthetic piecewise-linear curves, n=100, min_seg=8, k_max=4, 25 replicates per (true_k, noise) cell, true_k in (0, 1, 2, 3), noise levels {'low': 0.02, 'medium': 0.05, 'high': 0.1}.

## Conclusions

Six findings, in order of how directly they answer the original question: is a BIC-family sweep a sound backbone for multi-knee detection; how should it combine with false-positive control?

1. **DP beats greedy decisively; the mechanism is now measured, not assumed.** Every DP-based method outranks its greedy counterpart (DP+mBIC_sub: 0.85 exact-k accuracy vs. Greedy+mBIC_sub: 0.60; DP+BIC: 0.65 vs. Greedy+BIC: 0.49). Inspecting individual curves explains why greedy specifically *overselects* at **low** noise, which is the counterintuitive part: on `true_k=2`, low noise, seed 0, greedy's second split lands at index 50 instead of the true 65 (SSE 0.068 vs. DP's optimal 0.035 at the same k=2); its third breakpoint then closes most of that gap (0.068 → 0.032), so BIC correctly rewards that third breakpoint **because it is genuinely fixing greedy's own placement error**, not because it is fitting noise. At high noise, greedy's k=2 split (SSE 0.756) nearly matches DP's optimum (0.752); the correction is unnecessary and the overselection disappears. This is a specific, verified instance of the textbook motivation for Wild Binary Segmentation (Fryzlewicz 2014): greedy's early commitment errors don't just cost fit quality, they actively bias downstream model-selection criteria toward more breakpoints. **Practical conclusion: use DP (or an equivalent exact/near-exact search, e.g. PELT with the same penalty), not greedy, whenever the ~O(k·n²) cost is affordable**: it was 0.02s/curve here, against 0.0006s for greedy. The savings from greedy are not worth this failure mode at these sample sizes.

2. **Plain BIC overselects, exactly as the literature predicts; now that's a measured number, not a citation.** DP+BIC: 27% false-positive rate on genuinely flat data (`true_k=0`) and only 0.65 overall exact-k accuracy, the worst of the four DP-based criteria. Confirms Yao (1988) / Zhang & Siegmund (2007)'s claim empirically on this exact model class (piecewise-linear, not just piecewise-constant).

3. **The mBIC sign ambiguity is resolved empirically: subtractive wins, decisively.** `modified_bic_subtractive` (segment-length term penalizes uneven segments, matching the literature's stated intent) reaches 0.85 accuracy and 0% false positives at `true_k=0`. `modified_bic_additive` (the literal fork-reported formula, added straight to the fit term) still beats plain BIC (0.77 vs. 0.65) but is clearly worse than the subtractive form and leaves an 11% false-positive rate. **Conclusion: if implementing Zhang-Siegmund's mBIC from a secondary source, verify the sign of the segment-length term against real overselection behavior, don't trust the additive combination by default.**

4. **ICL and the FWER sequential gate both work and land in the same tier as mBIC_sub.** DP+ICL: 0.82 accuracy, 4% false-positive rate. DP+FWER: 0.82 accuracy, 0% false-positive rate. Both are close to mBIC_sub's 0.85 despite being structurally very different (ICL: entropy-corrected Bayesian criterion; FWER: permutation test with Bonferroni correction). Three independent corrections for the same known failure mode converge on similar performance, which is reassuring rather than redundant: it means the fix is robust to *how* you implement the correction, not an artifact of one specific formula. Runtime per curve:

   - ICL: 0.38s (Monte Carlo entropy sampling), the most expensive
   - FWER: 0.10s
   - mBIC: 0.02s, the same as plain BIC since it's a closed-form penalty swap: the cheapest way to get this benefit

5. **Combining mBIC_sub with an FWER confirmation is safe but adds no accuracy here and costs recall in the hardest cell.** `DP+mBIC_sub+FWERconfirm` (accept `min(mBIC_sub's k, FWER's k)`) scores 0.82, slightly *below* mBIC_sub alone (0.85), because it inherits FWER's extra caution in the hardest regime: at `true_k=3`, high noise, mBIC_sub alone still averages k_hat=1.84 (already a substantial undercount from the true 3); the FWER-confirmed combination pulls that down further to k_hat=1.16. Two independently conservative gates compound conservatism. **Conclusion: layering FWER confirmation on top of mBIC_sub is a legitimate belt-and-suspenders option when false positives are exceptionally costly, but it is not a free accuracy improvement: expect it to trade recall for even tighter false-positive control, concentrated exactly where detection is already hardest.**

6. **The honest limit of all methods is the same: underselection, not overselection, at high noise with several true breakpoints.** At `true_k=3`, high noise, every method's mean k_hat sits well below 3 (DP+mBIC_sub: 1.84, DP+FWER: 1.16, DP+ICL: 1.88). The failure mode at the hard end of this study is abstaining toward fewer knees, not hallucinating extra ones. That is the same direction elbow-helper's existing single-knee pipeline already errs toward by design (`NULL_NOT_REJECTED`, `BOOTSTRAP_UNSTABLE`, etc.), so a multi-knee extension built on these components would inherit a consistent, already-intended failure direction rather than a new one.

**Recommendation for elbow-helper's multi-knee extension**: search with DP (not greedy) up to a modest k_max, select k with the subtractive-sign modified BIC as the primary, cheap criterion, and optionally layer the FWER sequential gate on top when the cost of a false breakpoint is high enough to justify trading some recall for it. This is close to "Option A" from the earlier literature-only research (sequential gate reusing the existing single-knee gate architecture), now with actual numbers behind the choice of DP over greedy and the mBIC sign convention, neither of which the literature review alone could have settled.

**Scope note**: this comparison uses independent (discontinuous) per-segment OLS regression, the model class the changepoint literature being tested is actually stated for, not the continuous relu-basis broken-line model `elbow_helper`'s shipped single-knee pipeline uses. Reconciling those two modeling choices is a separate design decision for whenever this becomes a shipped `robust_knees` API, deliberately deferred here so the comparison stays faithful to the literature it's testing.

## Overall accuracy by method

| method | P(k_hat = true_k) | P(over) | P(under) | mean |bias| | mean seconds/curve |
|---|---|---|---|---|---|
| DP+BIC | 0.65 | 0.29 | 0.06 | 0.45 | 0.0229 |
| DP+FWER | 0.82 | 0.01 | 0.17 | 0.26 | 0.1022 |
| DP+ICL | 0.82 | 0.05 | 0.13 | 0.21 | 0.3787 |
| DP+mBIC_add | 0.77 | 0.15 | 0.08 | 0.28 | 0.0229 |
| DP+mBIC_sub | 0.85 | 0.02 | 0.13 | 0.17 | 0.0229 |
| DP+mBIC_sub+FWERconfirm | 0.82 | 0.01 | 0.17 | 0.26 | 0.1022 |
| Greedy+BIC | 0.49 | 0.44 | 0.07 | 0.61 | 0.0008 |
| Greedy+FWER | 0.59 | 0.21 | 0.21 | 0.50 | 0.0820 |
| Greedy+mBIC_add | 0.56 | 0.35 | 0.09 | 0.50 | 0.0008 |
| Greedy+mBIC_sub | 0.60 | 0.24 | 0.16 | 0.43 | 0.0008 |

## Accuracy by true_k and noise level


### DP+BIC

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 0.64 | 0.48 |
| 0 | medium | 0.80 | 0.28 |
| 0 | high | 0.76 | 0.52 |
| 1 | low | 0.68 | 1.56 |
| 1 | medium | 0.76 | 1.32 |
| 1 | high | 0.64 | 1.52 |
| 2 | low | 0.60 | 2.48 |
| 2 | medium | 0.64 | 2.48 |
| 2 | high | 0.76 | 2.00 |
| 3 | low | 0.60 | 3.40 |
| 3 | medium | 0.68 | 3.32 |
| 3 | high | 0.28 | 2.52 |

### DP+FWER

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 1.00 | 0.00 |
| 0 | medium | 1.00 | 0.00 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 1.00 | 1.00 |
| 1 | medium | 1.00 | 1.00 |
| 1 | high | 0.96 | 1.04 |
| 2 | low | 1.00 | 2.00 |
| 2 | medium | 0.96 | 2.04 |
| 2 | high | 0.20 | 1.08 |
| 3 | low | 0.96 | 3.04 |
| 3 | medium | 0.76 | 2.84 |
| 3 | high | 0.00 | 1.16 |

### DP+ICL

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 0.96 | 0.08 |
| 0 | medium | 0.92 | 0.12 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 0.96 | 1.04 |
| 1 | medium | 0.96 | 1.04 |
| 1 | high | 0.88 | 1.12 |
| 2 | low | 1.00 | 2.00 |
| 2 | medium | 0.92 | 2.08 |
| 2 | high | 0.36 | 1.40 |
| 3 | low | 0.92 | 3.08 |
| 3 | medium | 0.88 | 3.04 |
| 3 | high | 0.08 | 1.88 |

### DP+mBIC_add

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 0.88 | 0.16 |
| 0 | medium | 0.88 | 0.16 |
| 0 | high | 0.92 | 0.24 |
| 1 | low | 0.80 | 1.36 |
| 1 | medium | 0.92 | 1.08 |
| 1 | high | 0.80 | 1.24 |
| 2 | low | 0.84 | 2.16 |
| 2 | medium | 0.80 | 2.20 |
| 2 | high | 0.56 | 1.72 |
| 3 | low | 0.76 | 3.24 |
| 3 | medium | 0.80 | 3.20 |
| 3 | high | 0.28 | 2.28 |

### DP+mBIC_sub

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 1.00 | 0.00 |
| 0 | medium | 1.00 | 0.00 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 1.00 | 1.00 |
| 1 | medium | 1.00 | 1.00 |
| 1 | high | 0.92 | 1.08 |
| 2 | low | 1.00 | 2.00 |
| 2 | medium | 0.96 | 2.04 |
| 2 | high | 0.40 | 1.40 |
| 3 | low | 0.96 | 3.04 |
| 3 | medium | 0.92 | 3.00 |
| 3 | high | 0.08 | 1.84 |

### DP+mBIC_sub+FWERconfirm

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 1.00 | 0.00 |
| 0 | medium | 1.00 | 0.00 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 1.00 | 1.00 |
| 1 | medium | 1.00 | 1.00 |
| 1 | high | 0.96 | 1.04 |
| 2 | low | 1.00 | 2.00 |
| 2 | medium | 0.96 | 2.04 |
| 2 | high | 0.20 | 1.08 |
| 3 | low | 0.96 | 3.04 |
| 3 | medium | 0.76 | 2.84 |
| 3 | high | 0.00 | 1.16 |

### Greedy+BIC

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 0.68 | 0.40 |
| 0 | medium | 0.84 | 0.20 |
| 0 | high | 0.76 | 0.36 |
| 1 | low | 0.72 | 1.48 |
| 1 | medium | 0.88 | 1.16 |
| 1 | high | 0.76 | 1.32 |
| 2 | low | 0.00 | 3.32 |
| 2 | medium | 0.12 | 3.04 |
| 2 | high | 0.68 | 2.08 |
| 3 | low | 0.00 | 4.00 |
| 3 | medium | 0.28 | 3.72 |
| 3 | high | 0.16 | 2.48 |

### Greedy+FWER

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 1.00 | 0.00 |
| 0 | medium | 1.00 | 0.00 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 1.00 | 1.00 |
| 1 | medium | 1.00 | 1.00 |
| 1 | high | 0.96 | 1.04 |
| 2 | low | 0.00 | 3.00 |
| 2 | medium | 0.44 | 2.40 |
| 2 | high | 0.16 | 1.04 |
| 3 | low | 0.04 | 3.96 |
| 3 | medium | 0.44 | 2.44 |
| 3 | high | 0.00 | 1.12 |

### Greedy+mBIC_add

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 0.88 | 0.16 |
| 0 | medium | 0.88 | 0.16 |
| 0 | high | 0.96 | 0.12 |
| 1 | low | 0.88 | 1.16 |
| 1 | medium | 0.92 | 1.08 |
| 1 | high | 0.92 | 1.08 |
| 2 | low | 0.00 | 3.12 |
| 2 | medium | 0.12 | 2.96 |
| 2 | high | 0.40 | 1.76 |
| 3 | low | 0.00 | 4.00 |
| 3 | medium | 0.52 | 3.48 |
| 3 | high | 0.20 | 2.24 |

### Greedy+mBIC_sub

| true_k | noise | P(k_hat=true_k) | mean k_hat |
|---|---|---|---|
| 0 | low | 1.00 | 0.00 |
| 0 | medium | 1.00 | 0.00 |
| 0 | high | 1.00 | 0.00 |
| 1 | low | 1.00 | 1.00 |
| 1 | medium | 1.00 | 1.00 |
| 1 | high | 0.92 | 1.08 |
| 2 | low | 0.00 | 3.00 |
| 2 | medium | 0.28 | 2.72 |
| 2 | high | 0.36 | 1.36 |
| 3 | low | 0.04 | 3.96 |
| 3 | medium | 0.56 | 2.72 |
| 3 | high | 0.04 | 1.72 |

## False-positive rate on true_k = 0 specifically

This is the headline number the original research question was about:
does the method overselect breakpoints on data with no real structure?

| method | P(k_hat > 0 | true_k = 0) |
|---|---|
| DP+BIC | 0.27 |
| DP+FWER | 0.00 |
| DP+ICL | 0.04 |
| DP+mBIC_add | 0.11 |
| DP+mBIC_sub | 0.00 |
| DP+mBIC_sub+FWERconfirm | 0.00 |
| Greedy+BIC | 0.24 |
| Greedy+FWER | 0.00 |
| Greedy+mBIC_add | 0.09 |
| Greedy+mBIC_sub | 0.00 |

## Ranking by overall exact-k accuracy

1. **DP+mBIC_sub**: 0.85
2. **DP+mBIC_sub+FWERconfirm**: 0.82
3. **DP+ICL**: 0.82
4. **DP+FWER**: 0.82
5. **DP+mBIC_add**: 0.77
6. **DP+BIC**: 0.65
7. **Greedy+mBIC_sub**: 0.60
8. **Greedy+FWER**: 0.59
9. **Greedy+mBIC_add**: 0.56
10. **Greedy+BIC**: 0.49