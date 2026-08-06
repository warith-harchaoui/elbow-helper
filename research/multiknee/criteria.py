"""Model-selection criteria for choosing the number of breakpoints k.

Three criteria, deliberately of increasing sophistication:

- **Plain BIC**: the naive practice criticized throughout the changepoint
  literature (Yao 1988, Zhang & Siegmund 2007): plug the count of *regression*
  parameters into the textbook BIC formula and say nothing about the extra
  freedom spent choosing *where* the breakpoints go. Expected to overselect k.
- **Modified BIC (mBIC)**: Zhang & Siegmund's fix, charging `3` (not `1`) log
  n per breakpoint plus a segment-length term, verified against the paper via
  the sibling research fork. NOTE: their derivation is for piecewise-constant
  (mean-shift) segments; applying the same constant to piecewise-*linear*
  segments (this study's model) is an extrapolation on my part, not a result
  I can point to a citation for. Flagged, not hidden.
- **ICL**: Biernacki, Celeux & Govaert's entropy-penalized criterion,
  generalized to changepoints via the exact discrete segmentation posterior
  (Rigaill, Lebarbier & Robin 2012; Cleynen et al. 2012/2013), computed here
  by a forward log-partition DP plus Monte Carlo backward-sampling to
  estimate the posterior entropy (an approximation of the entropy term, not
  a closed form; sample count is a knob, more samples = less Monte Carlo
  noise at linear cost).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from segmentation import Segmentation, SegmentCostTable


def plain_bic(seg: Segmentation) -> float:
    """Naive BIC: charge only for the 2 regression params per segment.

    ``n_params = 2 * (k + 1)`` (slope + intercept per segment), plus 1 for
    the noise variance, times ``ln(n)``, added to the Gaussian fit term
    ``n * ln(sse / n)``. Does not charge anything for the freedom to place
    the k breakpoints anywhere among the ``n`` points, which is exactly the
    irregularity the literature says causes overselection.
    """
    n = seg.n
    sse = max(seg.sse, 1e-12)
    n_params = 2 * (seg.k + 1) + 1  # + 1 for sigma^2
    return float(n * np.log(sse / n) + n_params * np.log(n))


def _seg_len_term(seg: Segmentation) -> float:
    n = seg.n
    return sum(np.log(max(ell, 1) / n) for ell in seg.segment_lengths)


def modified_bic_additive(seg: Segmentation) -> float:
    """Zhang & Siegmund (2007) mBIC penalty, combined by literal addition.

    ``mBIC = n*ln(sse/n) + 3*m*ln(n) + sum(ln(l_j / n))``, taking the
    sibling research fork's reported penalty term
    (``3*m*ln(n) + sum(ln(l_j/n))``) and adding it straight to the Gaussian
    fit term, the way plain BIC's penalty is added.

    CAVEAT, found by testing rather than assumed: this combination makes
    the criterion *reward* uneven/short segments relative to balanced ones
    at fixed sse and k (``sum(ln(l_j/n))`` is least negative, i.e. adds the
    most, when segments are balanced): the opposite of the "penalizes
    short/uneven segments" behaviour the literature attributes to this term.
    That means either this additive combination is the wrong way to fold
    the fork-reported formula into a lower-is-better criterion or the
    fork's secondary-source formula itself needs the segment-length term
    negated to reproduce the claimed effect. See
    :func:`modified_bic_subtractive` for the sign-flipped alternative;
    :mod:`compare` runs both and reports which one actually curbs
    overselection empirically, rather than asserting an answer from memory.
    """
    n = seg.n
    sse = max(seg.sse, 1e-12)
    m = seg.k
    return float(n * np.log(sse / n) + 3 * m * np.log(n) + _seg_len_term(seg))


def modified_bic_subtractive(seg: Segmentation) -> float:
    """Zhang & Siegmund-inspired mBIC with the segment-length term negated.

    ``mBIC = n*ln(sse/n) + 3*m*ln(n) - sum(ln(l_j / n))``. Unlike
    :func:`modified_bic_additive`, this version does penalize uneven/short
    segments (the subtracted term is >= 0 and largest when segments are
    uneven), matching the literature's stated intent for this term. Which
    of the two actually matches Zhang & Siegmund's original formula (vs. a
    sign slip introduced by the secondary source or by my own
    reconstruction of "penalty term" -> "full criterion") is not resolved
    from memory; see the empirical comparison for which one behaves as
    advertised on data with a known true k.
    """
    n = seg.n
    sse = max(seg.sse, 1e-12)
    m = seg.k
    return float(n * np.log(sse / n) + 3 * m * np.log(n) - _seg_len_term(seg))


def _logsumexp(a: np.ndarray) -> float:
    m = np.max(a[np.isfinite(a)]) if np.isfinite(a).any() else -np.inf
    if not np.isfinite(m):
        return -np.inf
    return float(m + np.log(np.sum(np.exp(a - m))))


def estimate_noise_variance_richest_fit(
    x: np.ndarray, y: np.ndarray, richest: Segmentation
) -> float:
    """Pooled residual variance from the richest (largest-k) DP fit.

    A common pragmatic pilot estimate in the changepoint literature: fit a
    model rich enough to track genuine structure and treat its leftover
    residual variance as the noise floor. CAVEAT found by testing: on data
    with *no* real structure, the richest model already partly overfits the
    noise itself, which can bias this estimate low and, downstream, make
    ICL's entropy term too weak to counteract the spurious fit gain from
    extra breakpoints (see :func:`estimate_noise_variance_first_diff` for
    the alternative this comparison found necessary).
    """
    n = richest.n
    dof = max(1, n - 2 * (richest.k + 1))
    return float(max(richest.sse / dof, 1e-8))


def estimate_noise_variance_first_diff(y: np.ndarray) -> float:
    """Robust noise variance from the MAD of first differences of ``y``.

    Segmentation-independent (does not rely on any particular breakpoint
    count), so it cannot be biased by an overfit "richest" model the way
    :func:`estimate_noise_variance_richest_fit` can. Uses the same
    ``1.4826 / sqrt(2)`` MAD-of-differences scaling elbow-helper's shipped
    single-knee pipeline uses (``elbow_helper.numerics.robust_sigma_from_diffs``),
    reimplemented here to keep this research module self-contained.
    """
    y = np.asarray(y, dtype=float)
    d = np.diff(y)
    if d.size == 0:
        return 1e-8
    mad = np.median(np.abs(d - np.median(d)))
    sigma = 1.4826 * mad / np.sqrt(2.0)
    return float(max(sigma * sigma, 1e-8))


class _ForwardTables:
    """Forward log-partition function ``logZ[k][t]`` for k = 1..K segments."""

    def __init__(self, table: SegmentCostTable, n: int, k_max_segments: int,
                 sigma2: float, min_seg: int):
        self.table = table
        self.n = n
        self.sigma2 = sigma2
        self.min_seg = min_seg
        self.K = k_max_segments  # max number of *segments* (= max breakpoints + 1)

        self.logZ = [np.full(n + 1, -np.inf) for _ in range(self.K + 1)]
        for t in range(min_seg, n + 1):
            self.logZ[1][t] = self._ll(0, t)
        for k in range(2, self.K + 1):
            lo_t = min_seg * k
            for t in range(lo_t, n + 1):
                s_lo = min_seg * (k - 1)
                s_hi = t - min_seg
                if s_hi < s_lo:
                    continue
                vals = np.array([
                    self.logZ[k - 1][s] + self._ll(s, t)
                    for s in range(s_lo, s_hi + 1)
                ])
                self.logZ[k][t] = _logsumexp(vals)

    def _ll(self, i: int, j: int) -> float:
        m = j - i
        cost = self.table.cost(i, j)
        return float(-0.5 * cost / self.sigma2 - 0.5 * m * np.log(2 * np.pi * self.sigma2))

    def sample_segmentation(self, K: int, rng: np.random.Generator) -> List[int]:
        """Backward-sample one segmentation with exactly ``K`` segments."""
        cuts = []
        t = self.n
        for k in range(K, 1, -1):
            s_lo = self.min_seg * (k - 1)
            s_hi = t - self.min_seg
            candidates = np.arange(s_lo, s_hi + 1)
            log_w = np.array([
                self.logZ[k - 1][s] + self._ll(s, t) for s in candidates
            ])
            log_w -= _logsumexp(log_w)
            w = np.exp(log_w)
            w = w / w.sum()
            s = int(rng.choice(candidates, p=w))
            cuts.append(s)
            t = s
        cuts.reverse()
        return cuts

    def total_loglik(self, cuts: Sequence[int]) -> float:
        boundaries = (0, *cuts, self.n)
        return float(sum(
            self._ll(boundaries[i], boundaries[i + 1])
            for i in range(len(boundaries) - 1)
        ))


def icl_scores(
    dp_results: Sequence[Segmentation], x: np.ndarray, y: np.ndarray,
    sigma2: float, min_seg: int = 3, n_samples: int = 200, seed: int = 0,
) -> List[float]:
    """ICL-style score for each DP-optimal segmentation in ``dp_results``.

    Lower is better, matching the BIC/mBIC sign convention. Biernacki, Celeux
    & Govaert's original mixture-model ICL is ``BIC + 2*mean_entropy``: it
    starts from BIC (which already penalizes parameter count) and adds
    an *extra* entropy correction on top. It does not replace BIC's complexity
    penalty with bare integrated likelihood.

    An earlier version of this function built ``ICL(K) = -logZ_K(n) + H(K)``
    instead (integrated likelihood plus entropy, no separate parameter-count
    term), reasoning that the discrete segmentation posterior already
    "contains" a complexity notion. Testing this against pure-noise data
    (no true breakpoints) falsified that: ``logZ_K(n)`` grows by roughly
    4-5 nats per added segment purely from the combinatorial growth in the
    number of candidate segmentations being summed over; the entropy
    term did not grow fast enough to cancel it (measured, not assumed: on a
    60-point pure-noise line, ``logZ`` rose 89.8 -> 94.7 -> 99.0 -> 102.8
    across K=1..4 while entropy only rose 0 -> 3.7 -> 5.9 -> 7.9), so that
    construction systematically overselected k. This version instead adds
    the entropy correction on top of :func:`plain_bic`, the standard
    ``ICL = BIC + 2H`` form, which does carry an explicit complexity
    penalty.

    Parameters
    ----------
    dp_results : list of Segmentation
        The optimal segmentation for k = 0..k_max, as returned by
        :func:`segmentation.dp_optimal_partition` (this function needs the
        DP's exact segmentation at each k for the BIC term and separately
        builds the forward/backward tables for the entropy term).
    x, y : array-like
        The curve.
    sigma2 : float
        Noise variance for the Gaussian likelihood used only in the entropy
        estimate (see :func:`estimate_noise_variance_first_diff`).
    n_samples : int, optional
        Backward-sampling draws per k for the Monte Carlo entropy estimate.

    Returns
    -------
    list of float
        ``scores[k]`` for ``k = 0, ..., len(dp_results) - 1``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    table = SegmentCostTable(x, y)
    K_max = len(dp_results)  # dp_results[k] has K = k + 1 segments
    fwd = _ForwardTables(table, n, K_max, sigma2, min_seg)
    rng = np.random.default_rng(seed)

    scores = []
    for k, seg in enumerate(dp_results):
        K = k + 1
        base = plain_bic(seg)
        if K == 1:
            scores.append(base)
            continue
        logZ_K = fwd.logZ[K][n]
        if not np.isfinite(logZ_K):
            scores.append(np.inf)
            continue
        samples_ll = [
            fwd.total_loglik(fwd.sample_segmentation(K, rng))
            for _ in range(n_samples)
        ]
        entropy = logZ_K - float(np.mean(samples_ll))
        scores.append(float(base + 2.0 * entropy))
    return scores
