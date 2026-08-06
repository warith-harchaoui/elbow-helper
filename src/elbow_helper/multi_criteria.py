"""Phase M2: the subtractive-sign modified BIC for :func:`elbow_helper.robust_knees`.

Ported from the validated ``research/multiknee/criteria.py`` (see ``MATH-en.md``
section 14 and ``research/multiknee/RESULTS.md`` for the derivation, the
sign-convention ambiguity, and why the subtractive form is the one shipped:
0.85 overall exact-k accuracy and a 0% false-positive rate on flat data,
versus 0.77 and 11% for the literal additive form).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from .multi_segmentation import Segmentation

_EPS = 1e-12


def modified_bic(seg: Segmentation) -> float:
    """Zhang-Siegmund-inspired modified BIC, subtractive sign convention.

    ``mBIC = n*ln(sse/n) + 3*m*ln(n) - sum(ln(l_j / n))``, ``m = k`` the
    number of breakpoints and ``l_j`` the segment lengths. Lower is better.
    See ``MATH-en.md`` section 18 for the derivation and why the segment-length
    term is subtracted rather than added.

    Parameters
    ----------
    seg : Segmentation
        A candidate segmentation, carrying ``n``, ``sse``, ``k`` and
        ``segment_lengths``.

    Returns
    -------
    float
        The subtractive-sign modified BIC score; lower is better.
    """
    n = seg.n
    sse = max(seg.sse, _EPS)
    m = seg.k
    seg_len_term = sum(np.log(max(ell, 1) / n) for ell in seg.segment_lengths)
    return float(n * np.log(sse / n) + 3 * m * np.log(n) - seg_len_term)
