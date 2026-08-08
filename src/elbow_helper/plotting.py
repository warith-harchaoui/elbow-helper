"""Diagnostic figure for a :func:`~elbow_helper.pipeline.robust_knee` run.

Renders a self-contained SVG — no matplotlib, no Vega, no runtime image
library — so this stays a **core** feature (no ``[plot]`` extra to install)
rather than a lazy opt-in. The SVG-writing code below (the Catmull-Rom
spline helper, the responsive ``<svg>`` header, the additive dark-mode
block) is adapted from this project's sibling ``sprezzature-figures``
package (``scripts/make_elbow.py`` / ``_svg.py`` / ``_style.py``) and
copy-pasted in rather than taken as a dependency, then cut down and
re-specialised for this module's own data shapes (normalized ``x``/``y``
arrays and the :class:`~elbow_helper.pipeline.ClearKnee` /
:class:`~elbow_helper.pipeline.NoClearKnee` result types) — elbow-helper's
only runtime dependencies stay ``numpy`` and ``os-helper``.

The figure never shows a bare point estimate: the knee is paired with its
90% bootstrap interval and the supporting evidence (detection probability,
null-model p-value, slope contrast, a BIC-derived posterior model
probability, and a worst-case-normalized fit-quality score) in a compact
legend. When the evidence is too weak, the figure says so plainly
— a greyed, dashed curve and a reason — instead of drawing a marker that
implies more confidence than the data supports.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import os_helper as oh

from .config import RobustKneeConfig, RobustKneesConfig
from .multi_pipeline import robust_knees
from .numerics import ols_rss
from .pipeline import ClearKnee, robust_knee
from .preprocessing import Abstain, prepare_curve
from .segmented import _design_broken
from .smoothing import smooth_curve
from .types import Knees

# ------------------------------------------------------------------
# Canvas + house-style tokens (fixed, no CVD-accessibility variants or
# external palette file: this is a small diagnostic figure, not a
# publication dataviz product — see sprezzature-figures for that).
# ------------------------------------------------------------------
_WIDTH = 1000
_HEIGHT = 620
_PL, _PR, _PT, _PB = 96.0, 948.0, 188.0, 524.0
_PLOT_W = _PR - _PL
_PLOT_H = _PB - _PT

_INK = "#1D1D1F"
_SUBINK = "#6E6E73"
_HAIR = "#E5E5EA"
_BG = "#FFFFFF"
_CURVE = "#007AFF"
_CURVE_DEEP = "#0051A8"
_ACCENT = "#FF3B30"
_MUTED = "#8E8E93"
_GOOD = "#34C759"

_FONT = "Roboto, -apple-system, system-ui, sans-serif"
_FONT_MONO = "Roboto Mono, ui-monospace, SFMono-Regular, monospace"

_STRINGS = {
    "en": {
        "title_clear": "Knee detection: where the curve bends",
        "subtitle_clear": "The method flags the knee at x = {x}",
        "title_abstain": "No clear knee",
        "hint_abstain": "The evidence was too weak to report a point estimate.",
        "reason": "reason",
        "knee_pill": "Knee",
        "inset_title": "Evidence",
        "detection_rate": "detection probability",
        "null_p": "null p",
        "slope_contrast": "slope contrast",
        "bic": "posterior probability",
        "fit_quality": "fit quality",
        "x_axis": "x (normalized)",
        "y_axis": "y (scaled)",
        "data_label": "data",
        "smoothed_label": "smoothed",
        "title_multi_zero": "No breakpoints: one straight trend",
        "title_multi_one": "1 breakpoint detected",
        "title_multi_k": "{k} breakpoints detected",
        "subtitle_multi": "Each dashed line marks a confirmed change of slope",
    },
    "fr": {
        "title_clear": "Détection de coude : où la courbe plie",
        "subtitle_clear": "La méthode situe le coude à x = {x}",
        "title_abstain": "Aucun coude net",
        "hint_abstain": "L'évidence était trop faible pour une estimation ponctuelle.",
        "reason": "raison",
        "knee_pill": "Coude",
        "inset_title": "Preuves",
        "detection_rate": "probabilité de détection",
        "null_p": "p (nul)",
        "slope_contrast": "contraste de pente",
        "bic": "probabilité a posteriori",
        "fit_quality": "qualité d'ajustement",
        "x_axis": "x (normalisé)",
        "y_axis": "y (mis à l'échelle)",
        "data_label": "données",
        "smoothed_label": "lissé",
        "title_multi_zero": "Aucun point de rupture : une seule tendance",
        "title_multi_one": "1 point de rupture détecté",
        "title_multi_k": "{k} points de rupture détectés",
        "subtitle_multi": "Chaque ligne pointillée marque un changement de pente confirmé",
    },
}


def _strings(language: str) -> dict:
    return _STRINGS.get(language, _STRINGS["en"])


def _fmt(v: float) -> str:
    """Compact float formatting for SVG path data (one decimal, no trailing zero)."""
    return f"{v:.1f}".rstrip("0").rstrip(".")


def _nice_step(raw_step: float) -> float:
    """Round a step size up to a human-friendly 1/2/5 times a power of ten."""
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10**exponent)
    nice = 1 if fraction <= 1 else 2 if fraction <= 2 else 5 if fraction <= 5 else 10
    return nice * (10**exponent)


def _linear_ticks(lo: float, hi: float, target: int = 5) -> List[float]:
    """Evenly spaced, round tick values covering a raw (non-log) ``[lo, hi]`` span."""
    if hi <= lo:
        return [lo]
    step = _nice_step((hi - lo) / max(target - 1, 1))
    start = math.floor(lo / step) * step
    ticks, v = [], start
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            ticks.append(round(v, 10))
        v += step
    return ticks


def _log_ticks(lo: float, hi: float) -> List[float]:
    """1-3-10-per-decade tick values covering a positive ``[lo, hi]`` span."""
    lo = max(lo, 1e-12)
    exp_lo, exp_hi = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    ticks = []
    for e in range(exp_lo, exp_hi + 1):
        for mult in (1, 3):
            v = mult * (10**e)
            if lo * 0.999 <= v <= hi * 1.001:
                ticks.append(v)
    return ticks or [lo, hi]


def _fmt_tick(v: float) -> str:
    """Thousands-separated tick label; whole numbers drop the decimal part."""
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.2f}"


def _catmull_rom(pts: Sequence[Tuple[float, float]], tension: float = 6.0) -> str:
    """SVG ``C`` commands for a smooth Catmull-Rom spline through ``pts``.

    Assumes the caller already emitted the initial ``M`` to ``pts[0]``.
    Falls back to straight line-tos below three points.
    """
    n = len(pts)
    if n < 3:
        return "".join(f" L{_fmt(x)},{_fmt(y)}" for x, y in pts[1:])
    seg = []
    for i in range(n - 1):
        p0 = pts[i - 1] if i > 0 else pts[i]
        p1, p2 = pts[i], pts[i + 1]
        p3 = pts[i + 2] if i + 2 < n else pts[i + 1]
        c1x, c1y = p1[0] + (p2[0] - p0[0]) / tension, p1[1] + (p2[1] - p0[1]) / tension
        c2x, c2y = p2[0] - (p3[0] - p1[0]) / tension, p2[1] - (p3[1] - p1[1]) / tension
        seg.append(f" C{_fmt(c1x)},{_fmt(c1y)} {_fmt(c2x)},{_fmt(c2y)} {_fmt(p2[0])},{_fmt(p2[1])}")
    return "".join(seg)


def _svg_open(title_id: str, desc_id: str) -> str:
    """The responsive, accessible ``<svg>`` opening tag (no font embedding —
    a lightweight CSS font stack is used instead, unlike the full
    sprezzature-figures generators, which embed WOFF2 Roboto for
    publication use)."""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{_WIDTH}" '
        f'height="{_HEIGHT}" viewBox="0 0 {_WIDTH} {_HEIGHT}" '
        f'font-family="{_FONT}" role="img" aria-labelledby="{title_id} {desc_id}">'
    )


def _dark_mode_block() -> str:
    """Additive ``prefers-color-scheme: dark`` block flipping paper + ink."""
    rows = [
        f'[fill="{_BG}"]{{fill:#000;}}',
        f'[fill="{_INK}"]{{fill:#F5F5F7;}}',
        f'[fill="{_SUBINK}"]{{fill:#98989D;}}',
        f'[stroke="{_HAIR}"]{{stroke:#2C2C2E;}}',
    ]
    return "<style>@media (prefers-color-scheme: dark){" + "".join(rows) + "}</style>"


def _sx(x: float) -> float:
    return _PL + x * _PLOT_W


def _sy(y: float) -> float:
    return _PB - y * _PLOT_H


def _bic_posterior_probability(bic_improvement: float) -> float:
    """Convert a raw ΔBIC (nats) into a bounded, interpretable posterior probability.

    ``bic_improvement`` is already ``n * ln(RSS_single/n) - n * ln(RSS_broken/n)
    + k_diff * ln(n)`` (see :func:`elbow_helper.numerics.bic`) — the sample size
    ``n`` and the extra-parameter penalty are already inside it, so dividing by
    an arbitrary log base (10, 2, ...) would just rescale an already-composite,
    unbounded number without fixing the real problem: a raw ΔBIC has no natural
    ceiling, so "282.4" on its own says nothing about how much evidence that
    actually is.

    What does have a natural ceiling is the quantity ΔBIC approximates: twice
    the log Bayes factor between the broken-line and single-line models
    (Kass & Raftery, 1995, eq. 4 — the same approximation this project's
    ``min_bic_improvement`` gate is calibrated against, see ``MATH-en.tex``
    / ``MATH-fr.tex``, "Related Work"). That gives an approximate Bayes
    factor ``BF ≈ exp(bic_improvement / 2)`` — literally a likelihood ratio,
    since BIC is built from ``-2 ln(L)`` (see ``MATH-en.tex``'s "From raw
    likelihood to a bounded score" for the full derivation, natural log
    throughout, back to the Gaussian density itself) — and, under equal
    priors, a posterior model probability ``BF / (1 + BF)``, which is
    exactly the logistic/sigmoid of ``bic_improvement / 2``. A probability
    is bounded in ``[0, 1]`` (maximum entropy ``ln(2)`` nats at ``p = 0.5``,
    the point of total uncertainty between the two models), so it reads the
    same way regardless of curve length or noise scale: "99.9%" is
    unambiguous where "282.4" is not. :func:`_fit_quality_score` derives a
    different, sigmoid-free ``[0, 1]``-anchored score for the single fitted
    curve itself (not a comparison between two models), normalized against
    a deliberately pessimistic worst case rather than a Bayes factor; both
    are valid, non-equivalent normalizations of the same underlying NLL.

    Parameters
    ----------
    bic_improvement : float
        ``ClearKnee.bic_improvement``, in natural-log nats (``bic_single -
        bic_broken``; positive favours the knee model).

    Returns
    -------
    float
        The approximate posterior probability, under equal priors, that the
        broken-line (knee) model is correct, in ``[0, 1]``.

    Examples
    --------
    >>> round(_bic_posterior_probability(0.0), 3)
    0.5
    >>> _bic_posterior_probability(20.0) > 0.999
    True
    """
    # exp() overflows for very large bic_improvement; clip the odds' log
    # rather than the probability so the result still saturates smoothly
    # at 1.0 instead of raising OverflowError.
    log_odds = min(bic_improvement / 2.0, 700.0)
    odds = math.exp(log_odds)
    return odds / (1.0 + odds)


def _fit_quality_score(y_eval: np.ndarray, fit_eval: np.ndarray) -> float:
    """Score a fit against a deliberately pessimistic worst case.

    Unlike :func:`_bic_posterior_probability`, this does not compare two
    competing models; it scores how well ``fit_eval`` explains ``y_eval``,
    normalized so ``1.0`` is a perfect fit. The textbook ``R²`` baseline
    (predicting the sample mean) is not used here, since the mean is the
    *unique* MSE-minimizing constant and so is the easiest baseline to
    beat, letting ``R²`` swing negative for an otherwise reasonable fit.
    Instead the baseline is the single *observed* ``y_eval`` value that is
    hardest to predict everything else from (see ``MATH-en.tex`` /
    ``MATH-fr.tex``, "From raw likelihood to a bounded score"), a strictly
    higher and harder-to-beat MSE than the mean's, ``MSE_worst >
    Var(y_eval)`` always.

    Both arguments must already be in whatever space the figure actually
    displays, normalized ``[0, 1]`` by default, or the raw data units (and
    the log of them, under ``log_y``) when ``render_svg`` is called with
    ``raw_axis=True``: scoring a fit in a different space than the one a
    reader visually judges it in would silently misrepresent how good the
    fit looks on the page.

    Parameters
    ----------
    y_eval : numpy.ndarray
        The observed curve, in display space.
    fit_eval : numpy.ndarray
        The confirmed model's fitted curve at the same points, in the same
        display space.

    Returns
    -------
    float
        ``1 - MSE_fit / MSE_worst``, ``1.0`` at a perfect fit, ``0.0`` at a
        fit exactly as bad as the pessimistic baseline itself (in
        principle still unbounded below, for a fit worse than the
        baseline, though ``MSE_worst``'s deliberately high bar makes that
        rare for a fit that already passed the pipeline's BIC and
        cross-validation gates).
    """
    mse_fit = float(np.mean((y_eval - fit_eval) ** 2))
    mse_per_candidate = np.mean((y_eval[:, None] - y_eval[None, :]) ** 2, axis=0)
    mse_worst = float(np.max(mse_per_candidate))
    if mse_worst <= 0.0:
        return 1.0
    return 1.0 - mse_fit / mse_worst


def render_svg(
    x,
    y=None,
    *,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config: Optional[RobustKneeConfig] = None,
    language: str = "en",
    raw_axis: bool = False,
    log_y: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """Build the robust-knee diagnostic as a complete, standalone SVG string.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in :func:`elbow_helper.robust_knee`.
    curve, direction : str, optional
        Orientation; inferred from the data when omitted.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts forwarded to the pipeline.
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.
    raw_axis : bool, optional
        Display the curve in its original data units instead of the
        pipeline's internal normalized ``[0, 1]`` square. The knee marker
        and confidence band still come straight from ``result.knee_x`` /
        ``result.ci90``, already in data units; only the displayed points
        and smoothed curve are mapped back through ``prepared``'s stored
        ``x_lo``/``x_hi``/``y_lo``/``y_hi`` bounds. Default ``False``
        (the package's own normalized diagnostic view, unchanged).
    log_y : bool, optional
        Log-scale the y-axis. Only meaningful with ``raw_axis=True`` and a
        strictly positive ``y``; ignored otherwise.
    x_label, y_label : str, optional
        Axis captions to use with ``raw_axis=True``, where the default
        "x (normalized)" / "y (scaled)" captions no longer apply. Ignored
        when ``raw_axis`` is ``False``.

    Returns
    -------
    str
        A complete SVG document: the curve with its knee (or an honest
        abstention state) and the evidence that backs it.
    """
    strings = _strings(language)
    config = config or RobustKneeConfig()

    if y is None:
        y = x
        x = np.arange(len(np.asarray(y).ravel()), dtype=float)

    try:
        prepared = prepare_curve(x, y, curve, direction, config)
    except Abstain:
        prepared = None

    inferred_curve = prepared.curve if prepared is not None else curve
    inferred_direction = prepared.direction if prepared is not None else direction
    result = robust_knee(x, y, curve=inferred_curve, direction=inferred_direction, config=config)

    p: List[str] = []
    if isinstance(result, ClearKnee) and prepared is not None:
        title_txt = strings["title_clear"]
        subtitle_txt = strings["subtitle_clear"].format(x=f"{result.knee_x:.3g}")
    else:
        title_txt = strings["title_abstain"]
        reason = getattr(result, "reason", "unknown")
        subtitle_txt = f'{strings["reason"]}: {reason}'
    desc_txt = f"{title_txt}. {subtitle_txt}"

    p.append(_svg_open("kn-title", "kn-desc"))
    p.append(f'<title id="kn-title">{escape(title_txt)}</title>')
    p.append(f'<desc id="kn-desc">{escape(desc_txt)}</desc>')
    p.append(_dark_mode_block())
    p.append(f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="{_BG}"/>')
    p.append(
        f'<text x="{_PL:.0f}" y="72" font-size="29" font-weight="700" '
        f'fill="{_INK}">{escape(title_txt)}</text>'
    )
    p.append(
        f'<text x="{_PL:.0f}" y="106" font-size="17" fill="{_SUBINK}">'
        f'{escape(subtitle_txt)}</text>'
    )

    use_raw = raw_axis and prepared is not None

    def sx(v: float) -> float:
        if use_raw:
            return _PL + (v - prepared.x_lo) / (prepared.x_hi - prepared.x_lo) * _PLOT_W
        return _sx(v)

    def sy(v: float) -> float:
        if use_raw:
            if log_y:
                lo = math.log10(max(prepared.y_lo, 1e-12))
                hi = math.log10(max(prepared.y_hi, lo + 1e-12))
                return _PB - (math.log10(max(v, 1e-12)) - lo) / (hi - lo) * _PLOT_H
            return _PB - (v - prepared.y_lo) / (prepared.y_hi - prepared.y_lo) * _PLOT_H
        return _sy(v)

    # --- gridlines: raw/log data-unit ticks when requested, else the
    # pipeline's own fixed 0..1 normalized square. ---
    if use_raw:
        y_ticks = (
            _log_ticks(prepared.y_lo, prepared.y_hi)
            if log_y
            else _linear_ticks(prepared.y_lo, prepared.y_hi)
        )
        x_ticks = _linear_ticks(prepared.x_lo, prepared.x_hi)
        y_tick_labels = [_fmt_tick(v) for v in y_ticks]
        x_tick_labels = [_fmt_tick(v) for v in x_ticks]
    else:
        y_ticks = x_ticks = (0.0, 0.25, 0.5, 0.75, 1.0)
        y_tick_labels = x_tick_labels = [f"{v:.2f}" for v in y_ticks]

    for gv, label in zip(y_ticks, y_tick_labels):
        gy = sy(gv)
        p.append(
            f'<line x1="{_PL:.1f}" y1="{gy:.1f}" x2="{_PR:.1f}" y2="{gy:.1f}" '
            f'stroke="{_HAIR}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{_PL - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_SUBINK}">'
            f'{escape(label)}</text>'
        )
    p.append(
        f'<line x1="{_PL:.1f}" y1="{_PB:.1f}" x2="{_PR:.1f}" y2="{_PB:.1f}" '
        f'stroke="{_INK}" stroke-width="1.5"/>'
    )
    for gv, label in zip(x_ticks, x_tick_labels):
        p.append(
            f'<text x="{sx(gv):.1f}" y="{_PB + 26:.1f}" text-anchor="middle" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_INK}">{escape(label)}</text>'
        )
    p.append(
        f'<text x="{(_PL + _PR) / 2:.1f}" y="{_PB + 56:.1f}" text-anchor="middle" '
        f'font-size="15" fill="{_INK}">'
        f'{escape(x_label if use_raw and x_label else strings["x_axis"])}</text>'
    )
    p.append(
        f'<text x="30" y="{(_PT + _PB) / 2:.1f}" text-anchor="middle" font-size="15" '
        f'fill="{_INK}" transform="rotate(-90 30 {(_PT + _PB) / 2:.1f})">'
        f'{escape(y_label if use_raw and y_label else strings["y_axis"])}</text>'
    )

    if prepared is None:
        # Data failed even the pre-normalisation shape/range check (e.g. zero
        # range, or a global shape the locator cannot fit) — no curve to draw,
        # but still show the abstention card for visual parity with the
        # normal abstain case rather than leaving a bare axes frame.
        _emit_abstain_card(p, strings, getattr(result, "reason", None))
        p.append("</svg>")
        return "".join(p)

    xn, yn = prepared.x_norm, prepared.y_scaled
    is_clear = isinstance(result, ClearKnee)

    if use_raw:
        x_disp = prepared.x_lo + xn * (prepared.x_hi - prepared.x_lo)
        y_disp = prepared.y_lo + yn * (prepared.y_hi - prepared.y_lo)
    else:
        x_disp, y_disp = xn, yn
    pts = [(sx(float(px)), sy(float(py))) for px, py in zip(x_disp, y_disp)]

    # --- raw points + smoothed curve ---
    for px, py in pts:
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{_MUTED}" opacity="0.55"/>')
    smoothed = smooth_curve(yn, max(3, prepared.n // 12))
    smoothed_disp = (
        prepared.y_lo + smoothed * (prepared.y_hi - prepared.y_lo) if use_raw else smoothed
    )
    spts = [(sx(float(px)), sy(float(py))) for px, py in zip(x_disp, smoothed_disp)]
    line = [f'M{_fmt(spts[0][0])},{_fmt(spts[0][1])}']
    line.append(_catmull_rom(spts))
    stroke = _CURVE if is_clear else _MUTED
    dash = '' if is_clear else ' stroke-dasharray="6 5"'
    p.append(
        f'<path d="{"".join(line)}" fill="none" stroke="{stroke}" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"{dash}/>'
    )

    if is_clear:
        knee_x_disp = result.knee_x if use_raw else result.knee_x_norm
        kx = sx(float(knee_x_disp))
        ky = sy(float(np.interp(knee_x_disp, x_disp, smoothed_disp)))

        # CI band (bootstrap 90% interval; ci90 is already in original x
        # units — used directly under raw_axis, or re-normalised through
        # the same [x_lo, x_hi] span the algorithm itself used otherwise).
        if use_raw:
            lo_px, hi_px = sx(float(result.ci90[0])), sx(float(result.ci90[1]))
        else:
            lo_norm = (result.ci90[0] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
            hi_norm = (result.ci90[1] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
            lo_px, hi_px = sx(lo_norm), sx(hi_norm)
        p.append(
            f'<rect x="{lo_px:.1f}" y="{_PT - 6:.1f}" width="{(hi_px - lo_px):.1f}" '
            f'height="{(_PB - _PT + 6):.1f}" fill="{_ACCENT}" fill-opacity="0.10"/>'
        )
        p.append(f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="15" fill="{_ACCENT}" opacity="0.16"/>')
        p.append(
            f'<circle cx="{kx:.1f}" cy="{ky:.1f}" r="8" fill="{_BG}" '
            f'stroke="{_ACCENT}" stroke-width="3"/>'
        )
        call_txt = f'{strings["knee_pill"]} · x = {result.knee_x:.3g}'
        pill_w = 168.0
        callout_x = min(kx + 90.0, _PR - pill_w / 2 - 8)
        callout_y = max(ky - 70.0, _PT + 24)
        p.append(
            f'<line x1="{kx + 10:.1f}" y1="{ky - 10:.1f}" '
            f'x2="{callout_x - pill_w / 2 + 8:.1f}" y2="{callout_y:.1f}" '
            f'stroke="{_ACCENT}" stroke-width="1.6"/>'
        )
        p.append(
            f'<rect x="{callout_x - pill_w / 2:.1f}" y="{callout_y - 17:.1f}" '
            f'width="{pill_w:.0f}" height="34" rx="17" fill="{_ACCENT}"/>'
        )
        p.append(
            f'<text x="{callout_x:.1f}" y="{callout_y + 5:.1f}" text-anchor="middle" '
            f'font-family="{_FONT_MONO}" font-size="15" font-weight="700" '
            f'fill="{_BG}">{escape(call_txt)}</text>'
        )

        # Fit-quality score, evaluated in the space the broken-line model
        # was actually fit in (the algorithm's own robustly normalized
        # square, never log-transformed): the confirmed model is a linear
        # fit against normalized y, the same one BIC and cross-validation
        # scored internally, so that is the only space in which "how far
        # is the fit from the data" is the model's own criterion rather
        # than a different one invented for display. log_y only ever
        # changes how the y-axis is drawn, never what was fit or scored;
        # scoring a linear fit's predictions against log(y) would compare
        # it to a curve shape it was never asked to match.
        design = _design_broken(xn, float(result.knee_x_norm))
        coef, _ = ols_rss(design, yn)
        yhat_n = design @ coef
        fit_score = _fit_quality_score(yn, yhat_n)

        _emit_legend(p, result, strings, fit_score)
    else:
        _emit_abstain_card(p, strings, getattr(result, "reason", None))

    p.append("</svg>")
    return "".join(p)


def _emit_legend(p: List[str], result: "ClearKnee", strings: dict, fit_score: float) -> None:
    """Append a compact evidence legend: detection probability, null p, slope
    contrast, BIC-derived posterior probability, and worst-case-normalized fit
    quality.

    A plain labelled-swatch legend rather than a chart-bearing card: each
    row pairs a small dot (colour-keyed to the main curve/knee hues) with a
    ``label: value`` line, so the numbers that back the point estimate read
    at a glance without a second, chart-within-a-chart panel to parse.

    Parameters
    ----------
    p : list of str
        The SVG fragment list being assembled; extended in place.
    result : ClearKnee
        The pipeline result carrying the evidence values.
    strings : dict
        Chrome-text strings for the active language.
    fit_score : float
        The worst-case-normalized fit-quality score from
        :func:`_fit_quality_score`, already evaluated by the caller in
        this figure's own display space.
    """
    lx, ly, lw = 566.0, 196.0, 372.0
    rows = [
        (_CURVE, f'{strings["detection_rate"]}: {result.detection_rate:.0%}'),
        (_CURVE_DEEP, f'{strings["null_p"]}: {result.null_p_value:.3g}'),
        (_ACCENT, f'{strings["slope_contrast"]}: {result.slope_contrast:.0%}'),
        (_MUTED, f'{strings["bic"]}: {_bic_posterior_probability(result.bic_improvement):.1%}'),
        (_GOOD, f'{strings["fit_quality"]}: {fit_score:.0%}'),
    ]
    row_h = 32.0
    lh = 24.0 + row_h * len(rows)
    p.append(
        f'<rect x="{lx:.0f}" y="{ly:.0f}" width="{lw:.0f}" height="{lh:.0f}" '
        f'rx="16" fill="#F5F5F7" stroke="{_HAIR}" stroke-width="1"/>'
    )
    p.append(
        f'<text x="{lx + 20:.0f}" y="{ly + 30:.0f}" font-size="14" '
        f'font-weight="700" fill="{_INK}">{escape(strings["inset_title"])}</text>'
    )
    for i, (colour, line) in enumerate(rows):
        row_y = ly + 52 + i * row_h
        p.append(f'<circle cx="{lx + 26:.1f}" cy="{row_y - 5:.1f}" r="5" fill="{colour}"/>')
        p.append(
            f'<text x="{lx + 42:.1f}" y="{row_y:.1f}" font-family="{_FONT_MONO}" '
            f'font-size="13" fill="{_INK}">{escape(line)}</text>'
        )


def _emit_abstain_card(p: List[str], strings: dict, reason: Optional[str]) -> None:
    """Append the "no clear knee" explanation card, wrapping the reason by hand."""
    ix, iy, iw = 566.0, 196.0, 372.0
    body = reason or strings["hint_abstain"]
    words = str(body).split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) > 40 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    ih = 90.0 + 22.0 * min(len(lines), 5)
    p.append(
        f'<rect x="{ix:.0f}" y="{iy:.0f}" width="{iw:.0f}" height="{ih:.0f}" '
        f'rx="16" fill="#F5F5F7" stroke="{_HAIR}" stroke-width="1"/>'
    )
    p.append(
        f'<text x="{ix + 20:.0f}" y="{iy + 34:.0f}" font-size="15" '
        f'font-weight="700" fill="{_INK}">{escape(strings["title_abstain"])}</text>'
    )
    for i, line in enumerate(lines[:5]):
        p.append(
            f'<text x="{ix + 20:.0f}" y="{iy + 62 + i * 22:.0f}" font-size="13.5" '
            f'fill="{_SUBINK}">{escape(line)}</text>'
        )


def plot_diagnostics(
    x,
    y=None,
    *,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config: Optional[RobustKneeConfig] = None,
    out: Optional[str] = None,
    language: str = "en",
    raw_axis: bool = False,
    log_y: bool = False,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """Render the robust-knee diagnostic SVG and optionally save it.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in :func:`elbow_helper.robust_knee`.
    curve, direction : str, optional
        Orientation; inferred from the data when omitted.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts forwarded to the pipeline.
    out : str, optional
        If given, save the SVG to this path (parent directories are created).
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.
    raw_axis : bool, optional
        Display in original data units instead of the pipeline's internal
        normalized ``[0, 1]`` square; see :func:`render_svg`.
    log_y : bool, optional
        Log-scale the y-axis (only meaningful with ``raw_axis=True``).
    x_label, y_label : str, optional
        Axis captions to use with ``raw_axis=True``.

    Returns
    -------
    str
        The complete SVG document (also written to ``out`` when given).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 50)
    >>> y = np.where(x <= 0.3, 3 * x, 0.9 + 0.1 * (x - 0.3))
    >>> svg = plot_diagnostics(x, y)
    >>> svg.startswith("<svg")
    True
    """
    svg = render_svg(
        x,
        y,
        curve=curve,
        direction=direction,
        config=config,
        language=language,
        raw_axis=raw_axis,
        log_y=log_y,
        x_label=x_label,
        y_label=y_label,
    )
    if out:
        oh.make_directory(oh.folder_name_ext(out)[0])
        Path(out).write_text(svg, encoding="utf-8")
        oh.info(f"[elbow-helper] saved diagnostics to {out}")
    return svg


def render_svg_multi(
    x,
    y=None,
    *,
    config: Optional[RobustKneesConfig] = None,
    language: str = "en",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """Build the robust-knees (plural) diagnostic as a standalone SVG string.

    Unlike :func:`render_svg`, this wraps :func:`elbow_helper.robust_knees`
    and marks zero or more breakpoints with dashed lines rather than a
    single knee with a confidence band: there is no continuous broken-line
    fit or BIC-derived evidence legend here, only the breakpoint locations
    :func:`elbow_helper.robust_knees` itself reports, each an independent
    (discontinuous) segment boundary rather than :class:`ClearKnee`'s
    continuous broken-line model.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in
        :func:`elbow_helper.robust_knees`.
    config : RobustKneesConfig, optional
        Search size and false-positive-control settings.
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.
    x_label, y_label : str, optional
        Axis captions; default to the generic normalized-style captions
        when omitted.

    Returns
    -------
    str
        A complete SVG document: the curve with every detected breakpoint
        marked by a dashed line (or an honest abstention state).
    """
    strings = _strings(language)
    config = config or RobustKneesConfig()

    if y is None:
        y = x
        x = np.arange(len(np.asarray(y).ravel()), dtype=float)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(x)
    x, y = x[order], y[order]

    result = robust_knees(x, y, config=config)
    is_valid = isinstance(result, Knees)
    knees = result.knees if is_valid else []

    if is_valid:
        if len(knees) == 0:
            title_txt = strings["title_multi_zero"]
        elif len(knees) == 1:
            title_txt = strings["title_multi_one"]
        else:
            title_txt = strings["title_multi_k"].format(k=len(knees))
        subtitle_txt = strings["subtitle_multi"]
    else:
        title_txt = strings["title_abstain"]
        subtitle_txt = f'{strings["reason"]}: {getattr(result, "reason", "unknown")}'
    desc_txt = f"{title_txt}. {subtitle_txt}"

    x_lo, x_hi = float(x.min()), float(x.max())
    y_lo, y_hi = float(y.min()), float(y.max())
    pad = (y_hi - y_lo) * 0.08 or 1.0
    y_lo, y_hi = y_lo - pad, y_hi + pad

    def sx(v: float) -> float:
        return _PL + (v - x_lo) / (x_hi - x_lo) * _PLOT_W

    def sy(v: float) -> float:
        return _PB - (v - y_lo) / (y_hi - y_lo) * _PLOT_H

    p: List[str] = []
    p.append(_svg_open("kn-title", "kn-desc"))
    p.append(f'<title id="kn-title">{escape(title_txt)}</title>')
    p.append(f'<desc id="kn-desc">{escape(desc_txt)}</desc>')
    p.append(_dark_mode_block())
    p.append(f'<rect width="{_WIDTH}" height="{_HEIGHT}" fill="{_BG}"/>')
    p.append(
        f'<text x="{_PL:.0f}" y="72" font-size="29" font-weight="700" '
        f'fill="{_INK}">{escape(title_txt)}</text>'
    )
    p.append(
        f'<text x="{_PL:.0f}" y="106" font-size="17" fill="{_SUBINK}">'
        f'{escape(subtitle_txt)}</text>'
    )

    x_ticks = _linear_ticks(x_lo, x_hi)
    y_ticks = _linear_ticks(y_lo, y_hi)
    for gv in y_ticks:
        gy = sy(gv)
        p.append(
            f'<line x1="{_PL:.1f}" y1="{gy:.1f}" x2="{_PR:.1f}" y2="{gy:.1f}" '
            f'stroke="{_HAIR}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{_PL - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_SUBINK}">'
            f'{escape(_fmt_tick(gv))}</text>'
        )
    p.append(
        f'<line x1="{_PL:.1f}" y1="{_PB:.1f}" x2="{_PR:.1f}" y2="{_PB:.1f}" '
        f'stroke="{_INK}" stroke-width="1.5"/>'
    )
    for gv in x_ticks:
        p.append(
            f'<text x="{sx(gv):.1f}" y="{_PB + 26:.1f}" text-anchor="middle" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_INK}">'
            f'{escape(_fmt_tick(gv))}</text>'
        )
    p.append(
        f'<text x="{(_PL + _PR) / 2:.1f}" y="{_PB + 56:.1f}" text-anchor="middle" '
        f'font-size="15" fill="{_INK}">{escape(x_label or strings["x_axis"])}</text>'
    )
    p.append(
        f'<text x="30" y="{(_PT + _PB) / 2:.1f}" text-anchor="middle" font-size="15" '
        f'fill="{_INK}" transform="rotate(-90 30 {(_PT + _PB) / 2:.1f})">'
        f'{escape(y_label or strings["y_axis"])}</text>'
    )

    if not is_valid:
        _emit_abstain_card(p, strings, getattr(result, "reason", None))
        p.append("</svg>")
        return "".join(p)

    pts = [(sx(float(px)), sy(float(py))) for px, py in zip(x, y)]
    for px, py in pts:
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{_MUTED}" opacity="0.55"/>')
    smoothed = smooth_curve(y, max(3, len(y) // 12))
    spts = [(sx(float(px)), sy(float(py))) for px, py in zip(x, smoothed)]
    line = [f'M{_fmt(spts[0][0])},{_fmt(spts[0][1])}']
    line.append(_catmull_rom(spts))
    p.append(
        f'<path d="{"".join(line)}" fill="none" stroke="{_CURVE}" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'
    )

    for kn in knees:
        kx = sx(float(kn.x))
        p.append(
            f'<line x1="{kx:.1f}" y1="{_PT:.1f}" x2="{kx:.1f}" y2="{_PB:.1f}" '
            f'stroke="{_ACCENT}" stroke-width="2" stroke-dasharray="6 5"/>'
        )
        p.append(
            f'<text x="{kx:.1f}" y="{_PT - 8:.1f}" text-anchor="middle" '
            f'font-family="{_FONT_MONO}" font-size="12" font-weight="700" '
            f'fill="{_ACCENT}">{escape(f"{kn.x:.3g}")}</text>'
        )

    p.append("</svg>")
    return "".join(p)


def plot_multi_diagnostics(
    x,
    y=None,
    *,
    config: Optional[RobustKneesConfig] = None,
    out: Optional[str] = None,
    language: str = "en",
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> str:
    """Render the robust-knees (plural) diagnostic SVG and optionally save it.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in
        :func:`elbow_helper.robust_knees`.
    config : RobustKneesConfig, optional
        Search size and false-positive-control settings.
    out : str, optional
        If given, save the SVG to this path (parent directories are created).
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.
    x_label, y_label : str, optional
        Axis captions; default to the generic normalized-style captions
        when omitted.

    Returns
    -------
    str
        The complete SVG document (also written to ``out`` when given).

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 150)
    >>> y = np.piecewise(
    ...     x, [x < 0.4, x >= 0.4], [lambda t: 2 * t, lambda t: 0.8 + 0.1 * (t - 0.4)]
    ... )
    >>> svg = plot_multi_diagnostics(x, y)
    >>> svg.startswith("<svg")
    True
    """
    svg = render_svg_multi(
        x, y, config=config, language=language, x_label=x_label, y_label=y_label
    )
    if out:
        oh.make_directory(oh.folder_name_ext(out)[0])
        Path(out).write_text(svg, encoding="utf-8")
        oh.info(f"[elbow-helper] saved diagnostics to {out}")
    return svg


def render_svg_panels(svgs: Sequence[str], *, gap: float = 24.0) -> str:
    """Lay out several already-rendered diagnostic SVGs side by side.

    Each panel is whatever complete, standalone SVG the caller already
    produced, typically one :func:`render_svg` call per noise level or
    scenario (or a mix of :func:`render_svg` / :func:`render_svg_multi`
    outputs), run and rendered independently: its own pipeline call, its
    own title/subtitle, its own legend or abstention card. This only
    nests those SVGs into one shared canvas via SVG's native
    nested-viewport element, left to right. Useful for a controlled
    side-by-side comparison, such as the same shape at two noise levels.

    Parameters
    ----------
    svgs : sequence of str
        One complete, standalone SVG document string per panel.
    gap : float, optional
        Horizontal gap, in pixels, between panels.

    Returns
    -------
    str
        A complete SVG document containing one nested panel per input SVG.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 60)
    >>> y = np.where(x < 0.5, 0.8 * x, 0.4 + 0.2 * (x - 0.5))
    >>> left = render_svg(x, y + 0.01)
    >>> right = render_svg(x, y + 0.2)
    >>> svg = render_svg_panels([left, right])
    >>> svg.count("<svg") == 3
    True
    """
    n = len(svgs)
    total_w = _WIDTH * n + gap * max(n - 1, 0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_w:.0f}" '
        f'height="{_HEIGHT}" viewBox="0 0 {total_w:.0f} {_HEIGHT}" '
        f'font-family="{_FONT}" role="img">'
    ]
    for i, inner in enumerate(svgs):
        body = inner[inner.index(">") + 1 : inner.rindex("</svg>")]
        x_off = i * (_WIDTH + gap)
        parts.append(
            f'<svg x="{x_off:.1f}" y="0" width="{_WIDTH}" height="{_HEIGHT}" '
            f'viewBox="0 0 {_WIDTH} {_HEIGHT}">{body}</svg>'
        )
    parts.append("</svg>")
    return "".join(parts)


def plot_diagnostics_panels(
    svgs: Sequence[str],
    *,
    out: Optional[str] = None,
    gap: float = 24.0,
) -> str:
    """Render :func:`render_svg_panels` and optionally save it.

    Parameters
    ----------
    svgs : sequence of str
        One complete, standalone SVG document string per panel; see
        :func:`render_svg_panels`.
    out : str, optional
        If given, save the SVG to this path (parent directories are created).
    gap : float, optional
        Horizontal gap, in pixels, between panels.

    Returns
    -------
    str
        The complete SVG document (also written to ``out`` when given).
    """
    svg = render_svg_panels(svgs, gap=gap)
    if out:
        oh.make_directory(oh.folder_name_ext(out)[0])
        Path(out).write_text(svg, encoding="utf-8")
        oh.info(f"[elbow-helper] saved diagnostics to {out}")
    return svg
