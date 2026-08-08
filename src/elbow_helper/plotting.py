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
null-model p-value, slope contrast, and a BIC-derived posterior model
probability) in a compact legend. When the evidence is too weak, the figure says so plainly
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

from .config import RobustKneeConfig
from .pipeline import ClearKnee, robust_knee
from .preprocessing import Abstain, prepare_curve
from .smoothing import smooth_curve

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
        "bic": "model evidence",
        "x_axis": "x (normalized)",
        "y_axis": "y (scaled)",
        "data_label": "data",
        "smoothed_label": "smoothed",
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
        "bic": "évidence du modèle",
        "x_axis": "x (normalisé)",
        "y_axis": "y (mis à l'échelle)",
        "data_label": "données",
        "smoothed_label": "lissé",
    },
}


def _strings(language: str) -> dict:
    return _STRINGS.get(language, _STRINGS["en"])


def _fmt(v: float) -> str:
    """Compact float formatting for SVG path data (one decimal, no trailing zero)."""
    return f"{v:.1f}".rstrip("0").rstrip(".")


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
    section 14). That gives an approximate Bayes factor
    ``BF ≈ exp(bic_improvement / 2)`` — literally a likelihood ratio, since BIC
    is built from ``-2 ln(L)`` — and, under equal priors, a posterior model
    probability ``BF / (1 + BF)``, which is exactly the logistic/sigmoid of
    ``bic_improvement / 2``. A probability is bounded in ``[0, 1]`` (maximum
    entropy ``ln(2)`` nats at ``p = 0.5``, the point of total uncertainty
    between the two models), so it reads the same way regardless of curve
    length or noise scale: "99.9%" is unambiguous where "282.4" is not.

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


def render_svg(
    x,
    y=None,
    *,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config: Optional[RobustKneeConfig] = None,
    language: str = "en",
) -> str:
    """Build the robust-knee diagnostic as a complete, standalone SVG string.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in :func:`elbow_helper.robust_knee`.
    curve, direction : str, optional
        Kneedle orientation; inferred from the data when omitted.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts forwarded to the pipeline.
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.

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

    # --- gridlines (fixed 0..1 domain — both axes are already normalized) ---
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = _sy(frac)
        p.append(
            f'<line x1="{_PL:.1f}" y1="{gy:.1f}" x2="{_PR:.1f}" y2="{gy:.1f}" '
            f'stroke="{_HAIR}" stroke-width="1"/>'
        )
        p.append(
            f'<text x="{_PL - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_SUBINK}">'
            f'{frac:.2f}</text>'
        )
    p.append(
        f'<line x1="{_PL:.1f}" y1="{_PB:.1f}" x2="{_PR:.1f}" y2="{_PB:.1f}" '
        f'stroke="{_INK}" stroke-width="1.5"/>'
    )
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        p.append(
            f'<text x="{_sx(frac):.1f}" y="{_PB + 26:.1f}" text-anchor="middle" '
            f'font-family="{_FONT_MONO}" font-size="13" fill="{_INK}">{frac:.2f}</text>'
        )
    p.append(
        f'<text x="{(_PL + _PR) / 2:.1f}" y="{_PB + 56:.1f}" text-anchor="middle" '
        f'font-size="15" fill="{_INK}">{escape(strings["x_axis"])}</text>'
    )
    p.append(
        f'<text x="30" y="{(_PT + _PB) / 2:.1f}" text-anchor="middle" font-size="15" '
        f'fill="{_INK}" transform="rotate(-90 30 {(_PT + _PB) / 2:.1f})">'
        f'{escape(strings["y_axis"])}</text>'
    )

    if prepared is None:
        # Data failed even the pre-normalisation shape/range check (e.g. zero
        # range, or a global shape Kneedle cannot fit) — no curve to draw,
        # but still show the abstention card for visual parity with the
        # normal abstain case rather than leaving a bare axes frame.
        _emit_abstain_card(p, strings, getattr(result, "reason", None))
        p.append("</svg>")
        return "".join(p)

    xn, yn = prepared.x_norm, prepared.y_scaled
    is_clear = isinstance(result, ClearKnee)
    pts = [(_sx(float(px)), _sy(float(py))) for px, py in zip(xn, yn)]

    # --- raw points + smoothed curve ---
    for px, py in pts:
        p.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="{_MUTED}" opacity="0.55"/>')
    smoothed = smooth_curve(yn, max(3, prepared.n // 12))
    spts = [(_sx(float(px)), _sy(float(py))) for px, py in zip(xn, smoothed)]
    line = [f'M{_fmt(spts[0][0])},{_fmt(spts[0][1])}']
    line.append(_catmull_rom(spts))
    stroke = _CURVE if is_clear else _MUTED
    dash = '' if is_clear else ' stroke-dasharray="6 5"'
    p.append(
        f'<path d="{"".join(line)}" fill="none" stroke="{stroke}" '
        f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round"{dash}/>'
    )

    if is_clear:
        kx = _sx(float(result.knee_x_norm))
        ky = _sy(float(np.interp(result.knee_x_norm, xn, smoothed)))

        # CI band (bootstrap 90% interval, denormalised to knee_x then
        # re-normalised for pixel placement — ci90 is already in original x
        # units, so map it back through the same [x_lo, x_hi] span).
        lo_norm = (result.ci90[0] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
        hi_norm = (result.ci90[1] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
        lo_px, hi_px = _sx(lo_norm), _sx(hi_norm)
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
        _emit_legend(p, result, strings)
    else:
        _emit_abstain_card(p, strings, getattr(result, "reason", None))

    p.append("</svg>")
    return "".join(p)


def _emit_legend(p: List[str], result: "ClearKnee", strings: dict) -> None:
    """Append a compact evidence legend: detection probability, null p, slope contrast, BIC gain.

    A plain labelled-swatch legend rather than a chart-bearing card: each
    row pairs a small dot (colour-keyed to the main curve/knee hues) with a
    ``label: value`` line, so the four numbers that back the point estimate
    read at a glance without a second, chart-within-a-chart panel to parse.

    Parameters
    ----------
    p : list of str
        The SVG fragment list being assembled; extended in place.
    result : ClearKnee
        The pipeline result carrying the four evidence values.
    strings : dict
        Chrome-text strings for the active language.
    """
    lx, ly, lw = 566.0, 196.0, 372.0
    rows = [
        (_CURVE, f'{strings["detection_rate"]}: {result.detection_rate:.0%}'),
        (_CURVE_DEEP, f'{strings["null_p"]}: {result.null_p_value:.3g}'),
        (_ACCENT, f'{strings["slope_contrast"]}: {result.slope_contrast:.2f}'),
        (_MUTED, f'{strings["bic"]}: {_bic_posterior_probability(result.bic_improvement):.1%}'),
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
) -> str:
    """Render the robust-knee diagnostic SVG and optionally save it.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in :func:`elbow_helper.robust_knee`.
    curve, direction : str, optional
        Kneedle orientation; inferred from the data when omitted.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts forwarded to the pipeline.
    out : str, optional
        If given, save the SVG to this path (parent directories are created).
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.

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
    svg = render_svg(x, y, curve=curve, direction=direction, config=config, language=language)
    if out:
        oh.make_directory(oh.folder_name_ext(out)[0])
        Path(out).write_text(svg, encoding="utf-8")
        oh.info(f"[elbow-helper] saved diagnostics to {out}")
    return svg
