"""
generate_experiment_figures — the three new LIKELIHOOD-{fr,en}.tex figures,
as hand-authored SVG.

Reproduces, by direct Monte Carlo simulation, three asymptotic results
LIKELIHOOD-fr.tex / LIKELIHOOD-en.tex describe but never plot: the
exponential decay Cramer's theorem predicts, the Wald likelihood-ratio
process crossing an SPRT threshold, and the delta-method bias/variance of
the sample likelihood estimator against a direct Monte Carlo check.

No matplotlib, no seaborn, no Vega: every mark (axis, tick, line, point) is
placed by hand as SVG path/line/circle/text elements, the same convention
``sprezzature-figures``'s own ``make_<id>.py`` generators use (see
``make_residual.py`` for the reference pattern this file follows). numpy is
used only for the underlying simulation, never for drawing. House-style
tokens (palette, ink, secondary grey) are pulled from the sprezzature-figures
skill's ``_style``/``_svg`` modules when that skill is installed alongside
this repo, with a small self-contained fallback otherwise.

Each figure is written as an SVG, then rasterised to a matching PNG with
``rsvg-convert`` (librsvg) at 2x supersampling for crisp print embedding,
mirroring the PNG+SVG pair convention the three existing LIKELIHOOD figures
already follow.

Run directly: ``python figures/generate_experiment_figures.py``.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
_SKILL_CANDIDATES = [
    Path.home() / ".claude" / "skills" / "sprezzature-figures" / "scripts",
    Path.home() / ".opencode" / "skills" / "sprezzature-figures" / "scripts",
]
for _candidate in _SKILL_CANDIDATES:
    if (_candidate / "_style.py").is_file():
        sys.path.insert(0, str(_candidate))
        break

try:
    from _style import load_palette  # type: ignore
    from _svg import svg_open, xml_escape  # type: ignore
except ImportError:
    def load_palette(accessibility: str = "universal") -> Dict[str, str]:
        return {
            "Blue": "#007AFF", "Red": "#FF3B30", "Green": "#34C759",
            "Orange": "#FF9500", "Gray": "#8E8E93",
        }

    def svg_open(width: object, height: object, title_id: str, desc_id: str,
                 *, font_family: str = "Roboto, system-ui, sans-serif") -> str:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}" '
            f'font-family="{font_family}" role="img" '
            f'aria-labelledby="{title_id} {desc_id}">'
        )

    def xml_escape(text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

PALETTE = load_palette()
BLUE = PALETTE.get("Blue", "#007AFF")
RED = PALETTE.get("Red", "#FF3B30")
GREEN = PALETTE.get("Green", "#34C759")
ORANGE = PALETTE.get("Orange", "#FF9500")
INK = "#1D1D1F"
SUBTLE = "#6E6E73"
GRID = "#EEEEEE"
RULE = "#8E8E93"

_SUP = {"-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
        "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹"}


def pow10_label(k: int) -> str:
    """Return ``k``'s power of ten as a Unicode-superscript label ("10⁻²")."""
    if k == 0:
        return "1"
    if k == 1:
        return "10"
    return "10" + "".join(_SUP[c] for c in str(k))


LANG = {
    "fr": {
        "cramer_title": "La décroissance de Cramér : la moyenne se resserre vite",
        "cramer_sub": "p = (0,5 ; 0,3 ; 0,15 ; 0,05), écart toléré 0,1, 20 000 répliques par taille",
        "cramer_x": "taille d'échantillon n",
        "cramer_y": "Pr(|moyenne − cible| > 0,1)",
        "cramer_note": "presque une droite en échelle log : la vitesse que prédit Cramér",
        "cramer_floor": "sous le seuil détectable",
        "cramer_panel_a": "décroissance simulée, 7 tailles, 20 000 répliques",
        "cramer_panel_b": "convergence vers le taux exact I(z), 50 000 répliques",
        "cramer_rate_y": "taux empirique −ln(P̂)/n",
        "wald_title": "La martingale de Wald : deux trajectoires, deux verdicts",
        "wald_sub": "p₀ = 0,3, p₁ = 0,7, seuils du SPRT pour α = β = 0,05",
        "wald_x": "observation t",
        "wald_y": "somme cumulée du log-rapport",
        "wald_accept1": "accepter p₁",
        "wald_accept0": "accepter p₀",
        "wald_under0": "vraie loi = p₀",
        "wald_under1": "vraie loi = p₁",
        "conv_title": "Convergence de L̂ₙ : la simulation retrouve la méthode delta",
        "conv_sub_l": "biais de L̂ₙ, 100 000 répliques par taille",
        "conv_sub_r": "variance de L̂ₙ, 100 000 répliques par taille",
        "conv_x": "taille d'échantillon n",
        "conv_y_l": "biais empirique",
        "conv_y_r": "variance empirique",
        "conv_sim": "simulation",
        "conv_theory": "méthode delta",
    },
    "en": {
        "cramer_title": "Cramer's decay: the average tightens exponentially fast",
        "cramer_sub": "p=(0.5, 0.3, 0.15, 0.05), tolerance 0.1, 20,000 replicates per size",
        "cramer_x": "sample size n",
        "cramer_y": "Pr(|average − target| > 0.1)",
        "cramer_note": "near-straight on a log scale: the rate Cramer's theorem predicts",
        "cramer_floor": "below the detectable floor",
        "cramer_panel_a": "simulated decay, 7 sizes, 20,000 replicates",
        "cramer_panel_b": "convergence to the exact rate I(z), 50,000 replicates",
        "cramer_rate_y": "empirical rate −ln(P̂)/n",
        "wald_title": "Wald's martingale: two paths, two verdicts",
        "wald_sub": "p₀=0.3, p₁=0.7, SPRT thresholds for α=β=0.05",
        "wald_x": "observation t",
        "wald_y": "cumulative log-ratio sum",
        "wald_accept1": "accept p₁",
        "wald_accept0": "accept p₀",
        "wald_under0": "true law = p₀",
        "wald_under1": "true law = p₁",
        "conv_title": "Convergence of L̂ₙ: simulation matches the delta method",
        "conv_sub_l": "bias of L̂ₙ, 100,000 replicates per size",
        "conv_sub_r": "variance of L̂ₙ, 100,000 replicates per size",
        "conv_x": "sample size n",
        "conv_y_l": "empirical bias",
        "conv_y_r": "empirical variance",
        "conv_sim": "simulation",
        "conv_theory": "delta method",
    },
}


def _write_and_rasterize(svg: str, name: str, out_width_px: int) -> None:
    svg_path = HERE / f"{name}.svg"
    png_path = HERE / f"{name}.png"
    svg_path.write_text(svg, encoding="utf-8")
    if shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "-w", str(out_width_px * 2), "--keep-aspect-ratio",
             "-o", str(png_path), str(svg_path)],
            check=True,
        )
    else:
        print(f"WARNING: rsvg-convert not found; {png_path.name} not written", file=sys.stderr)
        return
    print(f"wrote {svg_path.name} + {png_path.name}")


# ----------------------------------------------------------------------
# Figure 1 -- Cramer's large-deviations decay, plus its exact rate
# ----------------------------------------------------------------------
_CAT_P = np.array([0.5, 0.3, 0.15, 0.05])
_CAT_LOGP = np.log(_CAT_P)
_CAT_MU = float((_CAT_P * _CAT_LOGP).sum())


def rate_function(a: float, iters: int = 200) -> Tuple[float, float]:
    """I(a) = sup_theta [theta*a - Lambda(theta)] via ternary search (Lambda convex)."""
    def obj(theta: float) -> float:
        Lambda = math.log(float(np.sum(_CAT_P * np.exp(theta * _CAT_LOGP))))
        return theta * a - Lambda

    lo, hi = -30.0, 30.0
    for _ in range(iters):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if obj(m1) < obj(m2):
            lo = m1
        else:
            hi = m2
    theta_star = (lo + hi) / 2
    return obj(theta_star), theta_star


def simulate_cramer(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    ns = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    reps = 20000
    eps = 0.1
    fracs = np.empty(ns.shape, dtype=float)
    for i, n in enumerate(ns):
        draws = rng.choice(4, size=(reps, n), p=_CAT_P)
        zbar = _CAT_LOGP[draws].mean(axis=1)
        fracs[i] = np.mean(np.abs(zbar - _CAT_MU) > eps)
    return ns, fracs, 1.0 / reps


def simulate_rate_convergence(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
    """Empirical -ln(P_hat)/n at growing n, next to the exact Legendre-transform rate."""
    eps = 0.1
    i_hi, _ = rate_function(_CAT_MU + eps)
    i_lo, _ = rate_function(_CAT_MU - eps)
    i_min = min(i_hi, i_lo)

    ns = np.array([10, 20, 30, 50, 100, 150, 200, 300, 400, 500])
    reps = 50000
    rate_hat = np.empty(ns.shape, dtype=float)
    for i, n in enumerate(ns):
        draws = rng.choice(4, size=(reps, n), p=_CAT_P)
        zbar = _CAT_LOGP[draws].mean(axis=1)
        frac = np.mean(np.abs(zbar - _CAT_MU) > eps)
        rate_hat[i] = -math.log(frac) / n if frac > 0 else np.nan
    return ns, rate_hat, i_min


def build_cramer_svg(lang: str, rng: np.random.Generator) -> str:
    t = LANG[lang]
    ns, fracs, floor = simulate_cramer(rng)
    fracs_plot = np.maximum(fracs, floor)
    ns_r, rate_hat, i_min = simulate_rate_convergence(rng)

    width, height = 1150, 600
    m_top, m_bottom = 150, 130
    panel_w = 400
    gap = 130
    px_left = 190
    px_right = px_left + panel_w + gap
    ph = height - m_top - m_bottom
    py = m_top

    parts: List[str] = []
    parts.append(svg_open(width, height, "cr-title", "cr-desc"))
    parts.append(f"<title id=\"cr-title\">{xml_escape(t['cramer_title'])}</title>")
    parts.append(f"<desc id=\"cr-desc\">{xml_escape(t['cramer_title'] + '. ' + t['cramer_sub'])}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(f'<text x="60" y="52" font-size="24" font-weight="700" fill="{INK}">'
                 f'{xml_escape(t["cramer_title"])}</text>')
    parts.append(f'<text x="60" y="82" font-size="16" fill="{SUBTLE}">'
                 f'{xml_escape(t["cramer_sub"])}</text>')

    # ---- left panel: P(|mean - target| > 0.1) vs n, log-log --------
    log_n_min, log_n_max = math.log10(8), math.log10(12000)
    y_top_k, y_bot_k = 0, -5

    def sx_l(n: float) -> float:
        return px_left + (math.log10(n) - log_n_min) / (log_n_max - log_n_min) * panel_w

    def sy_l(v: float) -> float:
        k = math.log10(v)
        return py + (y_top_k - k) / (y_top_k - y_bot_k) * ph

    parts.append(f'<text x="{px_left:.1f}" y="{py-18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
                 f'{xml_escape(t["cramer_panel_a"])}</text>')
    for k in range(y_bot_k, y_top_k + 1):
        gy = sy_l(10.0**k)
        parts.append(f'<line x1="{px_left:.1f}" y1="{gy:.1f}" x2="{px_left+panel_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px_left-12:.1f}" y="{gy+5:.1f}" font-size="13.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">'
                     f'{pow10_label(k)}</text>')
    ax_bottom_l = py + ph
    parts.append(f'<line x1="{px_left:.1f}" y1="{ax_bottom_l:.1f}" x2="{px_left+panel_w:.1f}" y2="{ax_bottom_l:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px_left:.1f}" y1="{py:.1f}" x2="{px_left:.1f}" y2="{ax_bottom_l:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    for n in ns:
        gx = sx_l(float(n))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom_l:.1f}" x2="{gx:.1f}" y2="{ax_bottom_l+5:.1f}" '
                     f'stroke="{INK}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom_l+24:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{px_left+panel_w/2:.1f}" y="{ax_bottom_l+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">{xml_escape(t["cramer_x"])}</text>')
    ytx, yty = px_left - 78, py + ph / 2
    parts.append(f'<text x="{ytx:.1f}" y="{yty:.1f}" font-size="14" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx:.1f} {yty:.1f})">{xml_escape(t["cramer_y"])}</text>')
    pts = [(sx_l(float(n)), sy_l(v)) for n, v in zip(ns, fracs_plot)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2.6" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
    for (x, y), raw in zip(pts, fracs):
        if raw < floor:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="#FFFFFF" '
                         f'stroke="{BLUE}" stroke-width="2.2"/>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{BLUE}" '
                         f'stroke="#FFFFFF" stroke-width="1.2"/>')
    if (fracs < floor).any():
        parts.append(f'<text x="{px_left+panel_w-8:.1f}" y="{py+18:.1f}" font-size="11.5" fill="{SUBTLE}" '
                     f'text-anchor="end">○ {xml_escape(t["cramer_floor"])}</text>')

    # ---- right panel: empirical rate -ln(P_hat)/n vs n, converging to I_min ----
    log_nr_min, log_nr_max = math.log10(8), math.log10(650)
    r_max = float(np.nanmax(rate_hat)) * 1.08
    r_min = 0.0

    def sx_r(n: float) -> float:
        return px_right + (math.log10(n) - log_nr_min) / (log_nr_max - log_nr_min) * panel_w

    def sy_r(v: float) -> float:
        return py + (r_max - v) / (r_max - r_min) * ph

    parts.append(f'<text x="{px_right:.1f}" y="{py-18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
                 f'{xml_escape(t["cramer_panel_b"])}</text>')
    r_ticks = [round(x, 3) for x in np.linspace(0, r_max, 5)]
    for v in r_ticks:
        gy = sy_r(v)
        parts.append(f'<line x1="{px_right:.1f}" y1="{gy:.1f}" x2="{px_right+panel_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px_right-12:.1f}" y="{gy+5:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.3f}</text>')
    ax_bottom_r = py + ph
    parts.append(f'<line x1="{px_right:.1f}" y1="{ax_bottom_r:.1f}" x2="{px_right+panel_w:.1f}" y2="{ax_bottom_r:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px_right:.1f}" y1="{py:.1f}" x2="{px_right:.1f}" y2="{ax_bottom_r:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    for n in [10, 30, 100, 300, 500]:
        gx = sx_r(float(n))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom_r:.1f}" x2="{gx:.1f}" y2="{ax_bottom_r+5:.1f}" '
                     f'stroke="{INK}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom_r+24:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{px_right+panel_w/2:.1f}" y="{ax_bottom_r+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">{xml_escape(t["cramer_x"])}</text>')
    ytx2, yty2 = px_right - 78, py + ph / 2
    parts.append(f'<text x="{ytx2:.1f}" y="{yty2:.1f}" font-size="14" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx2:.1f} {yty2:.1f})">{xml_escape(t["cramer_rate_y"])}</text>')

    i_min_y = sy_r(i_min)
    parts.append(f'<line x1="{px_right:.1f}" y1="{i_min_y:.1f}" x2="{px_right+panel_w:.1f}" y2="{i_min_y:.1f}" '
                 f'stroke="{RED}" stroke-width="1.8" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{px_right+panel_w-6:.1f}" y="{i_min_y-8:.1f}" font-size="13" fill="{RED}" '
                 f'text-anchor="end">I(z) = {i_min:.4f}</text>')

    valid = ~np.isnan(rate_hat)
    pts_r = [(sx_r(float(n)), sy_r(v)) for n, v in zip(ns_r[valid], rate_hat[valid])]
    d_r = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts_r)
    parts.append(f'<path d="{d_r}" fill="none" stroke="{ORANGE}" stroke-width="2.6" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y in pts_r:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{ORANGE}" '
                     f'stroke="#FFFFFF" stroke-width="1.2"/>')

    parts.append("</svg>")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Figure 2 -- Wald's likelihood-ratio martingale / SPRT
# ----------------------------------------------------------------------
def simulate_wald(rng: np.random.Generator) -> Tuple[List[np.ndarray], List[np.ndarray], float, float]:
    p0, p1 = 0.3, 0.7
    alpha = beta = 0.05
    a_hi = math.log((1 - beta) / alpha)
    a_lo = math.log(beta / (1 - alpha))
    step1 = math.log(p1 / p0)
    step0 = math.log((1 - p1) / (1 - p0))
    t_max = 100

    def simulate(true_p: float) -> np.ndarray:
        ys = rng.random(t_max) < true_p
        increments = np.where(ys, step1, step0)
        path = np.cumsum(increments)
        hit = np.where((path >= a_hi) | (path <= a_lo))[0]
        stop = int(hit[0]) if hit.size else t_max - 1
        return path[: stop + 1]

    under1 = [simulate(p1) for _ in range(3)]
    under0 = [simulate(p0) for _ in range(2)]
    return under0, under1, a_hi, a_lo


def build_wald_svg(lang: str, rng: np.random.Generator) -> str:
    t = LANG[lang]
    under0, under1, a_hi, a_lo = simulate_wald(rng)
    t_max_used = max(len(p) for p in under0 + under1)

    width, height = 1000, 580
    m_left, m_right, m_top, m_bottom = 140, 50, 150, 110
    plot_x, plot_y = m_left, m_top
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    x_max = max(10, int(math.ceil((t_max_used + 2) / 5.0) * 5))
    y_max = a_hi + 0.8
    y_min = a_lo - 0.8

    def sx(v: float) -> float:
        return plot_x + v / x_max * plot_w

    def sy(v: float) -> float:
        return plot_y + (y_max - v) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(svg_open(width, height, "wd-title", "wd-desc"))
    parts.append(f"<title id=\"wd-title\">{xml_escape(t['wald_title'])}</title>")
    parts.append(f"<desc id=\"wd-desc\">{xml_escape(t['wald_title'] + '. ' + t['wald_sub'])}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')

    parts.append(f'<text x="{m_left}" y="52" font-size="24" font-weight="700" fill="{INK}">'
                 f'{xml_escape(t["wald_title"])}</text>')
    parts.append(f'<text x="{m_left}" y="82" font-size="16" fill="{SUBTLE}">'
                 f'{xml_escape(t["wald_sub"])}</text>')

    y_ticks = sorted(set([round(a_lo, 2), 0.0, round(a_hi, 2)]))
    for v in y_ticks:
        gy = sy(v)
        parts.append(f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x+plot_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{plot_x-14:.1f}" y="{gy+5:.1f}" font-size="15" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:g}</text>')

    ax_bottom = plot_y + plot_h
    parts.append(f'<line x1="{plot_x:.1f}" y1="{ax_bottom:.1f}" x2="{plot_x+plot_w:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.6"/>')
    parts.append(f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.6"/>')

    step = 5 if x_max <= 40 else (10 if x_max <= 80 else 20)
    for xv in range(0, x_max + 1, step):
        gx = sx(float(xv))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom+6:.1f}" '
                     f'stroke="{INK}" stroke-width="1.4"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom+28:.1f}" font-size="15" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{xv}</text>')

    parts.append(f'<text x="{plot_x+plot_w/2:.1f}" y="{ax_bottom+62:.1f}" font-size="17" fill="{INK}" '
                 f'text-anchor="middle">{xml_escape(t["wald_x"])}</text>')
    ytx, yty = 40, plot_y + plot_h / 2
    parts.append(f'<text x="{ytx}" y="{yty:.1f}" font-size="16" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx} {yty:.1f})">{xml_escape(t["wald_y"])}</text>')

    # threshold rules
    for v, label, anchor_y in ((a_hi, t["wald_accept1"], "bottom"), (a_lo, t["wald_accept0"], "top")):
        gy = sy(v)
        parts.append(f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x+plot_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{RULE}" stroke-width="1.6" stroke-dasharray="7 5"/>')
        dy = -8 if anchor_y == "bottom" else 20
        parts.append(f'<text x="{plot_x+plot_w-8:.1f}" y="{gy+dy:.1f}" font-size="15" fill="{SUBTLE}" '
                     f'text-anchor="end">{xml_escape(label)}</text>')

    def draw_path(path: np.ndarray, color: str) -> None:
        pts = [(sx(0.0), sy(0.0))] + [(sx(float(i + 1)), sy(float(v))) for i, v in enumerate(path)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.2" '
                     f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>')
        ex, ey = pts[-1]
        parts.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="6" fill="{color}" '
                     f'stroke="#FFFFFF" stroke-width="1.4"/>')

    for path in under1:
        draw_path(path, GREEN)
    for path in under0:
        draw_path(path, RED)

    parts.append(f'<circle cx="{plot_x+22:.1f}" cy="{plot_y+22:.1f}" r="6" fill="{GREEN}"/>')
    parts.append(f'<text x="{plot_x+36:.1f}" y="{plot_y+27:.1f}" font-size="15" fill="{INK}">'
                 f'{xml_escape(t["wald_under1"])}</text>')
    parts.append(f'<circle cx="{plot_x+22:.1f}" cy="{plot_y+46:.1f}" r="6" fill="{RED}"/>')
    parts.append(f'<text x="{plot_x+36:.1f}" y="{plot_y+51:.1f}" font-size="15" fill="{INK}">'
                 f'{xml_escape(t["wald_under0"])}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Figure 3 -- bias/variance convergence of L_hat_n
# ----------------------------------------------------------------------
def simulate_convergence(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    l_inf = 1.0 / math.sqrt(2 * math.pi * math.e)
    sigma_z2 = 0.5
    ns = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    reps = 100_000
    emp_bias = np.empty(ns.shape, dtype=float)
    emp_var = np.empty(ns.shape, dtype=float)
    for i, n in enumerate(ns):
        y = rng.standard_normal(size=(reps, n))
        z = -0.5 * math.log(2 * math.pi) - 0.5 * y**2
        lhat = np.exp(z.mean(axis=1))
        emp_bias[i] = lhat.mean() - l_inf
        emp_var[i] = lhat.var()
    return ns, emp_bias, emp_var, l_inf, sigma_z2


def _loglog_panel(parts: List[str], px: float, py: float, pw: float, ph: float,
                   ns: np.ndarray, emp: np.ndarray, theory: np.ndarray,
                   title: str, xlabel: str, ylabel: str, sim_label: str, theory_label: str,
                   color: str) -> None:
    log_n_min, log_n_max = math.log10(4), math.log10(1300)
    all_vals = np.concatenate([emp, theory])
    y_top_k = int(math.ceil(math.log10(all_vals.max())))
    y_bot_k = int(math.floor(math.log10(all_vals.min())))

    def sx(n: float) -> float:
        return px + (math.log10(n) - log_n_min) / (log_n_max - log_n_min) * pw

    def sy(v: float) -> float:
        k = math.log10(v)
        return py + (y_top_k - k) / (y_top_k - y_bot_k) * ph

    parts.append(f'<text x="{px:.1f}" y="{py-18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
                 f'{xml_escape(title)}</text>')

    for k in range(y_bot_k, y_top_k + 1):
        gy = sy(10.0**k)
        parts.append(f'<line x1="{px:.1f}" y1="{gy:.1f}" x2="{px+pw:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px-12:.1f}" y="{gy+5:.1f}" font-size="13.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">'
                     f'{pow10_label(k)}</text>')

    ax_bottom = py + ph
    parts.append(f'<line x1="{px:.1f}" y1="{ax_bottom:.1f}" x2="{px+pw:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    for n in ns:
        gx = sx(float(n))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom+5:.1f}" '
                     f'stroke="{INK}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom+24:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{px+pw/2:.1f}" y="{ax_bottom+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">{xml_escape(xlabel)}</text>')
    ytx, yty = px - 78, py + ph / 2
    parts.append(f'<text x="{ytx:.1f}" y="{yty:.1f}" font-size="14" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx:.1f} {yty:.1f})">{xml_escape(ylabel)}</text>')

    tpts = [(sx(float(n)), sy(v)) for n, v in zip(ns, theory)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in tpts)
    parts.append(f'<path d="{d}" fill="none" stroke="{SUBTLE}" stroke-width="2.4"/>')

    for n, v in zip(ns, emp):
        x, y = sx(float(n)), sy(v)
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" '
                     f'stroke="#FFFFFF" stroke-width="1.2"/>')

    # Bottom-left corner is always empty for a decreasing log-log curve (low n
    # never lands at a low value), unlike top-left where the highest points sit.
    lx, ly = px + 14, ax_bottom - 46
    parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx+22:.1f}" y2="{ly:.1f}" '
                 f'stroke="{SUBTLE}" stroke-width="2.4"/>')
    parts.append(f'<text x="{lx+30:.1f}" y="{ly+4:.1f}" font-size="13" fill="{INK}">{xml_escape(theory_label)}</text>')
    parts.append(f'<circle cx="{lx+11:.1f}" cy="{ly+22:.1f}" r="5.5" fill="{color}"/>')
    parts.append(f'<text x="{lx+30:.1f}" y="{ly+26:.1f}" font-size="13" fill="{INK}">{xml_escape(sim_label)}</text>')


def build_convergence_svg(lang: str, rng: np.random.Generator) -> str:
    t = LANG[lang]
    ns, emp_bias, emp_var, l_inf, sigma_z2 = simulate_convergence(rng)
    theory_bias = l_inf * sigma_z2 / (2 * ns)
    theory_var = (l_inf**2) * sigma_z2 / ns

    width, height = 1150, 600
    m_top, m_bottom = 150, 130
    panel_w = 400
    gap = 130
    px_left = 190
    px_right = px_left + panel_w + gap
    ph = height - m_top - m_bottom
    py = m_top

    parts: List[str] = []
    parts.append(svg_open(width, height, "cv-title", "cv-desc"))
    parts.append(f"<title id=\"cv-title\">{xml_escape(t['conv_title'])}</title>")
    parts.append(f"<desc id=\"cv-desc\">{xml_escape(t['conv_title'])}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(f'<text x="60" y="52" font-size="24" font-weight="700" fill="{INK}">'
                 f'{xml_escape(t["conv_title"])}</text>')

    _loglog_panel(parts, px_left, py, panel_w, ph, ns, np.abs(emp_bias), theory_bias,
                  t["conv_sub_l"], t["conv_x"], t["conv_y_l"], t["conv_sim"], t["conv_theory"], BLUE)
    _loglog_panel(parts, px_right, py, panel_w, ph, ns, emp_var, theory_var,
                  t["conv_sub_r"], t["conv_x"], t["conv_y_r"], t["conv_sim"], t["conv_theory"], ORANGE)

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    for lang in ("fr", "en"):
        rng = np.random.default_rng(20260815)
        _write_and_rasterize(build_cramer_svg(lang, rng), f"largedev_cramer_{lang}", 1000)
        rng = np.random.default_rng(20260815)
        _write_and_rasterize(build_wald_svg(lang, rng), f"martingale_wald_{lang}", 1000)
        rng = np.random.default_rng(20260815)
        _write_and_rasterize(build_convergence_svg(lang, rng), f"estimator_convergence_{lang}", 1150)


if __name__ == "__main__":
    main()
