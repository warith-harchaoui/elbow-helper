"""
generate_likelihood_intro_figures -- real code behind the three
LIKELIHOOD-en.tex figures that predate this repo's figure-generation
scripts and had no reproducible source anywhere in the project:
``temperature_sweep_en.png``, ``classification_density_en.png``, and
``regression_density_en.png``. All three are explicitly synthetic,
hand-picked illustrative examples (the note says so directly: "the
p=(0.5,0.3,0.15,0.05) example," "45 simulated examples," "synthetic
regression data"), so being synthetic is not the gap; having no source
code to regenerate them was. This script closes that gap with the same
hand-rolled-SVG house style as this project's other generators: no
matplotlib, no seaborn, no Vega, numpy only for the underlying math.

Every number this script prints matches the ones already written into
LIKELIHOOD-en.tex's prose and captions (the geometric mean, the
arithmetic mean, the entropy floor and ceiling), so regenerating these
figures does not require touching the text.

Run directly: ``python figures/generate_likelihood_intro_figures.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_experiment_figures import (  # noqa: E402
    BLUE,
    RED,
    GREEN,
    ORANGE,
    INK,
    SUBTLE,
    GRID,
    svg_open,
    _write_and_rasterize,
    pow10_label,
)

CAT_P = np.array([0.5, 0.3, 0.15, 0.05])


# ----------------------------------------------------------------------
# Figure 1 -- temperature sweep
# ----------------------------------------------------------------------
def exp_neg_H_beta(beta: float, p: np.ndarray = CAT_P) -> float:
    logp = np.log(p)
    if abs(beta - 1.0) < 1e-9:
        return float(np.exp((p * logp).sum()))
    Z = float(np.sum(p**beta))
    H_beta = math.log(Z) / (1 - beta)
    return math.exp(-H_beta)


def build_temperature_sweep_svg() -> str:
    logp = np.log(CAT_P)
    geo_mean = float(np.exp((CAT_P * logp).sum()))
    arith_mean = float((CAT_P**2).sum())
    floor, ceil_ = 1.0 / len(CAT_P), float(CAT_P.max())

    width, height = 1150, 600
    m_top, m_bottom = 150, 130
    panel_w = 400
    gap = 130
    px_left = 190
    px_right = px_left + panel_w + gap
    ph = height - m_top - m_bottom
    py = m_top

    parts: List[str] = []
    parts.append(svg_open(width, height, "ts-title", "ts-desc"))
    parts.append(
        '<title id="ts-title">Jensen\'s inequality and the temperature sweep</title>'
    )
    parts.append(
        '<desc id="ts-desc">Bar chart of p with its geometric and arithmetic means, '
        "and exp(-H_beta(p)) swept across beta from its floor to its ceiling.</desc>"
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(
        f'<text x="60" y="52" font-size="22" font-weight="700" fill="{INK}">'
        f"Jensen&#8217;s inequality and the temperature sweep</text>"
    )
    parts.append(
        f'<text x="60" y="82" font-size="15" fill="{SUBTLE}">'
        f"p = (0.5, 0.3, 0.15, 0.05), K = 4 categories</text>"
    )

    # ---- left panel: bar chart with mean lines ----
    y_max = 0.62
    bx = px_left

    def sy_l(v: float) -> float:
        return py + (y_max - v) / y_max * ph

    parts.append(
        f'<text x="{bx:.1f}" y="{py - 18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
        f"bars vs. their two means</text>"
    )
    for v in (0.0, 0.2, 0.4, 0.6):
        gy = sy_l(v)
        parts.append(
            f'<line x1="{bx:.1f}" y1="{gy:.1f}" x2="{bx + panel_w:.1f}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{bx - 12:.1f}" y="{gy + 5:.1f}" font-size="13" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.1f}</text>'
        )
    ax_bottom_l = py + ph
    parts.append(
        f'<line x1="{bx:.1f}" y1="{ax_bottom_l:.1f}" x2="{bx + panel_w:.1f}" y2="{ax_bottom_l:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{bx:.1f}" y1="{py:.1f}" x2="{bx:.1f}" y2="{ax_bottom_l:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )

    bar_w = 62
    slot = panel_w / len(CAT_P)
    for i, pk in enumerate(CAT_P):
        cx = bx + slot * (i + 0.5)
        top = sy_l(float(pk))
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{top:.1f}" width="{bar_w:.1f}" '
            f'height="{ax_bottom_l - top:.1f}" fill="{BLUE}" fill-opacity="0.75" '
            f'stroke="{BLUE}" stroke-width="1.3"/>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{ax_bottom_l + 22:.1f}" font-size="13" fill="{INK}" '
            f'text-anchor="middle">k={i + 1}</text>'
        )
    parts.append(
        f'<text x="{bx + panel_w / 2:.1f}" y="{ax_bottom_l + 48:.1f}" font-size="14" fill="{INK}" '
        f'text-anchor="middle">category</text>'
    )

    gm_y, am_y = sy_l(geo_mean), sy_l(arith_mean)
    parts.append(
        f'<line x1="{bx:.1f}" y1="{gm_y:.1f}" x2="{bx + panel_w:.1f}" y2="{gm_y:.1f}" '
        f'stroke="{RED}" stroke-width="2" stroke-dasharray="7 4"/>'
    )
    parts.append(
        f'<text x="{bx + panel_w - 4:.1f}" y="{gm_y - 7:.1f}" font-size="12.5" fill="{RED}" '
        f'text-anchor="end">geometric mean {geo_mean:.3f}</text>'
    )
    parts.append(
        f'<line x1="{bx:.1f}" y1="{am_y:.1f}" x2="{bx + panel_w:.1f}" y2="{am_y:.1f}" '
        f'stroke="{ORANGE}" stroke-width="2" stroke-dasharray="7 4"/>'
    )
    parts.append(
        f'<text x="{bx + panel_w - 4:.1f}" y="{am_y - 7:.1f}" font-size="12.5" fill="{ORANGE}" '
        f'text-anchor="end">arithmetic mean {arith_mean:.3f}</text>'
    )

    # ---- right panel: beta sweep ----
    betas = np.logspace(-3, 3, 200)
    vals = np.array([exp_neg_H_beta(float(b)) for b in betas])
    log_b_min, log_b_max = -3, 3
    y2_min, y2_max = 0.22, 0.53

    def sx_r(b: float) -> float:
        return (
            px_right + (math.log10(b) - log_b_min) / (log_b_max - log_b_min) * panel_w
        )

    def sy_r(v: float) -> float:
        return py + (y2_max - v) / (y2_max - y2_min) * ph

    parts.append(
        f'<text x="{px_right:.1f}" y="{py - 18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
        f"exp(&#8722;H&#946;(p)) swept over &#946;</text>"
    )
    for v in (0.25, 0.3, 0.35, 0.4, 0.45, 0.5):
        gy = sy_r(v)
        parts.append(
            f'<line x1="{px_right:.1f}" y1="{gy:.1f}" x2="{px_right + panel_w:.1f}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{px_right - 12:.1f}" y="{gy + 5:.1f}" font-size="12.5" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.2f}</text>'
        )
    ax_bottom_r = py + ph
    parts.append(
        f'<line x1="{px_right:.1f}" y1="{ax_bottom_r:.1f}" x2="{px_right + panel_w:.1f}" y2="{ax_bottom_r:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{px_right:.1f}" y1="{py:.1f}" x2="{px_right:.1f}" y2="{ax_bottom_r:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    for k in range(log_b_min, log_b_max + 1):
        gx = sx_r(10.0**k)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{ax_bottom_r:.1f}" x2="{gx:.1f}" y2="{ax_bottom_r + 5:.1f}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{ax_bottom_r + 24:.1f}" font-size="12.5" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">'
            f"{pow10_label(k)}</text>"
        )
    parts.append(
        f'<text x="{px_right + panel_w / 2:.1f}" y="{ax_bottom_r + 50:.1f}" font-size="14" fill="{INK}" '
        f'text-anchor="middle">&#946; (hot &#8592; &#8594; cold)</text>'
    )

    pts = [(sx_r(float(b)), sy_r(float(v))) for b, v in zip(betas, vals)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>')
    b1x, b1y = sx_r(1.0), sy_r(geo_mean)
    parts.append(
        f'<circle cx="{b1x:.1f}" cy="{b1y:.1f}" r="6" fill="{RED}" stroke="#FFFFFF" stroke-width="1.4"/>'
    )
    parts.append(
        f'<text x="{b1x + 10:.1f}" y="{b1y - 10:.1f}" font-size="12.5" fill="{RED}">&#946;=1</text>'
    )
    parts.append(
        f'<text x="{px_right + 8:.1f}" y="{sy_r(floor) - 8:.1f}" font-size="12" fill="{SUBTLE}">'
        f"floor 1/K={floor:.2f}</text>"
    )
    parts.append(
        f'<text x="{px_right + 8:.1f}" y="{sy_r(ceil_) + 18:.1f}" font-size="12" fill="{SUBTLE}">'
        f"ceiling max p={ceil_:.2f}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


def simulate_classification(
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centroids = np.array([[0.0, 0.0], [3.2, 0.3], [1.5, 2.6]])
    covs = [
        np.array([[1.1, 0.3], [0.3, 1.0]]) * 0.9,
        np.array([[1.0, -0.15], [-0.15, 0.95]]) * 0.9,
        np.array([[0.9, -0.2], [-0.2, 1.2]]) * 0.9,
    ]
    n_per = 15
    tau = 1.3
    pts, labels = [], []
    for k, (c, cov) in enumerate(zip(centroids, covs)):
        pts.append(rng.multivariate_normal(c, cov, size=n_per))
        labels.append(np.full(n_per, k))
    pts = np.vstack(pts)
    labels = np.concatenate(labels)
    d2 = ((pts[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    logits = -d2 / (2 * tau**2)
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    true_probs = probs[np.arange(len(pts)), labels]
    return pts, labels, true_probs, centroids


def build_classification_density_svg(rng: np.random.Generator) -> str:
    pts, labels, true_probs, centroids = simulate_classification(rng)
    geo_mean = float(np.exp(np.mean(np.log(true_probs))))
    colors = [BLUE, ORANGE, GREEN]

    width, height = 1150, 600
    m_top, m_bottom = 150, 130
    panel_w = 400
    gap = 130
    px_left = 190
    px_right = px_left + panel_w + gap
    ph = height - m_top - m_bottom
    py = m_top

    parts: List[str] = []
    parts.append(svg_open(width, height, "cd-title", "cd-desc"))
    parts.append(
        '<title id="cd-title">Classification confidence, geometric mean of the true-class probabilities</title>'
    )
    parts.append(
        '<desc id="cd-desc">45 simulated points from 3 overlapping classes, dot size the '
        "true-class probability, and those same probabilities sorted with their geometric mean marked.</desc>"
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(
        f'<text x="60" y="52" font-size="22" font-weight="700" fill="{INK}">'
        f"Classification confidence, geometrically averaged</text>"
    )
    parts.append(
        f'<text x="60" y="82" font-size="15" fill="{SUBTLE}">'
        f"45 points, 3 overlapping classes, distance-to-centroid softmax model</text>"
    )

    # ---- left panel: 2D scatter, isotropic mapping ----
    all_x = pts[:, 0]
    all_y = pts[:, 1]
    x_min, x_max = all_x.min() - 0.9, all_x.max() + 0.9
    y_min, y_max = all_y.min() - 0.9, all_y.max() + 0.9
    data_w, data_h = x_max - x_min, y_max - y_min
    scale = min(panel_w / data_w, ph / data_h)
    plot_w_used, plot_h_used = data_w * scale, data_h * scale
    ox = px_left + (panel_w - plot_w_used) / 2
    oy = py + (ph - plot_h_used) / 2

    def sx(v: float) -> float:
        return ox + (v - x_min) * scale

    def sy(v: float) -> float:
        return oy + plot_h_used - (v - y_min) * scale

    parts.append(
        f'<text x="{px_left:.1f}" y="{py - 18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
        f"45 points, dot size = confidence</text>"
    )
    parts.append(
        f'<rect x="{px_left:.1f}" y="{py:.1f}" width="{panel_w}" height="{ph}" '
        f'fill="none" stroke="{GRID}" stroke-width="1.2"/>'
    )

    for c, col in zip(centroids, colors):
        parts.append(
            f'<circle cx="{sx(c[0]):.1f}" cy="{sy(c[1]):.1f}" r="{1.35 * scale:.1f}" '
            f'fill="none" stroke="{col}" stroke-width="1.6" stroke-dasharray="4 3"/>'
        )

    r_min, r_max = 3.0, 9.0
    for (x, y), lab, prob in zip(pts, labels, true_probs):
        r = r_min + (r_max - r_min) * float(prob)
        col = colors[int(lab)]
        parts.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{r:.1f}" fill="{col}" '
            f'fill-opacity="0.8" stroke="#FFFFFF" stroke-width="1"/>'
        )

    ly = py + ph + 40
    for i, (col, name) in enumerate(zip(colors, ["class 1", "class 2", "class 3"])):
        lx = px_left + i * 130
        parts.append(f'<circle cx="{lx:.1f}" cy="{ly:.1f}" r="6" fill="{col}"/>')
        parts.append(
            f'<text x="{lx + 14:.1f}" y="{ly + 4:.1f}" font-size="13" fill="{INK}">{name}</text>'
        )

    # ---- right panel: sorted probabilities ----
    sorted_probs = np.sort(true_probs)
    y2_min, y2_max = 0.0, 1.0

    def sx2(i: float) -> float:
        return px_right + (i + 0.5) / len(sorted_probs) * panel_w

    def sy2(v: float) -> float:
        return py + (y2_max - v) / (y2_max - y2_min) * ph

    parts.append(
        f'<text x="{px_right:.1f}" y="{py - 18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
        f"those 45 probabilities, sorted</text>"
    )
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = sy2(v)
        parts.append(
            f'<line x1="{px_right:.1f}" y1="{gy:.1f}" x2="{px_right + panel_w:.1f}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{px_right - 12:.1f}" y="{gy + 5:.1f}" font-size="12.5" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.2f}</text>'
        )
    ax_bottom = py + ph
    parts.append(
        f'<line x1="{px_right:.1f}" y1="{ax_bottom:.1f}" x2="{px_right + panel_w:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{px_right:.1f}" y1="{py:.1f}" x2="{px_right:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<text x="{px_right + panel_w / 2:.1f}" y="{ax_bottom + 30:.1f}" font-size="13.5" fill="{INK}" '
        f'text-anchor="middle">points, sorted low to high</text>'
    )

    bw = panel_w / len(sorted_probs) * 0.7
    for i, v in enumerate(sorted_probs):
        cx = sx2(float(i))
        top = sy2(float(v))
        parts.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{top:.1f}" width="{bw:.1f}" height="{ax_bottom - top:.1f}" '
            f'fill="{BLUE}" fill-opacity="0.75"/>'
        )

    gm_y = sy2(geo_mean)
    parts.append(
        f'<line x1="{px_right:.1f}" y1="{gm_y:.1f}" x2="{px_right + panel_w:.1f}" y2="{gm_y:.1f}" '
        f'stroke="{RED}" stroke-width="2.2" stroke-dasharray="7 4"/>'
    )
    parts.append(
        f'<text x="{px_right + 8:.1f}" y="{gm_y - 9:.1f}" font-size="13" fill="{RED}">'
        f"geometric mean {geo_mean:.3f}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


def gaussian_pdf(y: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    return (1.0 / (math.sqrt(2 * math.pi) * sigma)) * np.exp(
        -0.5 * ((y - mean) / sigma) ** 2
    )


def simulate_regression(rng: np.random.Generator, n: int = 24) -> dict:
    x = rng.uniform(-3, 3, n)
    sigma = 0.9
    eps = rng.normal(0, sigma, n)
    y = 2.0 + 0.5 * x + eps
    A = np.vstack([np.ones(n), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - (a + b * x)
    mse = float(np.mean(resid**2))
    sigma_hat = math.sqrt(mse)
    heights = gaussian_pdf(y, a + b * x, sigma_hat)
    geo_mean = float(np.exp(np.mean(np.log(heights))))
    return dict(
        x=x,
        y=y,
        a=float(a),
        b=float(b),
        sigma_hat=sigma_hat,
        mse=mse,
        geo_mean=geo_mean,
    )


def build_regression_density_svg(rng: np.random.Generator) -> str:
    d = simulate_regression(rng)
    x, y, a, b, sigma_hat = d["x"], d["y"], d["a"], d["b"], d["sigma_hat"]

    width, height = 1050, 600
    m_left, m_right, m_top, m_bottom = 130, 70, 150, 110
    px, py = m_left, m_top
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    order = np.argsort(x)
    picks = [order[int(round(q * (len(order) - 1)))] for q in (0.08, 0.38, 0.62, 0.92)]
    bell_tops = [a + b * float(x[i]) + 3.2 * sigma_hat for i in picks]
    bell_bots = [a + b * float(x[i]) - 3.2 * sigma_hat for i in picks]

    x_min, x_max = float(x.min()) - 0.6, float(x.max()) + 1.6
    y_min = min(float(y.min()) - 1.0, min(bell_bots) - 0.3)
    y_max = max(float(y.max()) + 1.0, max(bell_tops) + 0.3)

    def sx(v: float) -> float:
        return px + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return py + (y_max - v) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(svg_open(width, height, "rd-title", "rd-desc"))
    parts.append(
        '<title id="rd-title">The likelihood as a geometric mean of curve heights</title>'
    )
    parts.append(
        '<desc id="rd-desc">Synthetic regression data with the fitted line and four '
        "conditional-density curves, each marked at the height of the value actually observed.</desc>"
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(
        f'<text x="{m_left}" y="52" font-size="22" font-weight="700" fill="{INK}">'
        f"The likelihood as a geometric mean of curve heights</text>"
    )
    parts.append(
        f'<text x="{m_left}" y="82" font-size="15" fill="{SUBTLE}">'
        f"y = 2 + 0.5x + &#949;, &#949; ~ N(0, 0.9&#178;), fitted by least squares</text>"
    )

    for v in np.linspace(
        math.ceil(y_min),
        math.floor(y_max),
        int(math.floor(y_max) - math.ceil(y_min)) + 1,
    ):
        gy = sy(float(v))
        parts.append(
            f'<line x1="{px:.1f}" y1="{gy:.1f}" x2="{px + plot_w:.1f}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{px - 12:.1f}" y="{gy + 5:.1f}" font-size="13" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.0f}</text>'
        )
    ax_bottom = py + plot_h
    parts.append(
        f'<line x1="{px:.1f}" y1="{ax_bottom:.1f}" x2="{px + plot_w:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    parts.append(
        f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.5"/>'
    )
    for v in range(math.ceil(float(x.min())) - 1, math.floor(float(x.max())) + 2):
        if v < x_min or v > float(x.max()) + 0.4:
            continue
        gx = sx(float(v))
        parts.append(
            f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom + 5:.1f}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{ax_bottom + 24:.1f}" font-size="13" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{v}</text>'
        )
    parts.append(
        f'<text x="{px + plot_w / 2 - 40:.1f}" y="{ax_bottom + 48:.1f}" font-size="14" fill="{INK}" '
        f'text-anchor="middle">x</text>'
    )

    # fitted line, across the real data range only
    lx0, lx1 = float(x.min()) - 0.3, float(x.max()) + 0.3
    parts.append(
        f'<line x1="{sx(lx0):.1f}" y1="{sy(a + b * lx0):.1f}" x2="{sx(lx1):.1f}" y2="{sy(a + b * lx1):.1f}" '
        f'stroke="{INK}" stroke-width="2"/>'
    )

    # every point, small
    for xi, yi in zip(x, y):
        parts.append(
            f'<circle cx="{sx(float(xi)):.1f}" cy="{sy(float(yi)):.1f}" r="3.4" '
            f'fill="{SUBTLE}" fill-opacity="0.55"/>'
        )

    # four selected points get a conditional-density bell curve
    bell_px = 78.0
    max_dens = gaussian_pdf(np.array([0.0]), 0.0, sigma_hat)[0]
    bell_scale = bell_px / max_dens

    for idx in picks:
        x0, y0 = float(x[idx]), float(y[idx])
        mean0 = a + b * x0
        y_grid = np.linspace(mean0 - 3.2 * sigma_hat, mean0 + 3.2 * sigma_hat, 50)
        dens = gaussian_pdf(y_grid, mean0, sigma_hat)
        curve_pts = [
            (sx(x0) + d_ * bell_scale, sy(float(yy))) for yy, d_ in zip(y_grid, dens)
        ]
        base_top = (sx(x0), sy(float(y_grid[0])))
        base_bot = (sx(x0), sy(float(y_grid[-1])))
        path_d = (
            f"M {base_top[0]:.1f} {base_top[1]:.1f} "
            + " L ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in curve_pts)
            + f" L {base_bot[0]:.1f} {base_bot[1]:.1f} Z"
        )
        parts.append(
            f'<path d="{path_d}" fill="{BLUE}" fill-opacity="0.16" stroke="{BLUE}" stroke-width="1.6"/>'
        )
        parts.append(
            f'<line x1="{sx(x0):.1f}" y1="{sy(float(y_grid[0])):.1f}" x2="{sx(x0):.1f}" '
            f'y2="{sy(float(y_grid[-1])):.1f}" stroke="{SUBTLE}" stroke-width="1"/>'
        )
        dot_dens = gaussian_pdf(np.array([y0]), mean0, sigma_hat)[0]
        dot_x = sx(x0) + dot_dens * bell_scale
        parts.append(
            f'<line x1="{sx(x0):.1f}" y1="{sy(y0):.1f}" x2="{dot_x:.1f}" y2="{sy(y0):.1f}" '
            f'stroke="{RED}" stroke-width="1.3" stroke-dasharray="3 3"/>'
        )
        parts.append(
            f'<circle cx="{dot_x:.1f}" cy="{sy(y0):.1f}" r="5.5" fill="{RED}" '
            f'stroke="#FFFFFF" stroke-width="1.2"/>'
        )

    parts.append(
        f'<text x="{px + 16:.1f}" y="{py + 16:.1f}" font-size="13" fill="{SUBTLE}">'
        f"L = geometric mean of every height &#8776; {d['geo_mean']:.3f}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    print(
        "temperature sweep: geo_mean={:.4f} arith_mean={:.4f} floor={:.3f} ceiling={:.3f}".format(
            float(np.exp((CAT_P * np.log(CAT_P)).sum())),
            float((CAT_P**2).sum()),
            1.0 / len(CAT_P),
            float(CAT_P.max()),
        )
    )
    _write_and_rasterize(build_temperature_sweep_svg(), "temperature_sweep_en", 1150)

    rng = np.random.default_rng(7)
    svg = build_classification_density_svg(rng)
    _write_and_rasterize(svg, "classification_density_en", 1150)

    rng = np.random.default_rng(3)
    d = simulate_regression(rng)
    print(
        f"regression: a={d['a']:.3f} b={d['b']:.3f} mse={d['mse']:.3f} L={d['geo_mean']:.3f}"
    )
    rng = np.random.default_rng(3)
    svg = build_regression_density_svg(rng)
    _write_and_rasterize(svg, "regression_density_en", 1050)


if __name__ == "__main__":
    main()
