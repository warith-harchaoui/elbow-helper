"""
generate_gp_experiment -- a from-scratch Gaussian process regression, as
hand-authored SVG, for LIKELIHOOD-en.tex's Gaussian-process extension.

A Gaussian process puts a probability distribution directly over functions
rather than committing to one f(x;theta): for any finite set of x points,
the corresponding function values are jointly Gaussian, with covariance set
by a kernel. Fitting the kernel's own hyperparameters (how far apart two x
values can be before the function is allowed to look different, how much
noise separates the data from the curve) is done here by grid search on the
exact log marginal likelihood this note builds throughout,
ln p(y | X, theta) = -1/2 y^T K^-1 y - 1/2 ln|K| - n/2 ln(2*pi), the
multivariate-Gaussian generalization of every earlier section's scalar
Gaussian log-likelihood.

Real data: a 50-point subsample of the California housing dataset (Pace and
Barry, 1997; shipped with scikit-learn), one feature (median block income)
against the target (median house value), kept to one dimension and a small
n so the O(n^3) Cholesky solve stays exact and cheap and the posterior
band is easy to see by eye. No GPy, no GPflow, no scikit-learn
GaussianProcessRegressor: the kernel, the Cholesky solve, the marginal
likelihood, and the hyperparameter search are all plain numpy, the same
from-scratch discipline as every other model in this note.

No matplotlib, no seaborn, no Vega for drawing: every mark is placed by
hand as SVG, importing house-style tokens and primitives from
generate_experiment_figures.py as the other generators in this directory
do.

Run directly: ``python figures/generate_gp_experiment.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_experiment_figures import (  # noqa: E402
    BLUE, RED, ORANGE, INK, SUBTLE, GRID,
    svg_open, xml_escape, _write_and_rasterize,
)


# ----------------------------------------------------------------------
# Data and from-scratch GP
# ----------------------------------------------------------------------
def load_subset(n: int = 50, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    medinc = data.data[:, 0]  # median block income, $10,000s
    y = data.target  # median house value, $100,000s

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    x, yy = medinc[idx], y[idx]
    order = np.argsort(x)
    return x[order], yy[order]


def kernel(a: np.ndarray, b: np.ndarray, sigma_f2: float, ell: float) -> np.ndarray:
    """Squared-exponential (RBF) kernel: covariance between f(a) and f(b),
    decaying with squared distance, scaled by sigma_f2 and shaped by ell."""
    d2 = (a[:, None] - b[None, :]) ** 2
    return sigma_f2 * np.exp(-0.5 * d2 / ell**2)


def log_marginal_likelihood(xs: np.ndarray, y: np.ndarray, sigma_f2: float,
                             ell: float, sigma_n2: float) -> float:
    n = len(y)
    K = kernel(xs, xs, sigma_f2, ell) + sigma_n2 * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        return -np.inf
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return float(-0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * math.log(2 * math.pi))


def fit_gp(x: np.ndarray, y: np.ndarray) -> Dict[str, object]:
    """Type-II maximum likelihood by grid search: sigma_f2 is fixed to the
    sample variance of y (the standard, data-driven default for the signal
    scale), and (ell, sigma_n2) are chosen to maximize the log marginal
    likelihood above, profiling out sigma_n2 at each ell to trace the
    marginal-likelihood-vs-lengthscale curve the figure below plots."""
    mu_x, sd_x = float(x.mean()), float(x.std())
    xs = (x - mu_x) / sd_x
    sigma_f2 = float(np.var(y))

    ells = np.geomspace(0.05, 5.0, 60)
    sigma_n2s = np.geomspace(0.01, 2.0, 50)

    profile_lml = np.empty(len(ells))
    best = (-np.inf, ells[0], sigma_n2s[0])
    for i, ell in enumerate(ells):
        best_for_ell = -np.inf
        for sn2 in sigma_n2s:
            lml = log_marginal_likelihood(xs, y, sigma_f2, ell, sn2)
            if lml > best_for_ell:
                best_for_ell = lml
            if lml > best[0]:
                best = (lml, ell, sn2)
        profile_lml[i] = best_for_ell

    return {
        "mu_x": mu_x, "sd_x": sd_x, "sigma_f2": sigma_f2,
        "ells": ells, "profile_lml": profile_lml,
        "best_lml": best[0], "ell_star": best[1], "sigma_n2_star": best[2],
    }


def predict(x: np.ndarray, y: np.ndarray, fit: Dict[str, object],
            x_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu_x, sd_x = fit["mu_x"], fit["sd_x"]
    sigma_f2, ell, sigma_n2 = fit["sigma_f2"], fit["ell_star"], fit["sigma_n2_star"]
    xs = (x - mu_x) / sd_x
    xg = (x_grid - mu_x) / sd_x
    n = len(y)

    K = kernel(xs, xs, sigma_f2, ell) + sigma_n2 * np.eye(n)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))

    K_s = kernel(xs, xg, sigma_f2, ell)
    mu_star = K_s.T @ alpha
    v = np.linalg.solve(L, K_s)
    var_star = np.maximum(sigma_f2 - np.sum(v**2, axis=0), 1e-12)
    return mu_star, np.sqrt(var_star)


# ----------------------------------------------------------------------
# Figure 1 -- posterior mean and uncertainty band
# ----------------------------------------------------------------------
def build_posterior_svg(x: np.ndarray, y: np.ndarray, fit: Dict[str, object]) -> str:
    x_grid = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    mu_star, std_star = predict(x, y, fit, x_grid)

    width, height = 1050, 620
    m_left, m_right, m_top, m_bottom = 130, 70, 150, 110
    px, py = m_left, m_top
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    x_min, x_max = float(x_grid.min()), float(x_grid.max())
    y_lo_band = mu_star - 2.0 * std_star
    y_hi_band = mu_star + 2.0 * std_star
    y_min = min(float(y.min()), float(y_lo_band.min())) - 0.4
    y_max = max(float(y.max()), float(y_hi_band.max())) + 0.4

    def sx(v: float) -> float:
        return px + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return py + (y_max - v) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(svg_open(width, height, "gp-title", "gp-desc"))
    title = "California housing: a Gaussian process instead of one fitted curve"
    sub = (f"50 real blocks, median income vs. median house value; "
           f"ℓ* ≈ {fit['ell_star']:.2f}, σₙ² ≈ "
           f"{fit['sigma_n2_star']:.2f} chosen by maximizing the log marginal likelihood")
    parts.append(f"<title id=\"gp-title\">{xml_escape(title)}</title>")
    parts.append(f"<desc id=\"gp-desc\">{xml_escape(title + '. ' + sub)}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(f'<text x="{m_left}" y="52" font-size="22" font-weight="700" fill="{INK}">'
                 f'{xml_escape(title)}</text>')
    parts.append(f'<text x="{m_left}" y="82" font-size="15" fill="{SUBTLE}">{xml_escape(sub)}</text>')

    y_ticks = np.linspace(math.ceil(y_min * 2) / 2, math.floor(y_max * 2) / 2, 6)
    for v in y_ticks:
        gy = sy(float(v))
        parts.append(f'<line x1="{px:.1f}" y1="{gy:.1f}" x2="{px+plot_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px-12:.1f}" y="{gy+5:.1f}" font-size="13" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.1f}</text>')

    ax_bottom = py + plot_h
    parts.append(f'<line x1="{px:.1f}" y1="{ax_bottom:.1f}" x2="{px+plot_w:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    x_ticks = np.linspace(math.ceil(x_min), math.floor(x_max), 6)
    for v in x_ticks:
        gx = sx(float(v))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom+5:.1f}" '
                     f'stroke="{INK}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom+24:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{v:.0f}</text>')
    parts.append(f'<text x="{px+plot_w/2:.1f}" y="{ax_bottom+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">median block income ($10,000s)</text>')
    ytx, yty = px - 78, py + plot_h / 2
    parts.append(f'<text x="{ytx:.1f}" y="{yty:.1f}" font-size="14" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx:.1f} {yty:.1f})">median house value ($100,000s)</text>')

    # shaded +/-2 std band: forward along the top, backward along the bottom
    top_pts = [(sx(float(v)), sy(float(m))) for v, m in zip(x_grid, y_hi_band)]
    bot_pts = [(sx(float(v)), sy(float(m))) for v, m in zip(x_grid, y_lo_band)]
    band_d = ("M " + " L ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in top_pts)
              + " L " + " L ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in reversed(bot_pts)) + " Z")
    parts.append(f'<path d="{band_d}" fill="{BLUE}" fill-opacity="0.16"/>')

    mean_pts = [(sx(float(v)), sy(float(m))) for v, m in zip(x_grid, mu_star)]
    mean_d = "M " + " L ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in mean_pts)
    parts.append(f'<path d="{mean_d}" fill="none" stroke="{BLUE}" stroke-width="2.6"/>')

    for xv, yv in zip(x, y):
        parts.append(f'<circle cx="{sx(float(xv)):.1f}" cy="{sy(float(yv)):.1f}" r="4.5" '
                     f'fill="{INK}" fill-opacity="0.75"/>')

    lx, ly = px + 16, py + 16
    parts.append(f'<rect x="{lx-8:.1f}" y="{ly-14:.1f}" width="230" height="52" '
                 f'fill="#FFFFFF" fill-opacity="0.92" stroke="{GRID}" stroke-width="1"/>')
    parts.append(f'<circle cx="{lx+4:.1f}" cy="{ly:.1f}" r="4.5" fill="{INK}" fill-opacity="0.75"/>')
    parts.append(f'<text x="{lx+16:.1f}" y="{ly+4:.1f}" font-size="12.5" fill="{INK}">50 real blocks</text>')
    parts.append(f'<line x1="{lx-2:.1f}" y1="{ly+22:.1f}" x2="{lx+22:.1f}" y2="{ly+22:.1f}" '
                 f'stroke="{BLUE}" stroke-width="2.6"/>')
    parts.append(f'<text x="{lx+30:.1f}" y="{ly+26:.1f}" font-size="12.5" fill="{INK}">posterior mean</text>')
    parts.append(f'<rect x="{lx-2:.1f}" y="{ly+32:.1f}" width="24" height="10" '
                 f'fill="{BLUE}" fill-opacity="0.16"/>')
    parts.append(f'<text x="{lx+30:.1f}" y="{ly+41:.1f}" font-size="12.5" fill="{INK}">±2σ band</text>')

    parts.append("</svg>")
    return "\n".join(parts)


# ----------------------------------------------------------------------
# Figure 2 -- log marginal likelihood vs lengthscale
# ----------------------------------------------------------------------
def build_lml_svg(fit: Dict[str, object]) -> str:
    ells, profile = fit["ells"], fit["profile_lml"]

    width, height = 1050, 600
    m_left, m_right, m_top, m_bottom = 130, 70, 150, 110
    px, py = m_left, m_top
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    log_x_min, log_x_max = math.log10(float(ells.min())), math.log10(float(ells.max()))
    y_min, y_max = float(profile.min()) - 2.0, float(profile.max()) + 2.0

    def sx(v: float) -> float:
        return px + (math.log10(v) - log_x_min) / (log_x_max - log_x_min) * plot_w

    def sy(v: float) -> float:
        return py + (y_max - v) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(svg_open(width, height, "lml-title", "lml-desc"))
    title = "The marginal likelihood, profiled over the kernel's lengthscale"
    sub = "same 50 blocks; each point profiles out σₙ² at that ℓ by grid search"
    parts.append(f"<title id=\"lml-title\">{xml_escape(title)}</title>")
    parts.append(f"<desc id=\"lml-desc\">{xml_escape(title + '. ' + sub)}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(f'<text x="{m_left}" y="52" font-size="22" font-weight="700" fill="{INK}">'
                 f'{xml_escape(title)}</text>')
    parts.append(f'<text x="{m_left}" y="82" font-size="15" fill="{SUBTLE}">{xml_escape(sub)}</text>')

    y_ticks = np.linspace(math.ceil(y_min / 5) * 5, math.floor(y_max / 5) * 5, 6)
    for v in y_ticks:
        gy = sy(float(v))
        parts.append(f'<line x1="{px:.1f}" y1="{gy:.1f}" x2="{px+plot_w:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px-12:.1f}" y="{gy+5:.1f}" font-size="13" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.0f}</text>')

    ax_bottom = py + plot_h
    parts.append(f'<line x1="{px:.1f}" y1="{ax_bottom:.1f}" x2="{px+plot_w:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    for v in (0.1, 0.3, 1.0, 3.0):
        if ells.min() <= v <= ells.max():
            gx = sx(v)
            parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom+5:.1f}" '
                         f'stroke="{INK}" stroke-width="1.2"/>')
            parts.append(f'<text x="{gx:.1f}" y="{ax_bottom+24:.1f}" font-size="12.5" '
                         f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{v:g}</text>')
    parts.append(f'<text x="{px+plot_w/2:.1f}" y="{ax_bottom+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">lengthscale ℓ (standardized x units, log scale)</text>')
    ytx, yty = px - 90, py + plot_h / 2
    parts.append(f'<text x="{ytx:.1f}" y="{yty:.1f}" font-size="14" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx:.1f} {yty:.1f})">log marginal likelihood</text>')

    pts = [(sx(float(e)), sy(float(p))) for e, p in zip(ells, profile)]
    d = "M " + " L ".join(f"{cx:.1f} {cy:.1f}" for cx, cy in pts)
    parts.append(f'<path d="{d}" fill="none" stroke="{ORANGE}" stroke-width="2.6"/>')

    ell_star = float(fit["ell_star"])
    gx_star = sx(ell_star)
    parts.append(f'<line x1="{gx_star:.1f}" y1="{py:.1f}" x2="{gx_star:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{RED}" stroke-width="1.6" stroke-dasharray="6 4"/>')
    gy_star = sy(float(fit["best_lml"]))
    parts.append(f'<circle cx="{gx_star:.1f}" cy="{gy_star:.1f}" r="6" fill="{RED}" '
                 f'stroke="#FFFFFF" stroke-width="1.4"/>')
    # placed well below the peak (the plateau is nearly flat there, so a
    # label at the dot's own height would sit on top of the curve) and
    # anchored to end just left of it, so the string never runs past the
    # canvas edge the way a start-anchored label this close to the right
    # margin would.
    parts.append(f'<text x="{gx_star-10:.1f}" y="{gy_star+45:.1f}" font-size="13" fill="{RED}" '
                 f'text-anchor="end">ℓ* ≈ {ell_star:.2f}</text>')

    parts.append(f'<text x="{sx(0.09):.1f}" y="{sy(float(profile[2]))-14:.1f}" font-size="12.5" '
                 f'fill="{SUBTLE}" text-anchor="start">too small ℓ: interpolates noise</text>')
    parts.append(f'<text x="{sx(1.3):.1f}" y="{sy(y_min + 0.2 * (y_max - y_min)):.1f}" font-size="12.5" '
                 f'fill="{SUBTLE}" text-anchor="middle">too large ℓ: flattens toward a line</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    x, y = load_subset()
    fit = fit_gp(x, y)
    print(f"n={len(y)} ell*={fit['ell_star']:.4f} sigma_n2*={fit['sigma_n2_star']:.4f} "
          f"sigma_f2={fit['sigma_f2']:.4f} best_lml={fit['best_lml']:.4f}")

    _write_and_rasterize(build_posterior_svg(x, y, fit), "gp_posterior_housing_en", 1050)
    _write_and_rasterize(build_lml_svg(fit), "gp_marginal_likelihood_en", 1050)


if __name__ == "__main__":
    main()
