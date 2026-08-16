"""
generate_elbow_real_data -- two real-dataset experiments for
ELBOW-{fr,en}.tex, as hand-authored SVG.

Runs elbow-helper's own shipped ``robust_elbow`` on two curves built from
the real UCI Wine recognition dataset (Aeberhard, Coomans and de Vel,
1992; 178 real wines, 13 real chemical measurements, 3 real cultivars,
shipped with scikit-learn, no MNIST): a k-means inertia curve and a PCA
scree plot, mirroring the package's own synthetic k-means/PCA worked
examples but on real, messier data. Both curves make ``robust_elbow``
abstain (``NoClearKnee``) rather than confirm a knee, unlike the clean
synthetic blobs/signal-plus-noise construction those examples use --
exactly the honest-abstention design priority ELBOW-fr.tex / ELBOW-en.tex
already discuss (the "subtle knee at two noise levels" example), now
demonstrated on data nobody engineered to cooperate.

No matplotlib, no seaborn, no Vega: every mark is hand-placed SVG, reusing
the house-style tokens and SVG primitives from generate_experiment_figures.py.

Run directly: ``python figures/generate_elbow_real_data.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_experiment_figures import (  # noqa: E402
    BLUE,
    INK,
    SUBTLE,
    GRID,
    svg_open,
    xml_escape,
    _write_and_rasterize,
)

from elbow_helper import robust_elbow, RobustKneeConfig  # noqa: E402

# Same short-curve profile tests/test_real_world_examples.py uses for the
# synthetic k-means/PCA examples: elbow-helper's defaults are tuned for
# long, noisy measurement curves, and reject a dozen-point applied curve
# outright regardless of its shape.
_SHORT_CURVE_CONFIG = RobustKneeConfig(
    min_samples=6,
    min_side_points=2,
    min_consecutive_scales=2,
    min_sensitivity_support=0.5,
    min_slope_contrast=0.15,
    min_bic_improvement=3.0,
    min_cv_improvement=0.02,
    bootstrap_replicates=200,
    min_bootstrap_detection_rate=0.7,
    max_ci90_width=0.2,
    min_primary_cluster_rate=0.6,
    max_bootstrap_median_shift=0.08,
    null_replicates=200,
    max_null_p_value=0.05,
    random_seed=0,
    cv_folds=3,
)

LANG = {
    "fr": {
        "kmeans_title": "Le vin, sans grappes fabriquées : la courbe d'inertie réelle",
        "kmeans_sub": "178 vins réels, 13 mesures chimiques réelles, 3 cépages réels ; robust_elbow appelé pour de vrai",
        "kmeans_x": "nombre de groupes k",
        "kmeans_y": "inertie",
        "kmeans_candidate": "3 cépages réels (non confirmé)",
        "kmeans_note": "NoClearKnee : NO_PERSISTENT_CLUSTER",
        "pca_title": "Le vin, sans vérité connue : le vrai diagramme des éboulis",
        "pca_sub": "mêmes 178 vins, valeurs propres de la covariance des 13 mesures réelles ; robust_elbow appelé pour de vrai",
        "pca_x": "composante",
        "pca_y": "valeur propre",
        "pca_note": "NoClearKnee : WEAK_SLOPE_CHANGE",
    },
    "en": {
        "kmeans_title": "Wine, without manufactured blobs: the real inertia curve",
        "kmeans_sub": "178 real wines, 13 real chemical measurements, 3 real cultivars; robust_elbow called for real",
        "kmeans_x": "number of clusters k",
        "kmeans_y": "inertia",
        "kmeans_candidate": "3 real cultivars (unconfirmed)",
        "kmeans_note": "NoClearKnee: NO_PERSISTENT_CLUSTER",
        "pca_title": "Wine, without a known truth: the real scree plot",
        "pca_sub": "same 178 wines, covariance eigenvalues of the 13 real measurements; robust_elbow called for real",
        "pca_x": "component",
        "pca_y": "eigenvalue",
        "pca_note": "NoClearKnee: WEAK_SLOPE_CHANGE",
    },
}


def _kmeans_inertia(
    x: np.ndarray, k: int, seed: int, n_init: int = 15, max_iter: int = 100
) -> float:
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centroids = x[rng.choice(len(x), size=k, replace=False)].copy()
        for _ in range(max_iter):
            d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
            lab = d.argmin(1)
            newc = np.array(
                [
                    x[lab == j].mean(0) if np.any(lab == j) else centroids[j]
                    for j in range(k)
                ]
            )
            if np.allclose(newc, centroids):
                centroids = newc
                break
            centroids = newc
        d = ((x[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        inertia = float(d[np.arange(len(x)), lab].sum())
        if best is None or inertia < best:
            best = inertia
    return best


def compute() -> Dict[str, object]:
    from sklearn.datasets import load_wine

    data = load_wine()
    X = data.data
    Xs = (X - X.mean(0)) / X.std(0)

    ks = np.arange(1, 13)
    inertias = np.array([_kmeans_inertia(Xs, int(k), seed=97 * int(k)) for k in ks])
    r_km = robust_elbow(ks.astype(float), inertias, config=_SHORT_CURVE_CONFIG)

    cov = np.cov(Xs, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    comps = np.arange(1, len(eigvals) + 1)
    r_pca = robust_elbow(comps.astype(float), eigvals, config=_SHORT_CURVE_CONFIG)

    return {
        "n": X.shape[0],
        "n_features": X.shape[1],
        "ks": ks,
        "inertias": inertias,
        "km_result": type(r_km).__name__,
        "km_reason": getattr(r_km, "reason", None),
        "comps": comps,
        "eigvals": eigvals,
        "pca_result": type(r_pca).__name__,
        "pca_reason": getattr(r_pca, "reason", None),
    }


def _single_panel_svg(
    lang: str,
    title: str,
    sub: str,
    xlabel: str,
    ylabel: str,
    xs: np.ndarray,
    ys: np.ndarray,
    note: str,
    candidate_x: float = None,
    candidate_label: str = "",
) -> str:
    width, height = 1000, 580
    m_left, m_right, m_top, m_bottom = 130, 60, 150, 110
    plot_x, plot_y = m_left, m_top
    plot_w = width - m_left - m_right
    plot_h = height - m_top - m_bottom

    x_min, x_max = float(xs.min()) - 0.4, float(xs.max()) + 0.4
    y_min, y_max = 0.0, float(ys.max()) * 1.08

    def sx(v: float) -> float:
        return plot_x + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return plot_y + (y_max - v) / (y_max - y_min) * plot_h

    parts: List[str] = []
    parts.append(svg_open(width, height, "ew-title", "ew-desc"))
    parts.append(f'<title id="ew-title">{xml_escape(title)}</title>')
    parts.append(f'<desc id="ew-desc">{xml_escape(title + ". " + sub)}</desc>')
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(
        f'<text x="{m_left}" y="52" font-size="23" font-weight="700" fill="{INK}">'
        f"{xml_escape(title)}</text>"
    )
    parts.append(
        f'<text x="{m_left}" y="82" font-size="15" fill="{SUBTLE}">{xml_escape(sub)}</text>'
    )

    y_ticks = np.linspace(y_min, y_max, 5)
    for v in y_ticks:
        gy = sy(v)
        parts.append(
            f'<line x1="{plot_x:.1f}" y1="{gy:.1f}" x2="{plot_x + plot_w:.1f}" y2="{gy:.1f}" '
            f'stroke="{GRID}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{plot_x - 14:.1f}" y="{gy + 5:.1f}" font-size="14" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{v:.0f}</text>'
        )

    ax_bottom = plot_y + plot_h
    parts.append(
        f'<line x1="{plot_x:.1f}" y1="{ax_bottom:.1f}" x2="{plot_x + plot_w:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.6"/>'
    )
    parts.append(
        f'<line x1="{plot_x:.1f}" y1="{plot_y:.1f}" x2="{plot_x:.1f}" y2="{ax_bottom:.1f}" '
        f'stroke="{INK}" stroke-width="1.6"/>'
    )
    for v in xs:
        gx = sx(float(v))
        parts.append(
            f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom + 5:.1f}" '
            f'stroke="{INK}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{gx:.1f}" y="{ax_bottom + 26:.1f}" font-size="13" '
            f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{int(v)}</text>'
        )
    parts.append(
        f'<text x="{plot_x + plot_w / 2:.1f}" y="{ax_bottom + 58:.1f}" font-size="16" fill="{INK}" '
        f'text-anchor="middle">{xml_escape(xlabel)}</text>'
    )
    ytx, yty = 40, plot_y + plot_h / 2
    parts.append(
        f'<text x="{ytx}" y="{yty:.1f}" font-size="15" fill="{INK}" text-anchor="middle" '
        f'transform="rotate(-90 {ytx} {yty:.1f})">{xml_escape(ylabel)}</text>'
    )

    if candidate_x is not None:
        gx = sx(candidate_x)
        parts.append(
            f'<line x1="{gx:.1f}" y1="{plot_y:.1f}" x2="{gx:.1f}" y2="{ax_bottom:.1f}" '
            f'stroke="{SUBTLE}" stroke-width="1.6" stroke-dasharray="3 5"/>'
        )
        parts.append(
            f'<text x="{gx + 8:.1f}" y="{plot_y + 22:.1f}" font-size="13" fill="{SUBTLE}">'
            f"{xml_escape(candidate_label)}</text>"
        )

    pts = [(sx(float(v)), sy(float(w))) for v, w in zip(xs, ys)]
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(
        f'<path d="{d}" fill="none" stroke="{BLUE}" stroke-width="2.6" '
        f'stroke-linecap="round" stroke-linejoin="round"/>'
    )
    for x, y in pts:
        parts.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{BLUE}" '
            f'stroke="#FFFFFF" stroke-width="1.2"/>'
        )

    parts.append(
        f'<text x="{plot_x + plot_w - 10:.1f}" y="{plot_y + 22:.1f}" font-size="13.5" '
        f'fill="{SUBTLE}" text-anchor="end" font-family="Roboto Mono, monospace">'
        f"{xml_escape(note)}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    print(
        "running real k-means + PCA on the Wine dataset via elbow_helper.robust_elbow ..."
    )
    res = compute()
    print(f"  n={res['n']} features={res['n_features']}")
    print(f"  k-means: {res['km_result']} ({res['km_reason']})")
    print(f"  pca:     {res['pca_result']} ({res['pca_reason']})")

    for lang in ("fr", "en"):
        t = LANG[lang]
        svg_km = _single_panel_svg(
            lang,
            t["kmeans_title"],
            t["kmeans_sub"],
            t["kmeans_x"],
            t["kmeans_y"],
            res["ks"],
            res["inertias"],
            t["kmeans_note"],
            candidate_x=3.0,
            candidate_label=t["kmeans_candidate"],
        )
        _write_and_rasterize(svg_km, f"kmeans_wine_{lang}", 1000)

        svg_pca = _single_panel_svg(
            lang,
            t["pca_title"],
            t["pca_sub"],
            t["pca_x"],
            t["pca_y"],
            res["comps"],
            res["eigvals"],
            t["pca_note"],
        )
        _write_and_rasterize(svg_pca, f"pca_wine_{lang}", 1000)


if __name__ == "__main__":
    main()
