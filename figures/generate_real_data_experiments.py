"""
generate_real_data_experiments — real-data classification and regression
worked examples for LIKELIHOOD-{fr,en}.tex, as hand-authored SVG.

Trains two from-scratch numpy models by gradient descent on real, publicly
documented datasets (no MNIST), tracking every quantity this note builds
(cross-entropy, perplexity, Q_classif for classification; MSE, RMSE, PPL,
Q_reg for regression) at every training iteration, on both the training
split and a held-out validation split:

* Classification -- the Palmer Archipelago penguins dataset (Gorman, Williams
  and Fraser, 2014; distributed via the `palmerpenguins` project and its
  mirror in the seaborn-data repository, fetched here as a plain CSV, no
  seaborn import): 3 species, 4 real morphological measurements per bird.
  A multinomial softmax classifier is trained by batch gradient descent on
  cross-entropy.
* Regression -- the California housing dataset (Pace and Barry, 1997; shipped
  with scikit-learn), 8 real census-block features predicting median house
  value. A linear model and a one-hidden-layer MLP (30 ReLU units, He
  initialization, the architecture Geron's "Hands-On Machine Learning"
  (2019) uses for this exact dataset) are both trained by batch gradient
  descent on MSE, on the same split, so the gain from added model capacity
  is directly readable.

No model here is fit with a library solver: every one is plain numpy
gradient descent (forward pass and backpropagation both hand-written for
the MLP), so the per-iteration training/validation curves are genuine, not
approximated after the fact. No matplotlib, no seaborn, no Vega for drawing:
every mark is placed by hand as SVG, the same convention
`generate_experiment_figures.py` and sprezzature-figures's own `make_<id>.py`
generators use; this module imports its house-style tokens and SVG
primitives from that sibling script rather than redefining them.

Run directly: ``python figures/generate_real_data_experiments.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from generate_experiment_figures import (  # noqa: E402
    BLUE, RED, GREEN, ORANGE, INK, SUBTLE, GRID,
    svg_open, xml_escape, _write_and_rasterize,
)

PENGUINS_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"

LANG = {
    "fr": {
        "clf_title": "Pingouins de l'Antarctique : entropie croisée et score borné pendant l'entraînement",
        "clf_sub": "344 manchots, 3 espèces, 4 mesures réelles ; régression softmax entraînée par descente de gradient",
        "clf_panel_a_title": "entropie croisée",
        "clf_panel_a_y": "entropie croisée (nats)",
        "clf_panel_b_title": "score borné Q classif",
        "clf_panel_b_y": "Q classif = 1 − CE / ln K",
        "x_iter": "itération",
        "train": "entraînement",
        "val": "validation",
        "reg_title": "Logements californiens : erreur quadratique et score borné pendant l'entraînement",
        "reg_sub": "20 640 pâtés de maisons, 8 variables réelles du recensement, MSE en centaines de milliers de dollars au carré",
        "reg_panel_a_title": "erreur quadratique moyenne",
        "reg_panel_a_y": "MSE",
        "reg_panel_b_title": "score borné Q reg",
        "reg_panel_b_y": "Q reg = 1 − MSE / MSE du pire cas",
        "mlp_title": "Logements californiens : le même entraînement, avec un MLP à la place d'une droite",
        "mlp_sub": "même partage train/validation, mêmes 8 variables ; une couche cachée de 30 unités ReLU (init. He)",
    },
    "en": {
        "clf_title": "Antarctic penguins: cross-entropy and bounded score during training",
        "clf_sub": "344 penguins, 3 species, 4 real measurements; softmax regression trained by gradient descent",
        "clf_panel_a_title": "cross-entropy",
        "clf_panel_a_y": "cross-entropy (nats)",
        "clf_panel_b_title": "bounded score Q classif",
        "clf_panel_b_y": "Q classif = 1 − CE / ln K",
        "x_iter": "iteration",
        "train": "training",
        "val": "validation",
        "reg_title": "California housing: mean-squared error and bounded score during training",
        "reg_sub": "20,640 census blocks, 8 real features, MSE in hundred-thousand dollars squared",
        "reg_panel_a_title": "mean-squared error",
        "reg_panel_a_y": "MSE",
        "reg_panel_b_title": "bounded score Q reg",
        "reg_panel_b_y": "Q reg = 1 − MSE / worst-case MSE",
        "mlp_title": "California housing: the same training, an MLP standing in for a line",
        "mlp_sub": "same train/validation split, same 8 variables; one hidden layer of 30 ReLU units (He init)",
    },
}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_penguins() -> Dict[str, object]:
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(PENGUINS_URL).dropna().reset_index(drop=True)
    feats = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    X = df[feats].to_numpy(dtype=float)
    species = sorted(df["species"].unique())
    y = df["species"].map({s: i for i, s in enumerate(species)}).to_numpy()
    K = len(species)

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xval_s = (Xtr - mu) / sd, (Xval - mu) / sd
    n_tr, d = Xtr_s.shape

    def softmax(Z: np.ndarray) -> np.ndarray:
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    def cross_entropy(Xs: np.ndarray, ys: np.ndarray, W: np.ndarray, b: np.ndarray) -> float:
        P = softmax(Xs @ W + b)
        return float(-np.mean(np.log(P[np.arange(len(ys)), ys] + 1e-12)))

    W = np.zeros((d, K))
    b = np.zeros(K)
    lr, n_iters = 0.5, 400
    train_ce, val_ce = [], []
    for _ in range(n_iters):
        P = softmax(Xtr_s @ W + b)
        Y1h = np.eye(K)[ytr]
        grad_logits = (P - Y1h) / n_tr
        W -= lr * (Xtr_s.T @ grad_logits)
        b -= lr * grad_logits.sum(0)
        train_ce.append(cross_entropy(Xtr_s, ytr, W, b))
        val_ce.append(cross_entropy(Xval_s, yval, W, b))

    train_ce_arr, val_ce_arr = np.array(train_ce), np.array(val_ce)
    val_pred = softmax(Xval_s @ W + b).argmax(1)
    return {
        "n_iters": n_iters,
        "train_ce": train_ce_arr,
        "val_ce": val_ce_arr,
        "ln_k": math.log(K),
        "K": K,
        "n_total": len(df),
        "n_train": n_tr,
        "n_val": len(yval),
        "val_accuracy": float((val_pred == yval).mean()),
        "final_val_ce": float(val_ce_arr[-1]),
        "final_val_ppl": float(np.exp(val_ce_arr[-1])),
    }


def train_housing() -> Dict[str, object]:
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X, y = data.data, data.target  # y in $100,000 units

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xval_s = (Xtr - mu) / sd, (Xval - mu) / sd
    n_tr, d = Xtr_s.shape

    mse_worst = float(np.mean((np.full_like(yval, ytr.mean()) - yval) ** 2))

    w = np.zeros(d)
    b = float(ytr.mean())
    lr, n_iters = 0.3, 300
    train_mse, val_mse = [], []
    for _ in range(n_iters):
        resid = Xtr_s @ w + b - ytr
        w -= lr * (Xtr_s.T @ resid) / n_tr
        b -= lr * resid.mean()
        train_mse.append(float(np.mean((Xtr_s @ w + b - ytr) ** 2)))
        val_mse.append(float(np.mean((Xval_s @ w + b - yval) ** 2)))

    train_mse_arr, val_mse_arr = np.array(train_mse), np.array(val_mse)
    return {
        "n_iters": n_iters,
        "train_mse": train_mse_arr,
        "val_mse": val_mse_arr,
        "mse_worst": mse_worst,
        "n_total": len(y),
        "n_train": n_tr,
        "n_val": len(yval),
        "final_val_mse": float(val_mse_arr[-1]),
        "final_val_rmse": float(np.sqrt(val_mse_arr[-1])),
        "final_q_reg": 1.0 - float(val_mse_arr[-1]) / mse_worst,
    }


def train_housing_mlp(hidden: int = 30, lr: float = 0.1, n_iters: int = 3000) -> Dict[str, object]:
    """One hidden ReLU layer (He init) plus a linear output unit, the
    architecture Geron (2019) uses for this exact dataset, trained here by
    hand-written forward and backward passes rather than a Keras call, on
    the same standardized split `train_housing` uses so the two are directly
    comparable."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X, y = data.data, data.target

    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.2, random_state=42)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    Xtr_s, Xval_s = (Xtr - mu) / sd, (Xval - mu) / sd
    n_tr, d = Xtr_s.shape

    mse_worst = float(np.mean((np.full_like(yval, ytr.mean()) - yval) ** 2))

    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((d, hidden)) * np.sqrt(2.0 / d)  # He init (He et al., 2015)
    b1 = np.zeros(hidden)
    W2 = rng.standard_normal((hidden, 1)) * np.sqrt(2.0 / hidden)
    b2 = np.array([float(ytr.mean())])

    def forward(Xs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z1 = Xs @ W1 + b1
        A1 = np.maximum(Z1, 0.0)
        pred = (A1 @ W2 + b2).ravel()
        return Z1, A1, pred

    train_mse, val_mse = [], []
    for _ in range(n_iters):
        Z1, A1, pred = forward(Xtr_s)
        resid = pred - ytr
        dpred = (2.0 / n_tr) * resid
        dW2 = A1.T @ dpred.reshape(-1, 1)
        db2 = np.array([dpred.sum()])
        dA1 = dpred.reshape(-1, 1) @ W2.T
        dZ1 = dA1 * (Z1 > 0)  # ReLU gradient: 1 where the pre-activation fired, 0 where it didn't
        dW1 = Xtr_s.T @ dZ1
        db1 = dZ1.sum(0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        _, _, pred_tr = forward(Xtr_s)
        _, _, pred_val = forward(Xval_s)
        train_mse.append(float(np.mean((pred_tr - ytr) ** 2)))
        val_mse.append(float(np.mean((pred_val - yval) ** 2)))

    train_mse_arr, val_mse_arr = np.array(train_mse), np.array(val_mse)
    return {
        "n_iters": n_iters,
        "hidden": hidden,
        "train_mse": train_mse_arr,
        "val_mse": val_mse_arr,
        "mse_worst": mse_worst,
        "n_total": len(y),
        "n_train": n_tr,
        "n_val": len(yval),
        "final_val_mse": float(val_mse_arr[-1]),
        "final_val_rmse": float(np.sqrt(val_mse_arr[-1])),
        "final_q_reg": 1.0 - float(val_mse_arr[-1]) / mse_worst,
    }


# ----------------------------------------------------------------------
# Shared two-panel training-curve figure
# ----------------------------------------------------------------------
def _curve_panel(
    parts: List[str], px: float, py: float, pw: float, ph: float,
    n_iters: int, train_y: np.ndarray, val_y: np.ndarray,
    title: str, ylabel: str, xlabel: str, train_label: str, val_label: str,
    *, log_y: bool, y_range: Tuple[float, float] = None,
) -> None:
    xs = np.arange(1, n_iters + 1)

    def sx(i: float) -> float:
        return px + (i - 1) / (n_iters - 1) * pw

    if log_y:
        y_top_k = math.ceil(math.log10(max(train_y.max(), val_y.max())))
        y_bot_k = math.floor(math.log10(min(train_y.min(), val_y.min())))

        def sy(v: float) -> float:
            k = math.log10(v)
            return py + (y_top_k - k) / (y_top_k - y_bot_k) * ph

        y_ticks_vals = [10.0**k for k in range(y_bot_k, y_top_k + 1)]
        y_labels = [f"{v:g}" for v in y_ticks_vals]
    else:
        y_min, y_max = y_range
        def sy(v: float) -> float:
            return py + (y_max - v) / (y_max - y_min) * ph
        y_ticks_vals = list(np.linspace(y_min, y_max, 5))
        y_labels = [f"{v:.2f}" for v in y_ticks_vals]

    parts.append(f'<text x="{px:.1f}" y="{py-18:.1f}" font-size="16" font-weight="700" fill="{INK}">'
                 f'{xml_escape(title)}</text>')
    for v, lab in zip(y_ticks_vals, y_labels):
        gy = sy(v)
        parts.append(f'<line x1="{px:.1f}" y1="{gy:.1f}" x2="{px+pw:.1f}" y2="{gy:.1f}" '
                     f'stroke="{GRID}" stroke-width="1.2"/>')
        parts.append(f'<text x="{px-12:.1f}" y="{gy+5:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="end">{lab}</text>')

    ax_bottom = py + ph
    parts.append(f'<line x1="{px:.1f}" y1="{ax_bottom:.1f}" x2="{px+pw:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{ax_bottom:.1f}" '
                 f'stroke="{INK}" stroke-width="1.5"/>')
    x_ticks = [1] + [t for t in (n_iters // 4, n_iters // 2, 3 * n_iters // 4, n_iters)]
    for xv in x_ticks:
        gx = sx(float(xv))
        parts.append(f'<line x1="{gx:.1f}" y1="{ax_bottom:.1f}" x2="{gx:.1f}" y2="{ax_bottom+5:.1f}" '
                     f'stroke="{INK}" stroke-width="1.2"/>')
        parts.append(f'<text x="{gx:.1f}" y="{ax_bottom+24:.1f}" font-size="12.5" '
                     f'font-family="Roboto Mono, monospace" fill="{INK}" text-anchor="middle">{xv}</text>')
    parts.append(f'<text x="{px+pw/2:.1f}" y="{ax_bottom+50:.1f}" font-size="14.5" fill="{INK}" '
                 f'text-anchor="middle">{xml_escape(xlabel)}</text>')
    ytx, yty = px - 78, py + ph / 2
    parts.append(f'<text x="{ytx:.1f}" y="{yty:.1f}" font-size="13.5" fill="{INK}" text-anchor="middle" '
                 f'transform="rotate(-90 {ytx:.1f} {yty:.1f})">{xml_escape(ylabel)}</text>')

    for series, color, label, dash in ((train_y, BLUE, train_label, ""), (val_y, ORANGE, val_label, "6 4")):
        pts = [(sx(float(i)), sy(v)) for i, v in zip(xs, series)]
        d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2.4"{dash_attr}/>')

    lx, ly = px + 14, py + 20
    for color, label, dash in ((BLUE, train_label, ""), (ORANGE, val_label, "6 4")):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{lx:.1f}" y1="{ly:.1f}" x2="{lx+22:.1f}" y2="{ly:.1f}" '
                     f'stroke="{color}" stroke-width="2.6"{dash_attr}/>')
        parts.append(f'<text x="{lx+30:.1f}" y="{ly+4:.1f}" font-size="12.5" fill="{INK}">{xml_escape(label)}</text>')
        ly += 20


def _two_panel_figure(lang: str, title: str, sub: str, n_iters: int,
                       left_kwargs: dict, right_kwargs: dict) -> str:
    width, height = 1150, 600
    m_top, m_bottom = 150, 130
    panel_w = 400
    gap = 130
    px_left = 190
    px_right = px_left + panel_w + gap
    ph = height - m_top - m_bottom
    py = m_top

    parts: List[str] = []
    parts.append(svg_open(width, height, "rd-title", "rd-desc"))
    parts.append(f"<title id=\"rd-title\">{xml_escape(title)}</title>")
    parts.append(f"<desc id=\"rd-desc\">{xml_escape(title + '. ' + sub)}</desc>")
    parts.append(f'<rect width="{width}" height="{height}" fill="#FFFFFF"/>')
    parts.append(f'<text x="60" y="52" font-size="22" font-weight="700" fill="{INK}">'
                 f'{xml_escape(title)}</text>')
    parts.append(f'<text x="60" y="82" font-size="15" fill="{SUBTLE}">{xml_escape(sub)}</text>')

    _curve_panel(parts, px_left, py, panel_w, ph, n_iters, **left_kwargs)
    _curve_panel(parts, px_right, py, panel_w, ph, n_iters, **right_kwargs)

    parts.append("</svg>")
    return "\n".join(parts)


def build_classification_svg(lang: str, res: Dict[str, object]) -> str:
    t = LANG[lang]
    n_iters = res["n_iters"]
    q_train = 1 - res["train_ce"] / res["ln_k"]
    q_val = 1 - res["val_ce"] / res["ln_k"]
    return _two_panel_figure(
        lang, t["clf_title"], t["clf_sub"], n_iters,
        left_kwargs=dict(train_y=res["train_ce"], val_y=res["val_ce"], title=t["clf_panel_a_title"],
                         ylabel=t["clf_panel_a_y"], xlabel=t["x_iter"], train_label=t["train"],
                         val_label=t["val"], log_y=True),
        right_kwargs=dict(train_y=q_train, val_y=q_val, title=t["clf_panel_b_title"],
                          ylabel=t["clf_panel_b_y"], xlabel=t["x_iter"], train_label=t["train"],
                          val_label=t["val"], log_y=False, y_range=(0.0, 1.0)),
    )


def build_regression_svg(lang: str, res: Dict[str, object]) -> str:
    t = LANG[lang]
    n_iters = res["n_iters"]
    mse_worst = res["mse_worst"]
    q_train = 1 - res["train_mse"] / mse_worst
    q_val = 1 - res["val_mse"] / mse_worst
    mse_hi = float(max(res["train_mse"].max(), res["val_mse"].max())) * 1.05
    return _two_panel_figure(
        lang, t["reg_title"], t["reg_sub"], n_iters,
        left_kwargs=dict(train_y=res["train_mse"], val_y=res["val_mse"], title=t["reg_panel_a_title"],
                         ylabel=t["reg_panel_a_y"], xlabel=t["x_iter"], train_label=t["train"],
                         val_label=t["val"], log_y=False, y_range=(0.0, mse_hi)),
        right_kwargs=dict(train_y=q_train, val_y=q_val, title=t["reg_panel_b_title"],
                          ylabel=t["reg_panel_b_y"], xlabel=t["x_iter"], train_label=t["train"],
                          val_label=t["val"], log_y=False, y_range=(0.0, 1.0)),
    )


def build_regression_mlp_svg(lang: str, res: Dict[str, object]) -> str:
    t = LANG[lang]
    n_iters = res["n_iters"]
    mse_worst = res["mse_worst"]
    q_train = 1 - res["train_mse"] / mse_worst
    q_val = 1 - res["val_mse"] / mse_worst
    # log scale: the first gradient step from a random He-initialized net
    # overshoots by more than an order of magnitude before settling, the
    # same wide-dynamic-range shape build_classification_svg's cross-entropy
    # panel already handles with log_y rather than a clipped linear axis.
    return _two_panel_figure(
        lang, t["mlp_title"], t["mlp_sub"], n_iters,
        left_kwargs=dict(train_y=res["train_mse"], val_y=res["val_mse"], title=t["reg_panel_a_title"],
                         ylabel=t["reg_panel_a_y"], xlabel=t["x_iter"], train_label=t["train"],
                         val_label=t["val"], log_y=True),
        right_kwargs=dict(train_y=q_train, val_y=q_val, title=t["reg_panel_b_title"],
                          ylabel=t["reg_panel_b_y"], xlabel=t["x_iter"], train_label=t["train"],
                          val_label=t["val"], log_y=False, y_range=(0.0, 1.0)),
    )


def main() -> None:
    print("training penguins classifier...")
    clf_res = train_penguins()
    print(f"  n={clf_res['n_total']} train={clf_res['n_train']} val={clf_res['n_val']} "
          f"val_acc={clf_res['val_accuracy']:.4f} val_CE={clf_res['final_val_ce']:.4f} "
          f"val_PPL={clf_res['final_val_ppl']:.4f}")

    print("training housing regressor (linear)...")
    reg_res = train_housing()
    print(f"  n={reg_res['n_total']} train={reg_res['n_train']} val={reg_res['n_val']} "
          f"val_RMSE={reg_res['final_val_rmse']:.4f} MSE_worst={reg_res['mse_worst']:.4f} "
          f"Q_reg={reg_res['final_q_reg']:.4f}")

    print("training housing regressor (MLP, 30 ReLU units)...")
    mlp_res = train_housing_mlp()
    print(f"  n={mlp_res['n_total']} train={mlp_res['n_train']} val={mlp_res['n_val']} "
          f"val_RMSE={mlp_res['final_val_rmse']:.4f} MSE_worst={mlp_res['mse_worst']:.4f} "
          f"Q_reg={mlp_res['final_q_reg']:.4f}")

    for lang in ("fr", "en"):
        _write_and_rasterize(build_classification_svg(lang, clf_res), f"classification_training_penguins_{lang}", 1150)
        _write_and_rasterize(build_regression_svg(lang, reg_res), f"regression_training_housing_{lang}", 1150)
        _write_and_rasterize(build_regression_mlp_svg(lang, mlp_res), f"regression_training_housing_mlp_{lang}", 1150)


if __name__ == "__main__":
    main()
