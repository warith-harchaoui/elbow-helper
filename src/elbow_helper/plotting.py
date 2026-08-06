"""Optional diagnostic plotting.

Matplotlib is **not** a core dependency; it is imported lazily here so the base
install stays ``numpy + os-helper``. Install the extra with
``pip install elbow-helper[plot]``.

The plot never shows a bare point estimate: it pairs the located knee with its
bootstrap interval, the Kneedle difference curve that found it, and the decision
evidence.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import os_helper as oh

from .bootstrap import bootstrap_knee
from .candidates import generate_candidates
from .config import RobustKneeConfig
from .kneedle import KneeLocator
from .metrics import passes_basic_filters
from .pipeline import robust_knee
from .preprocessing import Abstain, prepare_curve
from .smoothing import smooth_curve


def _require_matplotlib():
    """Import matplotlib or raise a clear, actionable error.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401
        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "Diagnostic plotting needs matplotlib. Install it with "
            "`pip install elbow-helper[plot]`."
        ) from exc


def plot_diagnostics(
    x,
    y=None,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config: Optional[RobustKneeConfig] = None,
    out: Optional[str] = None,
    show: bool = False,
):
    """Draw the four-panel diagnostic for a robust-knee run.

    Panels: (1) raw + smoothed curve with the knee and its 90% interval;
    (2) candidate knees across smoothing window and sensitivity; (3) the Kneedle
    difference curve with the located peak; (4) the bootstrap distribution and
    the decision summary.

    Parameters
    ----------
    x, y : array-like
        The curve. ``y`` may be omitted, as in :func:`robust_knee`.
    curve, direction : str, optional
        Kneedle orientation (see :func:`robust_knee`). If omitted, inferred
        from the data.
    config : RobustKneeConfig, optional
        Thresholds and replicate counts.
    out : str, optional
        If given, save the figure to this path (parent dirs are created).
    show : bool, optional
        If ``True``, call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The assembled figure.
    """
    plt = _require_matplotlib()
    config = config or RobustKneeConfig()

    if y is None:
        y = x
        x = np.arange(len(np.asarray(y).ravel()), dtype=float)

    try:
        prepared = prepare_curve(x, y, curve, direction, config)
    except Abstain:
        prepared = None

    curve = prepared.curve if prepared is not None else curve
    direction = prepared.direction if prepared is not None else direction

    result = robust_knee(x, y, curve=curve, direction=direction, config=config)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    if prepared is not None:
        xn, yn = prepared.x_norm, prepared.y_scaled

        # Panel 1: raw + smoothed, knee marker, CI band.
        ax1.plot(xn, yn, ".", color="#8E8E93", ms=4, label="data (scaled)")
        ax1.plot(xn, smooth_curve(yn, max(3, prepared.n // 12)), "-",
                 color="#007AFF", lw=2, label="smoothed")
        if result.is_clear:
            ax1.axvline(result.knee_x_norm, color="#FF3B30", lw=1.6, label="knee")
            lo = (result.ci90[0] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
            hi = (result.ci90[1] - prepared.x_lo) / (prepared.x_hi - prepared.x_lo)
            ax1.axvspan(lo, hi, color="#FF3B30", alpha=0.12, label="90% CI")
        ax1.set_title("Curve + located knee")
        ax1.set_xlabel("x (normalized)")
        ax1.set_ylabel("y (scaled)")
        ax1.legend(loc="best", fontsize=8)

        # Panel 2: candidate scatter across window x sensitivity.
        cands = generate_candidates(prepared, config)
        for c in cands:
            ok = passes_basic_filters(c, prepared.n, config)
            ax2.scatter(c.knee_x_norm, c.window,
                        c=("#34C759" if ok else "#D1D1D6"),
                        s=25 + 8 * c.sensitivity, edgecolors="none", alpha=0.8)
        if result.is_clear:
            ax2.axvline(result.knee_x_norm, color="#FF3B30", lw=1.2)
        ax2.set_title("Candidates across scale-space")
        ax2.set_xlabel("knee x (normalized)")
        ax2.set_ylabel("smoothing window")
        ax2.set_xlim(0, 1)

        # Panel 3: difference curve at a mid smoothing scale.
        w = result.smoothing_window if result.is_clear else max(3, prepared.n // 12)
        kl = KneeLocator(xn, smooth_curve(yn, w), S=1.0, curve=curve,
                         direction=direction, online=True)
        ax3.plot(kl.x_difference, kl.y_difference, "-", color="#5856D6", lw=2)
        if kl.maxima_indices.size:
            ax3.plot(kl.x_difference[kl.maxima_indices],
                     kl.y_difference[kl.maxima_indices], "o",
                     color="#FF9500", ms=5, label="maxima")
        ax3.set_title("Kneedle difference curve")
        ax3.set_xlabel("x (normalized)")
        ax3.set_ylabel("difference")
        ax3.legend(loc="best", fontsize=8)

        # Panel 4: bootstrap distribution + decision text.
        if result.is_clear:
            boot = bootstrap_knee(prepared, result.knee_x_norm, config)
            if boot.knees:
                ax4.hist(boot.knees, bins=20, color="#007AFF", alpha=0.7)
                ax4.axvline(result.knee_x_norm, color="#FF3B30", lw=1.6)
            ax4.set_title("Bootstrap knee distribution")
            ax4.set_xlabel("knee x (normalized)")
        else:
            ax4.axis("off")

    summary = _decision_text(result)
    if not result.is_clear:
        ax4.axis("off")
    fig.text(0.52, 0.02, summary, fontsize=9, va="bottom", family="monospace")
    fig.suptitle("elbow-helper diagnostics", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    if out:
        oh.make_directory(oh.folder_name_ext(out)[0])
        fig.savefig(out, dpi=130)
        oh.info(f"[elbow-helper] saved diagnostics to {out}")
    if show:  # pragma: no cover
        plt.show()
    return fig


def _decision_text(result) -> str:
    """A short textual diagnostic summary, always with the evidence.

    Parameters
    ----------
    result : ClearKnee or NoClearKnee
        The pipeline's result, as returned by :func:`robust_knee`.

    Returns
    -------
    str
        A multi-line summary: the decision, the point estimate and its
        supporting evidence when clear, or the abstention reason otherwise.
    """
    if result.is_clear:
        return (
            f"Decision: CLEAR_KNEE\n"
            f"knee x            = {result.knee_x:.4g}\n"
            f"90% CI            = ({result.ci90[0]:.4g}, {result.ci90[1]:.4g})\n"
            f"detection rate    = {result.detection_rate:.2f}\n"
            f"slope contrast    = {result.slope_contrast:.2f}\n"
            f"BIC improvement   = {result.bic_improvement:.1f}\n"
            f"null-test p-value = {result.null_p_value:.3g}"
        )
    return (
        f"Decision: NO_CLEAR_KNEE\n"
        f"primary reason    = {result.reason}\n"
        f"(no point estimate reported without sufficient evidence)"
    )
