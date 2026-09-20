"""In-Pyodide endpoint logic of the elbow-helper web app.

Mirrors what the FastAPI surface (``elbow_helper.api``) computes, without
FastAPI: the page's ``backend-pyodide.js`` crosses the JS/Python boundary
with JSON strings only (no proxy lifetimes), through one entry point::

    glue_call(name, arg_json) -> json str   # {"ok": ...} or {"error": "..."}

Operations:

- ``version``   {} -> the installed elbow-helper version string
- ``preset``    {"id": ...} -> the named demo curve: x, y, curve, direction,
                per-language axis labels. Each preset reproduces one of the
                repo's ``examples/*.py`` scripts byte for byte (same NumPy
                seed, same formula), so the browser shows the same numbers
                the published EXAMPLES.md documents.
- ``analyze``   {"x", "y"?, "curve"?, "direction"?, "language", "x_label"?,
                "y_label"?, "log_y"?} -> {"result": <the CLI/HTTP JSON
                payload>, "svg": <the diagnostic figure>, "elapsed_s": ...}

``analyze`` runs the shipped default configuration with ``random_seed=0``:
the same validated thresholds and replicate counts as the published examples
(reduced replicate counts were measured to save little time while flipping
verdicts, because the no-knee null test needs at least 100 replicates for its
p-value to clear ``max_null_p_value=0.01``).

``render_svg`` recomputes the pipeline internally, so a naive analyze would
pay it twice. ``plotting`` looks ``robust_knee`` up in its own module globals
(``from .pipeline import robust_knee``), which makes the recomputation
avoidable from outside the package: ``analyze`` swaps that one name for a
memoized wrapper, computes the verdict first (after inferring orientation
exactly like ``render_svg`` does), and the figure's internal call becomes a
cache hit.
"""

from __future__ import annotations

import json
import time

import numpy as np

import elbow_helper.plotting as plotting
from elbow_helper import RobustKneeConfig
from elbow_helper._core_cli import _result_to_dict
from elbow_helper.plotting import render_svg
from elbow_helper.preprocessing import Abstain, prepare_curve

# --- presets: the repo's examples/*.py scripts, reproduced exactly -----------


def _clear_knee() -> dict:
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 80)
    knee = 0.30
    y = np.where(x <= knee, 3 * x, 3 * knee + 0.2 * (x - knee))
    y = y / y.max() + rng.normal(0, 0.02, x.size)
    return {
        "x": x,
        "y": y,
        "curve": "concave",
        "direction": "increasing",
        "x_label": {"en": "input (a.u.)", "fr": "entrée (u.a.)"},
        "y_label": {"en": "output (a.u.)", "fr": "sortie (u.a.)"},
    }


def _kmeans_elbow() -> dict:
    rng = np.random.default_rng(3)
    k = np.arange(1, 41, dtype=float)
    inertia = np.where(k <= 8, 1000 - 90 * k, 280 - 3 * (k - 8))
    inertia = inertia + rng.normal(0, 4.0, k.size)
    return {
        "x": k,
        "y": inertia,
        "curve": "convex",
        "direction": "decreasing",
        "x_label": {"en": "k (number of clusters)", "fr": "k (nombre de groupes)"},
        "y_label": {"en": "inertia", "fr": "inertie"},
    }


def _no_knee() -> dict:
    rng = np.random.default_rng(2)
    x = np.linspace(0, 1, 80)
    y = 0.2 + 0.5 * x + rng.normal(0, 0.01, x.size)
    return {
        "x": x,
        "y": y,
        "curve": "concave",
        "direction": "increasing",
        "x_label": {"en": "input (a.u.)", "fr": "entrée (u.a.)"},
        "y_label": {"en": "output (a.u.)", "fr": "sortie (u.a.)"},
    }


def _exponential_saturation() -> dict:
    tau = 1.0
    rng = np.random.default_rng(4)
    t = np.linspace(0, 5 * tau, 150)
    y = 1.0 - np.exp(-t / tau)
    y = y + rng.normal(0, 0.01, t.size)
    return {
        "x": t,
        "y": y,
        "curve": "concave",
        "direction": "increasing",
        "x_label": {"en": "time (in units of tau)", "fr": "temps (en unités de tau)"},
        "y_label": {"en": "response", "fr": "réponse"},
    }


def _queueing_latency() -> dict:
    baseline_ms = 8.0
    rng = np.random.default_rng(7)
    rho = np.linspace(0.02, 0.90, 150)
    latency_ms = baseline_ms / (1.0 - rho)
    latency_ms = latency_ms + rng.normal(0, 0.6, rho.size)
    return {
        "x": rho,
        "y": latency_ms,
        "curve": "convex",
        "direction": "increasing",
        "x_label": {"en": "utilisation ρ", "fr": "taux d'occupation ρ"},
        "y_label": {"en": "latency (ms)", "fr": "latence (ms)"},
    }


def _cache_hit_rate() -> dict:
    half_size = 200.0
    rng = np.random.default_rng(11)
    cache_size = np.linspace(10, 2000, 150)
    hit_rate = cache_size / (cache_size + half_size)
    hit_rate = hit_rate + rng.normal(0, 0.01, cache_size.size)
    return {
        "x": cache_size,
        "y": hit_rate,
        "curve": "concave",
        "direction": "increasing",
        "x_label": {"en": "cache size (entries)", "fr": "taille du cache (entrées)"},
        "y_label": {"en": "hit rate", "fr": "taux de succès"},
    }


def _bug_discovery_rate() -> dict:
    rng = np.random.default_rng(13)
    true_knee_day = 10.0
    days = np.linspace(1, 30, 120)
    bugs = np.where(
        days <= true_knee_day, 14.0 - 1.1 * days, 2.0 - 0.05 * (days - true_knee_day)
    )
    bugs = np.clip(bugs + rng.normal(0, 0.3, days.size), 0, None)
    return {
        "x": days,
        "y": bugs,
        "curve": "convex",
        "direction": "decreasing",
        "x_label": {"en": "day of testing", "fr": "jour de test"},
        "y_label": {"en": "new bugs found per day", "fr": "nouveaux bugs par jour"},
    }


PRESETS = {
    "clear_knee": _clear_knee,
    "kmeans_elbow": _kmeans_elbow,
    "no_knee": _no_knee,
    "exponential_saturation": _exponential_saturation,
    "queueing_latency": _queueing_latency,
    "cache_hit_rate": _cache_hit_rate,
    "bug_discovery_rate": _bug_discovery_rate,
}


def _op_version(_arg: dict) -> str:
    import elbow_helper

    return elbow_helper.__version__


def _op_preset(arg: dict) -> dict:
    preset_id = str(arg.get("id") or "")
    if preset_id not in PRESETS:
        raise ValueError(f"unknown preset: {preset_id!r}")
    data = PRESETS[preset_id]()
    return {
        "id": preset_id,
        "x": [round(float(v), 6) for v in data["x"]],
        "y": [round(float(v), 6) for v in data["y"]],
        "curve": data["curve"],
        "direction": data["direction"],
        "x_label": data["x_label"],
        "y_label": data["y_label"],
    }


def _op_analyze(arg: dict) -> dict:
    x = np.asarray(arg["x"], dtype=float)
    y = None if arg.get("y") is None else np.asarray(arg["y"], dtype=float)
    curve = arg.get("curve") or None
    direction = arg.get("direction") or None
    language = "fr" if arg.get("language") == "fr" else "en"
    x_label = arg.get("x_label") or None
    y_label = arg.get("y_label") or None
    log_y = bool(arg.get("log_y"))
    config = RobustKneeConfig(random_seed=0)

    if y is None:
        y = x
        x = np.arange(len(y), dtype=float)

    started = time.perf_counter()

    # One pipeline run for both the verdict and the figure: memoize
    # robust_knee for the duration of this call, then infer the orientation
    # the same way render_svg will, so its internal call hits the cache.
    cache: dict = {}
    real_robust_knee = plotting.robust_knee

    def memoized(mx, my=None, curve=None, direction=None, config=None):
        key = (
            np.asarray(mx, dtype=float).tobytes(),
            None if my is None else np.asarray(my, dtype=float).tobytes(),
            curve,
            direction,
            repr(config),
        )
        if key not in cache:
            cache[key] = real_robust_knee(
                mx, my, curve=curve, direction=direction, config=config
            )
        return cache[key]

    plotting.robust_knee = memoized
    try:
        try:
            prepared = prepare_curve(x, y, curve, direction, config)
            inferred_curve, inferred_direction = prepared.curve, prepared.direction
        except Abstain:
            inferred_curve, inferred_direction = curve, direction
        result = memoized(
            x, y, curve=inferred_curve, direction=inferred_direction, config=config
        )
        svg = render_svg(
            x,
            y,
            curve=curve,
            direction=direction,
            config=config,
            language=language,
            raw_axis=True,
            log_y=log_y,
            x_label=x_label,
            y_label=y_label,
        )
    finally:
        plotting.robust_knee = real_robust_knee

    return {
        "result": _result_to_dict(result),
        "svg": svg,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }


_OPS = {
    "version": _op_version,
    "preset": _op_preset,
    "analyze": _op_analyze,
}


def glue_call(name: str, arg_json: str) -> str:
    """Dispatch one operation; always return a JSON envelope, never raise.

    ``ValueError`` carries a message meant for the visitor (bad input); any
    other exception is reported with its type so the page can display it
    without dying.
    """
    try:
        arg = json.loads(arg_json) if arg_json else {}
        out = _OPS[name](arg)
        return json.dumps({"ok": out}, ensure_ascii=False)
    except ValueError as exc:
        return json.dumps({"error": str(exc)}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001 — the page must stay alive
        return json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)
