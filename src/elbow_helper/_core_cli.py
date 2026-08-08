"""Shared business logic — one core behind every delivery surface.

The suite convention (see this repo's `CODING.md` reference, `~/ai-helpers/
.private/CODING.md` §19) is that every surface — the argparse CLI, the click
twin, the FastAPI app, the MCP server — must call the *same* library
functions, never re-implement the decision logic. This module is that
shared core: thin functions that take already-parsed, primitive arguments
and return JSON-ready dicts (or a raw SVG string for diagnostics), so each
front end is a trivial adapter over these.

Consumes: :mod:`elbow_helper` (``robust_knee``/``robust_elbow``/
``RobustKneeConfig``/``KneeLocator``), :mod:`elbow_helper.plotting`, numpy,
``os_helper`` (logging).
Produces: :func:`do_knee`, :func:`do_elbow`, :func:`do_diagnostics`,
:func:`do_locator`.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np
import os_helper as oh

from .config import RobustKneeConfig
from .locator import KneeLocator
from .pipeline import robust_elbow, robust_knee
from .plotting import plot_diagnostics
from .types import KneeResult


def _jsonable(value: Any) -> Any:
    """Recursively coerce numpy scalars/arrays into plain JSON-safe Python values.

    ``KneeResult.diagnostics`` and several dataclass fields are built up
    from numpy computations; a stray ``numpy.float64``/``numpy.int64`` (as
    opposed to a plain ``float``/``int``) fails ``json.dumps`` with a
    ``TypeError`` that only surfaces the first time an HTTP/CLI caller
    actually serializes a result — this walks the whole structure up front
    so every surface gets a safe payload, not just the ones that happen to
    call ``round()`` first.

    Parameters
    ----------
    value : Any
        A (possibly nested) dict, list, tuple, numpy scalar, or numpy array.

    Returns
    -------
    Any
        The same structure with every numpy scalar/array replaced by a
        native ``int``/``float``/``list``.

    Examples
    --------
    >>> import numpy as np
    >>> _jsonable({"a": np.float64(1.5), "b": [np.int64(2), 3]})
    {'a': 1.5, 'b': [2, 3]}
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _result_to_dict(result: KneeResult) -> dict:
    """Serialize a :class:`ClearKnee`/:class:`NoClearKnee` to a JSON-ready dict.

    Parameters
    ----------
    result : KneeResult
        The tagged-union result from :func:`elbow_helper.robust_knee` or
        :func:`elbow_helper.robust_elbow`.

    Returns
    -------
    dict
        Every dataclass field, plus the derived ``is_clear`` property, all
        coerced through :func:`_jsonable`.

    Examples
    --------
    >>> from elbow_helper import RobustKneeConfig, robust_knee
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 60)
    >>> y = np.where(x <= 0.3, 3 * x, 0.9)
    >>> out = _result_to_dict(robust_knee(x, y, config=RobustKneeConfig(random_seed=0)))
    >>> out["is_clear"]
    True
    """
    payload = dataclasses.asdict(result)
    payload["is_clear"] = result.is_clear
    return _jsonable(payload)


def _config_from_overrides(overrides: Optional[dict]) -> RobustKneeConfig:
    """Build a :class:`RobustKneeConfig`, applying caller overrides on top of defaults.

    Parameters
    ----------
    overrides : dict, optional
        Field-name -> value pairs matching :class:`RobustKneeConfig`'s
        dataclass fields (e.g. ``{"bootstrap_replicates": 500}``). ``None``
        or an empty dict keeps every default.

    Returns
    -------
    RobustKneeConfig
        The default config, or ``config.with_(**overrides)`` when overrides
        were given.

    Raises
    ------
    TypeError
        If ``overrides`` names a field that does not exist on
        :class:`RobustKneeConfig` — surfaced as-is (from ``dataclasses.
        replace``) so the caller sees exactly which key was wrong.
    """
    config = RobustKneeConfig()
    if overrides:
        config = config.with_(**overrides)
    return config


def do_knee(
    x: Sequence[float],
    y: Optional[Sequence[float]] = None,
    *,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """Run :func:`elbow_helper.robust_knee` and return a JSON-ready result.

    Parameters
    ----------
    x : sequence of float
        The curve's x values (or, when ``y`` is omitted, the curve's y
        values themselves — see :func:`elbow_helper.robust_knee`).
    y : sequence of float, optional
        The curve's y values.
    curve, direction : str, optional
        Orientation hints; inferred from the data when omitted.
    config_overrides : dict, optional
        :class:`RobustKneeConfig` field overrides.

    Returns
    -------
    dict
        The serialized :class:`ClearKnee`/:class:`NoClearKnee`.

    Examples
    --------
    >>> x = list(range(60))
    >>> y = [min(3 * v, 54) / 54 for v in x]
    >>> do_knee(x, y)["is_clear"] in (True, False)
    True
    """
    config = _config_from_overrides(config_overrides)
    result = robust_knee(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float) if y is not None else None,
        curve=curve,
        direction=direction,
        config=config,
    )
    oh.info(f"[elbow-helper] knee: is_clear={result.is_clear}")
    return _result_to_dict(result)


def do_elbow(
    k: Sequence[float],
    inertia: Sequence[float],
    *,
    config_overrides: Optional[dict] = None,
) -> dict:
    """Run :func:`elbow_helper.robust_elbow` and return a JSON-ready result.

    Parameters
    ----------
    k : sequence of float
        The x axis (e.g. candidate cluster counts).
    inertia : sequence of float
        The y axis (convex, decreasing — e.g. k-means inertia).
    config_overrides : dict, optional
        :class:`RobustKneeConfig` field overrides.

    Returns
    -------
    dict
        The serialized :class:`ClearKnee`/:class:`NoClearKnee`.

    Examples
    --------
    >>> k = list(range(1, 21))
    >>> inertia = [1000 - 80 * v if v <= 8 else 360 - 2 * (v - 8) for v in k]
    >>> do_elbow(k, inertia)["is_clear"] in (True, False)
    True
    """
    config = _config_from_overrides(config_overrides)
    result = robust_elbow(
        np.asarray(k, dtype=float), np.asarray(inertia, dtype=float), config=config
    )
    oh.info(f"[elbow-helper] elbow: is_clear={result.is_clear}")
    return _result_to_dict(result)


def do_diagnostics(
    x: Sequence[float],
    y: Optional[Sequence[float]] = None,
    *,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    config_overrides: Optional[dict] = None,
    out: Optional[str] = None,
    language: str = "en",
) -> str:
    """Render the diagnostic SVG for a curve, optionally saving it.

    Parameters
    ----------
    x, y : sequence of float
        The curve; see :func:`elbow_helper.robust_knee`.
    curve, direction : str, optional
        Orientation hints; inferred from the data when omitted.
    config_overrides : dict, optional
        :class:`RobustKneeConfig` field overrides.
    out : str, optional
        If given, also write the SVG to this path.
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.

    Returns
    -------
    str
        The complete SVG document.

    Examples
    --------
    >>> x = list(range(60))
    >>> y = [min(3 * v, 54) / 54 for v in x]
    >>> do_diagnostics(x, y).startswith("<svg")
    True
    """
    config = _config_from_overrides(config_overrides)
    return plot_diagnostics(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float) if y is not None else None,
        curve=curve,
        direction=direction,
        config=config,
        out=out,
        language=language,
    )


def do_locator(
    x: Sequence[float],
    y: Sequence[float],
    *,
    sensitivity: float = 1.0,
    curve: Optional[str] = None,
    direction: Optional[str] = None,
    online: bool = True,
) -> dict:
    """Run the standalone locator (no bootstrap, no null test, no abstention gate).

    Parameters
    ----------
    x, y : sequence of float
        The curve.
    sensitivity : float, optional
        The locator's ``S`` sensitivity parameter (larger = more
        conservative). Defaults to ``1.0``.
    curve, direction : str, optional
        Orientation hints. Unlike :func:`do_knee` (backed by ``robust_knee``,
        which infers these from the data), the raw locator has no inference
        step — omitted values fall back to its own constructor defaults,
        ``"concave"``/``"increasing"``.
    online : bool, optional
        Whether the locator re-arms at every later, higher peak (the
        "online" traversal mode). Defaults to ``True``.

    Returns
    -------
    dict
        ``{"knee": float | None, "all_knees": list[float]}`` — the raw
        geometric result, with none of :func:`do_knee`'s uncertainty
        evidence (this is the building block, not the conservative
        pipeline; see :func:`do_knee` when a confidence estimate matters).

    Examples
    --------
    >>> x = list(range(60))
    >>> y = [min(3 * v, 54) / 54 for v in x]
    >>> out = do_locator(x, y)
    >>> "knee" in out and "all_knees" in out
    True
    """
    # KneeLocator's constructor (unlike robust_knee) has no shape-inference
    # step and validates curve/direction against a fixed choice set, so a
    # bare None would raise ValueError there — fall back to its own defaults.
    locator = KneeLocator(
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        S=sensitivity,
        curve=curve or "concave",
        direction=direction or "increasing",
        online=online,
    )
    return _jsonable({"knee": locator.knee, "all_knees": list(locator.all_knees)})
