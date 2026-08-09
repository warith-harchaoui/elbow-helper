"""click command-line interface — the optional, ergonomic CLI twin.

Identical behaviour to :mod:`elbow_helper.cli_argparse`, but built on click
for a nicer help/UX. click is an *optional* dependency (``pip install
'elbow-helper[cli]'``): the argparse CLI is the always-available one, so
importing this module lazily requires click and fails with an actionable
message if it is missing. Every command delegates to the shared
:mod:`elbow_helper._core_cli`, so the two CLIs can never drift apart.

Consumes: ``click`` (optional), ``elbow_helper._core_cli``, ``os_helper``.
Produces: :func:`cli` (the ``elbow-helper-click`` console-script entry point).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json
import logging
from typing import Optional

try:
    import click
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the click CLI needs the [cli] extra. Run: pip install 'elbow-helper[cli]' "
        "(the argparse CLI `elbow-helper` is always available without it)."
    ) from exc

import os_helper as oh

from . import _core_cli as core
from .cli_argparse import _load_series


def _emit(result: dict | str) -> None:
    """Print a core result as raw SVG (str) or indented JSON (dict).

    Parameters
    ----------
    result : dict or str
        A core function's result — a JSON-ready dict, or the raw SVG string
        from ``diagnostics``.
    """
    click.echo(result if isinstance(result, str) else json.dumps(result, indent=2))


def _resolve_xy(
    x_values: Optional[str],
    x_npy: Optional[str],
    x_csv: Optional[str],
    y_values: Optional[str],
    y_npy: Optional[str],
    y_csv: Optional[str],
) -> tuple[list, Optional[list]]:
    """Resolve x/y series from the click data options, with the y-only shorthand.

    Mirrors :func:`elbow_helper.cli_argparse._xy_from_args` for the click
    surface, reusing the same three-way loader so both CLIs read identical
    inputs identically.

    Parameters
    ----------
    x_values, x_npy, x_csv : str, optional
        The x-side inputs (mutually exclusive; see :func:`_load_series`).
    y_values, y_npy, y_csv : str, optional
        The y-side inputs.

    Returns
    -------
    tuple of (list, list or None)
        ``(x, y)``, with ``y`` ``None`` for the shorthand form.

    Raises
    ------
    click.UsageError
        If neither side had any input.
    """
    x = _load_series(x_values, x_npy, x_csv)
    y = _load_series(y_values, y_npy, y_csv)
    if x is None and y is None:
        raise click.UsageError(
            "no data given: pass --y-values/--y-npy/--y-csv (shorthand: y "
            "only) or both x and y series"
        )
    if x is None:
        return y, None
    if y is None:
        return x, None
    return x, y


# Every command below passes `help=` explicitly rather than let click derive
# it from the docstring: the docstring is full numpydoc (Parameters section
# and all, per house style), and click would otherwise print that raw RST as
# the command's --help text.


def _data_options(func):
    """Attach the shared x/y input + orientation/config options to a command.

    Parameters
    ----------
    func : Callable
        The click command function to decorate.

    Returns
    -------
    Callable
        ``func``, wrapped with every data option.
    """
    opts = [
        click.option("--x-values", help="comma-separated x values, e.g. 0,1,2,3"),
        click.option("--x-npy", help="path to a 1-D .npy file of x values"),
        click.option("--x-csv", help="PATH:COLUMN_INDEX into a CSV file for x"),
        click.option("--y-values", help="comma-separated y values"),
        click.option("--y-npy", help="path to a 1-D .npy file of y values"),
        click.option("--y-csv", help="PATH:COLUMN_INDEX into a CSV file for y"),
        click.option("--curve", type=click.Choice(["concave", "convex"]), default=None),
        click.option(
            "--direction", type=click.Choice(["increasing", "decreasing"]), default=None
        ),
        click.option(
            "--config-json",
            default=None,
            help="RobustKneeConfig field overrides as JSON, e.g. '{\"bootstrap_replicates\": 500}'",
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


@click.group()
@click.version_option(package_name="elbow-helper")
@click.option("-v", "--verbose", count=True, help="repeat for more logs")
def cli(verbose: int) -> None:
    """Noise-robust knee/elbow detection: a knee with uncertainty, or an explicit abstention.

    Parameters
    ----------
    verbose : int
        ``-v`` repeat count; raises the log level (0=WARNING, 1=INFO, 2+=DEBUG).
    """
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG)
    oh.init_logging(level=level, stdout=False)


@cli.command("knee", help="Locate a knee with uncertainty, or abstain.")
@_data_options
def knee_cmd(
    x_values,
    x_npy,
    x_csv,
    y_values,
    y_npy,
    y_csv,
    curve,
    direction,
    config_json,
) -> None:
    """Locate a knee with uncertainty, or abstain.

    Parameters
    ----------
    x_values, x_npy, x_csv, y_values, y_npy, y_csv : str, optional
        The curve inputs, from :func:`_data_options`.
    curve, direction : str, optional
        Orientation hints.
    config_json : str, optional
        ``RobustKneeConfig`` field overrides as a JSON object.
    """
    x, y = _resolve_xy(x_values, x_npy, x_csv, y_values, y_npy, y_csv)
    overrides = json.loads(config_json) if config_json else None
    _emit(
        core.do_knee(x, y, curve=curve, direction=direction, config_overrides=overrides)
    )


@cli.command("elbow", help="The convex/decreasing convenience (k-means elbow).")
@_data_options
def elbow_cmd(
    x_values,
    x_npy,
    x_csv,
    y_values,
    y_npy,
    y_csv,
    curve,
    direction,
    config_json,
) -> None:
    """The convex/decreasing convenience (k-means elbow).

    Parameters
    ----------
    x_values, x_npy, x_csv, y_values, y_npy, y_csv : str, optional
        The curve inputs, from :func:`_data_options`.
    curve, direction : str, optional
        Accepted for a uniform option surface across commands; unused —
        ``robust_elbow`` always pins convex/decreasing.
    config_json : str, optional
        ``RobustKneeConfig`` field overrides as a JSON object.
    """
    _ = curve, direction
    x, y = _resolve_xy(x_values, x_npy, x_csv, y_values, y_npy, y_csv)
    if y is None:
        raise click.UsageError(
            "elbow needs both an x and a y series (no y-only shorthand)"
        )
    overrides = json.loads(config_json) if config_json else None
    _emit(core.do_elbow(x, y, config_overrides=overrides))


@cli.command("diagnostics", help="Render the diagnostic SVG.")
@_data_options
@click.option("--out", default=None, help="write the SVG to this path")
@click.option("--language", type=click.Choice(["en", "fr"]), default="en")
def diagnostics_cmd(
    x_values,
    x_npy,
    x_csv,
    y_values,
    y_npy,
    y_csv,
    curve,
    direction,
    config_json,
    out,
    language,
) -> None:
    """Render the diagnostic SVG.

    Parameters
    ----------
    x_values, x_npy, x_csv, y_values, y_npy, y_csv : str, optional
        The curve inputs, from :func:`_data_options`.
    curve, direction : str, optional
        Orientation hints.
    config_json : str, optional
        ``RobustKneeConfig`` field overrides as a JSON object.
    out : str, optional
        Write the SVG to this path in addition to printing it.
    language : str, optional
        Chrome-text language, ``"en"`` (default) or ``"fr"``.
    """
    x, y = _resolve_xy(x_values, x_npy, x_csv, y_values, y_npy, y_csv)
    overrides = json.loads(config_json) if config_json else None
    _emit(
        core.do_diagnostics(
            x,
            y,
            curve=curve,
            direction=direction,
            config_overrides=overrides,
            out=out,
            language=language,
        )
    )


@cli.command("locator", help="The standalone locator (no uncertainty evidence).")
@_data_options
@click.option("--sensitivity", type=float, default=1.0)
@click.option("--online/--no-online", default=True)
def locator_cmd(
    x_values,
    x_npy,
    x_csv,
    y_values,
    y_npy,
    y_csv,
    curve,
    direction,
    config_json,
    sensitivity,
    online,
) -> None:
    """The standalone locator (no uncertainty evidence).

    Parameters
    ----------
    x_values, x_npy, x_csv, y_values, y_npy, y_csv : str, optional
        The curve inputs, from :func:`_data_options`.
    curve, direction : str, optional
        Orientation hints (falls back to the locator's own defaults).
    config_json : str, optional
        Accepted for a uniform option surface; unused — the standalone
        locator has no ``RobustKneeConfig``.
    sensitivity : float
        The locator's ``S`` sensitivity parameter.
    online : bool
        Whether the locator re-arms at every later, higher peak.
    """
    _ = config_json
    x, y = _resolve_xy(x_values, x_npy, x_csv, y_values, y_npy, y_csv)
    if y is None:
        raise click.UsageError("locator needs both an x and a y series")
    _emit(
        core.do_locator(
            x,
            y,
            sensitivity=sensitivity,
            curve=curve,
            direction=direction,
            online=online,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    cli()
