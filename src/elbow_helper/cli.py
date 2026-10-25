"""The elbow-helper command line, built with click.

The one command-line surface. It used to be the optional twin of an argparse
CLI, reachable under a ``-click`` suffix that named the parsing library rather
than the interface; both are gone and ``click`` is a core dependency now.
Every command still delegates to the shared :mod:`elbow_helper._core_cli`,
which the HTTP and MCP doors also call, so no door can answer differently
from another.

:func:`_load_series` moved here from the deleted argparse module: it is pure
input parsing with no argparse in it, and the CLI is its only caller.

Consumes: ``click``, ``elbow_helper._core_cli``, ``os_helper``.
Produces: :func:`cli` (the ``elbow-helper`` console-script entry point).

Author
------
Warith Harchaoui, <warithmetics@deraison.ai>
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional

import click
import numpy as np
import os_helper as oh

from . import __version__
from . import _core_cli as core


def _load_series(
    values: Optional[str], npy: Optional[str], csv_column: Optional[str]
) -> Optional[list]:
    """Load one numeric series from whichever of three flags was given.

    Parameters
    ----------
    values : str, optional
        Comma-separated inline floats, e.g. ``"1,2,3.5"``.
    npy : str, optional
        Path to a 1-D ``.npy`` array.
    csv_column : str, optional
        A ``path:column_index`` pair (0-based) into a CSV file.

    Returns
    -------
    list of float or None
        The loaded series, or ``None`` if all three inputs were omitted.

    Raises
    ------
    ValueError
        If more than one of the three inputs was given (ambiguous), or a
        ``csv_column`` spec is malformed.

    Examples
    --------
    >>> _load_series("1,2,3", None, None)
    [1.0, 2.0, 3.0]
    """
    given = [v for v in (values, npy, csv_column) if v is not None]
    if len(given) > 1:
        raise ValueError("give at most one of --values / --npy / --csv, not several")
    if values is not None:
        return [float(v) for v in values.split(",")]
    if npy is not None:
        return np.load(npy).astype(float).ravel().tolist()
    if csv_column is not None:
        path, _, col = csv_column.partition(":")
        if not col:
            raise ValueError("--csv expects PATH:COLUMN_INDEX, e.g. data.csv:1")
        col_idx = int(col)
        # Plain stdlib CSV read (no pandas dependency for a one-column pull).
        import csv as csv_module

        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv_module.reader(fh))
        # Skip a header row if the first cell in the target column isn't numeric.
        start = 0
        try:
            float(rows[0][col_idx])
        except (ValueError, IndexError):
            start = 1
        return [float(row[col_idx]) for row in rows[start:]]
    return None


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

    Resolves the x / y inputs for the click
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
        If neither side had any input, if a ``--*-csv`` spec is malformed, or
        if one side was named twice.
    """
    # _load_series raises a plain ValueError for a malformed PATH:COLUMN_INDEX
    # or for two inputs naming the same side. Those are the user mistyping the
    # command, which is what UsageError is for: it prints the message with the
    # usage line and exits 2, exactly like "no data given" below. Left bare,
    # the ValueError escaped the group and only the top-level main() wrapper
    # caught it -- so the same class of mistake exited 1 with no usage hint,
    # and nothing at all under click's own test runner.
    try:
        x = _load_series(x_values, x_npy, x_csv)
        y = _load_series(y_values, y_npy, y_csv)
    except ValueError as err:
        raise click.UsageError(str(err)) from err
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


# ``-h`` beside ``--help``: the short form is the one people type.
@click.group(context_settings={"help_option_names": ["-h", "--help"]})
# ``-V`` beside ``--version``, as the argparse door offered. ``-v`` is already
# --verbose, hence the capital.
@click.version_option(__version__, "-V", "--version", package_name="elbow-helper")
@click.option("-v", "--verbose", count=True, help="repeat for more logs")
def cli(verbose: int) -> None:
    """Noise-robust knee/elbow detection: a knee with uncertainty, or an explicit abstention.

    \f

    Everything past the form feed is for readers of the source: click truncates
    ``--help`` there. Without it the numpydoc block below reaches users as one
    reflowed paragraph of "Parameters ---------- verbose : int".

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


def main() -> None:
    """Console entry point (``elbow-helper``).

    click's own ``main()`` only special-cases ``ClickException``/``Abort``
    (and a broken pipe); a plain library exception (e.g. ``ValueError`` from
    ``locator`` on a malformed curve) would otherwise propagate as a raw
    Python traceback instead of a clean CLI error. This wraps the whole
    invocation and translates that last case into a one-line stderr message
    + exit 1 — click's own control flow (usage errors, ``--help``, an
    explicit ``sys.exit(1)`` in a subcommand) already raises ``SystemExit``,
    a ``BaseException`` this does not catch, so it passes through untouched.
    """
    try:
        cli()
    except Exception as err:  # noqa: BLE001 — last resort: see docstring
        click.echo(f"Error: {err}", err=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
