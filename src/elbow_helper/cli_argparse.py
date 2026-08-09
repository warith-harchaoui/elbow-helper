"""argparse command-line interface — the always-available CLI door.

The second of elbow-helper's surfaces (after the library) and, like every
other ``ai-helpers`` suite member's argparse twin, needs no third-party
dependency: argparse ships with Python, so ``elbow-helper`` works as a CLI
the moment the package is installed. Every subcommand is a thin shell over
:mod:`elbow_helper._core_cli`; the click twin (``elbow-helper-click``)
drives the exact same core.

Subcommands: ``knee``, ``elbow``, ``diagnostics``, ``locator``.

Consumes: ``elbow_helper._core_cli``, ``os_helper`` (logging to stderr).
Produces: :func:`main` (the ``elbow-helper`` console-script entry point).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional, Sequence

import numpy as np
import os_helper as oh

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


def _add_data_flags(p: argparse.ArgumentParser) -> None:
    """Attach the shared x/y input flags to a subparser.

    Every subcommand accepts the same three-way x/y input (inline values, a
    ``.npy`` path, or a CSV column) and the same orientation/config flags;
    only ``elbow``/``locator`` actually require both x *and* y at dispatch
    time (see :func:`_dispatch`) — ``knee``/``diagnostics`` allow the
    y-only shorthand from :func:`elbow_helper.robust_knee`.

    Parameters
    ----------
    p : argparse.ArgumentParser
        The subparser to attach flags to (mutated in place).
    """
    p.add_argument("--x-values", help="comma-separated x values, e.g. 0,1,2,3")
    p.add_argument("--x-npy", help="path to a 1-D .npy file of x values")
    p.add_argument("--x-csv", help="PATH:COLUMN_INDEX into a CSV file for x")
    p.add_argument("--y-values", help="comma-separated y values")
    p.add_argument("--y-npy", help="path to a 1-D .npy file of y values")
    p.add_argument("--y-csv", help="PATH:COLUMN_INDEX into a CSV file for y")
    p.add_argument("--curve", choices=["concave", "convex"], default=None)
    p.add_argument("--direction", choices=["increasing", "decreasing"], default=None)
    p.add_argument(
        "--config-json",
        default=None,
        help="RobustKneeConfig field overrides as JSON, e.g. '{\"bootstrap_replicates\": 500}'",
    )


def _xy_from_args(args: argparse.Namespace) -> tuple[list, Optional[list]]:
    """Resolve the x/y series an ``_add_data_flags`` subparser collected.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments from a parser built with :func:`_add_data_flags`.

    Returns
    -------
    tuple of (list, list or None)
        ``(x, y)``; ``y`` is ``None`` for the ``robust_knee(y)`` shorthand
        (a single series interpreted as y-values against an implicit
        ``0, 1, ..., n-1`` x axis). The shorthand triggers whichever of
        ``--x-*``/``--y-*`` was actually given when only one side was —
        a bare ``--y-values ...`` is the natural way to invoke it, but a
        bare ``--x-values ...`` works the same way (matching the library's
        own single-positional-argument shorthand).

    Raises
    ------
    ValueError
        If neither an x nor a y series was given at all.
    """
    x = _load_series(args.x_values, args.x_npy, args.x_csv)
    y = _load_series(args.y_values, args.y_npy, args.y_csv)
    if x is None and y is None:
        raise ValueError(
            "no data given: pass --y-values/--y-npy/--y-csv (shorthand: y "
            "only, per robust_knee(y)) or both x and y series"
        )
    if x is None:
        # Only --y-* was given: that series is the shorthand y-only input.
        return y, None
    if y is None:
        # Only --x-* was given: same shorthand, entered through the x flags.
        return x, None
    return x, y


def _config_overrides_from_args(args: argparse.Namespace) -> Optional[dict]:
    """Parse the ``--config-json`` flag into a dict, or ``None`` if unset.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments carrying an optional ``config_json`` string.

    Returns
    -------
    dict or None
        The parsed overrides mapping, or ``None``.
    """
    if not args.config_json:
        return None
    return json.loads(args.config_json)


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser with every subcommand.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser (used by :func:`main` and by the tests).

    Examples
    --------
    >>> parser = build_parser()
    >>> ns = parser.parse_args(["knee", "--y-values", "0,1,2,3"])
    >>> ns.command
    'knee'
    """
    parser = argparse.ArgumentParser(
        prog="elbow-helper",
        description="Noise-robust knee/elbow detection: a knee with uncertainty, or an explicit abstention.",
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="repeat for more logs"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_knee = sub.add_parser("knee", help="locate a knee with uncertainty, or abstain")
    _add_data_flags(p_knee)

    p_elbow = sub.add_parser(
        "elbow", help="the convex/decreasing convenience (k-means elbow)"
    )
    _add_data_flags(p_elbow)

    p_diag = sub.add_parser("diagnostics", help="render the diagnostic SVG")
    _add_data_flags(p_diag)
    p_diag.add_argument("--out", help="write the SVG to this path")
    p_diag.add_argument("--language", choices=["en", "fr"], default="en")

    p_loc = sub.add_parser(
        "locator", help="the standalone locator (no uncertainty evidence)"
    )
    _add_data_flags(p_loc)
    p_loc.add_argument("--sensitivity", type=float, default=1.0)
    p_loc.add_argument("--online", action="store_true", default=True)

    return parser


def _dispatch(args: argparse.Namespace) -> dict | str:
    """Route a parsed namespace to the matching core function.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments, including ``command``.

    Returns
    -------
    dict or str
        The core function's result (a JSON-ready dict, or the raw SVG for
        ``diagnostics``).
    """
    overrides = _config_overrides_from_args(args)
    if args.command == "knee":
        x, y = _xy_from_args(args)
        return core.do_knee(
            x, y, curve=args.curve, direction=args.direction, config_overrides=overrides
        )
    if args.command == "elbow":
        x, y = _xy_from_args(args)
        if y is None:
            raise ValueError("elbow needs both --x-* and --y-* (no y-only shorthand)")
        return core.do_elbow(x, y, config_overrides=overrides)
    if args.command == "diagnostics":
        x, y = _xy_from_args(args)
        return core.do_diagnostics(
            x,
            y,
            curve=args.curve,
            direction=args.direction,
            config_overrides=overrides,
            out=args.out,
            language=args.language,
        )
    if args.command == "locator":
        x, y = _xy_from_args(args)
        if y is None:
            raise ValueError("locator needs both --x-* and --y-*")
        return core.do_locator(
            x,
            y,
            sensitivity=args.sensitivity,
            curve=args.curve,
            direction=args.direction,
            online=args.online,
        )
    raise ValueError(f"unknown command {args.command!r}")  # pragma: no cover


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for the ``elbow-helper`` console script.

    Parameters
    ----------
    argv : sequence of str, optional
        Argument vector (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success).

    Examples
    --------
    >>> main(["knee", "--y-values", "0,0.5,0.9,0.95,0.97,0.98,0.99"])  # doctest: +SKIP
    0
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Logs go to stderr so stdout stays clean, pipeable JSON (a suite convention).
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    oh.init_logging(level=level, stdout=False)
    result = _dispatch(args)
    if isinstance(result, str):
        print(result)
    else:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
