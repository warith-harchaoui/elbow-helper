"""Tests for the argparse CLI door — always available, zero extra dependency.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import clear_knee_curve, elbow_curve
from elbow_helper import cli_argparse


def _run(capsys, argv: list[str]) -> str:
    """Drive ``cli_argparse.main(argv)`` and capture its stdout.

    Parameters
    ----------
    capsys : pytest fixture
        Captures stdout/stderr.
    argv : list of str
        Argument vector to pass to ``main``.

    Returns
    -------
    str
        Captured stdout.
    """
    assert cli_argparse.main(argv) == 0
    return capsys.readouterr().out


def _values_flag(arr: np.ndarray) -> str:
    """Format a numpy array as a comma-separated ``--*-values`` string."""
    return ",".join(f"{v:.6g}" for v in arr)


def test_root_help(capsys) -> None:
    """``elbow-helper --help`` lists every subcommand."""
    with pytest.raises(SystemExit) as exc:
        cli_argparse.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("knee", "elbow", "diagnostics", "locator"):
        assert cmd in out


def test_subcommand_help(capsys) -> None:
    """``elbow-helper knee --help`` documents the data-input flags."""
    with pytest.raises(SystemExit) as exc:
        cli_argparse.main(["knee", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--y-values" in out


def test_knee_y_only_shorthand(capsys) -> None:
    """A bare ``--y-values`` triggers the ``robust_knee(y)`` shorthand."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    out = _run(capsys, ["knee", "--y-values", _values_flag(y)])
    payload = json.loads(out)
    assert "is_clear" in payload


def test_knee_x_only_shorthand(capsys) -> None:
    """A bare ``--x-values`` triggers the same ``robust_knee(y)`` shorthand,
    entered through the x-side flags (``_xy_from_args``'s other branch)."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    out = _run(capsys, ["knee", "--x-values", _values_flag(y)])
    payload = json.loads(out)
    assert "is_clear" in payload


def test_load_series_from_npy(capsys, tmp_path) -> None:
    """``--y-npy`` loads a 1-D ``.npy`` array (never exercised before)."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    npy_path = tmp_path / "y.npy"
    np.save(npy_path, y)
    out = _run(capsys, ["knee", "--y-npy", str(npy_path)])
    payload = json.loads(out)
    assert "is_clear" in payload


def test_load_series_from_csv_with_header(capsys, tmp_path) -> None:
    """``--y-csv PATH:COLUMN`` skips a non-numeric header row automatically."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    csv_path = tmp_path / "y.csv"
    csv_path.write_text("value\n" + "\n".join(f"{v:.6g}" for v in y), encoding="utf-8")
    out = _run(capsys, ["knee", "--y-csv", f"{csv_path}:0"])
    payload = json.loads(out)
    assert "is_clear" in payload


def test_load_series_from_csv_without_header(capsys, tmp_path) -> None:
    """``--y-csv PATH:COLUMN`` reads every row when the first one is already numeric."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    csv_path = tmp_path / "y.csv"
    csv_path.write_text("\n".join(f"{v:.6g}" for v in y), encoding="utf-8")
    out = _run(capsys, ["knee", "--y-csv", f"{csv_path}:0"])
    payload = json.loads(out)
    assert "is_clear" in payload


def test_load_series_csv_malformed_spec_raises(capsys) -> None:
    """``--y-csv`` without a ``:COLUMN_INDEX`` suffix is a clear error, exit 1."""
    code = cli_argparse.main(["knee", "--y-csv", "data.csv"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "PATH:COLUMN_INDEX" in err


def test_knee_json_output_is_valid(capsys) -> None:
    """A clear-knee curve prints a JSON object with the expected fields."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    out = _run(
        capsys,
        [
            "knee",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
            "--config-json",
            '{"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}',
        ],
    )
    payload = json.loads(out)
    assert payload["is_clear"] is True
    assert "knee_x" in payload
    assert "ci90" in payload


def test_elbow_requires_both_series(capsys) -> None:
    """``elbow`` refuses the y-only shorthand with a clear error, exit 1."""
    code = cli_argparse.main(["elbow", "--y-values", "1,2,3,20,21,22"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "both" in err


def test_elbow_json_output(capsys) -> None:
    """A clear elbow curve routes through ``robust_elbow`` and returns JSON."""
    x, y = elbow_curve(seed=1, noise=0.02)
    out = _run(
        capsys,
        [
            "elbow",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
            "--config-json",
            '{"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}',
        ],
    )
    payload = json.loads(out)
    assert "is_clear" in payload


def test_diagnostics_writes_svg(capsys, tmp_path) -> None:
    """``diagnostics --out`` writes a real SVG file and echoes it to stdout."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    out_path = tmp_path / "diag.svg"
    out = _run(
        capsys,
        [
            "diagnostics",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
            "--out",
            str(out_path),
        ],
    )
    assert out.strip().startswith("<svg")
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<svg")


def test_locator_needs_x_and_y(capsys) -> None:
    """``locator`` refuses a y-only shorthand (it has no orientation inference)."""
    code = cli_argparse.main(["locator", "--y-values", "1,2,3,20,21,22"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "both" in err


def test_locator_json_output(capsys) -> None:
    """``locator`` returns the raw knee/all_knees payload."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    out = _run(
        capsys,
        ["locator", "--x-values", _values_flag(x), "--y-values", _values_flag(y)],
    )
    payload = json.loads(out)
    assert "knee" in payload
    assert "all_knees" in payload


def test_no_data_raises(capsys) -> None:
    """Calling ``knee`` with no data flags at all prints a clear error, exit 1."""
    code = cli_argparse.main(["knee"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "no data given" in err


def test_ambiguous_input_raises(capsys) -> None:
    """Passing two of --y-values/--y-npy/--y-csv at once is rejected, exit 1."""
    code = cli_argparse.main(["knee", "--y-values", "1,2,3", "--y-npy", "unused.npy"])
    assert code == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "at most one" in err


def test_parser_import_has_no_side_effects() -> None:
    """Building the parser does not touch the network, filesystem, or exit."""
    parser = cli_argparse.build_parser()
    ns = parser.parse_args(["knee", "--y-values", "1,2,3"])
    assert ns.command == "knee"


def test_locator_online_flag_can_be_disabled() -> None:
    """``--online`` defaults to True and ``--no-online`` actually clears it.

    Regression test: ``--online`` used to be declared with
    ``action="store_true", default=True``, which can only ever leave the
    value True; there was no way to pass ``online=False`` through this CLI,
    unlike the click twin's ``--online/--no-online`` pair.
    """
    parser = cli_argparse.build_parser()
    default_ns = parser.parse_args(["locator", "--x-values", "1,2", "--y-values", "1,2"])
    assert default_ns.online is True
    off_ns = parser.parse_args(
        ["locator", "--x-values", "1,2", "--y-values", "1,2", "--no-online"]
    )
    assert off_ns.online is False


def test_locator_no_online_flag_runs(capsys) -> None:
    """``locator --no-online`` dispatches successfully with ``online=False``."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    out = _run(
        capsys,
        [
            "locator",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
            "--no-online",
        ],
    )
    payload = json.loads(out)
    assert "knee" in payload
    assert "all_knees" in payload
