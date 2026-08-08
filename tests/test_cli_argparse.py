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


def test_knee_json_output_is_valid(capsys) -> None:
    """A clear-knee curve prints a JSON object with the expected fields."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    out = _run(
        capsys,
        [
            "knee", "--x-values", _values_flag(x), "--y-values", _values_flag(y),
            "--config-json", '{"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}',
        ],
    )
    payload = json.loads(out)
    assert payload["is_clear"] is True
    assert "knee_x" in payload
    assert "ci90" in payload


def test_elbow_requires_both_series(capsys) -> None:
    """``elbow`` refuses the y-only shorthand with a clear error."""
    with pytest.raises(ValueError, match="both"):
        cli_argparse.main(["elbow", "--y-values", "1,2,3,20,21,22"])


def test_elbow_json_output(capsys) -> None:
    """A clear elbow curve routes through ``robust_elbow`` and returns JSON."""
    x, y = elbow_curve(seed=1, noise=0.02)
    out = _run(
        capsys,
        [
            "elbow", "--x-values", _values_flag(x), "--y-values", _values_flag(y),
            "--config-json", '{"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}',
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
            "diagnostics", "--x-values", _values_flag(x), "--y-values", _values_flag(y),
            "--out", str(out_path),
        ],
    )
    assert out.strip().startswith("<svg")
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<svg")


def test_locator_needs_x_and_y() -> None:
    """``locator`` refuses a y-only shorthand (it has no orientation inference)."""
    with pytest.raises(ValueError, match="both"):
        cli_argparse.main(["locator", "--y-values", "1,2,3,20,21,22"])


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
    """Calling ``knee`` with no data flags at all raises a clear error."""
    with pytest.raises(ValueError, match="no data given"):
        cli_argparse.main(["knee"])


def test_ambiguous_input_raises() -> None:
    """Passing two of --y-values/--y-npy/--y-csv at once is rejected."""
    with pytest.raises(ValueError, match="at most one"):
        cli_argparse.main(["knee", "--y-values", "1,2,3", "--y-npy", "unused.npy"])


def test_parser_import_has_no_side_effects() -> None:
    """Building the parser does not touch the network, filesystem, or exit."""
    parser = cli_argparse.build_parser()
    ns = parser.parse_args(["knee", "--y-values", "1,2,3"])
    assert ns.command == "knee"
