"""Tests for the click CLI door — skipped cleanly when the ``[cli]`` extra is absent.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import clear_knee_curve, elbow_curve


def _values_flag(arr: np.ndarray) -> str:
    """Format a numpy array as a comma-separated ``--*-values`` string."""
    return ",".join(f"{v:.6g}" for v in arr)


@pytest.fixture
def runner():
    """A ``CliRunner``, skipped when ``click`` is not installed.

    Returns
    -------
    click.testing.CliRunner
        A runner for invoking :data:`elbow_helper.cli_click.cli`.
    """
    pytest.importorskip("click")
    from click.testing import CliRunner

    return CliRunner()


@pytest.fixture
def cli():
    """The click command group, skipped when ``click`` is not installed."""
    pytest.importorskip("click")
    from elbow_helper.cli_click import cli as group

    return group


def test_root_help(runner, cli) -> None:
    """``elbow-helper-click --help`` lists every subcommand."""
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("knee", "elbow", "diagnostics", "locator"):
        assert cmd in result.output


def test_knee_y_only_shorthand(runner, cli) -> None:
    """A bare ``--y-values`` triggers the shorthand and returns valid JSON."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    result = runner.invoke(cli, ["knee", "--y-values", _values_flag(y)])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "is_clear" in payload


def test_knee_json_output(runner, cli) -> None:
    """A clear-knee curve returns ``is_clear: true`` with the expected fields."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    result = runner.invoke(
        cli,
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
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["is_clear"] is True
    assert "ci90" in payload


def test_elbow_requires_both_series(runner, cli) -> None:
    """``elbow`` refuses the y-only shorthand with a usage error (exit code 2)."""
    result = runner.invoke(cli, ["elbow", "--y-values", "1,2,3,20,21,22"])
    assert result.exit_code == 2
    assert "both" in result.output


def test_elbow_json_output(runner, cli) -> None:
    """A clear elbow curve routes through ``robust_elbow`` and returns JSON."""
    x, y = elbow_curve(seed=1, noise=0.02)
    result = runner.invoke(
        cli,
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
    assert result.exit_code == 0
    assert "is_clear" in json.loads(result.output)


def test_diagnostics_writes_svg(runner, cli, tmp_path) -> None:
    """``diagnostics --out`` writes a real SVG file and echoes it to stdout."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    out_path = tmp_path / "diag.svg"
    result = runner.invoke(
        cli,
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
    assert result.exit_code == 0
    assert result.output.strip().startswith("<svg")
    assert out_path.exists()


def test_locator_needs_x_and_y(runner, cli) -> None:
    """``locator`` refuses a y-only shorthand (it has no orientation inference)."""
    result = runner.invoke(cli, ["locator", "--y-values", "1,2,3,20,21,22"])
    assert result.exit_code == 2
    assert "both" in result.output


def test_locator_json_output(runner, cli) -> None:
    """``locator`` returns the raw knee/all_knees payload."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    result = runner.invoke(
        cli,
        ["locator", "--x-values", _values_flag(x), "--y-values", _values_flag(y)],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "knee" in payload
    assert "all_knees" in payload


def test_no_data_raises_usage_error(runner, cli) -> None:
    """Calling ``knee`` with no data flags at all is a usage error."""
    result = runner.invoke(cli, ["knee"])
    assert result.exit_code == 2
    assert "no data given" in result.output


def test_main_prints_clean_error_on_a_real_library_exception(
    capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CliRunner.invoke (used by every test above) catches exceptions itself,
    # so it never exercises main() -- the actual `elbow-helper-click`
    # console-script entry point. Drive main() directly: an unparseable
    # --x-values raises a plain ValueError from float(), which used to
    # propagate as a raw Python traceback instead of a clean "Error: ..."
    # + exit 1.
    pytest.importorskip("click")
    from elbow_helper.cli_click import main

    monkeypatch.setattr(
        "sys.argv",
        ["elbow-helper-click", "locator", "--x-values", "", "--y-values", ""],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Error:" in capsys.readouterr().err
