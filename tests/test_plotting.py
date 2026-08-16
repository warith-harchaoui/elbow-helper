"""Tests for ``elbow_helper.plotting`` — no test file existed for it before.

The diagnostic SVG renderer was previously only exercised indirectly, via
the CLI's ``diagnostics`` subcommand test and via its own module doctests
(``tests/test_doctests.py``), leaving the abstain paths of ``render_svg``
and ``render_svg_multi``, ``raw_axis``/``log_y`` mode, and the ``out=``
file-writing branches of ``plot_multi_diagnostics``/
``plot_diagnostics_panels`` untested.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from conftest import clear_knee_curve, noisy_line
from elbow_helper import RobustKneesConfig
from elbow_helper.plotting import (
    plot_diagnostics_panels,
    plot_multi_diagnostics,
    render_svg,
    render_svg_multi,
    render_svg_panels,
)


def test_render_svg_clear_knee_shows_the_legend(fast_config) -> None:
    """A clear-knee curve renders the evidence legend, not the abstain card."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    svg = render_svg(x, y, curve="concave", direction="increasing", config=fast_config)
    assert svg.startswith("<svg")
    assert "detection probability" in svg
    assert "No clear knee" not in svg


def test_render_svg_abstain_shows_the_reason(fast_config) -> None:
    """A monotone line abstains and the reason code appears in the SVG."""
    x, y = noisy_line(seed=0)
    svg = render_svg(x, y, curve="concave", direction="increasing", config=fast_config)
    assert "No clear knee" in svg
    assert "reason" in svg


def test_render_svg_prepared_none_still_renders_a_card(fast_config) -> None:
    """A curve incompatible with a forced orientation fails before search
    entirely (``prepared is None``) and still renders a full abstain card
    rather than a bare axes frame."""
    x, y = clear_knee_curve(seed=1, noise=0.0)
    # Force the wrong shape: this curve is concave/increasing, not
    # convex/decreasing, so prepare_curve raises INCOMPATIBLE_GLOBAL_SHAPE
    # before any search runs.
    svg = render_svg(x, y, curve="convex", direction="decreasing", config=fast_config)
    assert svg.startswith("<svg")
    assert "No clear knee" in svg
    assert "INCOMPATIBLE_GLOBAL_SHAPE" in svg


def test_render_svg_raw_axis_and_log_y(fast_config) -> None:
    """``raw_axis=True`` with ``log_y=True`` maps back to data-unit, log-scaled ticks."""
    x = np.linspace(1.0, 100.0, 80)
    y = np.where(x <= 30.0, x, 30.0 + 0.05 * (x - 30.0)) + 1.0
    svg = render_svg(
        x,
        y,
        curve="concave",
        direction="increasing",
        config=fast_config,
        raw_axis=True,
        log_y=True,
        x_label="input",
        y_label="output",
    )
    assert svg.startswith("<svg")
    assert "input" in svg and "output" in svg


def test_render_svg_multi_valid_marks_every_breakpoint() -> None:
    """A clean two-breakpoint curve renders one dashed line per breakpoint."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 100)
    y = np.piecewise(
        x,
        [x < 0.3, (x >= 0.3) & (x < 0.65), x >= 0.65],
        [lambda t: 3 * t, lambda t: 0.9 + 0.2 * (t - 0.3), lambda t: 0.97 + 2.2 * (t - 0.65)],
    ) + rng.normal(0, 0.02, x.size)
    svg = render_svg_multi(
        x, y, config=RobustKneesConfig(random_seed=0, fwer_permutations=200)
    )
    assert svg.startswith("<svg")
    assert svg.count('stroke-dasharray="6 5"') >= 2


def test_render_svg_multi_invalid_input_shows_abstain_card() -> None:
    """Bad input (length mismatch) can't even be prepared: abstain card, not a crash."""
    svg = render_svg_multi([1.0, 2.0, 3.0], [1.0, 2.0])
    assert svg.startswith("<svg")
    assert "No clear knee" in svg
    assert "INVALID_INPUT" in svg


def test_render_svg_multi_empty_input_does_not_crash() -> None:
    """Regression test: an empty ``(x, y)`` used to crash on ``x.min()``
    (the same "zero-size array to reduction operation minimum which has no
    identity" failure mode CHANGELOG 0.1.3 already fixed once for
    ``KneeLocator``), instead of rendering the abstain card."""
    svg = render_svg_multi([], [])
    assert svg.startswith("<svg")
    assert "No clear knee" in svg


def test_plot_multi_diagnostics_writes_svg(tmp_path) -> None:
    """``plot_multi_diagnostics(out=...)`` writes the same SVG it returns."""
    rng = np.random.default_rng(3)
    x = np.linspace(0, 1, 100)
    y = 0.2 + 0.6 * x + rng.normal(0, 0.02, x.size)
    out_path = tmp_path / "multi.svg"
    svg = plot_multi_diagnostics(
        x, y, config=RobustKneesConfig(random_seed=0), out=str(out_path)
    )
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == svg


def test_render_svg_panels_nests_every_panel(fast_config) -> None:
    """``render_svg_panels`` nests N standalone SVGs side by side."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    left = render_svg(x, y + 0.01, config=fast_config)
    right = render_svg(x, y + 0.2, config=fast_config)
    svg = render_svg_panels([left, right])
    assert svg.count("<svg") == 3


def test_plot_diagnostics_panels_writes_svg(fast_config, tmp_path) -> None:
    """``plot_diagnostics_panels(out=...)`` writes the same SVG it returns."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    panels = [render_svg(x, y, config=fast_config)]
    out_path = tmp_path / "panels.svg"
    svg = plot_diagnostics_panels(panels, out=str(out_path))
    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8") == svg
