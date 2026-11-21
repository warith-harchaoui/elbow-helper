"""Every command line ends its ``--help`` on the same invitation to star the repository.

One blank line, then ``⭐️ on GitHub: <url>``, as the last line of the help of
each console script the package installs (the MCP and skill servers are not
command lines).
"""

import importlib
import sys

import pytest

STAR = "⭐️ on GitHub: https://github.com/warith-harchaoui/elbow-helper"

# Hand-listed rather than read from pyproject.toml: tomllib does not exist on
# Python 3.10, the floor, and an installed wheel has no pyproject.toml anyway.
ENTRY_POINTS = [
    ("elbow-helper", "elbow_helper.cli:main"),
]


@pytest.mark.parametrize(
    ("name", "target"), ENTRY_POINTS, ids=[n for n, _ in ENTRY_POINTS]
)
def test_the_help_ends_with_the_star_line(name, target, monkeypatch, capsys):
    module, attr = target.split(":")
    main = getattr(importlib.import_module(module), attr)
    monkeypatch.setattr(sys, "argv", [name, "--help"])
    try:
        main()
    except SystemExit as exc:
        assert exc.code in (0, None)
    lines = capsys.readouterr().out.rstrip().splitlines()
    assert lines[-1].strip() == STAR
    assert lines[-2].strip() == ""
