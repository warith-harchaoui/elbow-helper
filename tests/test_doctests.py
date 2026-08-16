"""Run every module's docstring doctests as part of the real test suite.

``pyproject.toml`` scopes ``pytest`` to ``testpaths = ["tests"]`` with no
``--doctest-modules``, so the ``>>> ...`` examples embedded in
``_core_cli.py``, ``cli_argparse.py``, ``plotting.py``, ``api.py`` and
``mcp_server.py`` were never actually executed by CI: they could drift from
real behaviour (the same failure mode CHANGELOG's 0.1.2 entry already fixed
once for ``EXAMPLES.md``) without anything catching it. This module closes
that gap by running ``doctest.testmod`` on each of them directly, so a stale
example fails ``pytest -q`` like any other test.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import doctest

import pytest


def _assert_doctests_pass(module) -> None:
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, f"{module.__name__}: {results.failed} doctest(s) failed"


def test_core_cli_doctests() -> None:
    """``_core_cli``'s JSON-serialization and dispatch examples stay accurate."""
    from elbow_helper import _core_cli

    _assert_doctests_pass(_core_cli)


def test_cli_argparse_doctests() -> None:
    """``cli_argparse``'s ``_load_series``/``build_parser`` examples stay accurate."""
    from elbow_helper import cli_argparse

    _assert_doctests_pass(cli_argparse)


def test_plotting_doctests() -> None:
    """``plotting``'s SVG-rendering and BIC-posterior examples stay accurate."""
    from elbow_helper import plotting

    _assert_doctests_pass(plotting)


def test_api_doctests() -> None:
    """``api``'s ``create_app`` example, skipped cleanly without the ``[api]`` extra."""
    pytest.importorskip("fastapi")
    from elbow_helper import api

    _assert_doctests_pass(api)


def test_mcp_server_doctests() -> None:
    """``mcp_server``'s ``build_server`` example, skipped without the ``[mcp]`` extra."""
    pytest.importorskip("fastapi_mcp")
    from elbow_helper import mcp_server

    _assert_doctests_pass(mcp_server)
