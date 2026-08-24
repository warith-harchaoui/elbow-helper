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


def _assert_doctests_pass(module) -> None:
    results = doctest.testmod(module, verbose=False)
    assert results.failed == 0, f"{module.__name__}: {results.failed} doctest(s) failed"


def test_every_module_with_doctests_stays_accurate() -> None:
    """Run every module's ``>>> ...`` doctest examples in one place.

    ``pyproject.toml`` scopes ``pytest`` to ``testpaths = ["tests"]`` with
    no ``--doctest-modules``, so these were never actually executed by CI
    without this: they could drift from real behavior (the same failure
    mode CHANGELOG's 0.1.2 entry already fixed once for ``EXAMPLES.md``)
    without anything catching it.
    """
    import importlib.util

    from elbow_helper import _core_cli, cli_argparse, plotting

    for module in (_core_cli, cli_argparse, plotting):
        _assert_doctests_pass(module)

    if importlib.util.find_spec("fastapi") is not None:
        from elbow_helper import api

        _assert_doctests_pass(api)

    if importlib.util.find_spec("fastapi_mcp") is not None:
        from elbow_helper import mcp_server

        _assert_doctests_pass(mcp_server)
