"""Browser stand-in for ``os_helper``, registered in ``sys.modules`` by
``backend-pyodide.js`` BEFORE the elbow-helper wheel installs.

The real ``os-helper`` declares ``psutil`` (a C extension absent from the
Pyodide distribution) plus five other dependencies the browser never needs.
``elbow_helper`` only ever touches six of its symbols: the logging quartet
(``info``/``warning``/``debug``/``init_logging``) and two path helpers used
solely by the ``plot_*(..., out=...)`` file-saving branch, which the web app
never takes (it calls the pure, string-returning ``render_svg``). So the stub
routes log lines to the browser console and implements the two path helpers
with the stdlib, byte-compatible with the real signatures.
"""

from __future__ import annotations

import os
import sys
import types


def _log(*args: object, **_kwargs: object) -> None:
    """Print to stdout, which Pyodide forwards to the browser console."""
    print(*args)


def _make_directory(path: str | None = None) -> None:
    """Create ``path`` (and parents) if given; the real helper does the same."""
    if path:
        os.makedirs(path, exist_ok=True)


def _folder_name_ext(path: str) -> tuple[str, str, str]:
    """Split ``path`` into (folder, stem, extension-without-dot), like os_helper."""
    folder = os.path.dirname(path)
    stem, ext = os.path.splitext(os.path.basename(path))
    return folder, stem, ext.lstrip(".")


def register() -> None:
    """Install the stub as ``sys.modules['os_helper']`` (idempotent)."""
    if "os_helper" in sys.modules:
        return
    module = types.ModuleType("os_helper")
    module.info = _log
    module.warning = _log
    module.debug = _log
    module.error = _log
    module.init_logging = lambda *a, **k: None
    module.make_directory = _make_directory
    module.folder_name_ext = _folder_name_ext
    sys.modules["os_helper"] = module


register()
