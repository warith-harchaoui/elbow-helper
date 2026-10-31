"""One guard: requirements.txt must declare exactly what the package declares.

Two files state this package's runtime dependencies, and until 2026-10-31
nothing compared them. They had drifted in ten of the suite's sixteen
dependents at once: `click` became a core dependency when the argparse door was
dropped, `pyproject.toml` recorded it, and `requirements.txt` did not — so
`pip install -r requirements.txt` built an environment whose command line could
not start, while `pip install -e .` built a working one. Neither reader could
notice, because a *missing* requirement fails only when something imports it
and a *surplus* one installs cleanly.

Floors drifted the same way: `os-helper>=1.5.0` sat in requirements.txt while
the package itself demanded `>=2.6.0`.

Deliberately regex, not `tomllib`: this file must run on the floor Python
(3.10), where `tomllib` is not in the standard library, and `[project]
dependencies` is a flat array of strings that needs no TOML parser.

Author
------
Warith HARCHAOUI, https://linkedin.com/in/warith-harchaoui
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _declared() -> set[str]:
    """The `[project] dependencies` array of pyproject.toml, as written."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.S | re.M)
    assert match, "pyproject.toml has no [project] dependencies array"
    # Comments inside the array quote specifiers too ("it was \">=1.0\""), so
    # they have to go before anything is read as a dependency.
    body = re.sub(r"#[^\n]*", "", match.group(1))
    return {m.group(1) for m in re.finditer(r'"([^"]+)"', body)}


def _pinned() -> set[str]:
    """The dependency lines of requirements.txt, comments stripped."""
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {s for s in (line.split("#")[0].strip() for line in lines) if s}


def test_requirements_txt_mirrors_the_package_metadata() -> None:
    declared, pinned = _declared(), _pinned()
    assert pinned == declared, (
        f"requirements.txt only: {sorted(pinned - declared)}; "
        f"pyproject.toml only: {sorted(declared - pinned)}"
    )
