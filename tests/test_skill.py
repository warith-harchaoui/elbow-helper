"""Tests for the elbow-helper Claude / OpenCode skill packaging.

The suite ships every helper with five surfaces: a library, two CLIs, a FastAPI
HTTP app, MCP mounted on that same app, and an agent skill. These tests cover
the fifth one: that ``skills/elbow-helper/SKILL.md`` exists, is well-formed against
the published Agent Skills standard, and that ``scripts/check_triggers.py``
reports full trigger-bucket coverage — so a thin or drifted description fails
the gate, not production.

Author
------
Warith HARCHAOUI — https://linkedin.com/in/warith-harchaoui/
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_SKILL_DIR = _REPO / "skills" / "elbow-helper"
_SKILL = _SKILL_DIR / "SKILL.md"


def _load_checker():
    """Import scripts/check_triggers.py (not an installed module) by path."""
    spec = importlib.util.spec_from_file_location(
        "check_triggers", _REPO / "scripts" / "check_triggers.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_skill_md_exists_and_has_frontmatter() -> None:
    """SKILL.md opens with YAML frontmatter carrying the right name."""
    assert _SKILL.exists(), "skills/elbow-helper/SKILL.md is missing"
    text = _SKILL.read_text(encoding="utf-8")
    assert text.startswith("---"), "SKILL.md must open with YAML frontmatter"
    assert "\nname: elbow-helper" in text
    assert "description:" in text


def test_progressive_disclosure_references_resolve() -> None:
    """Every references/*.md the body points at actually exists."""
    import re

    body = _SKILL.read_text(encoding="utf-8").split("---", 2)[-1]
    referenced = set(re.findall(r"references/([\w.-]+\.md)", body))
    assert referenced, "SKILL.md should defer depth to references/*.md"
    for ref in sorted(referenced):
        assert (_SKILL_DIR / "references" / ref).exists(), f"missing references/{ref}"


def test_triggers_catalogue_is_mirrored() -> None:
    """The catalogue exists for both audiences: repo root and skill references."""
    assert (_REPO / "TRIGGERS.md").exists(), "TRIGGERS.md missing at the repo root"
    assert (_SKILL_DIR / "references" / "triggers.md").exists(), (
        "skills/elbow-helper/references/triggers.md missing"
    )


def test_trigger_buckets_are_covered() -> None:
    """The description names every capability and carries a SKIP clause."""
    failures = _load_checker().check()
    assert not failures, "trigger coverage gaps:\n  - " + "\n  - ".join(failures)
