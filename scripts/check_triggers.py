"""Trigger-coverage checker for the elbow-helper Claude / OpenCode skill.

The suite's agent-surface rule is that a skill's triggers must be **exhaustive
and enforced**, not merely written down. A host model only ever sees the
``description:`` field of ``skills/elbow-helper/SKILL.md`` before deciding whether
to load the skill, so that one string — not the reference docs — is what has to
carry the triggers.

This script parses the YAML frontmatter and asserts that the ``description``:

* mentions at least one token from every required trigger bucket below,
* carries an explicit SKIP / negative-trigger clause so the skill does not
  over-fire,
* stays inside the published 1024-character cap for the field,
* names the skill in kebab-case, matching its folder,
* contains no XML angle brackets (a frontmatter security restriction),

and that the repo ships the companion catalogue (``TRIGGERS.md`` at the root,
mirrored into ``skills/elbow-helper/references/triggers.md``).

It is deliberately dependency-free (stdlib only) so it runs in the lightest CI
lane without installing anything.

Run it directly or through ``pytest tests/test_skill.py``::

    python scripts/check_triggers.py            # exit 0 = every bucket covered
    python scripts/check_triggers.py --list     # print the buckets and verdicts


Author
------
[Warith Harchaoui](https://www.linkedin.com/in/warith-harchaoui/)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Resolve paths from the repo root (this script lives in scripts/), so the
# checker works regardless of the caller's current directory.
_REPO = Path(__file__).resolve().parent.parent
_SKILL_DIR = _REPO / "skills" / "elbow-helper"
_SKILL_MD = _SKILL_DIR / "SKILL.md"
_ROOT_TRIGGERS = _REPO / "TRIGGERS.md"
_SKILL_TRIGGERS = _SKILL_DIR / "references" / "triggers.md"

# The published Agent Skills standard caps the description, because the field
# sits in the host model's system prompt for every conversation.
_MAX_DESCRIPTION = 1024

# Each bucket is (human label, [accepted lowercase tokens]). Coverage means the
# description contains ANY token in the bucket — synonyms are alternatives, not
# all-required — so paraphrases still count while every capability stays named.
_REQUIRED_BUCKETS: list[tuple[str, list[str]]] = [
    ("knee / elbow", ["knee", "elbow"]),
    ("choosing k", ["clusters", "k-means", "scree", "inertia", "wcss"]),
    ("diminishing returns", ["diminishing returns", "plateau", "saturat", "cutoff"]),
    ("uncertainty", ["uncertainty", "confiden", "noise", "robust"]),
    ("abstention", ["abstain", "noclearknee", "is there even"]),
    ("stopping criteria", ["stop", "iterat", "training"]),
    (
        "commands and symbols",
        ["elbow-helper", "elbow_helper", "robust_knee", "robust_elbow", "kneelocator"],
    ),
]

# A negative-trigger clause is mandatory: without a documented SKIP the skill
# over-fires on neighbouring requests the rest of the suite already handles.
_SKIP_MARKERS = ["skip when", "do not fire", "do not use", "skip:"]


def _frontmatter(text: str) -> str:
    """Return the raw YAML frontmatter block of a SKILL.md, or an empty string."""
    fm = re.search(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    return fm.group(1) if fm else ""


def _extract_description(text: str) -> str:
    """Return the ``description:`` value, lowercased and whitespace-collapsed.

    Parameters
    ----------
    text : str
        Full contents of ``SKILL.md``.

    Returns
    -------
    str
        The description text as one lowercase line. Empty when no frontmatter
        or no description was found (treated as a failure upstream).
    """
    block = _frontmatter(text)
    if not block:
        return ""
    # Match ``description:`` and everything up to the next top-level YAML key
    # (a line starting with a non-space character). This tolerates the ``>-``
    # folded-scalar style, where the value is indented across many lines.
    desc = re.search(
        r"^description:\s*(.*?)(?=\n[A-Za-z_][\w-]*:\s|\Z)",
        block,
        re.DOTALL | re.MULTILINE,
    )
    if not desc:
        return ""
    value = re.sub(r"\s+", " ", desc.group(1)).strip()
    # Drop a leading folded-scalar indicator if the regex captured it.
    value = re.sub(r"^>[-+]?\s*", "", value)
    return value.lower()


def check(verbose: bool = False) -> list[str]:
    """Return the list of failures (empty list = triggers fully enforced).

    Parameters
    ----------
    verbose : bool, optional
        When True, print every bucket with a ✓/✗ verdict to stdout.

    Returns
    -------
    list of str
        One human-readable message per uncovered bucket or broken rule.
    """
    if not _SKILL_MD.exists():
        return [f"SKILL.md not found at {_SKILL_MD}"]

    text = _SKILL_MD.read_text(encoding="utf-8")
    block = _frontmatter(text)
    description = _extract_description(text)
    if not description:
        return ["could not parse a `description:` field from SKILL.md frontmatter"]

    failures: list[str] = []

    # Every capability bucket must be represented by at least one token.
    for label, tokens in _REQUIRED_BUCKETS:
        hit = any(tok in description for tok in tokens)
        if verbose:
            print(f"  {'✓' if hit else '✗'} {label}: {tokens}")
        if not hit:
            failures.append(
                f"trigger bucket not covered in description: {label} ({tokens})"
            )

    # The SKIP clause guards against over-firing; its absence is a failure.
    has_skip = any(marker in description for marker in _SKIP_MARKERS)
    if verbose:
        print(f"  {'✓' if has_skip else '✗'} SKIP clause present")
    if not has_skip:
        failures.append("no SKIP / negative-trigger clause found in description")

    # Spec conformance: the description rides in the system prompt, so its size
    # is a hard budget rather than a style preference.
    if len(description) > _MAX_DESCRIPTION:
        failures.append(
            f"description is {len(description)} characters, over the {_MAX_DESCRIPTION} cap"
        )

    # The name must be kebab-case and match the folder, or hosts cannot load it.
    name = re.search(r"^name:\s*(.+)$", block, re.MULTILINE)
    name = name.group(1).strip() if name else ""
    if name != "elbow-helper":
        failures.append(
            f"frontmatter name {name!r} does not match folder 'elbow-helper'"
        )
    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", name):
        failures.append(f"frontmatter name {name!r} is not kebab-case")

    # Angle brackets are forbidden in frontmatter (prompt-injection surface).
    # The leading ``>-`` folded-scalar indicator is YAML, not an XML tag.
    scrubbed = re.sub(
        r"^description:\s*>[-+]?\s*$", "description:", block, flags=re.MULTILINE
    )
    if re.search(r"[<>]", scrubbed):
        failures.append("angle bracket found in frontmatter (forbidden by the spec)")

    # The spec forbids a README inside the skill folder; ours lives one level up.
    if (_SKILL_DIR / "README.md").exists():
        failures.append(
            "skills/elbow-helper/README.md exists — the spec forbids it inside a skill folder"
        )

    # The catalogue must exist on both sides: user-facing at the repo root, and
    # agent-facing under references/, so neither audience reads a stale list.
    if not _ROOT_TRIGGERS.exists():
        failures.append("TRIGGERS.md missing at the repo root")
    if not _SKILL_TRIGGERS.exists():
        failures.append("skills/elbow-helper/references/triggers.md missing")

    return failures


def main(argv: list[str] | None = None) -> int:
    """Console entry point. Exit 0 when every bucket is covered, 1 otherwise."""
    args = argv if argv is not None else sys.argv[1:]
    verbose = "--list" in args

    failures = check(verbose=verbose)
    if failures:
        # Surface every gap at once so a maintainer fixes them in one pass.
        print("elbow-helper skill triggers INCOMPLETE:", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("elbow-helper skill triggers: all buckets covered + SKIP clause present ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
