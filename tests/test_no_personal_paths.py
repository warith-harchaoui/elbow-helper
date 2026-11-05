"""No machine-specific path in a tracked file.

A path written in a tracked file travels with every clone, every wheel and every
published page. Two families of path never belong there:

1. **A path that names a machine or an account**, in any tracked text file:
   a home under ``/Users/``, ``/home/`` or ``C:\\Users\\``, a mounted volume
   under ``/Volumes/``, or a ``~/<folder>`` that only exists on one machine.
   Fictitious placeholders (``/home/user``, ``/Users/you``) are allowed: they
   name nobody.
2. **A hard-coded absolute path in code** (a Python or Rust string literal
   whose whole value is a filesystem path). A path comes from an environment
   variable, an argument or a config file; a default is computed at run time
   (relative to ``__file__``, ``tempfile``, a cache under ``Path.home()``).
   The only literals left are the ones an OS or a protocol imposes, listed in
   ``SYSTEM_PATHS`` and ``EXCEPTIONS`` below, each with its reason.

The scan reads ``git ls-files``, so it sees exactly what a clone sees. This
file is excluded from its own scan: it has to spell the patterns it hunts.
"""

from __future__ import annotations

import ast
import re
import subprocess
import warnings
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve().relative_to(REPO).as_posix()

# Placeholder account names: they appear in examples and name nobody.
PLACEHOLDER_USERS = {
    "user",
    "username",
    "you",
    "vous",
    "me",
    "moi",
    "alice",
    "bob",
    "bobby",
    "runner",
    "runneradmin",  # GitHub Actions accounts, on a throwaway VM
    "app",
    "dev",  # accounts inside a container image
}
# Placeholder volumes, same idea.
PLACEHOLDER_VOLUMES = {"some-unmounted-disk"}
# ``~/<x>`` that exist on any machine, or that are obviously examples.
GENERIC_HOME_DIRS = {
    "Downloads",
    "Documents",
    "Desktop",
    "Library",
    "AppData",  # OS standard
    "project",
    "projects",  # example folder in a docstring
    "...",
}

HOME = re.compile(
    r"(?<![\w.$-])(?:/Users/|/home/|[A-Za-z]:\\{1,2}Users\\{1,2})([A-Za-z][\w.-]*)"
)
VOLUME = re.compile(r"(?<![\w.$-])/Volumes/([\w.-]+)")
TILDE = re.compile(r"(?<![\w/~$-])~/([A-Za-z0-9_.][\w.-]*|\.\.\.)")
# The same thing spelled in code: ``Path.home() / "some-checkout"``.
HOME_JOIN = re.compile(
    r"""Path\.home\(\)\s*(?:/\s*|\.joinpath\(\s*)[rf]?["']([^"'/]+)"""
)

# Filesystem roots: a literal starting with one of these is a filesystem path,
# not a URL route (``"/health"``) or a regex.
ROOTS = (
    "Users|home|Volumes|tmp|var|opt|usr|etc|private|mnt|srv|data|root|app|"
    "Library|System|Applications|nix|snap|dev|proc|sys|bin|sbin|lib|run|media"
)
ABSOLUTE = re.compile(rf"^(?:/(?:{ROOTS})(?:/|$)|[A-Za-z]:\\)")

# Literals an OS or a protocol imposes: nobody can choose another value.
SYSTEM_PATHS = re.compile(
    r"^(?:"
    r"/dev/(?:null|stdin|stdout|stderr|video\S*|snd)"  # device files
    r"|/proc/\S+|/sys/\S+"  # Linux kernel interfaces
    r"|/usr/bin/env|/bin/(?:sh|bash|zsh)"  # interpreters
    r"|/(?:System/)?Library/Fonts\S*|/usr(?:/local)?/share/fonts"  # font dirs
    r"|/opt/homebrew|/usr/local|/usr"  # package-manager prefixes
    r"|/Applications/[\w ]+\.app\S*"  # where macOS installs an application
    r"|/etc/passwd|[A-Za-z]:\\Windows\\\S*"  # canonical path-traversal targets
    r")$"
)

# Literals the scan flags but that are right where they stand. Each entry:
# (file glob, literal regex, reason). Keep it short; justify every line.
EXCEPTIONS: list[tuple[str, str, str]] = [
    # none in this repository
]

TEXT_SUFFIXES = {
    ".py",
    ".rs",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".cfg",
    ".ini",
    ".sh",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".php",
    ".rst",
    ".svg",
    ".csv",
    ".tex",
    ".bib",
    ".env",
    ".example",
    ".lock",
    ".in",
    ".po",
    ".j2",
}
TEXT_NAMES = {"Dockerfile", "Makefile", "pre-push", "pre-commit", ".gitignore"}


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True
    ).stdout.decode()
    return [f for f in out.split("\0") if f and f != SELF]


def _is_text(rel: str) -> bool:
    p = Path(rel)
    return p.suffix.lower() in TEXT_SUFFIXES or p.name in TEXT_NAMES


def _excepted(rel: str, literal: str) -> bool:
    return any(
        Path(rel).match(glob) and re.fullmatch(rx, literal)
        for glob, rx, _why in EXCEPTIONS
    )


def personal_paths(rel: str, text: str) -> list[str]:
    """Paths naming an account, a volume or a one-machine ``~/`` folder."""
    found = []
    for n, line in enumerate(text.splitlines(), 1):
        for m in HOME.finditer(line):
            if m.group(1) not in PLACEHOLDER_USERS:
                found.append(f"{rel}:{n}: {m.group(0)}")
        for m in VOLUME.finditer(line):
            if m.group(1) not in PLACEHOLDER_VOLUMES:
                found.append(f"{rel}:{n}: {m.group(0)}")
        for m in TILDE.finditer(line):
            name = m.group(1)
            if name.startswith(".") or name in GENERIC_HOME_DIRS:
                continue  # dotfiles and caches are per-tool, not per-machine
            if rel.startswith(".github/workflows/"):
                continue  # the runner's home, rebuilt for every job
            found.append(f"{rel}:{n}: {m.group(0)}")
        for m in HOME_JOIN.finditer(line):
            name = m.group(1)
            if not name.startswith(".") and name not in GENERIC_HOME_DIRS:
                found.append(f"{rel}:{n}: {m.group(0)}")
    return [f for f in found if not _excepted(rel, f.split(": ", 1)[1])]


def _python_literals(text: str) -> list[tuple[int, str]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(text)
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    # Inside an f-string only the leading piece can start a path; the pieces
    # after a placeholder (``f"{root}/run"``) are relative to it.
    f_string_tails = {
        id(value)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for value in node.values[1:]
    }
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr) and node.values:
            first = node.values[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.append((node.lineno, first.value))
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
            and id(node) not in f_string_tails
        ):
            out.append((node.lineno, node.value))
    return out


RUST_STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _rust_literals(text: str) -> list[tuple[int, str]]:
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("//"):
            continue
        out.extend((n, m.group(1)) for m in RUST_STRING.finditer(line))
    return out


def hardcoded_paths(rel: str, text: str) -> list[str]:
    """Absolute filesystem paths written as a code literal."""
    if rel.endswith(".py"):
        try:
            literals = _python_literals(text)
        except SyntaxError:
            return [f"{rel}: does not parse, cannot be checked"]
    elif rel.endswith(".rs"):
        literals = _rust_literals(text)
    else:
        return []
    found = []
    for n, value in literals:
        lit = value.strip()
        if not ABSOLUTE.match(lit) or SYSTEM_PATHS.match(lit):
            continue
        if lit.startswith(("/path/to/", "/home/user/", "/Users/you/")):
            continue  # fictitious by construction
        if _excepted(rel, lit):
            continue
        found.append(f"{rel}:{n}: {lit!r}")
    return found


def _scan() -> tuple[list[str], list[str]]:
    personal, hardcoded = [], []
    for rel in _tracked():
        path = REPO / rel
        if not _is_text(rel) or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        personal += personal_paths(rel, text)
        hardcoded += hardcoded_paths(rel, text)
    return personal, hardcoded


@pytest.fixture(scope="module")
def scan() -> tuple[list[str], list[str]]:
    try:
        return _scan()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout (sdist): nothing tracked to scan")


def test_no_personal_path_in_tracked_files(scan):
    personal, _ = scan
    assert not personal, (
        "paths naming a machine or an account in tracked files "
        "(use an environment variable or an argument):\n" + "\n".join(personal)
    )


def test_no_hardcoded_absolute_path_in_code(scan):
    _, hardcoded = scan
    assert not hardcoded, (
        "hard-coded absolute paths in code (read an environment variable or an "
        "argument, or compute the default at run time):\n" + "\n".join(hardcoded)
    )


def test_the_scan_catches_what_it_hunts():
    """The patterns fire on a real path and stay quiet on a placeholder."""
    who = "war" + "ith"
    assert personal_paths("x.md", f"see /Users/{who}/notes")
    assert personal_paths("x.md", f"C:\\Users\\{who}\\x")
    assert personal_paths("x.md", "on /Volumes/" + "scratch/x")
    assert personal_paths("x.md", "run it from ~/" + "my-checkout")
    assert personal_paths("x.py", "D = Path.home() / " + '"my-checkout"')
    assert not personal_paths("x.py", 'C = Path.home() / ".cache" / "x"')
    assert not personal_paths("x.md", "/home/user/a ~/.cache/x ~/Downloads")
    assert hardcoded_paths("x.py", 'LOG = "/' + 'tmp/run.log"\n')
    assert hardcoded_paths("x.rs", 'let p = "/' + 'var/lib/x";\n')
    assert not hardcoded_paths("x.py", 'NULL = "/dev/null"\nR = "/health"\n')
    assert not hardcoded_paths("x.py", 'LOG = f"{root}/' + 'var/run.log"\n')
