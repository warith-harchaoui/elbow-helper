# Contributing

## Setup

```bash
pip install -e ".[cli,api,mcp,dev]"
```

Or via conda: `conda env create -f environment.yaml && conda activate elbow-helper`.

## Before you push

Install the repository's git hooks once per clone:

```bash
git config core.hooksPath .githooks
```

This activates `.githooks/pre-push`, which blocks a push on a lint or test
failure so a red local state never reaches CI: `ruff check`, `ruff format
--check`, then the full `pytest` suite (every marker, not just the default
subset). Bypass only when deliberate: `git push --no-verify`.

## What CI checks

`.github/workflows/test.yml` runs two fast, blocking jobs on every push and
pull request: `pytest` on Python 3.12, and `ruff check` plus `ruff format
--check`. A third job, `compat`, sweeps the full supported range (Python
3.10 through 3.13, matching `requires-python` in `pyproject.toml`) on a
weekly schedule and on manual dispatch, rather than on every push, so the
default gate stays fast while compatibility drift still gets caught.

## Style

Ruff enforces linting and formatting; run `ruff check .` and `ruff format .`
before committing, or let the pre-push hook catch it. No bare `except:`, no
mutable default arguments.

## Docs

`doc/ELBOW-en.tex` derives every formula the package runs; `doc/
LIKELIHOOD-en.tex` covers the general likelihood theory behind it. Compile
with `latexmk -pdf <name>.tex` from inside `doc/`. If a source change
affects behavior described in `README.md`, `LISEZ-MOI.md`, `EXAMPLES.md`,
`EXEMPLES.md`, or `CHANGELOG.md`, update those too: they drift out of sync
with the code easily.
