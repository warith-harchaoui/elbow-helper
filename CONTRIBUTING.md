# Contributing

## Setup

```bash
pip install -e ".[api,mcp,dev]"
```

Or via conda: `conda env create -f environment.yaml && conda activate elbow-helper`.

## Before you push

Install the repository's git hooks once per clone:

```bash
git config core.hooksPath .githooks
```

This activates `.githooks/pre-push`, which blocks a push on a lint or test
failure so a red local state never reaches CI: `ruff check`, `ruff format
--check`, then the full `pytest` suite with the marker filter cleared
(`-m ""`), so the tests marked `slow` that CI skips run here. Bypass only when deliberate: `git push --no-verify`.

## What CI checks

`.github/workflows/test.yml` runs two blocking jobs on every push and pull
request:

- `pytest`: on Python 3.12, `ruff check`, `ruff format --check`, then
  `pytest -m "not slow"`. Tests over about 2 s each (bootstrap sweeps,
  every-door round trips, doctests) carry the `slow` marker declared in
  `pyproject.toml`; CI skips them and the pre-push hook runs them.
- `floor`: on Python 3.10, `pip install .` then `import elbow_helper`, the
  promise `requires-python = ">=3.10"` makes. Install and import only, not
  the tests.

A third job, `compat`, runs the whole suite (`-m ""`) on Python 3.10 through
3.13 on a weekly schedule and on manual dispatch, rather than on every push,
so the default gate stays fast while compatibility drift still gets caught.

To reproduce the CI subset locally: `pytest -m "not slow"`; the whole suite:
`pytest -m ""`.

## Style

Ruff enforces linting and formatting; run `ruff check .` and `ruff format .`
before committing, or let the pre-push hook catch it. No bare `except:`, no
mutable default arguments.

## Docs

`doc/ELBOW-en.tex` derives every formula the package runs; `doc/
LIKELIHOOD-en.tex` covers the general likelihood theory behind it. Compile
with `latexmk -pdf <name>.tex` from inside `doc/`. If a source change
affects behavior described in `README.md`, `LISEZMOI.md`, `EXAMPLES.md`,
`EXEMPLES.md`, or `CHANGELOG.md`, update those too: they drift out of sync
with the code easily.
