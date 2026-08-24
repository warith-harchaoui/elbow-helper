# Changelog

All notable changes to `elbow-helper` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.6] - 2026-08-24

### Fixed
- **`elbow_helper.smoothing.smooth_curve`** (`method="moving_average"`): an
  even smoothing window produced an asymmetric kernel, shifting the whole
  smoothed curve by half a sample (found by testing: a linear ramp's moving
  average, which should reproduce the ramp exactly at every interior point,
  came out shifted by +0.5 for even windows). The window is now forced to
  the nearest odd size below it, matching the always-odd Gaussian kernel's
  centering. Affects `preprocessing.infer_curve_direction` and
  `preprocessing._direction_violation_rate`, both of which call
  `smooth_curve` with an unrounded, possibly-even window
  (`max(5, n // 8)`), so this could bias the auto-detected `curve`/
  `direction` and the direction-violation screen on some curve lengths.
- Ralph Eyeball Loop pass on `doc/ELBOW-en.tex`/`doc/LIKELIHOOD-en.tex`:
  two real layout defects found and fixed by rendering every table and
  `tikzpicture` and inspecting it — a Laplace-approximation figure's curve
  labels running into the plotted line at its peak (re-anchored), and two
  words hyphenated mid-word inside a narrow table column (rewrapped at the
  word boundary). Both documents recompile with zero overfull/underfull/
  undefined-reference warnings.

### Changed
- CI restructured for speed: `pytest` and `ruff` now run as two separate,
  fast jobs on Python 3.12 for every push and pull request, instead of one
  job sweeping four Python versions. The full 3.10-3.13 compatibility
  matrix still runs, on a weekly schedule and on manual dispatch, so the
  version guarantee documented in `README.md`/`LISEZ-MOI.md` holds without
  slowing down the default gate.
- Added a `pre-push` git hook (`.githooks/pre-push`, opt in via `git config
  core.hooksPath .githooks`, documented in the new `CONTRIBUTING.md`) that
  blocks a push on a lint or test failure locally, running `ruff check`,
  `ruff format --check`, and the full `pytest` suite before CI ever sees
  the commit.
- Consolidated the test suite from 114 tests to 85, with higher coverage
  (69 missed lines instead of 76, out of 1793). `test_cli_argparse.py`,
  `test_cli_click.py`, `test_api.py`, and `test_mcp_server.py` — 45 tests
  that mostly hand-duplicated the same handful of scenarios (a clear knee
  accepted, the y-only shorthand, elbow rejecting that shorthand,
  diagnostics returning an SVG, the locator's raw payload) once per door —
  are now `test_doors.py`: one behavioral test per scenario that drives
  every door (argparse CLI, click CLI, HTTP API, MCP) through a small
  adapter, so each scenario verifies parity across the whole surface
  instead of whichever doors someone remembered to copy it into.
  `test_doctests.py`'s five one-per-module tests are now one loop. Two new
  tests close a real coverage gap in `pipeline.py`'s bootstrap/null-test
  gate-failure paths and its numerical-safety-net exception handler,
  previously only 83% covered.

## [0.1.5] - 2026-08-17

### Fixed
- **`KneeLocator`** raised spurious NumPy `RuntimeWarning`s on degenerate
  curves: a single-point curve triggered `"Mean of empty slice"` computing
  the sensitivity threshold's mean step (never actually reached by a
  single-point curve, so `0.0` is a safe stand-in), and a constant or
  single-point curve's min-max normalization divided by a zero range
  (`0/0 = NaN` by design, correctly propagating to an abstain, but noisy).
  Both warnings are now suppressed without changing the abstain behavior;
  a regression test turns `RuntimeWarning` into an error to prove it.

### Documentation
- Glossed previously-bare technical terms (blocked cross-validation, the
  Bonferroni-gated permutation test, PCA) in `README.md`/`LISEZ-MOI.md`,
  and removed punctuation-dash asides from five module docstrings' prose.

## [0.1.4] - 2026-08-16

### Fixed
- **`elbow_helper.plotting.render_svg_multi`** (and `plot_multi_diagnostics`):
  an empty `(x, y)` crashed with `"zero-size array to reduction operation
  minimum which has no identity"` on `x.min()`; a length-mismatched
  `(x, y)` crashed with `IndexError` while sorting `y` by `x`'s own index
  order. Both failures hit before `robust_knees` ever got a chance to
  abstain with a clean reason code, the same failure mode CHANGELOG 0.1.3
  already fixed once for `KneeLocator`. Both inputs now render the abstain
  card instead.
- **argparse CLI**: `elbow-helper locator --online` was declared with
  `action="store_true", default=True`, so the flag could only ever leave
  `online` at its default `True`; there was no way to pass `online=False`
  through this CLI, unlike the click twin's `--online/--no-online` pair.
  Now uses `argparse.BooleanOptionalAction`, so `--online`/`--no-online`
  both work and match the click twin.
- **Docs**: `README.md`, `LISEZ-MOI.md`, `EXAMPLES.md`, `EXEMPLES.md`, and
  `PAYSAGE.md` still linked to `doc/ELBOW-fr.tex` / `doc/LIKELIHOOD-fr.tex`
  and their compiled PDFs, removed from `doc/` earlier this session when
  the math notes' scope narrowed to English only. Updated every reference
  to point at the English notes, with `LISEZ-MOI.md`/`EXEMPLES.md`/
  `PAYSAGE.md` now stating plainly that the math notes are English-only
  while the rest of the docs keep their French twin. Also cleaned the same
  dangling `ELBOW-fr.tex` mentions out of code comments/docstrings in
  `src/elbow_helper/plotting.py`, `tests/test_multiknee.py`, and
  `tests/test_real_world_examples.py`.

### Added
- **`tests/test_cli_argparse.py`**: `_load_series`'s `--*-npy`/`--*-csv`
  loading paths (shared by both CLI doors) and the x-only shorthand
  branch of `_xy_from_args` had no test in either CLI's test file.
  `cli_argparse.py` is now at 100% line coverage (was 77%).
- **`tests/test_plotting.py`**: `elbow_helper.plotting` (334 statements, the
  package's largest module) had no dedicated test file, only indirect
  coverage through the CLI's `diagnostics` subcommand and its own module
  doctests. Added direct tests for `render_svg`'s clear/abstain/
  `raw_axis`+`log_y` paths, `render_svg_multi`'s valid/invalid paths (the
  two crashes above were both found writing these), and the `out=`
  file-writing branches of `plot_multi_diagnostics`/
  `plot_diagnostics_panels`.
- **`tests/test_doctests.py`**: the `>>> ...` examples embedded in
  `_core_cli.py`, `cli_argparse.py`, `plotting.py`, `api.py`, and
  `mcp_server.py` were never actually executed by `pytest -q`
  (`pyproject.toml` scopes `testpaths` to `tests/` with no
  `--doctest-modules`), so they could drift from real behaviour
  unnoticed. Now run via `doctest.testmod` as part of the regular suite.

## [0.1.3] - 2026-08-15

### Fixed
- **`KneeLocator`**: an empty `(x, y)` input used to surface as a confusing
  `"zero-size array to reduction operation minimum which has no identity"`
  deep inside curve normalization, instead of a clear error at the actual
  boundary. Now raises `ValueError` immediately. Reachable via the
  unguarded `/locator` HTTP route and the CLI `locator` subcommand (unlike
  `robust_knee`/`robust_elbow`, which have their own numerical-safety net).
- **API**: `/locator`'s `ValueError` (e.g. the empty-input case above)
  collapsed into FastAPI's generic 500. Added a `ValueError` exception
  handler mapping it to 400.
- **CLI**: a library exception now prints one clean `Error: ...` line to
  stderr and exits 1, instead of a raw Python traceback, on both CLI
  twins. `elbow-helper-click`'s console-script entry point now points at a
  new `cli_click.main()` wrapper (was the bare `cli` group).

## [0.1.2] - 2026-08-14

### Added
- **`requirements.txt` / `requirements-dev.txt` / `environment.yaml`**, bringing
  the project in line with the rest of the suite's manifests. `requirements.txt`
  mirrors `pyproject.toml`'s `[project].dependencies`; `requirements-dev.txt`
  points at the `[dev]` extra; `environment.yaml` is a minimal conda env that
  hands off to `requirements.txt`, so it never drifts out of sync. README.md /
  LISEZ-MOI.md's Installation sections now mention the conda path.
- **`robust_knees` section in README.md / LISEZ-MOI.md**: the plural multi-knee
  API was fully public and tested but only mentioned in passing. Added a
  concise section to both, output verified against the actual code.

### Fixed
- **Three stale example outputs in `EXAMPLES.md` / `EXEMPLES.md`** no longer
  matched `examples/*.py`'s current behavior despite identical code and seed:
  `clear_knee.py` (`knee_x` 0.304 to 0.3418), `kmeans_elbow.py` (`knee_x` 8.0
  to 9), `no_knee.py` (abstention reason `INCOMPATIBLE_GLOBAL_SHAPE` to
  `NO_PERSISTENT_CLUSTER`). Updated all three to the verified current output.

## [0.1.1] - 2026-08-09

### Changed

- **CI ran `ruff check` but never `ruff format --check`**, so formatting
  drifted across 36 test files (line wrapping). Reformatted the tree and
  added the missing CI step.

## [0.1.0] - 2026-08-09

First public release: tagged, GitHub-released, and published to PyPI.

### Added
- **`doc/LIKELIHOOD-en.tex` / `doc/LIKELIHOOD-fr.tex`**, a new companion note
  factoring out the general likelihood foundation from `ELBOW-en.tex` /
  `ELBOW-fr.tex`, since that foundation does not depend on curve-fitting:
  why `L := exp(E[log p])` rather than the textbook raw product; Jensen's
  inequality and Shannon entropy; a temperature-indexed refinement
  (power-tilted families, Rényi entropy, the free-energy/entropy
  Legendre-transform dictionary borrowed from equilibrium statistical
  mechanics); Cramér's large deviations and the Chernoff information
  (distinguished from the Chernoff-Stein lemma); Wald's martingale and
  the Sequential Probability Ratio Test; a full estimator analysis of
  $L$ (bias and variance via the delta method, verified by Monte Carlo
  simulation) alongside a proof that the raw product is not an
  estimator of anything fixed; and two worked instantiations, each with
  its own generative-model diagram and a dedicated figure: categorical
  classification (a 45-example, 3-class simulation) and regression under
  Gaussian noise (conditional-density bells drawn on the fitted line).
  `ELBOW-en.tex` / `ELBOW-fr.tex`'s own Section 1 now states the Gaussian
  instantiation directly and points to the companion note for the
  general theory; `references.bib` gains `cramer1938`, `chernoff1952`,
  `wald1945`, `mackay2003information` and `touchette2009`. New figures
  `temperature_sweep`, `classification_density` and `regression_density`
  (EN+FR). `README.md` / `LISEZ-MOI.md` updated to reference both new
  documents.
- **`MATH-en.tex` / `MATH-fr.tex` renamed to `ELBOW-en.tex` /
  `ELBOW-fr.tex`** (`git mv`, history preserved), a clearer name for the
  project's own mathematics note; every reference across `doc/`, `src/`,
  `tests/`, `README.md`, `LISEZ-MOI.md`, `CHANGELOG.md`, `EXAMPLES.md`,
  `LANDSCAPE.md` and `PAYSAGE.md` updated accordingly. The four
  `research/multiknee/RESULTS.md` citations in `ELBOW-en.tex` /
  `ELBOW-fr.tex` were replaced by inline verified numbers and a new
  two-panel comparison figure (`multiknee_comparison`, EN+FR: exact-k
  accuracy ranked across ten DP/greedy times BIC/mBIC/ICL/FWER
  combinations, and false-positive rate on flat data), citing
  `research/multiknee/compare.py` instead of the static summary file.
- **CLI, HTTP API, and MCP surfaces**, mirroring the `ai-helpers` suite's
  standard architecture (`~/ai-helpers/.private/CODING.md` §19-§23): a new
  `elbow_helper._core_cli` shared core (`do_knee`/`do_elbow`/
  `do_diagnostics`/`do_locator`, JSON-safe serialization of `ClearKnee`/
  `NoClearKnee` including numpy-scalar coercion) backs four adapters:
  `cli_argparse.py` (always installed; subcommands `knee`/`elbow`/
  `diagnostics`/`locator`, three ways to pass a curve: inline comma-separated
  values, a `.npy` path, or a CSV column), `cli_click.py` (`[cli]` extra,
  identical commands), `api.py` (`[api]` extra, FastAPI; each route carries
  an explicit `operation_id`), and `mcp_server.py` (`[mcp]` extra,
  `fastapi-mcp` mounted on a copy of the API app; never a standalone tool
  server). New console scripts `elbow-helper`, `elbow-helper-click`,
  `elbow-helper-mcp`. New tests: `test_cli_argparse.py`, `test_cli_click.py`,
  `test_api.py`, `test_mcp_server.py`. New README.md/LISEZ-MOI.md "CLI / API
  / MCP" section. `pyproject.toml` description/keywords also had a stray
  pre-Acknowledgements "Kneedle" mention cleaned up (missed by the earlier
  sweep; `.toml` wasn't in that pass's file-type filter).
- **`EXAMPLES.md`**: a runnable cookbook (a clear knee, the k-means elbow, an
  explicit abstention, a `1 - exp(-t/tau)` saturation curve, a queueing-
  latency capacity-planning knee, cache-sizing diminishing returns, a
  release-engineering "when to stop testing" knee, the diagnostic figure,
  the standalone locator, configuration tuning), linked from both README.md
  and LISEZ-MOI.md, each generated figure now embedded directly in the page.
  `examples/exponential_saturation.py`, `examples/queueing_latency.py`,
  `examples/cache_hit_rate.py` and `examples/bug_discovery_rate.py` added
  alongside the existing example scripts.
- **Four new worked-example sections in `ELBOW-en.tex` / `ELBOW-fr.tex`**:
  an exponential-saturation knee (`1 - exp(-t/tau)`, matching
  `examples/exponential_saturation.py`), a convex/increasing queueing-
  latency knee (`examples/queueing_latency.py`), a concave/increasing
  cache-sizing knee built from a Michaelis-Menten curve rather than an
  exponential (`examples/cache_hit_rate.py`), and a convex/decreasing
  release-engineering knee (`examples/bug_discovery_rate.py`), each with
  its own generated figure. Fixed a real `elbow_helper.plotting.render_svg`
  layout bug surfaced by the queueing-latency and exponential-saturation
  figures: the knee callout pill could overlap the fixed evidence-legend
  box whenever the knee sat high and left-of-center (an early-saturating or
  fast-rising curve); the callout now falls back to sit left of the legend
  when the two would collide.
- **`robust_knee`**: a conservative wrapper around a from-scratch difference-curve
  implementation. Reports a knee with an uncertainty estimate, or explicitly
  abstains, rather than always returning a point estimate on a noisy or
  knee-less curve. The whole package depends on `numpy` + `os-helper` only.
- **`robust_knees`** (multi-knee API): segmentation-based detection of
  multiple knees along a curve, with a full bilingual (FR/EN) mathematical
  writeup and real-world worked examples (k-means, PCA).
- Bilingual competitive-landscape page (`LANDSCAPE.md` / `PAYSAGE.md`), with
  the positioning map generated via `standpoint` (replacing an earlier
  Vega-Lite quadrant chart).

### Changed
- **`elbow_helper.plotting` no longer uses matplotlib.** `plot_diagnostics`
  now renders a hand-authored, self-contained SVG (Catmull-Rom spline, house
  palette, additive dark-mode block; no runtime image library) instead of a
  four-panel matplotlib figure. The `[plot]` extra is gone: diagnostics are a
  **core** feature, nothing extra to install. The new figure shows the curve
  with the located knee and its 90% CI band (or an honest greyed/dashed
  abstention state with the reason, when the evidence is too weak) next to a
  compact evidence legend: detection probability, null-model p-value, slope
  contrast, and a BIC-derived posterior model probability. That last number
  is not the raw `bic_improvement` nats: a raw ΔBIC has no natural
  ceiling, so it is converted through Kass and Raftery's (1995) Bayes-factor
  approximation (`BF ~= exp(bic_improvement / 2)`, posterior probability
  `BF / (1 + BF)`) into a bounded [0, 1] reading, e.g. "99.9%", that means
  the same thing regardless of curve length or noise scale
  (`ClearKnee.bic_improvement` itself is untouched, still raw nats, still
  the same public API field). Supports `language="fr"`.
  `examples/diagnostic_plot.py` and the README/LISEZ-MOI.md Diagnostics
  sections updated to match, with a real generated figure
  (`assets/diagnostics.svg`) embedded in both.
- **`MATH-en.md` / `MATH-fr.md` replaced by `ELBOW-en.tex` / `ELBOW-fr.tex`**:
  the mathematics writeup is now native LaTeX (`amsmath`, `natbib` against
  `references.bib`, numbered equations, a proper table of contents) rather
  than markdown, given the target audience (readers from the end of high
  school through a Ph.D. in applied mathematics). Compile with
  `latexmk -pdf` (or `pdflatex` + `bibtex` + `pdflatex` ×2); `ELBOW-en.pdf` /
  `ELBOW-fr.pdf` are the checked-in compiled copies. Every reference to the
  old `.md` filenames (README, LISEZ-MOI.md, LANDSCAPE.md, PAYSAGE.md, and
  source docstrings) updated to point at the `.tex` sources.
- **The locator module and its test file renamed for naming consistency**
  (`src/elbow_helper/locator.py`, `tests/test_locator.py`), and every
  prose mention of the algorithm's name across the codebase, tests, README,
  LISEZ-MOI.md, LANDSCAPE.md, PAYSAGE.md and the ELBOW-en/fr.tex writeups
  reworded to describe it generically ("the locator", "the difference-curve
  method", ...). The name now appears exactly once per language, in the
  README/LISEZ-MOI.md Acknowledgements section, where the original paper is
  cited by title. No public API change: `KneeLocator`, `KneeCandidate` and
  every other exported name were already free of the word.

### Fixed
- CI push trigger was scoped to `branches: [main]` while the repo's actual
  default branch is `master`; the test workflow had never once run on a
  real push. Corrected to `master`.
- `requires-python = ">=3.9"` while the hard dependency `os-helper` requires
  `>=3.10` on every published version, so the 3.9 CI matrix leg could never
  resolve. Dropped 3.9, floor is now `>=3.10`.
- Assorted `ruff` cleanup: unused imports (`dataclasses.field`,
  `typing.Callable`/`Tuple`/`Optional`) and dead local variables in the
  `research/multiknee/` prototype scripts and `elbow_helper/clustering.py`.

