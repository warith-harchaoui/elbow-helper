# Changelog

All notable changes to `elbow-helper` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Not yet tagged or published — first public commits, still under active
development.

### Added
- **`EXAMPLES.md`**: a runnable cookbook (a clear knee, the k-means elbow, an
  explicit abstention, a `1 - exp(-t/tau)` saturation curve, the diagnostic
  figure, the standalone locator, configuration tuning), linked from both
  README.md and LISEZ-MOI.md. `examples/exponential_saturation.py` added
  alongside the existing example scripts.
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
  palette, additive dark-mode block — no runtime image library) instead of a
  four-panel matplotlib figure. The `[plot]` extra is gone: diagnostics are a
  **core** feature, nothing extra to install. The new figure shows the curve
  with the located knee and its 90% CI band (or an honest greyed/dashed
  abstention state with the reason, when the evidence is too weak) next to a
  compact evidence legend: detection probability, null-model p-value, slope
  contrast, and a BIC-derived posterior model probability. That last number
  is not the raw `bic_improvement` nats — a raw ΔBIC has no natural
  ceiling, so it is converted through Kass and Raftery's (1995) Bayes-factor
  approximation (`BF ~= exp(bic_improvement / 2)`, posterior probability
  `BF / (1 + BF)`) into a bounded [0, 1] reading, e.g. "99.9%", that means
  the same thing regardless of curve length or noise scale
  (`ClearKnee.bic_improvement` itself is untouched, still raw nats, still
  the same public API field). Supports `language="fr"`.
  `examples/diagnostic_plot.py` and the README/LISEZ-MOI.md Diagnostics
  sections updated to match, with a real generated figure
  (`assets/diagnostics.svg`) embedded in both.
- **`MATH-en.md` / `MATH-fr.md` replaced by `MATH-en.tex` / `MATH-fr.tex`** —
  the mathematics writeup is now native LaTeX (`amsmath`, `natbib` against
  `references.bib`, numbered equations, a proper table of contents) rather
  than markdown, given the target audience (readers from the end of high
  school through a Ph.D. in applied mathematics). Compile with
  `latexmk -pdf` (or `pdflatex` + `bibtex` + `pdflatex` ×2); `MATH-en.pdf` /
  `MATH-fr.pdf` are the checked-in compiled copies. Every reference to the
  old `.md` filenames (README, LISEZ-MOI.md, LANDSCAPE.md, PAYSAGE.md, and
  source docstrings) updated to point at the `.tex` sources.

### Changed
- **The locator module and its test file renamed for naming consistency**
  (`src/elbow_helper/locator.py`, `tests/test_locator.py`), and every
  prose mention of the algorithm's name across the codebase, tests, README,
  LISEZ-MOI.md, LANDSCAPE.md, PAYSAGE.md and the MATH-en/fr.tex writeups
  reworded to describe it generically ("the locator", "the difference-curve
  method", ...). The name now appears exactly once per language, in the
  README/LISEZ-MOI.md Acknowledgements section, where the original paper is
  cited by title. No public API change — `KneeLocator`, `KneeCandidate` and
  every other exported name were already free of the word.

### Fixed
- CI push trigger was scoped to `branches: [main]` while the repo's actual
  default branch is `master` — the test workflow had never once run on a
  real push. Corrected to `master`.
- `requires-python = ">=3.9"` while the hard dependency `os-helper` requires
  `>=3.10` on every published version, so the 3.9 CI matrix leg could never
  resolve. Dropped 3.9, floor is now `>=3.10`.
- Assorted `ruff` cleanup: unused imports (`dataclasses.field`,
  `typing.Callable`/`Tuple`/`Optional`) and dead local variables in the
  `research/multiknee/` prototype scripts and `elbow_helper/clustering.py`.

### Notes
- Not yet tagged, released, or published to PyPI.
