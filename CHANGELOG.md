# Changelog

All notable changes to `elbow-helper` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Not yet tagged or published — first public commits, still under active
development.

### Added
- **`robust_knee`**: a conservative wrapper around a from-scratch Kneedle
  implementation. Reports a knee with an uncertainty estimate, or explicitly
  abstains, rather than always returning a point estimate on a noisy or
  knee-less curve. Core depends on `numpy` + `os-helper` only; plotting is a
  lazy `[plot]` extra.
- **`robust_knees`** (multi-knee API): segmentation-based detection of
  multiple knees along a curve, with a full bilingual (FR/EN) mathematical
  writeup and real-world worked examples (k-means, PCA).
- Bilingual competitive-landscape page (`LANDSCAPE.md` / `PAYSAGE.md`), with
  the positioning map generated via `standpoint` (replacing an earlier
  Vega-Lite quadrant chart).

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
