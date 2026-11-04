# elbow-helper CLI reference

One CLI, `elbow-helper`, built with click and shipped with the core package.
It dispatches to the same `_core_cli` implementation the HTTP API and the MCP
tools call, so every door answers the same way.

## The four commands

```bash
elbow-helper knee        DATA [--curve concave|convex] [--direction increasing|decreasing] [--config-json JSON]
elbow-helper elbow       DATA [--config-json JSON]
elbow-helper locator     DATA [--curve …] [--direction …] [--sensitivity 1.0] [--online/--no-online]
elbow-helper diagnostics DATA [--curve …] [--direction …] [--config-json JSON] [--out figure.svg] [--language en|fr]
```

`DATA` is one input per side, each given one of three ways:

| Side | Inline | `.npy` file | CSV column |
|---|---|---|---|
| x | `--x-values 1,2,3` | `--x-npy x.npy` | `--x-csv data.csv:0` |
| y | `--y-values 9,5,4` | `--y-npy y.npy` | `--y-csv data.csv:1` |

The CSV form is `PATH:COLUMN_INDEX` (0-based); a non-numeric first cell is
taken as a header and skipped. Give only y and x defaults to `0, 1, …, n-1`.
A curve needs at least 20 points (`min_samples`), else the answer is
`NoClearKnee` with `INSUFFICIENT_DATA`.

`-v` / `-vv` before the subcommand raise the log level (logs go to stderr);
`-V` prints the version.

| Command | Curve shape | Returns |
|---|---|---|
| `knee` | increasing, saturating | `ClearKnee` with uncertainty, or `NoClearKnee` |
| `elbow` | decreasing | the same, for the mirrored case |
| `locator` | either | the **raw** point estimate, no decision procedure |
| `diagnostics` | either | an SVG showing why the verdict came out that way |

**Pick the command from the curve's direction.** Explained variance, recall
vs k, coverage → `knee`. k-means inertia / WCSS, training loss, error vs
capacity → `elbow`. The wrong one does not raise; it answers a different
question.

## Reading the result

Every field of the result dataclass, as indented JSON, plus `is_clear`:

```json
{"reason": "CLEAR_KNEE", "knee_x": 6.0, "ci90": [5.0, 7.0], "detection_rate": 0.98, "...": "...", "is_clear": true}
{"reason": "NULL_NOT_REJECTED", "detail": "...", "diagnostics": {"...": "..."}, "is_clear": false}
```

A `ClearKnee` carries `knee_x` with its 90 % bootstrap interval `ci90`; a
`NoClearKnee` carries its `Reason` (`INSUFFICIENT_DATA`, `ALL_CANDIDATES_WEAK`,
`BOOTSTRAP_UNSTABLE`, `NULL_NOT_REJECTED`, …).

`NoClearKnee` is a **result**. Report it with its reason. Do not re-run with
looser settings to force a number, and do not quietly substitute `locator`'s
estimate — that is the over-trust this package exists to prevent.

## The decision criteria

A candidate becomes a `ClearKnee` only if it is all five of:

1. **Strong** — the bend is large relative to the curve's own scale.
2. **Unique** — no rival candidate of comparable strength elsewhere.
3. **Persistent** — it survives smoothing and resampling.
4. **Reproducible** — repeated runs land in the same place.
5. **Unlikely under a no-knee model** — a straight line or noise would not
   produce it by chance.

Fail any one and the answer is `NoClearKnee`, with the `Reason` naming which.

## Configuration

`RobustKneeConfig` (and `RobustKneesConfig` for the multi-knee path, which is
library-only) carries the thresholds behind those five criteria. On the
command line, `--config-json '{"bootstrap_replicates": 500}'` overrides
`RobustKneeConfig` fields. The shipped values are **a validated starting
point, not a universal constant** — this is a 0.x package, and the repo's
README says so plainly.

When a domain's curves are systematically mis-called:

1. Render `diagnostics` first and look at *which* criterion rejected the
   candidate.
2. Move that one knob, not all of them.
3. Say in your report which knob moved and why.

Loosening the config until an answer appears is not tuning; it is defeating
the decision procedure.

## The diagnostic figure

```bash
elbow-helper diagnostics --y-csv curve.csv:1 --x-csv curve.csv:0 --out figure.svg
```

Hand-authored SVG — no matplotlib, no extra install. It plots the curve, the
candidates considered, and the criterion each one failed. Render it whenever
the verdict surprises the user: it settles the argument faster than prose.

## Output contract

- Verdict commands always print JSON on stdout.
- `diagnostics` prints the SVG itself (and also writes it with `--out`).
- An abstention exits `0` — it is a valid answer, not an error.
- A mistyped command (no data, two inputs for one side, a bad `PATH:COL`)
  exits `2` with the usage line; any other failure prints `Error: …` on
  stderr and exits `1`.
