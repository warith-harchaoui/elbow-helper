# elbow-helper CLI reference

One CLI, `elbow-helper`, built with click and shipped with the core package.
It
dispatch to the same `_core_cli` implementation, so flags match exactly.

## The four commands

```bash
elbow-helper knee        --input curve.csv [--json]
elbow-helper elbow       --input curve.csv [--json]
elbow-helper locator     --input curve.csv [--json]
elbow-helper diagnostics --input curve.csv --output figure.svg
```

| Command | Curve shape | Returns |
|---|---|---|
| `knee` | increasing, saturating | `ClearKnee` with uncertainty, or `NoClearKnee` |
| `elbow` | decreasing | the same, for the mirrored case |
| `locator` | either | the **raw** point estimate, no decision procedure |
| `diagnostics` | either | an SVG showing why the verdict came out that way |

Input is a two-column CSV (`x`, `y`). `x` must be monotonic.

**Pick the command from the curve's direction.** Explained variance, recall
vs k, coverage → `knee`. k-means inertia / WCSS, training loss, error vs
capacity → `elbow`. The wrong one does not raise; it answers a different
question.

## Reading the result

```json
{"verdict": "ClearKnee", "x": 5, "uncertainty": 1, "y": 142.7}
{"verdict": "NoClearKnee", "reason": "no candidate survived the no-knee model"}
```

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

`RobustKneeConfig` (and `RobustKneesConfig` for the multi-knee path) carries
the thresholds behind those five criteria. The shipped values are **a
validated starting point, not a universal constant** — this is a v0.1.x
package, and the repo's README says so plainly.

When a domain's curves are systematically mis-called:

1. Render `diagnostics` first and look at *which* criterion rejected the
   candidate.
2. Move that one knob, not all of them.
3. Say in your report which knob moved and why.

Loosening the config until an answer appears is not tuning; it is defeating
the decision procedure.

## The diagnostic figure

```bash
elbow-helper diagnostics --input curve.csv --output figure.svg
```

Hand-authored SVG — no matplotlib, no extra install. It plots the curve, the
candidates considered, and the criterion each one failed. Render it whenever
the verdict surprises the user: it settles the argument faster than prose.

## Output contract

- Verdict commands print a human-readable line, or JSON with `--json`.
- `diagnostics` prints the SVG path it wrote.
- An abstention exits `0` — it is a valid answer, not an error.
- Genuine failures (unreadable input, non-monotonic `x`) are logged through
  `os_helper` and exit non-zero.
