# TRIGGERS: elbow-helper

This is the user-facing, exhaustive catalogue of what `elbow-helper` can do and
the natural-language phrasings, commands, functions, and situations that should
invoke it.

`elbow-helper` decides **whether a curve has a defensible knee or elbow**, and
if so where, with uncertainty. It abstains when the evidence is weak. It is
pure NumPy, local, and depends on nothing but `numpy` and `os-helper`. It does
**not** do change-point detection, peak finding, curve fitting, clustering, or
forecasting.

## Operations → how to invoke

| Intent | CLI | Library | API |
|--------|-----|---------|-----|
| Knee of an increasing, saturating curve | `elbow-helper knee` | `robust_knee` | `POST /knee` |
| Elbow of a decreasing curve | `elbow-helper elbow` | `robust_elbow` | `POST /elbow` |
| Several knees on one curve | n/a | `robust_knees` | n/a |
| Raw point estimate, no decision | `elbow-helper locator` | `KneeLocator` | `POST /locator` |
| SVG explaining the verdict | `elbow-helper diagnostics` | `elbow_helper.plotting` | `POST /diagnostics` |
| Tune the decision procedure | `--config` flags | `RobustKneeConfig`, `RobustKneesConfig` | request body |

Every command is also reachable through the click CLI (`elbow-helper-click …`,
same flags), and the MCP surface mounts on the same FastAPI app at `/mcp`.

## Natural-language phrasings that should fire

- **Choosing k**: "how many clusters should I use", "pick k for k-means",
  "what k does the elbow say", "optimal number of clusters", "elbow method",
  "scree plot", "where does the scree plot bend".
- **Diminishing returns**: "where do diminishing returns start", "past what
  point is extra budget wasted", "the curve flattens — where", "point of
  saturation", "where is the knee of this curve", "where does it plateau".
- **Stopping criteria**: "when should I stop training", "how many iterations
  are enough", "where does the loss stop improving", "early-stopping point",
  "how many epochs", "how many components / dimensions to keep".
- **Cutoffs in ranked data**: "find the cutoff in this sorted list of
  scores", "where do the relevant results end", "threshold this ranking",
  "top-N but N is unknown".
- **Doubting the knee**: "is there even an elbow here", "the elbow method
  gives nonsense on this curve", "I don't trust this knee", "is this knee
  real", "how confident is that elbow", "my curve is noisy".
- **Explicit mentions**: `elbow-helper`, `elbow-helper-click`,
  `elbow-helper-mcp`, `import elbow_helper`, `robust_knee`, `robust_elbow`,
  `robust_knees`, `KneeLocator`, `KneeResult`, `ClearKnee`, `NoClearKnee`,
  `MultiKneeResult`, `RobustKneeConfig`, `kneedle`, "the Kneedle algorithm",
  "install / run elbow-helper".

## Inputs it accepts

- Two 1-D NumPy arrays or sequences: `x` (monotonic) and `y`.
- **Increasing, saturating** curves (explained variance, recall vs k,
  coverage) → `robust_knee`.
- **Decreasing** curves (k-means inertia / WCSS, training loss, error vs
  capacity) → `robust_elbow`.
- A CSV of two columns via the CLI, or a JSON body of two arrays via the API.

Choosing the wrong direction does not raise; it answers a different question.

## The decision, and what abstention means

A candidate becomes a `ClearKnee` only if it is **strong** (the bend is large
relative to the curve), **unique** (no rival candidate of comparable
strength), **persistent** (it survives smoothing and resampling),
**reproducible**, and **unlikely under a no-knee model**. Otherwise the result
is `NoClearKnee` with a `Reason`.

`NoClearKnee` is a **result**, not an error and not a reason to retry with
looser settings. Falling back to a bare `KneeLocator` estimate to produce a
number defeats the package's entire purpose.

## When NOT to use elbow-helper (SKIP)

- Change-point / breakpoint detection in a time series (`ruptures`-style).
- Peak, trough, or zero-crossing finding.
- Curve fitting, regression, or smoothing for its own sake.
- Choosing a *clustering algorithm* (as opposed to its k), or running the
  clustering itself.
- Outlier or anomaly detection.
- Picking an approximate-nearest-neighbour backend → `ann-router`.
- Picking a local model → `best-engine-ai-helper`.

## See also

- the repo's `README.md`: why it exists, install, limitations.
- the repo's `EXAMPLES.md`: runnable recipes.
- the repo's `LANDSCAPE.md`: how it compares with `kneed` and hand-rolled
  elbow heuristics.
- the web app at <https://deraison.ai/elbow-helper>: the pipeline in a
  browser, nothing uploaded.
