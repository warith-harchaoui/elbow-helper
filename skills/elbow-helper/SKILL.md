---
name: elbow-helper
description: >-
  Noise-robust knee and elbow detection that reports uncertainty and abstains
  when the evidence is weak. Ordinary knee heuristics always return a point,
  even on a straight line or pure noise; this one asks whether any candidate
  is strong, unique, persistent, reproducible and unlikely under a no-knee
  model, and says `NoClearKnee` when it is not. Use when the user asks "how
  many clusters", "where is the elbow in this scree / inertia / WCSS curve",
  "pick k for k-means", "where do diminishing returns start", "when should I
  stop training / iterating", "where is the knee of this curve", "is there
  even an elbow here", "find the cutoff in this sorted list of scores", or
  names `elbow-helper`, `robust_knee`, `robust_elbow`, `KneeLocator`,
  `kneedle`. Do NOT use for change-point detection in a time series, peak
  finding, fitting a curve, or choosing a clustering algorithm.
license: BSD-3-Clause
metadata:
  author: Warith Harchaoui
  package: elbow-helper
  suite: ai-helpers
  homepage: https://deraison.ai/ai-helpers
---

# elbow-helper — a knee with uncertainty, or an honest abstention

A knee on a curve of diminishing returns marks the point past which extra
input buys little extra output: more clusters in k-means, more iterations in
an optimiser, more budget on a channel. Existing heuristics are good at
*proposing* that point, but they always propose one — on a straight line, on
noise, on anything.

`elbow-helper` wraps a from-scratch knee locator in a conservative decision
procedure that answers the harder question:

> Is any candidate knee **strong, unique, persistent, reproducible**, and
> **unlikely under a no-knee model**? If not, say so.

The design priority is minimising false-positive knees, **even at the cost of
more abstentions**. An answer of `NoClearKnee` is a result, not a failure.

## Before anything: verify it is installed

```bash
elbow-helper --help                # argparse CLI (ships with the package)
python -c "import elbow_helper"    # library import check
```

Install:

```bash
pip install elbow-helper                   # numpy + os-helper, diagnostics included
pip install 'elbow-helper[cli]'            # + the click twin
pip install 'elbow-helper[api]'            # + FastAPI HTTP surface
pip install 'elbow-helper[mcp]'            # + MCP tools over the same app
```

The whole package depends on **numpy and os-helper only** — no scipy, no
scikit-learn, no matplotlib. The diagnostic figure is hand-authored SVG.

## What it does, and how to invoke it

| Intent | CLI | Library |
|---|---|---|
| Knee of an increasing, saturating curve | `elbow-helper knee` | `robust_knee` |
| Elbow of a decreasing curve (inertia, loss) | `elbow-helper elbow` | `robust_elbow` |
| Several knees on one curve | — | `robust_knees` |
| The raw point estimate, no decision | `elbow-helper locator` | `KneeLocator` |
| An SVG explaining the verdict | `elbow-helper diagnostics` | `elbow_helper.plotting` |

```python
import numpy as np
from elbow_helper import robust_elbow, ClearKnee, NoClearKnee

k = np.arange(1, 16)
inertia = ...                       # k-means WCSS for each k

result = robust_elbow(k, inertia)
match result:
    case ClearKnee():
        print("k =", result.x, "±", result.uncertainty)
    case NoClearKnee():
        print("no defensible elbow:", result.reason)
```

**The return type is the answer.** `ClearKnee` carries the location and its
uncertainty; `NoClearKnee` carries a `Reason`. Code that reads `.x` without
matching on the type has thrown away the whole point of the package.

For the decision criteria, the config knobs and the flag matrix, read
`references/cli-reference.md`. For the HTTP API, MCP tools and the web app,
read `references/surfaces.md`. For the exhaustive trigger list, read
`references/triggers.md`.

## Rules of thumb

- **Knee or elbow depends on the curve's direction.** Increasing and
  saturating (explained variance, recall vs k) → `robust_knee`. Decreasing
  (inertia, WCSS, loss) → `robust_elbow`. Choosing wrong does not error, it
  answers the wrong question.
- **Abstention is the feature.** When the answer is `NoClearKnee`, report
  that and its reason. Do not fall back to a raw `KneeLocator` estimate to
  produce a number — that is exactly the over-trust the package exists to
  prevent.
- **Report the uncertainty, never just the point.** "k = 5 ± 1" is the
  result; "k = 5" is a lossy summary of it.
- **`KneeLocator` is the unguarded primitive.** Use it only when you
  explicitly want a point estimate with no decision procedure attached, and
  say so to the user.
- **The thresholds in `RobustKneeConfig` are a validated starting point, not
  a universal constant** — this is a v0.1.x package. If a domain curve is
  systematically mis-called, tune the config and say which knob moved.
- **Render the diagnostic SVG when the verdict surprises the user.** It shows
  *why* a candidate was rejected, which settles the argument faster than
  prose.
- **After running, report the verdict, the location with its uncertainty or
  the abstention reason** — do not re-run with looser settings to force an
  answer.
