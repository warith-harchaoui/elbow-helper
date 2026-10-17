# elbow-helper non-CLI surfaces

## 1. Library — the primary surface

```python
from elbow_helper import robust_knee, robust_elbow, robust_knees
from elbow_helper import ClearKnee, NoClearKnee, KneeLocator, RobustKneeConfig
```

Public API (`elbow_helper.__all__`): `robust_knee`, `robust_elbow`,
`robust_knees`, `RobustKneeConfig`, `RobustKneesConfig`, `KneeLocator`,
`KneeResult`, `ClearKnee`, `NoClearKnee`, `MultiKneeResult`, `Knees`,
`InvalidKnees`, `KneeEstimate`, `Reason`.

### The return type is the answer

```python
result = robust_elbow(k, inertia)
match result:
    case ClearKnee():
        use_k(result.x, result.uncertainty)
    case NoClearKnee():
        report_abstention(result.reason)
```

`ClearKnee` carries the location **and its uncertainty**; `NoClearKnee`
carries a `Reason`. Code that reads `.x` without matching on the type has
discarded the only thing that distinguishes this package from a bare
heuristic.

`robust_knees` returns a `MultiKneeResult` (`Knees` or `InvalidKnees`) for
curves that may bend more than once.

`KneeLocator` is the **unguarded primitive** — the from-scratch point
estimator with no decision procedure. Reach for it only when you explicitly
want that, and say so to the user.

### Dependencies

`numpy` and `os-helper`, nothing else. No scipy, no scikit-learn, no
statsmodels, no joblib. `elbow_helper.plotting` writes hand-authored SVG, so
even the diagnostic figure needs no matplotlib.

## 2. CLIs — argparse (core) and click (`[cli]`)

`elbow-helper` and `elbow-helper-click`, both over a shared `_core_cli`. See
`cli-reference.md`.

## 3. HTTP API — FastAPI (`[api]`)

```bash
pip install 'elbow-helper[api]'
python -m elbow_helper.api          # 127.0.0.1:8000, OpenAPI docs at /docs
```

| Method | Route | `operation_id` |
|---|---|---|
| POST | `/knee` | `knee` |
| POST | `/elbow` | `elbow` |
| POST | `/locator` | `locator` |
| POST | `/diagnostics` | `diagnostics` |

Each takes a JSON body of two arrays and returns the verdict. The explicit
`operation_id`s are what the MCP tools are named after — they are a public
contract, not an implementation detail.

## 4. MCP — `fastapi-mcp` over the *same* app (`[mcp]`)

```bash
pip install 'elbow-helper[mcp]'
elbow-helper-mcp                    # same app, MCP mounted at /mcp
```

`elbow_helper.mcp_server` takes the FastAPI app from `elbow_helper.api` and
mounts `FastApiMCP` on it, so one entry point covers both doors and the two
surfaces cannot drift.

Register it with Claude Code:

```bash
claude mcp add elbow-helper --transport http http://127.0.0.1:8000/mcp
```

Every tool is read-only: the package computes a verdict and writes nothing
beyond an SVG you asked for.

## 5. Web app

<https://deraison.ai/elbow-helper> — the pipeline in a browser: paste a
curve, see the verdict and the diagnostic figure. No install, and nothing is
uploaded. Hand it to a human who wants to sanity-check a curve once rather
than script anything.

## Where this sits in the suite

elbow-helper is the suite's **decision-with-abstention** member, next to
`ann-router` (which backend) and `best-engine-ai-helper` (which model). All
three answer a choice question with a justification rather than a bare
answer; elbow-helper is the one allowed to reply "there is no defensible
answer here".
