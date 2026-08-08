# EXAMPLES.md

Runnable recipes for `elbow-helper`. Each one is also a standalone script
under `examples/` — copy the code, or just run the file. See
[`README.md`](README.md) for the full API reference and the pipeline this
all rests on.

## 1. A clear, saturating knee

The most common case: a curve that rises, bends, and flattens, with enough
signal for the pipeline to commit to a point estimate.

```python
import numpy as np
from elbow_helper import RobustKneeConfig, robust_knee

rng = np.random.default_rng(1)
x = np.linspace(0.0, 1.0, 80)
knee = 0.30
y = np.where(x <= knee, 3.0 * x, 3.0 * knee + 0.2 * (x - knee))
y = y / y.max() + rng.normal(0, 0.02, x.size)

result = robust_knee(
    x, y, curve="concave", direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    print(f"  true knee ~ {knee}, located at {result.knee_x:.3f}")
    print(f"  90% CI    = ({result.ci90[0]:.3f}, {result.ci90[1]:.3f})")
    print(f"  detection = {result.detection_rate:.2f}, null p = {result.null_p_value:.3g}")
```

```text
ClearKnee(knee_x=0.304, ci90=(0.296, 0.315), detection_rate=0.98, null_p=0.00498)
  true knee ~ 0.3, located at 0.304
  90% CI    = (0.296, 0.315)
  detection = 0.98, null p = 0.00498
```

The point estimate never travels alone: `ci90` bounds it, `detection_rate`
says how often an independent resample of the noise finds the same knee, and
`null_p_value` says how surprising this knee would be if the curve were
secretly just a straight line plus noise.

Full script: [`examples/clear_knee.py`](examples/clear_knee.py).

## 2. The k-means "elbow" (convex, decreasing)

`robust_elbow` is `robust_knee` with `curve`/`direction` pinned to the
classic scree-plot shape: inertia falling fast, then flattening as extra
clusters stop paying for themselves.

```python
import numpy as np
from elbow_helper import RobustKneeConfig, robust_elbow

rng = np.random.default_rng(3)
k = np.arange(1, 41, dtype=float)
inertia = np.where(k <= 8, 1000 - 90 * k, 280 - 3 * (k - 8))
inertia = inertia + rng.normal(0, 4.0, k.size)

result = robust_elbow(k, inertia, config=RobustKneeConfig(random_seed=0))

print(result)
if result.is_clear:
    print(f"  elbow at k = {result.knee_x:.1f}  (true k = 8)")
    print(f"  90% CI     = ({result.ci90[0]:.1f}, {result.ci90[1]:.1f})")
```

```text
ClearKnee(knee_x=8.0, ci90=(7.6, 8.4), detection_rate=1.00, null_p=0.00498)
  elbow at k = 8.0  (true k = 8)
  90% CI     = (7.6, 8.4)
```

Full script: [`examples/kmeans_elbow.py`](examples/kmeans_elbow.py).

## 3. Explicit abstention, not a fake knee

A noisy but genuinely monotone straight line has no knee. The pipeline says
so, with a machine-readable reason, instead of returning a plausible-looking
but spurious point.

```python
import numpy as np
from elbow_helper import RobustKneeConfig, robust_knee

rng = np.random.default_rng(2)
x = np.linspace(0.0, 1.0, 80)
y = 0.2 + 0.5 * x + rng.normal(0, 0.01, x.size)  # monotone, no knee

result = robust_knee(
    x, y, curve="concave", direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
print(f"  reason = {result.reason}")
```

```text
NoClearKnee(reason='INCOMPATIBLE_GLOBAL_SHAPE')
  reason = INCOMPATIBLE_GLOBAL_SHAPE
```

Every abstention carries one of the fifteen reason codes documented in the
README's "Abstention reason codes" section — `result.diagnostics` has the
full numeric trail behind the decision, useful for tuning `RobustKneeConfig`
against your own curve family.

Full script: [`examples/no_knee.py`](examples/no_knee.py).

## 4. A saturating curve: `1 - exp(-t / tau)`

Charging voltages, learning curves, dose-response curves — anything that
rises fast then asymptotically approaches a ceiling — share this shape. It
is a useful stress test: unlike a broken-line knee, it has no true slope
discontinuity, it is smooth everywhere.

```python
import numpy as np
from elbow_helper import RobustKneeConfig, robust_knee

tau = 1.0
rng = np.random.default_rng(4)
t = np.linspace(0.0, 5.0 * tau, 150)
y = 1.0 - np.exp(-t / tau)
y = y + rng.normal(0, 0.01, t.size)

result = robust_knee(
    t, y, curve="concave", direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    print(f"  knee at t = {result.knee_x:.3f}  (tau = {tau})")
```

```text
ClearKnee(knee_x=1.980, ci90=(1.946, 2.114), detection_rate=0.99, null_p=0.00498)
  knee at t = 1.980  (tau = 1.0)
```

Worth knowing before reading too much into the exact number: the located
knee sits consistently around **1.9 tau**, not 1 tau. The textbook "time
constant" tau marks `1 - e^-1 ≈ 63%` of the rise — mathematically clean, but
not where a curve visually reads as "flattened". The pipeline (like a human
eye) settles on roughly 2 time constants (`1 - e^-2 ≈ 85%`), the same
informal "practically settled" convention used in RC-circuit and
control-theory contexts. This ratio is a property of the *observation
window* as much as of the curve: measured here over `t ∈ [0, 5·tau]`, it
would shift under a narrower or wider window, since the decision is about
maximum deviation from the chord spanning whatever range it is actually
given, not an intrinsic property of the infinite curve.

Full script: [`examples/exponential_saturation.py`](examples/exponential_saturation.py).

## 5. The diagnostic figure

`elbow_helper.plotting` needs no extra install — it writes a self-contained
SVG by hand, no matplotlib.

```python
from elbow_helper import RobustKneeConfig
from elbow_helper.plotting import plot_diagnostics

plot_diagnostics(
    x, y, curve="concave", direction="increasing",
    config=RobustKneeConfig(random_seed=0),
    out="diagnostics.svg",
    language="en",   # or "fr"
)
```

The figure shows the curve with its located knee and 90% CI band (or, on
abstention, a greyed dashed curve and the reason) next to a compact evidence
legend: detection probability, null-model p-value, slope contrast, and a
BIC-derived posterior model probability — a bounded `[0, 1]` reading (e.g.
"99.9%"), not the raw, unbounded `bic_improvement` nats the API itself
returns, since a raw log-likelihood difference has no natural scale to
compare against.

Full script: [`examples/diagnostic_plot.py`](examples/diagnostic_plot.py).

## 6. The standalone locator

The from-scratch bump-peak locator underneath `robust_knee` is usable on its
own, without the full conservative pipeline (no bootstrap, no null test, no
persistence clustering — just the geometric peak-finding step):

```python
from elbow_helper import KneeLocator

kl = KneeLocator(x, y, S=1.0, curve="concave", direction="increasing", online=True)
kl.knee, kl.all_knees
```

Useful for exploring a curve interactively, or as a building block in your
own pipeline; `robust_knee` is the one to reach for when you need the
uncertainty estimate and the abstention guarantee.

## 7. Tuning the configuration

Every threshold lives in `RobustKneeConfig`, a frozen dataclass. The shipped
defaults favour speed (modest replicate counts, a run finishes in seconds);
raise them for a validation-grade run:

```python
from elbow_helper import RobustKneeConfig

config = RobustKneeConfig(bootstrap_replicates=100, null_replicates=200)
validation_config = config.with_(bootstrap_replicates=500, null_replicates=1000)
```

`cluster_tolerance` and `max_neighbor_shift` are the two most worth
recalibrating for your own data: they sit slightly above the 0.05 reference
value to absorb the locator's discretisation jitter at modest sample sizes
(n around 60-100) — tighten them for cleaner, larger datasets, loosen them
for noisier or shorter ones.
