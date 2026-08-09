"""Example: a 1 - exp(-t/tau) saturation curve (charging, learning, dose-response).

Run::

    python examples/exponential_saturation.py

A curve that rises fast then asymptotically flattens toward 1, the shape of an
RC-circuit charging voltage, a learning curve, or a dose-response saturation.
Unlike a genuine broken-line knee, it has no true slope discontinuity — it is
smooth everywhere — so it is a useful stress test for the locator, and a
reminder that the located knee is not simply "one time constant" (see below).

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_knee

tau = 1.0
rng = np.random.default_rng(4)
t = np.linspace(0.0, 5.0 * tau, 150)
y = 1.0 - np.exp(-t / tau)
y = y + rng.normal(0, 0.01, t.size)  # light noise: a clean charging curve

result = robust_knee(
    t,
    y,
    curve="concave",
    direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    ratio = result.knee_x / tau
    print(f"  knee at t = {result.knee_x:.3f}  (tau = {tau}, knee/tau = {ratio:.2f})")
    print(f"  90% CI    = ({result.ci90[0]:.3f}, {result.ci90[1]:.3f})")

# Note on the knee/tau ratio: it lands consistently around 1.9, not 1.0. The
# textbook "time constant" tau marks 1 - e^-1 ~ 63% of the rise, which is a
# mathematically clean but visually unsatisfying "knee" — a human eye (and
# this locator) reads the curve as having genuinely flattened only once it
# clears roughly 2 time constants (1 - e^-2 ~ 85%), the same "practically
# settled" convention used informally in RC-circuit and control-theory
# contexts. This ratio is also a property of the *observation window*, not
# of the curve alone: it was measured here over t in [0, 5*tau]; a narrower
# or wider window shifts it, since the locator's decision is about maximum
# deviation from the chord spanning whatever range it is actually given.
