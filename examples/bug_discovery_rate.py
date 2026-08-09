"""Example: bugs found per day of testing, a "when to stop QA" knee.

Run::

    python examples/bug_discovery_rate.py

Reliability-growth models describe defect discovery during testing as
fast at first (the easy, high-impact bugs), then slow (the rare, deep
ones): a curve that falls steeply, bends, and flattens into a long, low
tail. This is the same convex/decreasing family as the k-means inertia
and PCA scree examples elsewhere in this project, run on a release
engineering question instead of a clustering or dimensionality one: past
which testing day does another day of QA stop paying for itself.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_knee

true_knee_day = 10.0
rng = np.random.default_rng(13)
days = np.linspace(1, 30, 120)
bugs_per_day = np.where(
    days <= true_knee_day,
    14.0 - 1.1 * days,
    2.0 - 0.05 * (days - true_knee_day),
)
bugs_per_day = np.clip(bugs_per_day + rng.normal(0, 0.3, days.size), 0, None)

result = robust_knee(
    days,
    bugs_per_day,
    curve="convex",
    direction="decreasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    print(f"  knee at day = {result.knee_x:.1f}  (true knee = {true_knee_day:.0f})")
    print(f"  90% CI      = ({result.ci90[0]:.1f}, {result.ci90[1]:.1f})")

# Note: the located knee sits about one day past the true break, the same
# small, systematic "one to two units past the true bend" offset the PCA
# scree example shows. Reading knee_x as "stop testing exactly here"
# rather than "the discovery rate has now genuinely flattened, a day or
# two either side" over-trusts the point estimate past what it claims.
