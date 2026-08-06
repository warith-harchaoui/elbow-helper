"""Example: the classic k-means "elbow" (convex, decreasing inertia curve).

Run::

    python examples/kmeans_elbow.py

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_elbow

# An inertia-like scree curve: steep drop until the real structure (k=8) is
# captured, then a slow flat tail. (The elbow sits well inside the sweep so it
# clears the "enough points on each side" filter.)
rng = np.random.default_rng(3)
k = np.arange(1, 41, dtype=float)
inertia = np.where(k <= 8, 1000 - 90 * k, 280 - 3 * (k - 8))
inertia = inertia + rng.normal(0, 4.0, k.size)

result = robust_elbow(k, inertia, config=RobustKneeConfig(random_seed=0))

print(result)
if result.is_clear:
    print(f"  elbow at k = {result.knee_x:.1f}  (true k = 8)")
    print(f"  90% CI     = ({result.ci90[0]:.1f}, {result.ci90[1]:.1f})")
