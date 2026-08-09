"""Example: cache hit rate vs. cache size, a capacity-sizing knee.

Run::

    python examples/cache_hit_rate.py

For a working set with roughly power-law-popular items, hit rate as a
function of cache size ``C`` follows a Michaelis-Menten-shaped curve,
``C / (C + K)``, with ``K`` the cache size at which half the working set is
already resident: fast early gains, then a long, flattening tail. This is
the standard "how big should my cache be" question, distinct in
functional form from the ``1 - exp(-t/tau)`` saturation curve elsewhere in
this project (a rational function, not an exponential) even though both
are concave and increasing.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_knee

K = 200.0  # items: cache size at which hit rate = 50%
rng = np.random.default_rng(11)
cache_size = np.linspace(10, 2000, 150)
hit_rate = cache_size / (cache_size + K)
hit_rate = hit_rate + rng.normal(0, 0.01, cache_size.size)

result = robust_knee(
    cache_size,
    hit_rate,
    curve="concave",
    direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    knee_hit_rate = result.knee_x / (result.knee_x + K)
    print(
        f"  knee at cache size = {result.knee_x:.0f} items  ({result.knee_x / K:.2f}x K)"
    )
    print(f"  hit rate at knee   = {knee_hit_rate:.1%}")
    print(f"  90% CI             = ({result.ci90[0]:.0f}, {result.ci90[1]:.0f})")

# Note: the knee lands around 3.5x K, not at K itself. K only marks the
# 50%-hit-rate point, a property of the formula's algebra, not of where
# the *marginal* return on cache size actually stops paying off. Buying
# cache up to K leaves a lot of cheap hit-rate on the table; buying past
# roughly 3.5x K is spending real memory for a shrinking sliver of hits.
