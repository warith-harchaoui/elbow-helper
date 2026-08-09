"""Example: a clear, saturating knee is accepted with an uncertainty interval.

Run::

    python examples/clear_knee.py

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_knee

rng = np.random.default_rng(1)
x = np.linspace(0.0, 1.0, 80)
knee = 0.30
y = np.where(x <= knee, 3.0 * x, 3.0 * knee + 0.2 * (x - knee))
y = y / y.max() + rng.normal(0, 0.02, x.size)

result = robust_knee(
    x,
    y,
    curve="concave",
    direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    print(f"  true knee ~ {knee}, located at {result.knee_x:.3f}")
    print(f"  90% CI    = ({result.ci90[0]:.3f}, {result.ci90[1]:.3f})")
    print(
        f"  detection = {result.detection_rate:.2f}, null p = {result.null_p_value:.3g}"
    )
