"""Example: a noisy straight line yields an explicit abstention, not a fake knee.

Run::

    python examples/no_knee.py
"""

from __future__ import annotations

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
print(f"  reason      = {result.reason}")
print(f"  diagnostics = {result.diagnostics}")
