"""Example: render the robust-knee diagnostic SVG (no extra install needed).

    python examples/diagnostic_plot.py

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig
from elbow_helper.plotting import plot_diagnostics

rng = np.random.default_rng(1)
x = np.linspace(0.0, 1.0, 80)
knee = 0.30
y = np.where(x <= knee, 3.0 * x, 3.0 * knee + 0.2 * (x - knee))
y = y / y.max() + rng.normal(0, 0.02, x.size)

plot_diagnostics(
    x,
    y,
    curve="concave",
    direction="increasing",
    config=RobustKneeConfig(random_seed=0),
    out="diagnostics.svg",
)
print("wrote diagnostics.svg")
