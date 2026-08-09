"""Example: p99 latency vs. utilization, a queueing "knee" (capacity planning).

Run::

    python examples/queueing_latency.py

A single-server queue's response time grows like ``1 / (1 - rho)`` in its
utilization ``rho`` (fraction of capacity used): a classic M/M/1-style
blow-up. Latency stays close to the baseline service time for most of the
range, then rises sharply as ``rho`` approaches 1. Unlike the saturation
curves elsewhere in this project, this one is convex and increasing: the
"knee" here is the point past which adding a little more load costs a lot
more latency, the number an SRE actually wants before setting an
autoscaling or admission-control threshold.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import numpy as np

from elbow_helper import RobustKneeConfig, robust_knee

baseline_ms = 8.0
rng = np.random.default_rng(7)
rho = np.linspace(0.02, 0.90, 150)
latency_ms = baseline_ms / (1.0 - rho)
latency_ms = latency_ms + rng.normal(0, 0.6, rho.size)

result = robust_knee(
    rho,
    latency_ms,
    curve="convex",
    direction="increasing",
    config=RobustKneeConfig(random_seed=0),
)

print(result)
if result.is_clear:
    knee_latency = baseline_ms / (1.0 - result.knee_x)
    print(
        f"  knee at utilization = {result.knee_x:.3f}  ({result.knee_x:.0%} of capacity)"
    )
    print(
        f"  latency at knee     = {knee_latency:.1f} ms  ({knee_latency / baseline_ms:.1f}x baseline)"
    )
    print(f"  90% CI              = ({result.ci90[0]:.3f}, {result.ci90[1]:.3f})")

# Note: the knee lands around rho ~ 0.60, well before the mathematical
# blow-up at rho = 1. A queue is already paying a real latency tax at 60%
# utilization; the curve only *looks* fine up to there because 1/(1-rho)
# is still small on an absolute scale. Sizing capacity off "is it still
# roughly flat" instead of off this located knee is a common, avoidable
# way capacity plans run out of headroom earlier than expected.
