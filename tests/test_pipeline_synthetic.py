"""End-to-end pipeline tests: clear knees accepted, no-knee curves abstained.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import types

from elbow_helper import ClearKnee, NoClearKnee, Reason, robust_elbow, robust_knee

from conftest import clear_knee_curve, elbow_curve, noisy_line, two_knee_curve


def test_clear_knee_is_accepted_with_uncertainty(fast_config):
    x, y = clear_knee_curve(seed=1, knee_frac=0.3, noise=0.02)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, ClearKnee)
    assert r.reason == Reason.CLEAR_KNEE
    assert abs(r.knee_x - 0.3) < 0.1  # near the true knee
    assert r.ci90[1] > r.ci90[0]  # a real interval
    assert r.detection_rate >= fast_config.min_bootstrap_detection_rate
    assert r.null_p_value <= fast_config.max_null_p_value


def test_elbow_is_accepted(fast_config):
    x, y = elbow_curve(seed=1, knee_frac=0.25, noise=0.02)
    r = robust_elbow(x, y, fast_config)
    assert isinstance(r, ClearKnee)
    assert abs(r.knee_x - 0.25) < 0.12


def test_noisy_line_abstains(fast_config):
    x, y = noisy_line(seed=2)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, NoClearKnee)


def test_two_knees_do_not_force_a_single_answer(fast_config):
    x, y = two_knee_curve(seed=1)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    # Either explicit multi-knee abstention, or at least not a false-confident
    # single knee wedged between the two real ones.
    if isinstance(r, ClearKnee):
        assert not (0.4 < r.knee_x < 0.6)
    else:
        assert isinstance(r, NoClearKnee)


def test_result_is_always_tagged_union(fast_config):
    for seed in range(3):
        x, y = clear_knee_curve(seed=seed)
        r = robust_knee(x, y, "concave", "increasing", fast_config)
        assert isinstance(r, (ClearKnee, NoClearKnee))
        assert r.is_clear == isinstance(r, ClearKnee)


def test_abstains_when_the_bootstrap_or_null_gate_fails(fast_config, monkeypatch):
    """A candidate that clears search and segmentation but fails a later
    gate abstains with that gate's own reason, not a false accept.

    These two gates sit deep inside ``robust_knee``'s search-confirmation
    branch (only reached once a real candidate survives everything before
    them), which a hand-picked synthetic curve cannot reliably steer into
    on demand; monkeypatching the gate's own verdict is the direct way to
    exercise both abstain paths deterministically.
    """
    import elbow_helper.pipeline as pipeline_module

    x, y = clear_knee_curve(seed=1, knee_frac=0.3, noise=0.02)

    fake_boot = types.SimpleNamespace(
        passes=False,
        reason=Reason.BOOTSTRAP_UNSTABLE,
        detection_rate=0.1,
        ci90=(0.1, 0.5),
        ci90_width=0.4,
        primary_cluster_rate=0.2,
        secondary_cluster_rate=0.3,
        median_shift=0.2,
    )
    monkeypatch.setattr(pipeline_module, "bootstrap_knee", lambda *a, **k: fake_boot)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, NoClearKnee)
    assert r.reason == Reason.BOOTSTRAP_UNSTABLE
    assert r.diagnostics["bootstrap"]["passes"] is False

    monkeypatch.undo()
    fake_null = types.SimpleNamespace(
        passes=False, reason=Reason.NULL_NOT_REJECTED, p_value=0.5, null_replicates=150
    )
    monkeypatch.setattr(pipeline_module, "no_knee_null_test", lambda *a, **k: fake_null)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, NoClearKnee)
    assert r.reason == Reason.NULL_NOT_REJECTED
    assert r.diagnostics["null"]["passes"] is False


def test_unexpected_internal_error_abstains_instead_of_crashing(
    fast_config, monkeypatch
):
    """The numerical-safety-net ``except Exception`` never lets a stray bug
    in a downstream stage crash the caller: it abstains with the error
    recorded in diagnostics, the same honesty contract every other gate
    follows."""
    import elbow_helper.pipeline as pipeline_module

    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic failure for coverage")

    monkeypatch.setattr(pipeline_module, "run_search", _boom)
    x, y = clear_knee_curve(seed=1)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, NoClearKnee)
    assert r.reason == Reason.INTERNAL_NUMERICAL_FAILURE
    assert "synthetic failure" in r.diagnostics["error"]
