"""End-to-end pipeline tests: clear knees accepted, no-knee curves abstained."""

from __future__ import annotations

from elbow_helper import ClearKnee, NoClearKnee, Reason, robust_elbow, robust_knee

from conftest import clear_knee_curve, elbow_curve, noisy_line, two_knee_curve


def test_clear_knee_is_accepted_with_uncertainty(fast_config):
    x, y = clear_knee_curve(seed=1, knee_frac=0.3, noise=0.02)
    r = robust_knee(x, y, "concave", "increasing", fast_config)
    assert isinstance(r, ClearKnee)
    assert r.reason == Reason.CLEAR_KNEE
    assert abs(r.knee_x - 0.3) < 0.1              # near the true knee
    assert r.ci90[1] > r.ci90[0]                  # a real interval
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
