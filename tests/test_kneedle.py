"""Tests for the from-scratch, numpy-only Kneedle port."""

from __future__ import annotations

import numpy as np

from elbow_helper.kneedle import KneeLocator, _argrelextrema


def test_argrelextrema_matches_clip_semantics():
    a = np.array([0.0, 2.0, 1.0, 3.0, 0.5])
    maxima = _argrelextrema(a, np.greater_equal)
    # indices 1 and 3 are strict local maxima; endpoints handled by clip.
    assert 1 in maxima and 3 in maxima


def test_concave_increasing_knee_location():
    # Saturating curve: knee near x = 20.
    x = np.arange(1, 51, dtype=float)
    y = 1.0 - np.exp(-(x) / 8.0)
    kl = KneeLocator(x, y, S=1.0, curve="concave", direction="increasing")
    assert kl.knee is not None
    assert 5 <= kl.knee <= 20


def test_convex_decreasing_elbow_location():
    k = np.arange(1, 11, dtype=float)
    inertia = np.array([1000, 560, 330, 210, 165, 138, 120, 108, 100, 95], float)
    kl = KneeLocator(k, inertia, S=1.0, curve="convex", direction="decreasing")
    assert kl.knee == 4  # the classic elbow of this scree curve


def test_online_collects_multiple_and_records_align():
    x = np.arange(1, 51, dtype=float)
    y = 1.0 - np.exp(-x / 8.0)
    kl = KneeLocator(x, y, S=1.0, curve="concave", direction="increasing", online=True)
    assert len(kl.all_knee_records) == len(kl.all_knees)
    for rec in kl.all_knee_records:
        assert 0 <= rec["threshold_index"] < x.size


def test_transform_y_orientations_are_finite():
    y = np.linspace(0, 1, 10) ** 2
    for curve in ("concave", "convex"):
        for direction in ("increasing", "decreasing"):
            out = KneeLocator.transform_y(y.copy(), direction, curve)
            assert np.all(np.isfinite(out))
            assert out.shape == y.shape


def test_invalid_arguments_raise():
    x = np.arange(1, 11, dtype=float)
    y = np.sqrt(x)
    try:
        KneeLocator(x, y, curve="banana", direction="increasing")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for invalid curve")
