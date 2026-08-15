"""Tests for the FastAPI door — skipped cleanly when the ``[api]`` extra is absent.

The HTTP surface must return the same decisions as the library, so these
drive the app through FastAPI's ``TestClient`` and check every route.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import pytest

from conftest import clear_knee_curve, elbow_curve, noisy_line


@pytest.fixture
def client():
    """A ``TestClient`` over a fresh app, skipped when ``[api]`` is absent.

    Returns
    -------
    fastapi.testclient.TestClient
        A client wrapping :func:`elbow_helper.api.create_app`'s app.
    """
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from elbow_helper.api import create_app

    return TestClient(create_app())


def test_knee_endpoint_clear(client) -> None:
    """``POST /knee`` on a clear-knee curve returns ``is_clear: true``."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    resp = client.post(
        "/knee",
        json={
            "x": x.tolist(),
            "y": y.tolist(),
            "config_overrides": {
                "random_seed": 0,
                "bootstrap_replicates": 60,
                "null_replicates": 120,
            },
        },
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["is_clear"] is True
    assert "ci90" in payload


def test_knee_endpoint_y_only_shorthand(client) -> None:
    """``POST /knee`` accepts the y-only shorthand (``y`` omitted from the body)."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    resp = client.post("/knee", json={"x": y.tolist()})
    assert resp.status_code == 200
    assert "is_clear" in resp.json()


def test_knee_endpoint_abstains_on_a_straight_line(client) -> None:
    """``POST /knee`` on a noisy straight line abstains, not a fake knee."""
    x, y = noisy_line(seed=2, noise=0.01)
    resp = client.post("/knee", json={"x": x.tolist(), "y": y.tolist()})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["is_clear"] is False
    assert "reason" in payload


def test_knee_endpoint_validation_error_on_missing_x(client) -> None:
    """A body missing the required ``x`` field is a 422, not a 500."""
    resp = client.post("/knee", json={})
    assert resp.status_code == 422


def test_elbow_endpoint(client) -> None:
    """``POST /elbow`` routes through ``robust_elbow`` and returns JSON."""
    x, y = elbow_curve(seed=1, noise=0.02)
    resp = client.post(
        "/elbow",
        json={
            "k": x.tolist(),
            "inertia": y.tolist(),
            "config_overrides": {
                "random_seed": 0,
                "bootstrap_replicates": 60,
                "null_replicates": 120,
            },
        },
    )
    assert resp.status_code == 200
    assert "is_clear" in resp.json()


def test_diagnostics_endpoint_returns_svg(client) -> None:
    """``POST /diagnostics`` returns an SVG document with the right content type."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    resp = client.post("/diagnostics", json={"x": x.tolist(), "y": y.tolist()})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert resp.text.startswith("<svg")


def test_diagnostics_endpoint_language_fr(client) -> None:
    """``POST /diagnostics`` with ``language: "fr"`` renders French chrome text."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    resp = client.post(
        "/diagnostics", json={"x": x.tolist(), "y": y.tolist(), "language": "fr"}
    )
    assert resp.status_code == 200
    assert "coude" in resp.text.lower()


def test_locator_endpoint(client) -> None:
    """``POST /locator`` returns the raw knee/all_knees payload."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    resp = client.post("/locator", json={"x": x.tolist(), "y": y.tolist()})
    assert resp.status_code == 200
    payload = resp.json()
    assert "knee" in payload
    assert "all_knees" in payload


def test_locator_endpoint_validation_error_on_missing_y(client) -> None:
    """``/locator`` requires both x and y (no shorthand) — a 422 on a missing y."""
    resp = client.post("/locator", json={"x": [1, 2, 3]})
    assert resp.status_code == 422


def test_locator_endpoint_empty_curve_returns_400_not_500(client) -> None:
    # /locator calls the raw KneeLocator directly (no abstention gate in
    # front of it, unlike /knee and /elbow), so an empty curve used to
    # surface as an unhandled ValueError -> FastAPI's generic 500,
    # indistinguishable from an actual server bug. It's a client-input
    # problem: 400.
    resp = client.post("/locator", json={"x": [], "y": []})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_openapi_operation_ids_are_stable(client) -> None:
    """Every route carries the operation_id the MCP door allowlists against."""
    schema = client.get("/openapi.json").json()
    op_ids = {
        op["operationId"]
        for path in schema["paths"].values()
        for op in path.values()
        if "operationId" in op
    }
    assert {"knee", "elbow", "diagnostics", "locator"} <= op_ids
