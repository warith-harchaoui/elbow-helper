"""Tests for the MCP door — skipped cleanly when the ``[mcp]`` extra is absent.

``elbow_helper.mcp_server`` mounts ``fastapi-mcp`` on a copy of the REST app
(see ``elbow_helper.api``), so these drive the real MCP wire protocol
(JSON-RPC over the Streamable HTTP transport ``fastapi-mcp`` mounts at
``/mcp``) through FastAPI's ``TestClient`` — not just "the app object
exists". The session manager needs the app's lifespan running, hence
``with TestClient(app) as``.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import pytest

from conftest import clear_knee_curve


@pytest.fixture
def client():
    """A ``TestClient`` (lifespan started) over an MCP-mounted app.

    Returns
    -------
    fastapi.testclient.TestClient
        A client wrapping a fresh app with :func:`build_server` mounted.
    """
    pytest.importorskip("fastapi_mcp")
    from fastapi.testclient import TestClient

    from elbow_helper.api import create_app
    from elbow_helper.mcp_server import build_server

    app = create_app()
    build_server(app)
    with TestClient(app) as c:
        yield c


def _mcp_call(
    client, method: str, params: dict, session_id: str | None = None
) -> tuple:
    """POST one JSON-RPC message to /mcp; returns (response, mcp-session-id).

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        A client over an MCP-mounted app.
    method : str
        The JSON-RPC method (``"initialize"``, ``"tools/list"``, ``"tools/call"``).
    params : dict
        The JSON-RPC ``params`` payload.
    session_id : str, optional
        An existing MCP session id to send in the ``mcp-session-id`` header.

    Returns
    -------
    response : httpx.Response
        The raw HTTP response.
    session_id : str or None
        The ``mcp-session-id`` response header, if present.
    """
    headers = {"accept": "application/json, text/event-stream"}
    if session_id:
        headers["mcp-session-id"] = session_id
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers=headers,
    )
    return resp, resp.headers.get("mcp-session-id")


def _mcp_session(client) -> str:
    """Complete the MCP initialize handshake and return the session id.

    Parameters
    ----------
    client : fastapi.testclient.TestClient
        A client over an MCP-mounted app.

    Returns
    -------
    str
        The negotiated ``mcp-session-id``.
    """
    resp, session_id = _mcp_call(
        client,
        "initialize",
        {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        },
    )
    assert resp.status_code == 200
    assert session_id
    return session_id


def test_build_server_name_defaults_to_app_title() -> None:
    """build_server()'s MCP server name defaults to the app's title."""
    pytest.importorskip("fastapi_mcp")
    from elbow_helper.api import create_app
    from elbow_helper.mcp_server import build_server

    mcp = build_server(create_app())
    assert mcp.name == "elbow-helper"


def test_mcp_route_is_mounted_on_the_app() -> None:
    """mcp_server.app serves /mcp alongside the plain REST routes."""
    pytest.importorskip("fastapi_mcp")
    from elbow_helper.mcp_server import app

    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)
    # The plain REST routes still work on the same app — MCP is additive.
    assert any(getattr(r, "path", "") == "/knee" for r in app.routes)


def test_exposed_tools_are_exactly_the_four_operations(client) -> None:
    """tools/list returns exactly knee/elbow/diagnostics/locator, nothing leaked."""
    session_id = _mcp_session(client)
    resp, _ = _mcp_call(client, "tools/list", {}, session_id)
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["result"]["tools"]}
    assert names == {"knee", "elbow", "diagnostics", "locator"}


def test_knee_tool_call_matches_the_library_decision(client) -> None:
    """tools/call on 'knee' returns the same decision the library would."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    session_id = _mcp_session(client)
    resp, _ = _mcp_call(
        client,
        "tools/call",
        {
            "name": "knee",
            "arguments": {
                "x": x.tolist(),
                "y": y.tolist(),
                "config_overrides": {
                    "random_seed": 0,
                    "bootstrap_replicates": 60,
                    "null_replicates": 120,
                },
            },
        },
        session_id,
    )
    assert resp.status_code == 200
    payload = resp.json()["result"]
    assert payload["isError"] is False
    assert '"is_clear": true' in payload["content"][0]["text"]


def test_locator_tool_call(client) -> None:
    """tools/call on 'locator' returns a knee/all_knees payload."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    session_id = _mcp_session(client)
    resp, _ = _mcp_call(
        client,
        "tools/call",
        {"name": "locator", "arguments": {"x": x.tolist(), "y": y.tolist()}},
        session_id,
    )
    body = resp.json()["result"]
    assert body["isError"] is False
    assert "all_knees" in body["content"][0]["text"]
