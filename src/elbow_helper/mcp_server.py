"""MCP server — the agent-tool door (optional ``[mcp]`` extra), via fastapi-mcp.

The fourth of elbow-helper's surfaces lets an AI agent call the pipeline as
Model-Context-Protocol tools by mounting ``fastapi-mcp`` on a copy of
:mod:`elbow_helper.api`'s FastAPI app: the four REST operations already
tagged with an explicit ``operation_id`` there (``knee``/``elbow``/
``diagnostics``/``locator``) become the four MCP tools here, automatically,
with the same request/response schema. Rename or extend a route in
``api.py`` and this door follows without a second edit: MCP must never be
a standalone, hand-written tool server that could drift from the REST API.

MCP is served over **Streamable HTTP** (mounted at ``/mcp`` on a running
app), not stdio: an HTTP-based MCP client pointed at the running server's
``/mcp`` endpoint is what ``fastapi-mcp`` offers. The same four tools are
also just REST endpoints an agent (or a human with curl) can hit directly.

Run it with::

    pip install 'elbow-helper[mcp]'
    uvicorn elbow_helper.mcp_server:app --port 8021   # or: python -m elbow_helper.mcp_server

MCP endpoint: ``http://127.0.0.1:8021/mcp``. Runs on a different port from
:mod:`elbow_helper.api`'s 8020 by default so both doors can run side by side.

Consumes: ``fastapi-mcp`` (optional, pulls in ``fastapi``/``mcp``),
:mod:`elbow_helper.api`.
Produces: :data:`app` (the ASGI application), :func:`build_server`.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from fastapi_mcp import FastApiMCP
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the MCP server needs the [mcp] extra. Run: pip install 'elbow-helper[mcp]'"
    ) from exc

from .api import create_app

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

# The only REST operations exposed as MCP tools — matches the operation_id
# set on each route in elbow_helper.api.create_app(). An explicit allowlist
# (rather than excluding FastAPI's auto /docs, /openapi.json) so a future
# route added to the API door doesn't silently become an MCP tool without a
# deliberate opt-in.
_EXPOSED_OPERATIONS = ["knee", "elbow", "diagnostics", "locator"]


def build_server(app: FastAPI) -> FastApiMCP:
    """Mount the MCP server (knee/elbow/diagnostics/locator tools) onto ``app``.

    Parameters
    ----------
    app : fastapi.FastAPI
        The app to mount onto — typically a fresh
        :func:`elbow_helper.api.create_app` instance, so this module's
        :data:`app` stays independent of :mod:`elbow_helper.api`'s own
        module-level one.

    Returns
    -------
    fastapi_mcp.FastApiMCP
        The mounted server descriptor (``.name`` defaults to ``app.title``).

    Examples
    --------
    >>> from elbow_helper.api import create_app
    >>> mcp = build_server(create_app())  # doctest: +SKIP
    >>> mcp.name  # doctest: +SKIP
    'elbow-helper'
    """
    mcp = FastApiMCP(app, include_operations=_EXPOSED_OPERATIONS)
    mcp.mount_http()
    return mcp


# A fresh app (not elbow_helper.api's module-level one) with the MCP server
# mounted at /mcp, so `uvicorn elbow_helper.mcp_server:app` works out of the box.
app = create_app()
build_server(app)


def main() -> None:
    """Run the MCP-mounted app over HTTP (the ``python -m elbow_helper.mcp_server`` path)."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8021)


if __name__ == "__main__":  # pragma: no cover
    main()
