"""FastAPI HTTP surface — the web-API door (optional ``[api]`` extra).

The third of elbow-helper's surfaces exposes the pipeline over HTTP so a
service can ask "where's the knee in this curve?" without a Python import.
FastAPI + uvicorn live behind the ``[api]`` extra, imported lazily here so
the core package never pays for a web stack it does not use — the whole
point of the package staying at ``numpy`` + ``os-helper`` for everyone who
just wants the library. Every endpoint delegates to
:mod:`elbow_helper._core_cli`, so the HTTP behaviour matches the CLI and
library exactly.

Each route carries an explicit ``operation_id`` (``knee``/``elbow``/
``diagnostics``/``locator``) — not just OpenAPI hygiene:
:mod:`elbow_helper.mcp_server` mounts ``fastapi-mcp`` on a copy of this same
app and selects exactly these operation ids as the exposed MCP tools, so the
id *is* the tool name an agent calls. Rename a route here and the MCP door
renames with it, automatically.

Run it with::

    pip install 'elbow-helper[api]'
    uvicorn elbow_helper.api:app --reload   # or: python -m elbow_helper.api

Consumes: ``fastapi`` (optional), ``elbow_helper._core_cli``.
Produces: :data:`app` (the ASGI application), :func:`create_app`.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

from typing import Optional

try:
    from fastapi import FastAPI, Request, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "the HTTP API needs the [api] extra. Run: pip install 'elbow-helper[api]'"
    ) from exc

from . import _core_cli as core
from . import __version__


class CurveRequest(BaseModel):
    """Request body for ``/knee`` and ``/diagnostics``: a curve plus knobs.

    ``y`` may be omitted, mirroring :func:`elbow_helper.robust_knee`'s
    shorthand where a single series is treated as y-values against an
    implicit ``0, 1, ..., n-1`` x axis.
    """

    x: list[float]
    y: Optional[list[float]] = None
    curve: Optional[str] = None
    direction: Optional[str] = None
    config_overrides: Optional[dict] = None


class ElbowRequest(BaseModel):
    """Request body for ``/elbow``: an explicit x/y pair, both required."""

    k: list[float]
    inertia: list[float]
    config_overrides: Optional[dict] = None


class DiagnosticsRequest(CurveRequest):
    """Request body for ``/diagnostics``: a curve plus the SVG's language."""

    language: str = "en"


class LocatorRequest(BaseModel):
    """Request body for ``/locator``: an explicit x/y pair, no config overrides."""

    x: list[float]
    y: list[float]
    sensitivity: float = 1.0
    curve: Optional[str] = None
    direction: Optional[str] = None
    online: bool = True


def create_app() -> FastAPI:
    """Build and return the FastAPI application.

    Returns
    -------
    FastAPI
        The ASGI app with the ``/knee``, ``/elbow``, ``/diagnostics`` and
        ``/locator`` routes wired to the shared core.

    Examples
    --------
    >>> app = create_app()  # doctest: +SKIP
    >>> [r.path for r in app.routes]  # doctest: +SKIP
    ['/openapi.json', '/docs', ..., '/knee', '/elbow', '/diagnostics', '/locator']
    """
    app = FastAPI(
        title="elbow-helper",
        description=(
            "Noise-robust knee/elbow detection: a knee with uncertainty, "
            "or an explicit abstention."
        ),
        version=__version__,
    )

    @app.exception_handler(ValueError)
    async def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """Map a library ``ValueError`` to HTTP 400 instead of a generic 500.

        ``/knee``/``/elbow``/``/diagnostics`` go through ``robust_knee``/
        ``robust_elbow``, which never raise (a numerical-safety-net
        ``except Exception`` inside them converts any internal failure to a
        ``NoClearKnee``/diagnostics result instead). ``/locator`` is the
        exception: it calls the raw, unguarded ``KneeLocator`` directly (by
        design — it is the building block, not the conservative pipeline),
        so a malformed request (e.g. an empty curve) surfaces here as a
        ``ValueError`` that would otherwise fall through to FastAPI's
        generic 500, indistinguishable from an actual server bug.
        """
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.post("/knee", operation_id="knee")
    def knee(body: CurveRequest) -> dict:
        """Locate a knee with uncertainty, or abstain.

        Parameters
        ----------
        body : CurveRequest
            The curve and orientation/config knobs.

        Returns
        -------
        dict
            The serialized ``ClearKnee``/``NoClearKnee``.
        """
        return core.do_knee(
            body.x,
            body.y,
            curve=body.curve,
            direction=body.direction,
            config_overrides=body.config_overrides,
        )

    @app.post("/elbow", operation_id="elbow")
    def elbow(body: ElbowRequest) -> dict:
        """The convex/decreasing convenience (k-means elbow).

        Parameters
        ----------
        body : ElbowRequest
            The ``k`` (x) and ``inertia`` (y) series, plus config overrides.

        Returns
        -------
        dict
            The serialized ``ClearKnee``/``NoClearKnee``.
        """
        return core.do_elbow(
            body.k, body.inertia, config_overrides=body.config_overrides
        )

    @app.post("/diagnostics", operation_id="diagnostics")
    def diagnostics(body: DiagnosticsRequest) -> Response:
        """Render the diagnostic SVG.

        Parameters
        ----------
        body : DiagnosticsRequest
            The curve, orientation/config knobs, and chrome-text language.

        Returns
        -------
        fastapi.Response
            The SVG document, ``Content-Type: image/svg+xml``.
        """
        svg = core.do_diagnostics(
            body.x,
            body.y,
            curve=body.curve,
            direction=body.direction,
            config_overrides=body.config_overrides,
            language=body.language,
        )
        return Response(content=svg, media_type="image/svg+xml")

    @app.post("/locator", operation_id="locator")
    def locator(body: LocatorRequest) -> dict:
        """The standalone locator (no uncertainty evidence).

        Parameters
        ----------
        body : LocatorRequest
            The curve, sensitivity, orientation hints, and traversal mode.

        Returns
        -------
        dict
            ``{"knee": float | None, "all_knees": list[float]}``.
        """
        return core.do_locator(
            body.x,
            body.y,
            sensitivity=body.sensitivity,
            curve=body.curve,
            direction=body.direction,
            online=body.online,
        )

    return app


# Module-level ASGI app so `uvicorn elbow_helper.api:app` works out of the box.
app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8020)
