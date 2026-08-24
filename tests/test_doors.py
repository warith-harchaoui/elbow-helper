"""Cross-door behavioral parity tests.

The argparse CLI, the click CLI, the HTTP API, and the MCP server are all
thin adapters over :mod:`elbow_helper._core_cli` (see its module
docstring): none of them implements its own decision logic. The old suite
tested each door with its own near-identical copy of the same handful of
scenarios (four "knee returns is_clear" tests, four "elbow needs both
series" tests, and so on), which multiplied test count without multiplying
what was actually being checked.

This module drives the same scenarios through every door inside one test
function each, via a small per-door adapter below, so a scenario now
verifies parity across the whole surface in one place instead of four
places that could silently drift apart. Door-specific mechanics that do not
generalize (file-loading flags, the ``--online`` toggle, the MCP session
handshake, click's raw ``main()`` traceback wrapping, the OpenAPI
operation-id contract) stay as their own small tests below the shared
scenarios.

Author
------
Warith Harchaoui, <warith.harchaoui@deraison.ai>
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from conftest import clear_knee_curve, elbow_curve, noisy_line


def _values_flag(arr: np.ndarray) -> str:
    """Format a numpy array as a comma-separated ``--*-values`` string."""
    return ",".join(f"{v:.6g}" for v in arr)


@dataclasses.dataclass
class DoorResult:
    """A door invocation's outcome, normalized across all four doors.

    Attributes
    ----------
    ok : bool
        Whether the door reported success (exit 0 / HTTP 2xx / ``isError``
        False), regardless of that door's own status-code vocabulary.
    payload : Any
        The parsed JSON dict for knee/elbow/locator, the raw SVG string for
        diagnostics, or ``None`` on failure.
    error_text : str
        Whatever error text the door produced, for failure-path assertions
        and for surfacing in a failed test's message.
    """

    ok: bool
    payload: Any
    error_text: str = ""


class ArgparseDoor:
    name = "argparse"

    def __init__(self, capsys):
        self._capsys = capsys

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        from elbow_helper import cli_argparse

        code = cli_argparse.main(argv)
        out = self._capsys.readouterr()
        return code, out.out, out.err

    def knee(self, x=None, y=None, config_overrides=None) -> DoorResult:
        argv = ["knee"]
        if x is not None:
            argv += ["--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        if config_overrides:
            argv += ["--config-json", json.dumps(config_overrides)]
        code, out, err = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, err)

    def elbow(self, x, y=None, config_overrides=None) -> DoorResult:
        argv = ["elbow", "--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        if config_overrides:
            argv += ["--config-json", json.dumps(config_overrides)]
        code, out, err = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, err)

    def diagnostics(self, x, y) -> DoorResult:
        argv = [
            "diagnostics",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
        ]
        code, out, err = self._run(argv)
        return DoorResult(code == 0, out if code == 0 else None, err)

    def locator(self, x=None, y=None) -> DoorResult:
        argv = ["locator"]
        if x is not None:
            argv += ["--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        code, out, err = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, err)


class ClickDoor:
    name = "click"

    def __init__(self, capsys):
        pytest.importorskip("click")
        from click.testing import CliRunner

        from elbow_helper.cli_click import cli

        self._runner = CliRunner()
        self._cli = cli

    def _run(self, argv: list[str]) -> tuple[int, str]:
        result = self._runner.invoke(self._cli, argv)
        return result.exit_code, result.output

    def knee(self, x=None, y=None, config_overrides=None) -> DoorResult:
        argv = ["knee"]
        if x is not None:
            argv += ["--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        if config_overrides:
            argv += ["--config-json", json.dumps(config_overrides)]
        code, out = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, out)

    def elbow(self, x, y=None, config_overrides=None) -> DoorResult:
        argv = ["elbow", "--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        if config_overrides:
            argv += ["--config-json", json.dumps(config_overrides)]
        code, out = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, out)

    def diagnostics(self, x, y) -> DoorResult:
        argv = [
            "diagnostics",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
        ]
        code, out = self._run(argv)
        return DoorResult(code == 0, out if code == 0 else None, out)

    def locator(self, x=None, y=None) -> DoorResult:
        argv = ["locator"]
        if x is not None:
            argv += ["--x-values", _values_flag(x)]
        if y is not None:
            argv += ["--y-values", _values_flag(y)]
        code, out = self._run(argv)
        return DoorResult(code == 0, json.loads(out) if code == 0 else None, out)


class ApiDoor:
    name = "api"

    def __init__(self, capsys):
        pytest.importorskip("fastapi")
        pytest.importorskip("httpx")
        from fastapi.testclient import TestClient

        from elbow_helper.api import create_app

        self._client = TestClient(create_app())

    def knee(self, x=None, y=None, config_overrides=None) -> DoorResult:
        body: dict = {}
        if x is not None:
            body["x"] = list(x)
        if y is not None:
            body["y"] = list(y)
        if config_overrides:
            body["config_overrides"] = config_overrides
        resp = self._client.post("/knee", json=body)
        ok = resp.status_code == 200
        return DoorResult(ok, resp.json() if ok else None, resp.text)

    def elbow(self, x, y=None, config_overrides=None) -> DoorResult:
        body: dict = {"k": list(x)}
        if y is not None:
            body["inertia"] = list(y)
        if config_overrides:
            body["config_overrides"] = config_overrides
        resp = self._client.post("/elbow", json=body)
        ok = resp.status_code == 200
        return DoorResult(ok, resp.json() if ok else None, resp.text)

    def diagnostics(self, x, y) -> DoorResult:
        resp = self._client.post("/diagnostics", json={"x": list(x), "y": list(y)})
        ok = resp.status_code == 200
        return DoorResult(ok, resp.text if ok else None, resp.text)

    def locator(self, x=None, y=None) -> DoorResult:
        body: dict = {}
        if x is not None:
            body["x"] = list(x)
        if y is not None:
            body["y"] = list(y)
        resp = self._client.post("/locator", json=body)
        ok = resp.status_code == 200
        return DoorResult(ok, resp.json() if ok else None, resp.text)


class McpDoor:
    name = "mcp"

    def __init__(self, capsys):
        pytest.importorskip("fastapi_mcp")
        from fastapi.testclient import TestClient

        from elbow_helper.api import create_app
        from elbow_helper.mcp_server import build_server

        app = create_app()
        build_server(app)
        self._client = TestClient(app)
        self._client.__enter__()  # start the lifespan; closed by pytest's GC
        self._session_id = None
        self._session_id = self._handshake()

    def _call(self, method: str, params: dict) -> dict:
        headers = {"accept": "application/json, text/event-stream"}
        if self._session_id:
            headers["mcp-session-id"] = self._session_id
        return self._client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            headers=headers,
        )

    def _handshake(self) -> str:
        resp = self._call(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        )
        return resp.headers.get("mcp-session-id")

    def _tool_call(self, name: str, arguments: dict) -> DoorResult:
        resp = self._call("tools/call", {"name": name, "arguments": arguments})
        if resp.status_code != 200:
            return DoorResult(False, None, resp.text)
        body = resp.json()["result"]
        text = body["content"][0]["text"]
        ok = not body["isError"]
        payload = None
        if ok:
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = text  # diagnostics: raw SVG, not JSON
        return DoorResult(ok, payload, "" if ok else text)

    def knee(self, x=None, y=None, config_overrides=None) -> DoorResult:
        args: dict = {}
        if x is not None:
            args["x"] = list(x)
        if y is not None:
            args["y"] = list(y)
        if config_overrides:
            args["config_overrides"] = config_overrides
        return self._tool_call("knee", args)

    def elbow(self, x, y=None, config_overrides=None) -> DoorResult:
        args: dict = {"k": list(x)}
        if y is not None:
            args["inertia"] = list(y)
        if config_overrides:
            args["config_overrides"] = config_overrides
        return self._tool_call("elbow", args)

    def diagnostics(self, x, y) -> DoorResult:
        return self._tool_call("diagnostics", {"x": list(x), "y": list(y)})

    def locator(self, x=None, y=None) -> DoorResult:
        args: dict = {}
        if x is not None:
            args["x"] = list(x)
        if y is not None:
            args["y"] = list(y)
        return self._tool_call("locator", args)


def _all_doors(capsys) -> list:
    """Every installed door (argparse is always available; the rest are
    skipped by their constructor's ``importorskip`` when the optional
    extra is absent, so this list shrinks gracefully in a minimal install).
    """
    doors = [ArgparseDoor(capsys)]
    for cls in (ClickDoor, ApiDoor, McpDoor):
        try:
            doors.append(cls(capsys))
        except pytest.skip.Exception:
            continue
    return doors


def test_knee_clear_is_accepted_across_every_door(capsys) -> None:
    """A clear-knee curve returns ``is_clear: true`` with a CI, everywhere."""
    x, y = clear_knee_curve(seed=2, noise=0.02)
    overrides = {"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}
    for door in _all_doors(capsys):
        r = door.knee(x, y, config_overrides=overrides)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert r.payload["is_clear"] is True, door.name
        assert "ci90" in r.payload, door.name


def test_knee_abstains_on_a_straight_line_across_every_door(capsys) -> None:
    """A noisy straight line abstains, with a reason, everywhere."""
    x, y = noisy_line(seed=2, noise=0.01)
    for door in _all_doors(capsys):
        r = door.knee(x, y)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert r.payload["is_clear"] is False, door.name
        assert "reason" in r.payload, door.name


def test_knee_y_only_shorthand_works_across_every_door(capsys) -> None:
    """A bare y series (x omitted) triggers ``robust_knee(y)`` everywhere."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    for door in _all_doors(capsys):
        r = door.knee(x=y)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert "is_clear" in r.payload, door.name


def test_elbow_is_accepted_across_every_door(capsys) -> None:
    """A clear elbow curve routes through ``robust_elbow`` everywhere."""
    x, y = elbow_curve(seed=1, noise=0.02)
    overrides = {"random_seed": 0, "bootstrap_replicates": 60, "null_replicates": 120}
    for door in _all_doors(capsys):
        r = door.elbow(x, y, config_overrides=overrides)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert "is_clear" in r.payload, door.name


def test_elbow_and_locator_reject_the_y_only_shorthand_across_every_door(
    capsys,
) -> None:
    """Neither ``elbow`` nor ``locator`` has ``knee``'s y-only inference."""
    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    for door in _all_doors(capsys):
        assert not door.elbow(x=y).ok, f"{door.name}: elbow"
        assert not door.locator(x=y).ok, f"{door.name}: locator"


def test_diagnostics_returns_an_svg_document_across_every_door(capsys) -> None:
    """Every door's ``diagnostics`` returns a real, parseable SVG document."""
    x, y = clear_knee_curve(seed=1, noise=0.02)
    for door in _all_doors(capsys):
        r = door.diagnostics(x, y)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert r.payload.strip().startswith("<svg"), door.name


def test_locator_returns_knee_and_all_knees_across_every_door(capsys) -> None:
    """The standalone locator's raw payload shape is identical everywhere."""
    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    for door in _all_doors(capsys):
        r = door.locator(x, y)
        assert r.ok, f"{door.name}: {r.error_text}"
        assert "knee" in r.payload, door.name
        assert "all_knees" in r.payload, door.name


# --- Door-specific mechanics: not generalizable across all four doors -----


def test_root_help_lists_every_subcommand(capsys) -> None:
    """``--help`` on both CLI twins lists every subcommand (argparse and
    click each render help text their own way, so this stays CLI-only)."""
    from elbow_helper import cli_argparse

    with pytest.raises(SystemExit) as exc:
        cli_argparse.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("knee", "elbow", "diagnostics", "locator"):
        assert cmd in out

    click = pytest.importorskip("click")
    _ = click
    from click.testing import CliRunner

    from elbow_helper.cli_click import cli

    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for cmd in ("knee", "elbow", "diagnostics", "locator"):
        assert cmd in result.output


def test_argparse_loads_series_from_every_file_format(tmp_path, capsys) -> None:
    """``--y-npy``, and ``--y-csv`` with or without a header row, all load
    the same series (three input formats beyond the plain ``--y-values``
    comma list already covered by the cross-door scenarios above)."""
    from elbow_helper import cli_argparse

    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)

    npy_path = tmp_path / "y.npy"
    np.save(npy_path, y)
    assert cli_argparse.main(["knee", "--y-npy", str(npy_path)]) == 0
    assert "is_clear" in json.loads(capsys.readouterr().out)

    csv_header = tmp_path / "y_header.csv"
    csv_header.write_text(
        "value\n" + "\n".join(f"{v:.6g}" for v in y), encoding="utf-8"
    )
    assert cli_argparse.main(["knee", "--y-csv", f"{csv_header}:0"]) == 0
    assert "is_clear" in json.loads(capsys.readouterr().out)

    csv_plain = tmp_path / "y_plain.csv"
    csv_plain.write_text("\n".join(f"{v:.6g}" for v in y), encoding="utf-8")
    assert cli_argparse.main(["knee", "--y-csv", f"{csv_plain}:0"]) == 0
    assert "is_clear" in json.loads(capsys.readouterr().out)


def test_argparse_input_edge_cases_are_clear_errors(capsys) -> None:
    """A malformed ``--y-csv`` spec, no data at all, and two data flags at
    once are each a clear ``Error: ...`` on stderr, exit 1 — never a raw
    traceback. Building the parser itself has no side effects."""
    from elbow_helper import cli_argparse

    assert cli_argparse.main(["knee", "--y-csv", "data.csv"]) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "PATH:COLUMN_INDEX" in err

    assert cli_argparse.main(["knee"]) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "no data given" in err

    assert cli_argparse.main(["knee", "--y-values", "1,2,3", "--y-npy", "x.npy"]) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "at most one" in err

    parser = cli_argparse.build_parser()
    ns = parser.parse_args(["knee", "--y-values", "1,2,3"])
    assert ns.command == "knee"


def test_argparse_online_flag_defaults_true_and_no_online_disables_it(
    capsys,
) -> None:
    """``--online`` defaults to True; ``--no-online`` actually clears it and
    the locator still dispatches successfully with ``online=False``.

    Regression test: ``--online`` used to be declared with
    ``action="store_true", default=True``, which can only ever leave the
    value True; there was no way to pass ``online=False`` through this CLI,
    unlike the click twin's ``--online/--no-online`` pair.
    """
    from elbow_helper import cli_argparse

    parser = cli_argparse.build_parser()
    default_ns = parser.parse_args(
        ["locator", "--x-values", "1,2", "--y-values", "1,2"]
    )
    assert default_ns.online is True
    off_ns = parser.parse_args(
        ["locator", "--x-values", "1,2", "--y-values", "1,2", "--no-online"]
    )
    assert off_ns.online is False

    x, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    code = cli_argparse.main(
        [
            "locator",
            "--x-values",
            _values_flag(x),
            "--y-values",
            _values_flag(y),
            "--no-online",
        ]
    )
    assert code == 0
    assert "all_knees" in json.loads(capsys.readouterr().out)


def test_click_input_edge_cases(capsys) -> None:
    """click's own ``_resolve_xy`` mirrors argparse's ``_xy_from_args`` (same
    "no data given" usage error, same y-only shorthand branch), on its own
    code path -- worth its own check since the two never share code."""
    door = ClickDoor(capsys)
    no_data = door._run(["knee"])
    assert no_data[0] == 2
    assert "no data given" in no_data[1]

    _, y = clear_knee_curve(seed=1, noise=0.0, n=60)
    r = door.knee(y=y)
    assert r.ok, r.error_text
    assert "is_clear" in r.payload


def test_click_main_wraps_a_library_exception_as_a_clean_error(
    capsys, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CliRunner.invoke (used everywhere above) catches exceptions itself, so
    # it never exercises main() -- the actual `elbow-helper-click` console-
    # script entry point. Drive main() directly: an unparseable --x-values
    # raises a plain ValueError from float(), which used to propagate as a
    # raw Python traceback instead of a clean "Error: ..." + exit 1.
    pytest.importorskip("click")
    from elbow_helper.cli_click import main

    monkeypatch.setattr(
        "sys.argv",
        ["elbow-helper-click", "locator", "--x-values", "", "--y-values", ""],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Error:" in capsys.readouterr().err


def test_api_diagnostics_language_fr_and_validation_errors(capsys) -> None:
    """The FR chrome-text switch, and the two 4xx edge cases the app-level
    abstention gate doesn't cover: a schema-invalid body (422) and a
    locator call on a genuinely empty curve, which has no abstention gate
    in front of it and used to surface as a generic 500 (400 instead)."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from elbow_helper.api import create_app

    client = TestClient(create_app())
    x, y = clear_knee_curve(seed=1, noise=0.02)

    resp = client.post(
        "/diagnostics", json={"x": x.tolist(), "y": y.tolist(), "language": "fr"}
    )
    assert resp.status_code == 200
    assert "coude" in resp.text.lower()

    assert client.post("/knee", json={}).status_code == 422
    assert client.post("/locator", json={"x": [1, 2, 3]}).status_code == 422

    resp = client.post("/locator", json={"x": [], "y": []})
    assert resp.status_code == 400
    assert "empty" in resp.json()["detail"]


def test_api_openapi_operation_ids_are_stable(capsys) -> None:
    """Every route carries the operation_id the MCP door allowlists against."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from elbow_helper.api import create_app

    client = TestClient(create_app())
    schema = client.get("/openapi.json").json()
    op_ids = {
        op["operationId"]
        for path in schema["paths"].values()
        for op in path.values()
        if "operationId" in op
    }
    assert {"knee", "elbow", "diagnostics", "locator"} <= op_ids


def test_mcp_server_construction(capsys) -> None:
    """``build_server``'s name defaults to the app's title, and the MCP
    route is mounted additively alongside the plain REST routes."""
    pytest.importorskip("fastapi_mcp")
    from elbow_helper.api import create_app
    from elbow_helper.mcp_server import app, build_server

    mcp = build_server(create_app())
    assert mcp.name == "elbow-helper"
    assert any(getattr(r, "path", "") == "/mcp" for r in app.routes)
    assert any(getattr(r, "path", "") == "/knee" for r in app.routes)


def test_mcp_exposes_exactly_the_four_operations(capsys) -> None:
    """``tools/list`` returns exactly knee/elbow/diagnostics/locator."""
    door = McpDoor(capsys)
    resp = door._call("tools/list", {})
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["result"]["tools"]}
    assert names == {"knee", "elbow", "diagnostics", "locator"}
