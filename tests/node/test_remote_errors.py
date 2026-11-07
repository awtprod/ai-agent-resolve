"""The remote link's error contract, one row at a time (S15a item 5).

Unreachable (OSError) vs refused (ValueError), with real HTTP on loopback and fault servers.
"""

from __future__ import annotations

import asyncio
import dataclasses
import http.client
import socket
import traceback
from datetime import UTC, datetime
from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from aar.hub import approvals, control, nodes
from aar.hub.approvals import ApprovalAlreadyConsumed, ApprovalScope
from aar.hub.common import InvalidInput
from aar.hub.control import Stopped
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import presence_to_hub
from aar.node.hublink import HubUnreachable, UnknownHubProject
from aar.node.locator import ProjectLocator
from aar.node.remote import FeedConflict, HubRefused, NodeNotEnrolled

from .remote_support import (
    ServerThread,
    fault_app,
    hub_app,
    remote_link,
    remote_transport,
    write_token,
)

HUMAN = Actor("human", "andrew")


def _hub(tmp_path: Path, node_id: str = "node-1") -> tuple[EventLog, Path, str]:
    log = EventLog(str(tmp_path / "hub.db"))
    _, token = nodes.enroll(log, node_id, HUMAN)
    token_file = write_token(tmp_path, token)
    return log, token_file, node_id


# -- unreachable (OSError) ----------------------------------------------------------------


@pytest.mark.parametrize("status", [500, 503, 429])
def test_5xx_429_503_are_unreachable(tmp_path: Path, status: int) -> None:
    async def responder(_: Request) -> Response:
        # A parseable refusal envelope: still unreachable because the STATUS decides, not the
        # body (so "map 503/429 to a refusal" is a killable mutation).
        return JSONResponse(
            {"ok": False, "error": {"type": "stopped", "message": "busy"}, "request_id": "r"},
            status_code=status,
        )

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(HubUnreachable):
            link.control_status()


def test_a_slow_reply_past_the_read_timeout_is_unreachable(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        await asyncio.sleep(0.6)  # < 1s, but past the read timeout below
        return JSONResponse({"ok": True, "result": None, "replayed": False, "request_id": "r"})

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file, read_timeout=0.3)
        with pytest.raises(HubUnreachable):
            link.control_status()


def test_a_non_json_200_is_unreachable(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        return PlainTextResponse("not json", status_code=200)

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(HubUnreachable):
            link.control_status()


def test_an_over_cap_response_is_unreachable(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        return PlainTextResponse("x" * 5000, status_code=200)

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file, response_cap=1024)
        with pytest.raises(HubUnreachable):
            link.control_status()


def test_connection_refused_is_unreachable(tmp_path: Path) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens here now
    token_file = write_token(tmp_path, "tok")
    link = remote_link(f"http://127.0.0.1:{port}", token_file, connect_timeout=2)
    with pytest.raises(HubUnreachable):
        link.control_status()


def test_dns_failure_is_unreachable(tmp_path: Path) -> None:
    token_file = write_token(tmp_path, "tok")
    # .invalid is reserved (RFC 2606) and never resolves; no real network is touched.
    link = remote_link("http://nonexistent.invalid:8789", token_file, connect_timeout=2)
    with pytest.raises(HubUnreachable):
        link.control_status()


def test_tls_failure_is_unreachable(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        return JSONResponse({"ok": True, "result": None, "replayed": False, "request_id": "r"})

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        # Point an https client at a plain-http server: the TLS handshake fails.
        https_url = f"https://127.0.0.1:{server.port}"
        link = remote_link(https_url, token_file, connect_timeout=2, read_timeout=2)
        with pytest.raises(HubUnreachable):
            link.control_status()


def test_the_hub_restarting_between_calls(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path)
    with ServerThread(hub_app(log)) as server:
        link = remote_link(server.base_url, token_file)
        link.control_status()  # up: ok
        port = server.port
    # Server stopped: the next call is unreachable.
    link_down = remote_link(f"http://127.0.0.1:{port}", token_file, connect_timeout=2)
    with pytest.raises(HubUnreachable):
        link_down.control_status()
    # Restarted (new port): a fresh link works again.
    with ServerThread(hub_app(log)) as server2:
        link_up = remote_link(server2.base_url, token_file)
        link_up.control_status()


# -- refusals (ValueError) ----------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"synthetic-alpha\nsynthetic-beta",
        b"synthetic-alpha\r\nsynthetic-beta",
        b"synthetic-alpha\rsynthetic-beta",
        b"synthetic-alpha synthetic-beta",
        b"synthetic-alpha\x00synthetic-beta",
        b"synthetic-alpha\tsynthetic-beta",
        b"synthetic-alpha\xc3\xa9synthetic-beta",
        b"synthetic-alpha\xffsynthetic-beta",
        b"synthetic-alpha" + b"x" * (4097 - len(b"synthetic-alpha")),
        b"synthetic-alpha:synthetic-beta",
        b"synthetic-alpha=synthetic-beta",
    ],
    ids=[
        "lf",
        "crlf",
        "cr",
        "space",
        "nul",
        "tab",
        "non-ascii",
        "invalid-utf8",
        "cap-plus-one",
        "punctuation",
        "interior-padding",
    ],
)
def test_invalid_token_is_private_and_never_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bytes
) -> None:
    received = []
    connections = []

    async def responder(request: Request) -> Response:
        received.append(request.headers.get("authorization"))
        return Response(status_code=401)

    connection_class = http.client.HTTPConnection

    def connect(*args, **kwargs):
        connections.append(True)
        return connection_class(*args, **kwargs)

    token_file = write_token(tmp_path, "placeholder")
    token_file.write_bytes(raw)
    with ServerThread(fault_app(responder)) as server:
        monkeypatch.setattr(http.client, "HTTPConnection", connect)
        link = remote_link(server.base_url, token_file)
        with pytest.raises(NodeNotEnrolled) as exc:
            link.control_status()
        formatted = "".join(traceback.format_exception(exc.value))
        for fragment in ("synthetic-alpha", "synthetic-beta"):
            assert fragment not in str(exc.value)
            assert fragment not in formatted
        assert not connections
        assert not received


def test_enrolled_token_with_trailing_newline_authenticates(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path)
    token_file.write_bytes(token_file.read_bytes() + b"\r\n")
    with ServerThread(hub_app(log)) as server:
        remote_link(server.base_url, token_file).control_status()


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize(
    "body",
    [
        b"<html>sensitive-refusal</html>",
        b"",
        b"{sensitive-refusal",
        b"\xffsensitive-refusal",
        b"sensitive-refusal" * 1000,
    ],
    ids=["html", "empty", "invalid-json", "invalid-utf8", "oversized"],
)
def test_auth_status_is_permanent_regardless_of_body(
    tmp_path: Path, status: int, body: bytes
) -> None:
    async def responder(_: Request) -> Response:
        return Response(body, status_code=status)

    token_file = write_token(tmp_path, "sensitive-token")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file, response_cap=1024)
        with pytest.raises(NodeNotEnrolled) as exc:
            link.control_status()
        assert "sensitive" not in str(exc.value)


@pytest.mark.parametrize("status", [401, 403])
def test_auth_status_is_permanent_before_slow_body(tmp_path: Path, status: int) -> None:
    async def chunks():
        await asyncio.sleep(5.0)
        yield b"sensitive-refusal"

    async def responder(_: Request) -> Response:
        return StreamingResponse(chunks(), status_code=status)

    token_file = write_token(tmp_path, "sensitive-token")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file, read_timeout=1.0)
        with pytest.raises(NodeNotEnrolled) as exc:
            link.control_status()
        assert "sensitive" not in str(exc.value)


def test_401_and_403_are_node_not_enrolled(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path)
    with ServerThread(hub_app(log)) as server:
        # 401: a token the hub never enrolled.
        bad_file = tmp_path / "bad_token"
        bad_file.write_text("not-a-real-token")
        bad_file.chmod(0o600)
        with pytest.raises(NodeNotEnrolled):
            remote_link(server.base_url, bad_file).control_status()
        # 403: a body node_id that is not the token's node (node_mismatch).
        link = remote_link(server.base_url, token_file)
        with pytest.raises(NodeNotEnrolled):
            link._client.call("control_status", {"node_id": "someone-else"})


def test_a_stop_is_the_stopped_exception(tmp_path: Path) -> None:
    log, token_file, node_id = _hub(tmp_path)
    control.stop(log, control.StopScope.node(node_id), "safety", HUMAN)
    with ServerThread(hub_app(log)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(Stopped):
            link.assert_running(node_id=node_id)


def test_an_already_consumed_token_across_nodes_is_an_approval_error(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path, "node-1")
    _, other_token = nodes.enroll(log, "node-2", HUMAN)
    other_file = tmp_path / "token2"
    other_file.write_text(other_token)
    other_file.chmod(0o600)
    granted = approvals.grant(log, ApprovalScope.PROJECT_WRITE, "proj-x", 3600, "ok", HUMAN)
    actor = Actor("agent", "agent-1#job-1")
    with ServerThread(hub_app(log)) as server:
        remote_link(server.base_url, token_file).consume_approval(
            granted.token_id, ApprovalScope.PROJECT_WRITE, "proj-x", actor
        )
        with pytest.raises(ApprovalAlreadyConsumed):
            remote_link(server.base_url, other_file).consume_approval(
                granted.token_id, ApprovalScope.PROJECT_WRITE, "proj-x", actor
            )


def test_an_unknown_project_is_unknown_hub_project(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path)
    with ServerThread(hub_app(log)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(UnknownHubProject):
            link.branches(ProjectLocator((), "Nope", "uid-404"))


def test_a_hub_invalid_input_is_invalid_input(tmp_path: Path) -> None:
    log, token_file, _ = _hub(tmp_path)
    granted = approvals.grant(log, ApprovalScope.PROJECT_WRITE, "proj-x", 3600, "ok", HUMAN)
    with ServerThread(hub_app(log)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(InvalidInput):
            link.consume_approval(
                granted.token_id, ApprovalScope.PROJECT_WRITE, "y" * 300, Actor("agent", "a#1")
            )


def test_a_409_feed_conflict_maps_to_feed_conflict(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        return JSONResponse(
            {
                "ok": False,
                "error": {"type": "feed_conflict", "message": "replaced"},
                "request_id": "r",
            },
            status_code=409,
        )

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(FeedConflict):
            link.control_status()


def test_an_unknown_4xx_type_is_hub_refused(tmp_path: Path) -> None:
    async def responder(_: Request) -> Response:
        return JSONResponse(
            {"ok": False, "error": {"type": "weird_code", "message": "no"}, "request_id": "r"},
            status_code=400,
        )

    token_file = write_token(tmp_path, "tok")
    with ServerThread(fault_app(responder)) as server:
        link = remote_link(server.base_url, token_file)
        with pytest.raises(HubRefused) as exc:
            link.control_status()
        assert exc.value.code == "weird_code"


def test_feed_transport_keeps_the_idempotency_key_on_resend(tmp_path: Path) -> None:
    log, token_file, node_id = _hub(tmp_path)
    spec = presence_to_hub(node_id, active=True, at=datetime.now(UTC))
    spec = dataclasses.replace(spec, idempotency_key=f"presence:{node_id}:test:1")
    with ServerThread(hub_app(log)) as server:
        transport = remote_transport(server.base_url, token_file)
        transport.deliver(spec, node_id=node_id)
        transport.deliver(spec, node_id=node_id)  # a resend of the same spec: same key
    # The hub dedupes on the kept key: exactly one event.
    matching = [
        e for e in log.iter_events(types=(spec.type,)) if e.idempotency_key == spec.idempotency_key
    ]
    assert len(matching) == 1
