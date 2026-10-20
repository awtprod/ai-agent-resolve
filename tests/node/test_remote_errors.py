"""The remote link's error contract, one row at a time (S15a item 5).

Unreachable (OSError) vs refused (ValueError), with real HTTP on loopback and fault servers.
"""

from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response

from aar.hub import approvals, control, nodes
from aar.hub.approvals import ApprovalAlreadyConsumed, ApprovalScope
from aar.hub.common import InvalidInput
from aar.hub.control import Stopped
from aar.hub.events import Actor, EventLog
from aar.node.hublink import HubUnreachable, UnknownHubProject
from aar.node.locator import ProjectLocator
from aar.node.remote import FeedConflict, HubRefused, NodeNotEnrolled

from .remote_support import ServerThread, fault_app, hub_app, remote_link, write_token

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
        return PlainTextResponse("busy", status_code=status)

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
