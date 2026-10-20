"""The node API over real HTTP on loopback (S15a item 3): auth, limits, ops, no token leak."""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from nodeapi_support import ServerThread, post

from aar.hub import approvals, control, nodes
from aar.hub.approvals import ApprovalScope
from aar.hub.events import Actor, EventLog
from aar.hub.leaseboard import presence_to_hub
from aar.hub.nodeapi import build_app

HUMAN = Actor("human", "andrew")


@dataclasses.dataclass
class Hub:
    base_url: str
    log: EventLog
    db_path: str


@pytest.fixture(scope="module")
def hub(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Hub]:
    db = tmp_path_factory.mktemp("nodeapi") / "hub.db"
    log = EventLog(str(db))
    with ServerThread(build_app(log)) as server:
        yield Hub(server.base_url, log, str(db))
    log.close()


def _enroll(hub: Hub, node_id: str) -> str:
    _, token = nodes.enroll(hub.log, node_id, HUMAN)
    return token


# -- routing and method -------------------------------------------------------------------


def test_unknown_route_is_404_and_non_post_is_405(hub: Hub) -> None:
    token = _enroll(hub, "route-node")
    assert post(hub.base_url, "", token=token).status == 404
    assert post(f"{hub.base_url}/nope", "assert_running", token=token).status == 404
    assert post(hub.base_url, "assert_running", token=token, method="GET").status == 405
    assert post(hub.base_url, "no_such_op", token=token).status == 404


# -- auth ---------------------------------------------------------------------------------


def test_missing_or_bad_token_is_401_without_detail(hub: Hub) -> None:
    for resp in (
        post(hub.base_url, "control_status"),  # no Authorization header
        post(hub.base_url, "control_status", token="not-a-token"),
    ):
        assert resp.status == 401
        assert resp.data["error"]["type"] == "unauthorized"
        assert "detail" not in resp.data["error"]


def test_a_revoked_token_is_rejected_on_the_next_request(hub: Hub) -> None:
    token = _enroll(hub, "revoke-node")
    assert post(hub.base_url, "control_status", {"request_id": "r"}, token=token).status == 200
    nodes.revoke(hub.log, "revoke-node", HUMAN)
    # Auth is checked before the body, so a revoked token is 401 regardless of request_id.
    assert post(hub.base_url, "control_status", {"request_id": "r"}, token=token).status == 401


# -- request limits -----------------------------------------------------------------------


def test_wrong_content_type_is_400(hub: Hub) -> None:
    token = _enroll(hub, "ct-node")
    resp = post(
        hub.base_url, "control_status", {"request_id": "r"}, token=token, content_type="text/plain"
    )
    assert resp.status == 400
    assert resp.data["error"]["type"] == "invalid_request"


def test_body_over_1_mib_is_413(hub: Hub) -> None:
    token = _enroll(hub, "big-node")
    raw = b'{"request_id": "r", "pad": "' + b"x" * (1 << 20) + b'"}'
    resp = post(hub.base_url, "control_status", token=token, raw=raw)
    assert resp.status == 413
    assert resp.data["error"]["type"] == "payload_too_large"


def test_non_object_body_and_missing_request_id_are_400(hub: Hub) -> None:
    token = _enroll(hub, "shape-node")
    assert post(hub.base_url, "control_status", token=token, raw=b"[1,2,3]").status == 400
    # missing request_id
    resp = post(hub.base_url, "control_status", {}, token=token)
    assert resp.status == 400 and resp.data["error"]["type"] == "invalid_request"


def test_body_node_id_must_match_the_token(hub: Hub) -> None:
    token = _enroll(hub, "self-node")
    resp = post(
        hub.base_url, "control_status", {"request_id": "r", "node_id": "someone-else"}, token=token
    )
    assert resp.status == 403
    assert resp.data["error"]["type"] == "node_mismatch"


# -- ops ----------------------------------------------------------------------------------


def test_assert_running_and_a_node_scoped_stop(hub: Hub) -> None:
    token = _enroll(hub, "stop-node")
    ok = post(hub.base_url, "assert_running", {"request_id": "r1"}, token=token)
    assert ok.status == 200 and ok.data == {
        "ok": True,
        "result": None,
        "replayed": False,
        "request_id": "r1",
    }
    # A stop scoped to this node bites at once (423 Locked, type stopped, with the record).
    control.stop(hub.log, control.StopScope.node("stop-node"), "safety", HUMAN)
    stopped = post(hub.base_url, "assert_running", {"request_id": "r2"}, token=token)
    assert stopped.status == 423
    assert stopped.data["error"]["type"] == "stopped"
    assert stopped.data["error"]["detail"]["scope"] == "node:stop-node"
    control.resume(hub.log, control.StopScope.node("stop-node"), "cleared", HUMAN)


def test_control_status(hub: Hub) -> None:
    token = _enroll(hub, "status-node")
    resp = post(hub.base_url, "control_status", {"request_id": "r"}, token=token)
    assert resp.status == 200
    assert resp.data["result"]["running"] in (True, False)
    assert isinstance(resp.data["result"]["stops"], list)


def test_approval_read_and_idempotent_consume(hub: Hub) -> None:
    token = _enroll(hub, "appr-node")
    granted = approvals.grant(hub.log, ApprovalScope.PROJECT_WRITE, "proj-a", 3600, "ok", HUMAN)
    actor = {"kind": "agent", "id": "agent-1#job-1"}

    read = post(
        hub.base_url, "approval", {"request_id": "r", "token_id": granted.token_id}, token=token
    )
    assert read.status == 200 and read.data["result"]["token_id"] == granted.token_id

    consume_body = {
        "request_id": "c1",
        "token_id": granted.token_id,
        "scope": "project.write",
        "target": "proj-a",
        "actor": actor,
    }
    first = post(hub.base_url, "consume_approval", consume_body, token=token)
    assert first.status == 200 and first.data["replayed"] is False
    assert first.data["result"]["consumed_at"] is not None
    # The consume recorded via_node from the token.
    consumed = next(
        e
        for e in hub.log.iter_events(types=("approval.consumed",))
        if e.payload["token_id"] == granted.token_id
    )
    assert consumed.payload["via_node"] == "appr-node"

    # A lost-ack retry (same node, same fingerprint, different request_id) replays.
    replay = post(
        hub.base_url, "consume_approval", {**consume_body, "request_id": "c2"}, token=token
    )
    assert replay.status == 200 and replay.data["replayed"] is True
    assert replay.data["result"] == first.data["result"]

    # Another node may not replay it.
    other = _enroll(hub, "appr-node-2")
    denied = post(
        hub.base_url, "consume_approval", {**consume_body, "request_id": "c3"}, token=other
    )
    assert denied.status == 409 and denied.data["error"]["type"] == "already_consumed"


def test_consume_refuses_a_human_body_actor(hub: Hub) -> None:
    token = _enroll(hub, "human-actor-node")
    granted = approvals.grant(hub.log, ApprovalScope.PROJECT_WRITE, "proj-h", 3600, "ok", HUMAN)
    resp = post(
        hub.base_url,
        "consume_approval",
        {
            "request_id": "r",
            "token_id": granted.token_id,
            "scope": "project.write",
            "target": "proj-h",
            "actor": {"kind": "human", "id": "andrew"},
        },
        token=token,
    )
    assert resp.status == 403 and resp.data["error"]["type"] == "actor_not_allowed"


def test_feed_deliver_keeps_the_idempotency_key_on_resend(hub: Hub) -> None:
    token = _enroll(hub, "feed-node")
    spec = presence_to_hub("feed-node", active=True, at=datetime.now(UTC))
    spec = dataclasses.replace(spec, idempotency_key="presence:feed-node:test:1")
    body = {
        "request_id": "f1",
        "spec": {
            "type": spec.type,
            "payload": spec.payload,
            "idempotency_key": spec.idempotency_key,
        },
    }
    first = post(hub.base_url, "feed_deliver", body, token=token)
    assert first.status == 200
    again = post(hub.base_url, "feed_deliver", {**body, "request_id": "f2"}, token=token)
    assert again.status == 200
    # The resent spec is a no-op: exactly one event under that key.
    matching = [
        e
        for e in hub.log.iter_events(types=(spec.type,))
        if e.idempotency_key == "presence:feed-node:test:1"
    ]
    assert len(matching) == 1


# -- no token leak ------------------------------------------------------------------------


def test_the_token_appears_in_neither_the_logs_nor_the_event_log(
    hub: Hub, caplog: pytest.LogCaptureFixture
) -> None:
    token = _enroll(hub, "secret-node")
    with caplog.at_level(logging.DEBUG, logger="aar.hub.nodeapi"):
        assert post(hub.base_url, "control_status", {"request_id": "r"}, token=token).status == 200
    assert token not in caplog.text
    import json as _json

    dump = _json.dumps([e.payload for e in hub.log.iter_events()])
    assert token not in dump
    # Only the sha256 is ever stored.
    assert nodes.token_sha256(token) in dump
