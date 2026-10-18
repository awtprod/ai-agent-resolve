"""S12f item 6: tests that pin the guards the validator's mutants removed without a red.

Each test names the mutant id it pins (G09 ... W16). Every scenario forges the shape
straight into the log or drives the console, the way a buggy or hostile writer would.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import InvalidInput, skipped_summary
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import lease_board, lease_event_to_hub
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
OTHER_AGENT = Actor("agent", "agent-2")
NODE = Actor("node", "node-1")
OTHER_NODE = Actor("node", "node-2")
BASE = "http://127.0.0.1:8788"
SECRET = b"0123456789abcdef0123456789abcdef"
NOW = datetime.now(UTC).replace(microsecond=0)
ISO = NOW.isoformat()
LATER = (NOW + timedelta(hours=1)).isoformat()
LONG_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%S.") + "1" * 80 + "+00:00"  # valid, 106 chars


class Hub:
    def __init__(self, root: Path) -> None:
        self.log = EventLog(root / "hub.db")
        self.store = ArtifactStore(root / "artifacts", self.log)
        self.pid = Registry(self.log).create_project("demo-show", HUMAN).id
        self.graph = WorkGraph(self.log, self.pid)
        self.op = self.graph.add_node("op", AGENT, title="build").id
        self.artifact = self.graph.add_node("artifact", AGENT, title="render").id
        render = self.store.put(b"render", actor=AGENT, mime="video/mp4", project_id=self.pid)
        self.review = reviews.request_review(
            self.log, self.store, self.pid, self.op, [render], AGENT, title="cut v1", fps=24
        ).id

    def forge(self, event_type: str, actor: Actor, payload: Any, *, project: bool = True) -> str:
        project_id = self.pid if project else None
        return self.log.append(event_type, actor, payload, project_id=project_id).id

    def client(self) -> TestClient:
        app = build_app(
            self.log, self.store, ConsoleConfig(operator="operator-1", csrf_secret=SECRET)
        )
        return TestClient(app, base_url=BASE, client=("127.0.0.1", 50000), follow_redirects=False)


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def _graph_skips(hub: Hub) -> set[str]:
    return {item.event_id for item in WorkGraph(hub.log, hub.pid).skipped()}


def _review_skips(hub: Hub) -> set[str]:
    return {item.event_id for item in reviews.review_view(hub.log, hub.pid, hub.review)[1]}


# -- G09, G10, G12, G13: forged node and edge shapes -----------------------------------


@pytest.mark.parametrize(
    ("mutant", "patch"),
    [
        ("node-id", {"node_id": "n" * 200}),
        ("kind", {"kind": "scene"}),
        ("status", {"status": "maybe"}),
        ("G09-title", {"title": 5}),
        ("G10-data", {"data": ["not", "an", "object"]}),
    ],
)
def test_forged_node_shapes_are_skipped(hub: Hub, mutant: str, patch: dict[str, Any]) -> None:
    payload = {"node_id": "nd_forged", "kind": "op", "status": "pending", "title": "t", "data": {}}
    forged = hub.forge("graph.node_added", AGENT, {**payload, **patch})
    graph = WorkGraph(hub.log, hub.pid)
    assert forged in _graph_skips(hub), mutant
    assert graph.node("nd_forged") is None and graph.node("n" * 200) is None


@pytest.mark.parametrize(
    ("mutant", "patch"),
    [
        ("G12-type", {"type": "depends"}),
        ("unknown-dst", {"dst": "nd_missing"}),
        ("src-kind", {"type": "produces", "src": "ARTIFACT", "dst": "ARTIFACT"}),
        ("dst-kind", {"type": "produces", "src": "OP", "dst": "OP"}),
        ("G13-data", {"data": ["not", "an", "object"]}),
    ],
)
def test_forged_edge_shapes_are_skipped(hub: Hub, mutant: str, patch: dict[str, Any]) -> None:
    ids = {"OP": hub.op, "ARTIFACT": hub.artifact}
    payload = {"edge_id": "edg_forged", "type": "blocks", "src": hub.op, "dst": hub.artifact}
    payload["data"] = {}
    payload.update({k: ids.get(v, v) if isinstance(v, str) else v for k, v in patch.items()})
    forged = hub.forge("graph.edge_added", AGENT, payload)
    graph = WorkGraph(hub.log, hub.pid)
    assert forged in _graph_skips(hub), mutant
    assert all(edge.id != "edg_forged" for edge in graph.edges())


# -- R02, R04, R05, R06, R07: withdrawal rules ----------------------------------------


def test_R02_a_withdrawal_whose_payload_is_not_an_object_is_skipped(hub: Hub) -> None:
    forged = hub.forge("review.withdrawn", AGENT, ["not", "an", "object"])
    review, skipped = reviews.review_view(hub.log, hub.pid, hub.review)
    assert review is not None and review.withdrawal is None
    assert forged in {item.event_id for item in skipped}
    assert hub.client().get(f"/reviews/{hub.pid}/{hub.review}").status_code == 200


def test_R04_a_withdrawal_without_a_text_reason_is_skipped(hub: Hub) -> None:
    # Written like withdraw() would (cancel by the requester, then the event), but
    # with a reason that is not text.
    hub.forge("graph.node_status_changed", AGENT, {"node_id": hub.review, "to": "cancelled"})
    forged = hub.forge("review.withdrawn", AGENT, {"review_id": hub.review, "reason": 5})
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.status == "cancelled"
    assert review.withdrawal is None
    assert forged in _review_skips(hub)


def test_R05_a_withdrawal_shows_only_on_a_cancelled_review(hub: Hub) -> None:
    # A withdrawal event credited to the right change, but on a review the graph
    # still holds open (its status change was skipped as illegal), must not show.
    hub.forge("graph.node_status_changed", AGENT, {"node_id": hub.review, "to": "pending"})
    forged = hub.forge("review.withdrawn", AGENT, {"review_id": hub.review, "reason": "superseded"})
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.pending and review.withdrawal is None
    assert forged in _review_skips(hub)


def test_R06_a_decided_review_cannot_be_withdrawn(hub: Hub) -> None:
    reviews.decide(hub.log, hub.pid, hub.review, "approve", "ok", HUMAN)
    before = hub.log.latest_seq()
    with pytest.raises(reviews.ReviewNotOpen):
        reviews.withdraw(hub.log, hub.pid, hub.review, "too late", AGENT)
    assert hub.log.latest_seq() == before


def test_R07_withdraw_validates_its_payload_before_writing(hub: Hub) -> None:
    before = hub.log.latest_seq()
    with pytest.raises(InvalidInput):
        reviews.withdraw(hub.log, hub.pid, "r" * 500, "superseded", AGENT)
    assert hub.log.latest_seq() == before


def test_another_agent_cannot_withdraw_and_the_requester_can(hub: Hub) -> None:
    with pytest.raises(reviews.NotRequester):
        reviews.withdraw(hub.log, hub.pid, hub.review, "mine now", OTHER_AGENT)
    withdrawal = reviews.withdraw(hub.log, hub.pid, hub.review, "superseded", AGENT)
    assert withdrawal.by == AGENT


# -- R11: the add_note validator --------------------------------------------------------


def test_R11_a_note_whose_derived_timecode_overflows_is_refused_not_lost(hub: Hub) -> None:
    # 10**9 frames at 24 fps is over 11,000 hours: HHHHH:MM:SS:FF does not fit the
    # timecode format, so the fold would skip it. The writer must refuse it instead.
    before = hub.log.latest_seq()
    with pytest.raises(InvalidInput, match="timecode"):
        reviews.add_note(hub.log, hub.pid, hub.review, 10**9, "far away", AGENT)
    assert hub.log.latest_seq() == before


# -- C04: a non-object control payload --------------------------------------------------


def test_C04_a_control_event_that_is_not_an_object_is_skipped(hub: Hub) -> None:
    forged = hub.forge("control.stopped", AGENT, ["all", "halt"], project=False)
    state = control.status(hub.log)
    assert state.running is True
    assert forged in {item.event_id for item in state.skipped}
    control.assert_running(hub.log)
    assert hub.client().get("/control").status_code == 200


# -- L03: the lease TTL upper bound -----------------------------------------------------


def _lease_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "node_id": "node-1",
        "lease_id": "lease-x",
        "scope": "project",
        "key": ["demo-show"],
        "holder": {"kind": "agent", "id": "agent-1"},
        "acquired_at": ISO,
        "expires_at": LATER,
        "ttl": 3600.0,
        "at": ISO,
        "by": None,
    }
    payload.update(overrides)
    return payload


def test_L03_a_lease_ttl_past_the_upper_bound_is_skipped_and_refused(hub: Hub) -> None:
    forged = hub.forge("lease.granted", NODE, _lease_payload(ttl=2e9), project=False)
    board = lease_board(hub.log)
    assert board.leases == ()
    assert forged in {item.event_id for item in board.skipped}
    events: list[LeaseEvent] = []
    LeaseManager(clock=lambda: 10.0, sink=events.append).acquire(
        Holder.agent("agent-1"), Scope.PROJECT, ("demo-show",), ttl=2e9
    )
    with pytest.raises(InvalidInput, match="ttl"):
        lease_event_to_hub(events[0], "node-1", wall_offset=NOW.timestamp() - 10.0)


# -- A04, A05, A06: duplicate grant, double consume, revoke after use -------------------


def _grant_payload(token_id: str, **overrides: Any) -> dict[str, Any]:
    payload = {
        "token_id": token_id,
        "scope": "db.switch",
        "target": "Local Database",
        "reason": "sandbox",
        "ttl_seconds": 3600.0,
        "granted_at": ISO,
        "expires_at": LATER,
    }
    payload.update(overrides)
    return payload


def test_A04_a_duplicate_grant_does_not_resurrect_a_consumed_token(hub: Hub) -> None:
    token = approvals.grant(hub.log, "db.switch", "Local Database", 3600, "sandbox", HUMAN)
    approvals.consume(hub.log, token.token_id, "db.switch", "Local Database", NODE)
    forged = hub.forge("approval.granted", HUMAN, _grant_payload(token.token_id), project=False)
    view = approvals.approvals_view(hub.log)
    assert [t.state(NOW) for t in view.tokens] == ["consumed"]
    assert forged in {item.event_id for item in view.skipped}
    with pytest.raises(approvals.ApprovalAlreadyConsumed):
        approvals.consume(hub.log, token.token_id, "db.switch", "Local Database", OTHER_NODE)


def test_A05_a_second_consume_does_not_change_who_used_the_token(hub: Hub) -> None:
    token = approvals.grant(hub.log, "db.switch", "Local Database", 3600, "sandbox", HUMAN)
    approvals.consume(hub.log, token.token_id, "db.switch", "Local Database", NODE)
    forged = hub.forge(
        "approval.consumed",
        OTHER_NODE,
        {"token_id": token.token_id, "scope": "db.switch", "target": "Local Database", "at": LATER},
        project=False,
    )
    used = approvals.get_approval(hub.log, token.token_id)
    assert used is not None and used.consumed_by == NODE
    assert forged in {item.event_id for item in approvals.approvals_view(hub.log).skipped}


def test_A06_a_revoke_after_use_does_not_rewrite_the_token(hub: Hub) -> None:
    token = approvals.grant(hub.log, "db.switch", "Local Database", 3600, "sandbox", HUMAN)
    approvals.consume(hub.log, token.token_id, "db.switch", "Local Database", NODE)
    forged = hub.forge(
        "approval.revoked",
        HUMAN,
        {"token_id": token.token_id, "reason": "afterwards", "at": LATER},
        project=False,
    )
    used = approvals.get_approval(hub.log, token.token_id)
    assert used is not None and used.revoked_at is None and used.revoke_reason is None
    assert forged in {item.event_id for item in approvals.approvals_view(hub.log).skipped}


# -- S02, S03, S04, S05: skip cap, bool, empty text, timestamp cap ----------------------


def test_S02_skipped_summaries_list_at_most_fifty(hub: Hub) -> None:
    for _ in range(60):
        hub.forge("control.stopped", AGENT, {"scope": 7, "reason": "junk"}, project=False)
    summary = skipped_summary(control.status(hub.log).skipped)
    assert summary["count"] == 60
    assert len(summary["events"]) == 50


def test_S03_a_bool_is_not_a_frame(hub: Hub) -> None:
    forged = hub.forge(
        "review.note_added",
        AGENT,
        {"review_id": hub.review, "frame": True, "timecode": "00:00:00:01", "text": "x"},
    )
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.notes == ()
    assert forged in _review_skips(hub)


def test_S04_empty_text_is_not_text(hub: Hub) -> None:
    forged = hub.forge(
        "approval.granted", HUMAN, _grant_payload("apv_empty", target=""), project=False
    )
    assert approvals.get_approval(hub.log, "apv_empty") is None
    assert forged in {item.event_id for item in approvals.approvals_view(hub.log).skipped}


def test_S05_an_over_long_timestamp_is_not_a_timestamp(hub: Hub) -> None:
    assert len(LONG_ISO) > 64 and datetime.fromisoformat(LONG_ISO)  # parses, but too long
    forged = hub.forge(
        "approval.granted", HUMAN, _grant_payload("apv_long", granted_at=LONG_ISO), project=False
    )
    assert approvals.get_approval(hub.log, "apv_long") is None
    assert forged in {item.event_id for item in approvals.approvals_view(hub.log).skipped}


# -- W11, W07, W13, W16: console ----------------------------------------------------------


def test_W11_the_skipped_notice_lists_fifty_and_counts_the_rest(hub: Hub) -> None:
    for _ in range(60):
        hub.forge("control.stopped", AGENT, {"scope": 7, "reason": "junk"}, project=False)
    page = hub.client().get("/control").text
    assert "60 malformed events ignored" in page
    assert len(re.findall(r"<li><code>[0-9A-Z]{26}</code>", page)) == 50
    assert "and 10 more" in page


def test_W07_a_chunked_body_over_the_cap_is_refused_while_streaming(hub: Hub) -> None:
    def chunks() -> Any:
        for _ in range(70):
            yield b"reason=" + b"x" * 1024 + b"&"

    response = hub.client().post(
        "/control/stop",
        content=chunks(),
        headers={
            "Origin": BASE,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "body_too_large"
    assert control.status(hub.log).running is True


def test_W13_pages_only_post_forms_to_the_console_itself(hub: Hub) -> None:
    for path in ("/", "/control", "/approvals", "/leases"):
        csp = hub.client().get(path).headers["content-security-policy"]
        assert "form-action 'self'" in csp, path


def test_W16_resuming_a_running_scope_is_a_409_not_stopped(hub: Hub) -> None:
    client = hub.client()
    page = client.get("/control").text
    start = page.index('action="/control/stop"')
    section = page[start : page.index("</form>", start)]
    csrf = re.search(r'name="csrf" value="([^"]*)"', section)
    assert csrf is not None
    response = client.post(
        "/control/resume",
        data={"csrf": csrf.group(1), "idem": "f" * 32, "scope": "all", "reason": "x"},
        headers={"Origin": BASE, "Accept": "application/json"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_stopped"
