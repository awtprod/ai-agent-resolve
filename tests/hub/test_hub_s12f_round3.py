"""S12f round 3: torn reads, unhashable forged fields, attribution and retry gaps."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import SkippedEvent
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import lease_board
from aar.hub.registry import ProjectSource, Registry

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
NODE = Actor("node", "node-1")
NOW = datetime.now(UTC).replace(microsecond=0)
ISO = NOW.isoformat()
LATER = (NOW + timedelta(hours=1)).isoformat()


class Hub:
    def __init__(self, root: Path) -> None:
        self.db = root / "hub.db"
        self.log = EventLog(self.db)
        self.store = ArtifactStore(root / "artifacts", self.log)
        self.pid = Registry(self.log).create_project("demo-show", HUMAN).id
        self.op = WorkGraph(self.log, self.pid).add_node("op", AGENT, title="build").id
        self.render = self.store.put(b"render", actor=AGENT, mime="video/mp4", project_id=self.pid)
        self.review = self.request()

    def request(self) -> str:
        return reviews.request_review(
            self.log, self.store, self.pid, self.op, [self.render], AGENT, title="cut", fps=24
        ).id

    def skipped(self, review_id: str) -> list[SkippedEvent]:
        return list(reviews.review_view(self.log, self.pid, review_id)[1])


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


# -- item 1: one snapshot -----------------------------------------------------------------


@pytest.mark.parametrize("act", ["decide", "withdraw"])
def test_a_commit_between_reads_never_makes_a_real_act_look_forged(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, act: str
) -> None:
    writer = EventLog(hub.db)  # a second connection, as another process would have
    real_read = hub.log.read
    fired: list[bool] = []

    def read_then_commit(*args: Any, **kwargs: Any) -> Any:
        result = real_read(*args, **kwargs)
        if not fired:  # commit right after the first read of this view
            fired.append(True)
            if act == "decide":
                reviews.decide(writer, hub.pid, hub.review, "approve", "ship it", HUMAN)
            else:
                reviews.withdraw(writer, hub.pid, hub.review, "superseded", AGENT)
        return result

    monkeypatch.setattr(hub.log, "read", read_then_commit)
    review, skipped = reviews.review_view(hub.log, hub.pid, hub.review)
    assert fired
    assert [s.reason for s in skipped] == []
    assert review is not None
    # Whole before or whole after the commit, never half.
    assert (review.status, review.decision, review.withdrawal) == ("active", None, None)
    monkeypatch.undo()
    after = reviews.get_review(hub.log, hub.pid, hub.review)
    assert after is not None and not after.pending
    assert (after.decision or after.withdrawal) is not None


# -- item 2: unhashable forged fields ----------------------------------------------------

LEASE = {
    "node_id": "node-1",
    "lease_id": "lease-x",
    "scope": "project",
    "key": ["demo-show"],
    "holder": {"kind": "agent", "id": "agent-1"},
    "acquired_at": ISO,
    "expires_at": LATER,
    "ttl": 3600.0,
    "at": ISO,
    "by": {"kind": "human", "id": "operator-1"},
}
GRANT = {
    "token_id": "apv_forged",
    "scope": "db.switch",
    "target": "t",
    "reason": "r",
    "granted_at": ISO,
    "expires_at": LATER,
}


def _set(payload: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in payload.items()}
    if "." in path:
        outer, inner = path.split(".")
        out[outer] = {**out[outer], inner: value}
    else:
        out[path] = value
    return out


def _cases() -> list[tuple[str, str, Actor, dict[str, Any] | None]]:
    """(event type, field path, actor, base payload or None for graph/registry ones)."""
    cases: list[tuple[str, str, Actor, dict[str, Any] | None]] = [
        ("graph.node_added", "kind", AGENT, None),
        ("graph.node_added", "status", AGENT, None),
        ("graph.node_status_changed", "to", AGENT, None),
        ("graph.edge_added", "type", AGENT, None),
        ("approval.granted", "scope", HUMAN, GRANT),
        ("project.created", "sources.kind", HUMAN, None),
        ("project.source_added", "source.kind", HUMAN, None),
    ]
    for kind in ("granted", "renewed", "released", "expired", "preempted"):
        for path in ("scope", "holder", "holder.kind", "by", "by.kind"):
            cases.append((f"lease.{kind}", path, NODE, LEASE))
    return cases


def _payload(hub: Hub, event_type: str, path: str, base: Any, bad: Any) -> dict[str, Any]:
    if event_type == "graph.node_added":
        payload = {"node_id": "nd_bad", "kind": "op", "status": "pending", "title": "t", "data": {}}
        return _set(payload, path, bad)
    if event_type == "graph.node_status_changed":
        return {"node_id": hub.op, "to": bad}
    if event_type == "graph.edge_added":
        return {"edge_id": "e_bad", "type": bad, "src": hub.op, "dst": hub.op, "data": {}}
    if event_type == "project.created":
        source = {"kind": bad, "locator": {"path": "x"}, "host": None}
        return {"project_id": "prj_bad", "name": "x", "sources": [source], "aliases": []}
    if event_type == "project.source_added":
        return {"project_id": hub.pid, "source": {"kind": bad, "locator": {}, "host": None}}
    return _set(base, path, bad)


@pytest.mark.parametrize("bad", [[], {}], ids=["list", "dict"])
@pytest.mark.parametrize(
    ("event_type", "path", "actor", "base"), _cases(), ids=[f"{c[0]}:{c[1]}" for c in _cases()]
)
def test_an_unhashable_forged_field_is_skipped_and_nothing_breaks(
    hub: Hub, event_type: str, path: str, actor: Actor, base: Any, bad: Any
) -> None:
    project_id = hub.pid if event_type.startswith("graph.") else None
    if event_type.startswith("project."):
        project_id = "prj_bad" if event_type == "project.created" else hub.pid
    forged = hub.log.append(
        event_type, actor, _payload(hub, event_type, path, base, bad), project_id=project_id
    ).id
    # Every projection still reads, and lists the forged event where it applies.
    graph = WorkGraph(hub.log, hub.pid)
    registry = Registry(hub.log)
    view = approvals.approvals_view(hub.log)
    board = lease_board(hub.log)
    scan = reviews.scan_reviews(hub.log)
    control.status(hub.log)
    skipped = {
        s.event_id for s in (*graph.skipped(), *registry.skipped(), *view.skipped, *board.skipped)
    }
    assert forged in skipped
    assert [r.id for r in scan.reviews] == [hub.review]
    # And every writer still writes.
    graph.add_node("op", AGENT, title="after")
    registry.create_project("after", HUMAN)
    approvals.grant(hub.log, "db.switch", "Local Database", 60, "after", HUMAN)
    reviews.add_note(hub.log, hub.pid, hub.review, 1, "after", AGENT)


def test_writers_refuse_unhashable_arguments_cleanly(hub: Hub) -> None:
    graph = WorkGraph(hub.log, hub.pid)
    with pytest.raises(ValueError):
        graph.add_node([], AGENT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        graph.add_node("op", AGENT, status={})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        graph.set_status(hub.op, [], AGENT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        graph.add_edge({}, hub.op, hub.op, AGENT)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ProjectSource([], {"path": "x"})  # type: ignore[arg-type]


# -- item 3: test gaps ---------------------------------------------------------------------


def test_N02_N04_a_decision_not_adjacent_to_its_status_change_is_skipped(hub: Hub) -> None:
    graph = WorkGraph(hub.log, hub.pid)
    with hub.log.transaction() as txn:
        graph.set_status_in(txn, hub.review, "done", HUMAN)  # no decision written
    WorkGraph(hub.log, hub.pid).add_node("op", AGENT, title="unrelated")
    forged = hub.log.append(
        "review.decided",
        HUMAN,
        {"review_id": hub.review, "decision": "approve", "reason": "later"},
        project_id=hub.pid,
    ).id
    review = reviews.get_review(hub.log, hub.pid, hub.review)
    assert review is not None and review.status == "done" and review.decision is None
    assert forged in {s.event_id for s in hub.skipped(hub.review)}


def test_G13_the_fold_refuses_an_illegal_transition(hub: Hub) -> None:
    graph = WorkGraph(hub.log, hub.pid)
    graph.set_status(hub.op, "active", AGENT)
    graph.set_status(hub.op, "done", AGENT)
    forged = hub.log.append(
        "graph.node_status_changed", AGENT, {"node_id": hub.op, "to": "pending"}, project_id=hub.pid
    ).id
    fresh = WorkGraph(hub.log, hub.pid)
    node = fresh.node(hub.op)
    assert node is not None and node.status == "done"
    assert forged in {s.event_id for s in fresh.skipped()}


def test_R07_no_notes_on_a_withdrawn_review(hub: Hub) -> None:
    reviews.withdraw(hub.log, hub.pid, hub.review, "superseded", AGENT)
    before = hub.log.latest_seq()
    with pytest.raises(reviews.ReviewNotOpen):
        reviews.add_note(hub.log, hub.pid, hub.review, 12, "late", AGENT)
    assert hub.log.latest_seq() == before


@pytest.mark.parametrize("start", ["pending", "blocked"])
def test_N08_deciding_a_pending_or_blocked_review_credits_the_decision(
    hub: Hub, start: str
) -> None:
    graph = WorkGraph(hub.log, hub.pid)
    node = graph.add_node("review", AGENT, title="made by hand", status=start)
    graph.add_edge("reviews", node.id, hub.op, AGENT)
    decision = reviews.decide(hub.log, hub.pid, node.id, "reject", "redo", HUMAN)
    review = reviews.get_review(hub.log, hub.pid, node.id)
    assert review is not None and review.decision == decision and review.status == "failed"
    assert hub.skipped(node.id) == []


def test_N33_a_grant_retry_whose_stored_token_id_is_a_list_conflicts(hub: Hub) -> None:
    hub.log.append(
        "approval.granted",
        HUMAN,
        {"scope": "db.switch", "target": "x", "ttl_seconds": 60.0, "reason": "r", "token_id": []},
        idempotency_key="k-grant",
    )
    with pytest.raises(IdempotencyConflict):
        approvals.grant(hub.log, "db.switch", "x", 60, "r", HUMAN, idempotency_key="k-grant")


def test_N35_N36_skip_reasons_are_cut_at_exactly_200_characters() -> None:
    kept = SkippedEvent("e", 1, "t", "x" * 200)
    cut = SkippedEvent("e", 1, "t", "x" * 201)
    assert kept.reason == "x" * 200
    assert len(cut.reason) == 200 and cut.reason == "x" * 199 + "…"


# -- item 4: public timecode matcher ---------------------------------------------------


def test_the_timecode_matcher_is_public_and_the_private_alias_still_imports() -> None:
    from aar.hub.reviews import _TIMECODE_RE, TIMECODE_RE, is_timecode

    assert _TIMECODE_RE is TIMECODE_RE
    assert is_timecode("01:00:12:05") and is_timecode("00:00:00;29")
    for bad in ("1:00:12:05", "01:60:00:00", "01:00:12", 5, None, ["01:00:00:00"]):
        assert not is_timecode(bad)


# -- item 5: name and title caps at the writers ----------------------------------------


def test_writers_refuse_huge_names_and_titles(hub: Hub) -> None:
    registry = Registry(hub.log)
    before = hub.log.latest_seq()
    with pytest.raises(ValueError, match="over 500"):
        registry.create_project("n" * 501, HUMAN)
    with pytest.raises(ValueError, match="over 500"):
        registry.rename_project(hub.pid, "n" * 3_000_000, HUMAN)
    with pytest.raises(ValueError, match="over 500"):
        registry.add_alias(hub.pid, "a" * 501, HUMAN)
    with pytest.raises(ValueError, match="over 1000"):
        WorkGraph(hub.log, hub.pid).add_node("op", AGENT, title="t" * 1001)
    assert hub.log.latest_seq() == before
    # At the limit is fine.
    assert registry.rename_project(hub.pid, "n" * 500, HUMAN).name == "n" * 500
    assert WorkGraph(hub.log, hub.pid).add_node("op", AGENT, title="t" * 1000).title
