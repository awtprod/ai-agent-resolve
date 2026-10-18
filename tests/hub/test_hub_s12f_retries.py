"""S12f: retries against malformed stored events, capped skip reasons, deep payloads.

Every scenario forges the bad shape straight into the log (as a buggy or hostile
writer could) and checks the hub answers with a clean typed error or a working read.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from starlette.testclient import TestClient

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import MAX_SKIP_REASON
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog, IdempotencyConflict
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import lease_board
from aar.hub.mcp_server import ELIDED, build_server
from aar.hub.registry import ProjectSource, Registry

HUMAN = Actor("human", "operator-1")
OTHER_HUMAN = Actor("human", "operator-2")
AGENT = Actor("agent", "agent-1")
NODE = Actor("node", "node-1")
BASE = "http://127.0.0.1:8788"
SECRET = b"0123456789abcdef0123456789abcdef"


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


def _project(log: EventLog) -> str:
    return Registry(log).create_project("demo-show", HUMAN, project_id="prj_demo").id


def _nest(depth: int) -> Any:
    value: Any = "leaf"
    for _ in range(depth):
        value = [value]
    return value


# -- item 1: a stored payload that is not an object ---------------------------------


def test_every_retry_against_a_non_object_stored_payload_is_a_conflict(log: EventLog) -> None:
    pid = _project(log)
    op = WorkGraph(log, pid).add_node("op", AGENT, title="build").id
    for key, event_type in (
        ("k-stop", "control.stopped"),
        ("k-resume", "control.resumed"),
        ("k-grant", "approval.granted"),
        ("k-node", "graph.node_added"),
        ("k-project", "project.created"),
        ("k-note", "review.note_added"),
    ):
        log.append(event_type, HUMAN, ["not", "an", "object"], idempotency_key=key)
    store = ArtifactStore(log_path(log).parent / "artifacts", log)
    render = store.put(b"render", actor=AGENT, mime="video/mp4", project_id=pid)
    review = reviews.request_review(log, store, pid, op, [render], AGENT, title="cut", fps=24)
    before = log.latest_seq()
    attempts = [
        lambda: control.stop(log, "all", "halt", HUMAN, idempotency_key="k-stop"),
        lambda: control.resume(log, "all", "ok", HUMAN, idempotency_key="k-resume"),
        lambda: approvals.grant(log, "db.switch", "x", 60, "r", HUMAN, idempotency_key="k-grant"),
        lambda: WorkGraph(log, pid).add_node("op", AGENT, idempotency_key="k-node"),
        lambda: Registry(log).create_project("x", HUMAN, idempotency_key="k-project"),
        lambda: reviews.add_note(log, pid, review.id, 1, "x", AGENT, idempotency_key="k-note"),
    ]
    for attempt in attempts:
        with pytest.raises(IdempotencyConflict):
            attempt()
    assert log.latest_seq() == before


def log_path(log: EventLog) -> Path:
    return Path(log._path)


def test_resume_retry_survives_a_malformed_stop_in_the_log(log: EventLog) -> None:
    # control.resume's retry path used to call .get() on every earlier stop's payload.
    control.stop(log, "all", "halt", HUMAN)
    log.append("control.stopped", AGENT, ["junk"])
    lifted = control.resume(log, "all", "checked", HUMAN, idempotency_key="r1")
    again = control.resume(log, "all", "checked", HUMAN, idempotency_key="r1")
    assert again == lifted
    assert again.reason == "halt"
    assert len(log.read(types=["control.resumed"])) == 1


def test_console_double_submitted_resume_with_junk_in_the_log_is_not_a_500(
    log: EventLog, tmp_path: Path
) -> None:
    _project(log)
    store = ArtifactStore(tmp_path / "artifacts", log)
    app = build_app(log, store, ConsoleConfig(operator="operator-1", csrf_secret=SECRET))
    client = TestClient(app, base_url=BASE, client=("127.0.0.1", 50000), follow_redirects=False)
    control.stop(log, "all", "halt", HUMAN)
    log.append("control.stopped", AGENT, ["junk"])
    page = client.get("/control").text
    start = page.index('action="/control/resume"')
    section = page[start : page.index("</form>", start)]
    fields = {
        name: re.search(rf'name="{name}" value="([^"]*)"', section).group(1)  # type: ignore[union-attr]
        for name in ("csrf", "idem")
    }
    data = {**fields, "scope": "all", "reason": "checked"}
    first = client.post("/control/resume", data=data, headers={"Origin": BASE})
    second = client.post("/control/resume", data=data, headers={"Origin": BASE})
    assert (first.status_code, second.status_code) == (303, 303)
    assert control.status(log).running is True


def test_a_retry_whose_stored_event_the_projection_skipped_is_a_conflict(
    log: EventLog,
) -> None:
    pid = _project(log)
    # Same request fields under the key, but missing the id the fold needs.
    log.append(
        "graph.node_added",
        AGENT,
        {"kind": "op", "status": "pending", "title": None, "data": {}},
        project_id=pid,
        idempotency_key="k-node",
    )
    log.append(
        "graph.edge_added",
        AGENT,
        {"type": "blocks", "src": "a", "dst": "b", "data": {}},
        project_id=pid,
        idempotency_key="k-edge",
    )
    log.append(
        "project.created",
        HUMAN,
        {"name": "x", "sources": [], "aliases": []},
        project_id="prj_x",
        idempotency_key="k-project",
    )
    log.append(
        "approval.granted",
        HUMAN,
        {"scope": "db.switch", "target": "x", "ttl_seconds": 60.0, "reason": "r"},
        idempotency_key="k-grant",
    )
    graph = WorkGraph(log, pid)
    a = graph.add_node("op", AGENT).id
    b = graph.add_node("op", AGENT).id
    with pytest.raises(IdempotencyConflict):
        graph.add_node("op", AGENT, idempotency_key="k-node")
    log.append(
        "graph.edge_added",
        AGENT,
        {"type": "blocks", "src": a, "dst": b, "data": {}},
        project_id=pid,
        idempotency_key="k-edge2",
    )
    with pytest.raises(IdempotencyConflict):
        graph.add_edge("blocks", a, b, AGENT, idempotency_key="k-edge2")
    with pytest.raises(IdempotencyConflict):
        Registry(log).create_project("x", HUMAN, idempotency_key="k-project")
    with pytest.raises(IdempotencyConflict):
        approvals.grant(log, "db.switch", "x", 60, "r", HUMAN, idempotency_key="k-grant")


# -- item 2: skip reasons are capped ------------------------------------------------


def test_skip_reasons_quoting_agent_text_are_capped(log: EventLog) -> None:
    pid = _project(log)
    graph = WorkGraph(log, pid)
    node = graph.add_node("op", AGENT).id
    long_id = "a" * 5000
    log.append(
        "graph.node_status_changed", AGENT, {"node_id": node, "to": "x" * 5000}, project_id=pid
    )
    log.append(
        "approval.granted",
        Actor("agent", long_id),
        {
            "token_id": "apv_x",
            "scope": "db.switch",
            "target": "t",
            "reason": "r",
            "granted_at": "2026-10-17T12:00:00+00:00",
            "expires_at": "2026-10-17T13:00:00+00:00",
        },
    )
    log.append("control.resumed", Actor("agent", long_id), {"scope": "all", "reason": "r"})
    log.append(
        "lease.released",
        Actor("agent", long_id),
        {
            "node_id": "node-1",
            "lease_id": "l1",
            "scope": "project",
            "key": ["demo-show"],
            "holder": {"kind": "agent", "id": "agent-1"},
            "acquired_at": "2026-10-17T12:00:00+00:00",
            "expires_at": None,
            "ttl": None,
            "at": "2026-10-17T12:00:00+00:00",
            "by": None,
        },
    )
    reasons = [s.reason for s in WorkGraph(log, pid).skipped()]
    reasons += [s.reason for s in approvals.approvals_view(log).skipped]
    reasons += [s.reason for s in control.status(log).skipped]
    reasons += [s.reason for s in lease_board(log).skipped]
    assert len(reasons) == 4
    for reason in reasons:
        assert len(reason) == MAX_SKIP_REASON and reason.endswith("…"), reason[:60]


# -- item 5: approvals retried by a different human ---------------------------------


def test_grant_and_revoke_retried_by_a_different_human_conflict(log: EventLog) -> None:
    token = approvals.grant(
        log, "db.switch", "Local Database", 60, "sandbox", HUMAN, idempotency_key="g1"
    )
    before = log.latest_seq()
    with pytest.raises(IdempotencyConflict):
        approvals.grant(
            log, "db.switch", "Local Database", 60, "sandbox", OTHER_HUMAN, idempotency_key="g1"
        )
    approvals.revoke(log, token.token_id, "not needed", HUMAN, idempotency_key="v1")
    before = log.latest_seq()
    with pytest.raises(IdempotencyConflict):
        approvals.revoke(log, token.token_id, "not needed", OTHER_HUMAN, idempotency_key="v1")
    assert log.latest_seq() == before
    # The same human retrying is still a no-op.
    assert (
        approvals.revoke(log, token.token_id, "not needed", HUMAN, idempotency_key="v1").revoked_by
        == HUMAN
    )
    assert log.latest_seq() == before


# -- item 7: deeply nested payloads -------------------------------------------------


def _call(server: Any, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    async def body() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, args)
            assert result.is_error is False, (tool, result.content)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(body())


def test_mcp_tools_answer_with_payloads_nested_past_the_serializer_limit(
    log: EventLog, tmp_path: Path
) -> None:
    pid = _project(log)
    deep = _nest(400)
    WorkGraph(log, pid).add_node("op", AGENT, title="deep", data={"x": deep})
    Registry(log).add_source(pid, ProjectSource("drp_file", {"path": "demo.drp", "x": deep}), HUMAN)
    store = ArtifactStore(tmp_path / "artifacts", log)
    artifact = store.put(b"x", actor=AGENT, meta={"x": deep})
    server = build_server(log, Registry(log), store)
    graph = _call(server, "work_graph", {"project_id": pid})
    tail = _call(server, "events_tail", {"limit": 1000})
    projects = _call(server, "projects_list", {})
    project = _call(server, "project_get", {"project_id": pid})
    meta = _call(server, "artifact_meta", {"artifact_id": artifact})
    for result in (graph, tail, projects, project, meta):
        assert ELIDED in json.dumps(result)
    node = next(n for n in graph["nodes"] if n["title"] == "deep")
    assert ELIDED in json.dumps(node["payload"])
    # Shallow data is untouched.
    shallow = _call(server, "events_tail", {"limit": 3})["events"][0]
    assert ELIDED not in json.dumps(shallow)


def test_bounded_marks_exactly_the_levels_past_the_cap() -> None:
    from aar.hub.mcp_server import MAX_RESULT_DEPTH, _bounded

    value = _nest(MAX_RESULT_DEPTH + 5)
    out = _bounded({"x": value})
    depth = 0
    cursor: Any = out
    while isinstance(cursor, dict | list) and cursor:
        cursor = cursor["x"] if isinstance(cursor, dict) else cursor[0]
        depth += 1
    assert cursor == ELIDED and depth == MAX_RESULT_DEPTH
    assert _bounded({"x": _nest(10)}) == {"x": _nest(10)}
