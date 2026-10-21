"""Malformed and forged events appended straight to the log must never break a read.

The log is append-only, so a bad event can never be removed. Every projection must
skip it, list it, and keep serving: every console page returns 200 and shows the
skipped ids, every MCP tool answers, and STOP and resume keep working.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from starlette.testclient import TestClient

from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import lease_board
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry

AGENT = Actor("agent", "agent-1")
HUMAN = Actor("human", "operator-1")
NODE = Actor("node", "node-1")
BASE = "http://127.0.0.1:8788"
ORIGIN = {"Origin": BASE}
SECRET = b"0123456789abcdef0123456789abcdef"
XSS = "<script>alert(1)</script>"
ISO = datetime.now(UTC).replace(microsecond=0).isoformat()
SEED = Path(__file__).resolve().parents[2] / "scripts" / "seed_console_demo.py"


def _seed(root: Path) -> None:
    spec = importlib.util.spec_from_file_location("seed_console_demo", SEED)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.seed(root, b"demo render bytes")


class Forged:
    """A seeded demo hub plus forged events, with the ids the projections must skip."""

    def __init__(self, root: Path) -> None:
        _seed(root)
        self.log = EventLog(root / "hub.db")
        self.store = ArtifactStore(root / "artifacts", self.log)
        self.store_root = root / "artifacts"
        registry = Registry(self.log)
        self.show = next(p.id for p in registry.list() if p.name == "demo-show")
        pending = reviews.list_reviews(self.log, self.show, pending_only=True)
        self.review = pending[0].id
        self.subject = pending[0].subject or ""
        active = [t for t in approvals.list_approvals(self.log) if t.revoked_at is None]
        self.token = next(t.token_id for t in active if t.consumed_at is None)
        self.bad: dict[str, str] = {}  # event id -> label, every one must be skipped
        self.junk_render = self._junk_artifact()
        self._forge()

    def add(self, label: str, event_type: str, actor: Actor, payload: Any, **kw: Any) -> str:
        event = self.log.append(event_type, actor, payload, **kw)
        self.bad[event.id] = label
        return event.id

    def _junk_artifact(self) -> str:
        # Claim the metadata key of a render before it is stored, with a junk payload.
        data = b"render whose metadata record is junk"
        artifact_id = hashlib.sha256(data).hexdigest()
        self.log.append(
            "artifact.added", AGENT, ["junk"], idempotency_key=f"artifact:{artifact_id}"
        )
        assert self.store.put(data, actor=AGENT, mime="video/mp4") == artifact_id
        reviews.request_review(
            self.log, self.store, self.show, self.subject, [artifact_id], AGENT, title="junk meta"
        )
        return artifact_id

    def _forge(self) -> None:
        show, review = self.show, self.review
        lease = {
            "node_id": "node-1",
            "lease_id": "lease-good",
            "scope": "project",
            "key": ["demo-show"],
            "holder": {"kind": "agent", "id": "agent-1"},
            "acquired_at": ISO,
            "expires_at": None,
            "ttl": None,
            "at": ISO,
            "by": None,
        }
        # The three repros from the review.
        self.add(
            "status change for an unknown node",
            "graph.node_status_changed",
            AGENT,
            {"node_id": "nd_missing", "to": "done"},
            project_id=show,
        )
        self.add(
            "lease acquired_at is an int",
            "lease.granted",
            NODE,
            {**lease, "lease_id": "lease-int", "acquired_at": 5},
        )
        # Well-formed: it must show up, and sorting it beside the int one must not crash.
        self.log.append("lease.granted", NODE, lease)
        # A structurally valid review node whose data is junk: not skipped, but every
        # page showing it must still render (huge int fps used to overflow float()).
        self.log.append(
            "graph.node_added",
            AGENT,
            {
                "node_id": "nd_junk_data",
                "kind": "review",
                "status": "active",
                "title": "t" * 5000,
                "data": {"fps": 10**400, "renders": "nope", "subject": 7, "timeline": 5},
            },
            project_id=show,
        )
        self.add(
            "note frame 10**400",
            "review.note_added",
            AGENT,
            {"review_id": review, "frame": 10**400, "timecode": "00:00:01:00", "text": "huge"},
            project_id=show,
        )
        # Wrong shapes, missing fields, duplicates and forged privilege.
        self.add("graph payload is a list", "graph.node_added", AGENT, ["a", 1], project_id=show)
        self.add(
            "edge without dst",
            "graph.edge_added",
            AGENT,
            {"edge_id": "edg_x", "type": "reviews", "src": review},
            project_id=show,
        )
        self.add(
            "re-add an existing review as done",
            "graph.node_added",
            AGENT,
            {"node_id": review, "kind": "review", "status": "done", "title": "x", "data": {}},
            project_id=show,
        )
        self.add(
            "agent closes a review by raw status change",
            "graph.node_status_changed",
            AGENT,
            {"node_id": review, "to": "done"},
            project_id=show,
        )
        self.add(
            "decision 'maybe'",
            "review.decided",
            HUMAN,
            {"review_id": review, "decision": "maybe", "reason": "x"},
            project_id=show,
        )
        self.add(
            "agent decision",
            "review.decided",
            AGENT,
            {"review_id": review, "decision": "approve", "reason": "trust me"},
            project_id=show,
        )
        self.add(
            "NaN smuggled as a string frame",
            "review.note_added",
            AGENT,
            {"review_id": review, "frame": "NaN", "timecode": 5, "text": "x"},
            project_id=show,
        )
        self.add(
            "note without text",
            "review.note_added",
            AGENT,
            {"review_id": review, "frame": 1, "timecode": "00:00:00:01"},
            project_id=show,
        )
        self.add(
            "rename of an unknown project",
            "project.renamed",
            HUMAN,
            {"project_id": "prj_missing", "name": "x"},
            project_id="prj_missing",
        )
        self.add(
            "project with junk sources",
            "project.created",
            HUMAN,
            {"project_id": "prj_junk", "name": "junk", "sources": "junk"},
            project_id="prj_junk",
        )
        self.add(
            "grant expiring at 'NaN'",
            "approval.granted",
            HUMAN,
            {
                "token_id": "apv_nan",
                "scope": "db.switch",
                "target": "Local Database",
                "reason": "x",
                "granted_at": ISO,
                "expires_at": "NaN",
            },
        )
        self.add(
            "agent revokes a token",
            "approval.revoked",
            AGENT,
            {"token_id": self.token, "reason": "agent revoke", "at": ISO},
        )
        self.add("consume without at", "approval.consumed", AGENT, {"token_id": self.token})
        self.add("stop with int scope", "control.stopped", AGENT, {"scope": 123, "reason": "x"})
        self.add("stop with int reason", "control.stopped", AGENT, {"scope": "all", "reason": 5})
        self.add(
            "timeline key with three parts",
            "lease.granted",
            NODE,
            {**lease, "lease_id": "lease-arity", "scope": "timeline", "key": ["a", "b", "c"]},
        )
        self.add(
            "ttl 'inf' as a string",
            "lease.granted",
            NODE,
            {**lease, "lease_id": "lease-ttl", "ttl": "inf"},
        )
        self.add("release without a lease id", "lease.released", NODE, {"node_id": "node-1"})
        self.add(
            "lease written by an agent for node-1",
            "lease.granted",
            AGENT,
            {**lease, "lease_id": "lease-forged"},
        )
        self.add(
            "presence active='yes'",
            "presence.changed",
            NODE,
            {"node_id": "node-1", "active": "yes", "at": ISO},
        )
        self.add("presence payload is a list", "presence.changed", NODE, ["junk"])
        self.add(
            "script tag in a skip reason",
            "graph.node_status_changed",
            AGENT,
            {"node_id": review, "to": XSS},
            project_id=show,
        )

    def client(self) -> TestClient:
        app = build_app(
            self.log, self.store, ConsoleConfig(operator="operator-1", csrf_secret=SECRET)
        )
        return TestClient(app, base_url=BASE, client=("127.0.0.1", 50000), follow_redirects=False)

    def server(self) -> Any:
        reader = EventLog(self.log_path, read_only=True)
        return build_server(reader, Registry(reader), ArtifactStore(self.store_root, reader))

    @property
    def log_path(self) -> Path:
        return self.store_root.parent / "hub.db"


@pytest.fixture
def forged(tmp_path: Path) -> Forged:
    return Forged(tmp_path)


def _call(server: Any, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    async def body() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, args or {})
            assert result.is_error is False, (tool, result.content)
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(body())


def _form(html: str, action: str) -> dict[str, str]:
    start = html.index(f'action="{action}"')
    section = html[start : html.index("</form>", start)]
    found = {
        name: re.search(rf'name="{name}" value="([^"]*)"', section) for name in ("csrf", "idem")
    }
    assert all(found.values()), action
    return {name: match.group(1) for name, match in found.items() if match}


def test_every_page_renders_and_lists_the_skipped_events(forged: Forged) -> None:
    client = forged.client()
    scan = reviews.scan_reviews(forged.log)
    pages = ["/", "/leases", "/approvals", "/control"]
    pages += [f"/reviews/{r.project_id}/{r.id}" for r in scan.reviews]
    shown: set[str] = set()
    for path in pages:
        response = client.get(path)
        assert response.status_code == 200, (path, response.text[:300])
        shown.update(re.findall(r"<li><code>([0-9A-Z]{26})</code>", response.text))
        assert XSS not in response.text, path  # skip reasons quote agent text: escaped
        assert "malformed event" in response.text, path
    missing = {label for event_id, label in forged.bad.items() if event_id not in shown}
    assert missing == set()


def test_every_mcp_tool_answers_and_reports_what_it_skipped(forged: Forged) -> None:
    server = forged.server()
    calls: list[tuple[str, dict[str, Any]]] = [
        ("projects_list", {}),
        ("project_get", {"project_id": forged.show}),
        ("events_tail", {"limit": 1000}),
        ("work_graph", {"project_id": forged.show}),
        ("artifact_meta", {"artifact_id": forged.junk_render}),
        ("reviews_pending", {}),
        ("review_get", {"project_id": forged.show, "review_id": forged.review}),
        ("control_status", {}),
        ("lease_board", {}),
        ("approvals_list", {"include_inactive": True}),
    ]
    reported: set[str] = set()
    for tool, args in calls:
        result = _call(server, tool, args)
        if "skipped" in result:
            assert result["skipped"]["count"] == len(result["skipped"]["events"])
            reported.update(item["event_id"] for item in result["skipped"]["events"])
    missing = {label for event_id, label in forged.bad.items() if event_id not in reported}
    assert missing == set()
    meta = _call(server, "artifact_meta", {"artifact_id": forged.junk_render})
    assert meta["found"] is True and meta["meta"]["mime"] is None


def test_forged_events_change_nothing_they_claim(forged: Forged) -> None:
    review = reviews.get_review(forged.log, forged.show, forged.review)
    assert review is not None
    assert review.status == "active" and review.pending and review.decision is None
    assert all(note.frame < 10**9 for note in review.notes)
    token = approvals.get_approval(forged.log, forged.token)
    assert token is not None and token.state(datetime.now(UTC)) == "active"
    assert approvals.get_approval(forged.log, "apv_nan") is None
    assert control.status(forged.log).running is True
    assert Registry(forged.log).get("prj_junk") is None
    board = lease_board(forged.log)
    held = {lease.lease_id for lease in board.leases}
    assert "lease-good" in held
    assert not held & {"lease-int", "lease-arity", "lease-ttl", "lease-forged"}
    assert [p.human_id for p in board.presence] == ["operator-1"]  # the junk did not erase it
    graph = WorkGraph(forged.log, forged.show)
    assert {s.event_id for s in graph.skipped()} <= set(forged.bad)
    # The real human decision still works on the review the forgeries targeted.
    reviews.decide(forged.log, forged.show, forged.review, "approve", "checked", HUMAN)


def test_stop_and_resume_still_work_through_the_console(forged: Forged) -> None:
    client = forged.client()
    server = forged.server()
    stop_form = _form(client.get("/control").text, "/control/stop")
    stopped = client.post(
        "/control/stop", data={**stop_form, "scope": "all", "reason": "halt"}, headers=ORIGIN
    )
    assert stopped.status_code == 303
    assert _call(server, "control_status")["running"] is False
    # A resume an agent appends straight to the log does not lift it.
    forged.log.append("control.resumed", AGENT, {"scope": "all", "reason": "agent resume"})
    assert _call(server, "control_status")["running"] is False
    resume_form = _form(client.get("/control").text, "/control/resume")
    resumed = client.post(
        "/control/resume",
        data={**resume_form, "scope": "all", "reason": "checked"},
        headers=ORIGIN,
    )
    assert resumed.status_code == 303
    assert _call(server, "control_status")["running"] is True


def test_the_stop_page_survives_its_other_panels_failing(
    forged: Forged, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = forged.client()
    stop_form = _form(client.get("/control").text, "/control/stop")
    client.post(
        "/control/stop", data={**stop_form, "scope": "all", "reason": "halt"}, headers=ORIGIN
    )

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("panel dependency failed")

    # Break the lease board and the registry as the STOP page sees them.
    monkeypatch.setattr("aar.hub.console.lease_board", broken)
    monkeypatch.setattr("aar.hub.console.Registry", broken)
    page = client.get("/control")
    assert page.status_code == 200
    assert "STOP ALL" in page.text and 'action="/control/resume"' in page.text
    assert "panel failed to load" in page.text and "panel dependency failed" not in page.text
    resume_form = _form(page.text, "/control/resume")
    resumed = client.post(
        "/control/resume",
        data={**resume_form, "scope": "all", "reason": "checked"},
        headers=ORIGIN,
    )
    assert resumed.status_code == 303
    assert control.status(forged.log).running is True
    stop_again = _form(client.get("/control").text, "/control/stop")
    assert (
        client.post(
            "/control/stop", data={**stop_again, "scope": "all", "reason": "x"}, headers=ORIGIN
        ).status_code
        == 303
    )
