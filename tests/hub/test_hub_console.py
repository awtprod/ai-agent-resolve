"""Tests for the human supervision console, driven through Starlette's TestClient."""

from __future__ import annotations

import asyncio
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from starlette.routing import Route
from starlette.testclient import TestClient

from aar.cli import main
from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import HumanRequired
from aar.hub.console import WRITE_ROUTES, ConsoleConfig, CsrfSigner, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import record_lease_event
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

AGENT = Actor("agent", "agent-1")
HUMAN = Actor("human", "operator-1")
BASE = "http://127.0.0.1:8788"
ORIGIN = {"Origin": BASE}
XSS = '<script>alert("xss")</script>'
XSS_ESCAPED = "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;"
SECRET = b"0123456789abcdef0123456789abcdef"
RENDER = bytes(range(256)) * 4  # 1024 bytes with distinct positions


class Hub:
    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.log = EventLog(tmp_path / "hub.db")
        self.store_root = tmp_path / "artifacts"
        self.store = ArtifactStore(self.store_root, self.log)
        self.project = Registry(self.log).create_project("demo-show", HUMAN).id
        graph = WorkGraph(self.log, self.project)
        self.op = graph.add_node("op", AGENT, title="build").id
        self.render = self.store.put(RENDER, actor=AGENT, mime="video/mp4", project_id=self.project)
        self.xss_review = reviews.request_review(
            self.log, self.store, self.project, self.op, [self.render], AGENT, title=XSS, fps=24
        ).id
        reviews.add_note(self.log, self.project, self.xss_review, 48, XSS, AGENT)
        self.plain_review = reviews.request_review(
            self.log, self.store, self.project, self.op, [self.render], AGENT, title="acme v2"
        ).id

    def client(self, config: ConsoleConfig | None = None, **kwargs: Any) -> TestClient:
        config = config or ConsoleConfig(operator="operator-1", csrf_secret=SECRET)
        app = build_app(self.log, self.store, config, clock=kwargs.pop("clock", _now))
        kwargs.setdefault("client", ("127.0.0.1", 50000))
        return TestClient(app, base_url=BASE, follow_redirects=False, **kwargs)

    def decide_path(self, review_id: str) -> str:
        return f"/reviews/{self.project}/{review_id}/decide"

    def count(self, event_type: str) -> int:
        return len(self.log.read(types=[event_type]))


def _now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def _form(html: str, action: str) -> dict[str, str]:
    """Return the hidden csrf and idem fields of the form posting to ``action``."""
    start = html.index(f'action="{action}"')
    section = html[start : html.index("</form>", start)]
    csrf = re.search(r'name="csrf" value="([^"]*)"', section)
    idem = re.search(r'name="idem" value="([^"]*)"', section)
    assert csrf and idem, f"no form for {action}"
    return {"csrf": csrf.group(1), "idem": idem.group(1)}


def _decide_form(client: TestClient, hub: Hub, review_id: str) -> dict[str, str]:
    page = client.get(f"/reviews/{hub.project}/{review_id}").text
    return _form(page, hub.decide_path(review_id))


# -- escaping -----------------------------------------------------------------


def test_agent_strings_render_escaped_on_every_page(hub: Hub) -> None:
    clock_now = _now()
    approvals.grant(hub.log, "db.switch", XSS, timedelta(hours=1), XSS, HUMAN)
    control.stop(hub.log, "project:" + hub.project, XSS, AGENT)
    Registry(hub.log).create_project(XSS, HUMAN)
    manager_events: list[LeaseEvent] = []
    manager = LeaseManager(clock=lambda: 100.0, sink=manager_events.append)
    manager.acquire(Holder.agent(XSS), Scope.PROJECT, ("demo-show",), ttl=3600)
    offset = clock_now.timestamp() - 100.0
    for event in manager_events:
        record_lease_event(hub.log, event, "node-1", wall_offset=offset)

    client = hub.client()
    pages = [
        "/",
        f"/reviews/{hub.project}/{hub.xss_review}",
        "/leases",
        "/approvals",
        "/control",
    ]
    for path in pages:
        response = client.get(path)
        assert response.status_code == 200, path
        body = response.text
        assert XSS not in body, path
        assert XSS_ESCAPED in body, path
        # Every script on the page carries the CSP nonce; injected ones could not.
        nonce = re.search(
            r"script-src 'nonce-([^']+)'", response.headers["content-security-policy"]
        )
        assert nonce is not None
        assert body.count("<script") == body.count(f'<script nonce="{nonce.group(1)}"') == 1


def test_flash_codes_are_not_reflected(hub: Hub) -> None:
    body = hub.client().get("/", params={"done": "<i>flash-probe</i>"}).text
    assert "flash-probe" not in body and 'class="flash"' not in body
    assert 'class="flash"' in hub.client().get("/", params={"done": "approved"}).text


# -- writes need POST, CSRF, Origin and a human ----------------------------------


def test_every_write_route_is_post_only_and_get_is_rejected(hub: Hub) -> None:
    client = hub.client()
    app = client.app.app  # inside the HostGuard
    posts = {r.path for r in app.routes if isinstance(r, Route) and "POST" in (r.methods or ())}
    assert posts == set(WRITE_ROUTES)
    for route in app.routes:
        if isinstance(route, Route) and route.path in posts:
            assert route.methods == {"POST"}
    before = hub.log.latest_seq()
    for path in WRITE_ROUTES:
        concrete = path.format(project_id=hub.project, review_id=hub.plain_review)
        response = client.get(concrete, params={"decision": "approve", "reason": "x"})
        assert response.status_code == 405, concrete
        assert "POST" in response.headers["allow"]
    # And no read page writes anything either.
    for path in ("/", "/leases", "/approvals", "/control", f"/artifacts/{hub.render}"):
        assert client.get(path).status_code == 200
    assert hub.log.latest_seq() == before


def test_post_without_or_with_a_bad_csrf_token_is_rejected(hub: Hub) -> None:
    client = hub.client()
    form = _decide_form(client, hub, hub.plain_review)
    base = {"decision": "approve", "reason": "ok", "idem": form["idem"]}
    other_identity = CsrfSigner(SECRET, max_age=3600).issue("human:operator-2")
    expired = CsrfSigner(SECRET, max_age=3600, now=lambda: 1.0).issue("human:operator-1")
    for csrf in (None, "", "1.deadbeef", form["csrf"] + "0", other_identity, expired):
        data = dict(base) if csrf is None else {**base, "csrf": csrf}
        response = client.post(hub.decide_path(hub.plain_review), data=data, headers=ORIGIN)
        assert response.status_code == 403, csrf
        assert "csrf_failed" in response.text
    assert hub.count(reviews.EVENT_DECIDED) == 0
    # The real token works, so the refusals above were the token's fault.
    ok = client.post(
        hub.decide_path(hub.plain_review), data={**base, "csrf": form["csrf"]}, headers=ORIGIN
    )
    assert ok.status_code == 303
    assert hub.count(reviews.EVENT_DECIDED) == 1


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Origin": "http://evil.example"},
        {"Origin": "null"},
        {"Origin": "http://127.0.0.1:9999"},
        {"Referer": "http://evil.example/page"},
    ],
)
def test_cross_origin_writes_are_rejected(hub: Hub, headers: dict[str, str]) -> None:
    client = hub.client()
    form = _decide_form(client, hub, hub.plain_review)
    data = {**form, "decision": "approve", "reason": "ok"}
    response = client.post(hub.decide_path(hub.plain_review), data=data, headers=headers)
    assert response.status_code == 403
    assert hub.count(reviews.EVENT_DECIDED) == 0


def test_same_origin_referer_is_accepted_without_origin(hub: Hub) -> None:
    client = hub.client()
    form = _decide_form(client, hub, hub.plain_review)
    data = {**form, "decision": "reject", "reason": "redo"}
    headers = {"Referer": f"{BASE}/reviews/{hub.project}/{hub.plain_review}"}
    assert (
        client.post(hub.decide_path(hub.plain_review), data=data, headers=headers).status_code
        == 303
    )


def test_without_an_identity_the_console_is_read_only(hub: Hub) -> None:
    client = hub.client(ConsoleConfig(csrf_secret=SECRET))
    page = client.get("/").text
    assert "Read-only" in page and 'name="csrf"' not in page
    forged = CsrfSigner(SECRET, max_age=3600).issue("")
    for path, data in (
        (hub.decide_path(hub.plain_review), {"decision": "approve", "reason": "ok"}),
        ("/control/resume", {"scope": "all", "reason": "ok"}),
        ("/control/stop", {"scope": "all", "reason": "ok"}),
    ):
        response = client.post(
            path, data={**data, "csrf": forged, "idem": "a" * 32}, headers=ORIGIN
        )
        assert response.status_code == 403 and "no_identity" in response.text
    assert hub.count(reviews.EVENT_DECIDED) == 0 and hub.count(control.EVENT_STOPPED) == 0


def test_agents_cannot_decide_or_resume(hub: Hub) -> None:
    control.stop(hub.log, "all", "agent saw smoke", AGENT)  # anyone may stop
    for actor in (AGENT, Actor("node", "node-1")):
        with pytest.raises(HumanRequired):
            reviews.decide(hub.log, hub.project, hub.plain_review, "approve", "ok", actor)
        with pytest.raises(HumanRequired):
            control.resume(hub.log, "all", "ok", actor)
    assert not control.status(hub.log).running
    assert hub.count(reviews.EVENT_DECIDED) == 0


def test_tailscale_login_is_trusted_only_from_loopback_and_only_when_enabled(hub: Hub) -> None:
    login = {"Tailscale-User-Login": "operator-2@acme.example"}
    trusting = ConsoleConfig(trust_tailscale_headers=True, csrf_secret=SECRET)

    behind_serve = hub.client(trusting)
    assert "operator-2@acme.example" in behind_serve.get("/", headers=login).text

    remote_peer = hub.client(trusting, client=("100.64.0.9", 50000))
    assert "Read-only" in remote_peer.get("/", headers=login).text

    not_enabled = hub.client(ConsoleConfig(csrf_secret=SECRET))
    assert "Read-only" in not_enabled.get("/", headers=login).text

    malformed = {"Tailscale-User-Login": "<b>evil</b>"}
    with_operator = ConsoleConfig(
        operator="operator-1", trust_tailscale_headers=True, csrf_secret=SECRET
    )
    assert "Read-only" in hub.client(with_operator).get("/", headers=malformed).text

    # The decision is recorded as the tailnet user.
    page = behind_serve.get(f"/reviews/{hub.project}/{hub.plain_review}", headers=login).text
    form = _form(page, hub.decide_path(hub.plain_review))
    response = behind_serve.post(
        hub.decide_path(hub.plain_review),
        data={**form, "decision": "approve", "reason": "ok"},
        headers={**ORIGIN, **login},
    )
    assert response.status_code == 303
    review = reviews.get_review(hub.log, hub.project, hub.plain_review)
    assert review is not None and review.decision is not None
    assert review.decision.by == Actor("human", "operator-2@acme.example")


def test_unknown_host_headers_are_refused(hub: Hub) -> None:
    client = hub.client()
    assert client.get("/", headers={"Host": "evil.example:8788"}).status_code == 421
    assert client.get("/", headers={"Host": "[::1]:8788"}).status_code == 200
    assert client.get("/", headers={"Host": "localhost"}).status_code == 200
    named = ConsoleConfig(
        operator="operator-1", allowed_hosts=("console.acme.example",), csrf_secret=SECRET
    )
    assert hub.client(named).get("/", headers={"Host": "console.acme.example"}).status_code == 200


# -- review flow -------------------------------------------------------------------


def test_double_submit_approves_once(hub: Hub) -> None:
    client = hub.client()
    form = _decide_form(client, hub, hub.plain_review)
    data = {**form, "decision": "approve", "reason": "ship it", "return_to": "queue"}
    first = client.post(hub.decide_path(hub.plain_review), data=data, headers=ORIGIN)
    second = client.post(hub.decide_path(hub.plain_review), data=data, headers=ORIGIN)
    assert first.status_code == second.status_code == 303
    assert first.headers["location"] == "/?done=approved"
    assert hub.count(reviews.EVENT_DECIDED) == 1
    changes = [
        e.payload["to"]
        for e in hub.log.read(types=["graph.node_status_changed"])
        if e.payload["node_id"] == hub.plain_review
    ]
    assert changes == ["done"]
    # A fresh form for the decided review is a conflict, shown as a structured error.
    again = {**data, "idem": "b" * 32, "decision": "reject"}
    conflict = client.post(
        hub.decide_path(hub.plain_review),
        data=again,
        headers={**ORIGIN, "Accept": "application/json"},
    )
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "already_decided"


def test_queue_shows_pending_reviews_with_render_and_notes(hub: Hub) -> None:
    body = hub.client().get("/").text
    assert "2 waiting for a human" in body
    assert f'src="/artifacts/{hub.render}#t=0.1"' in body
    assert "00:00:02:00" in body and 'data-seek="2.000"' in body
    assert "acme v2" in body


def test_note_from_the_console_lands_at_its_frame(hub: Hub) -> None:
    client = hub.client()
    path = f"/reviews/{hub.project}/{hub.xss_review}"
    form = _form(client.get(path).text, f"{path}/notes")
    response = client.post(
        f"{path}/notes", data={**form, "frame": "60", "text": "cut here"}, headers=ORIGIN
    )
    assert response.status_code == 303
    review = reviews.get_review(hub.log, hub.project, hub.xss_review)
    assert review is not None
    assert [(n.frame, n.timecode, n.author) for n in review.notes if n.text == "cut here"] == [
        (60, "00:00:02:12", HUMAN)
    ]
    bad = client.post(
        f"{path}/notes", data={**form, "idem": "c" * 32, "frame": "-3", "text": "x"}, headers=ORIGIN
    )
    assert bad.status_code == 422


# -- approvals and stop through the console -----------------------------------------


def test_grant_and_revoke_from_the_console(hub: Hub) -> None:
    client = hub.client()
    form = _form(client.get("/approvals").text, "/approvals/grant")
    data = {
        **form,
        "scope": "db.switch",
        "target": "Local Database",
        "ttl": "900",
        "reason": "sandbox",
    }
    assert client.post("/approvals/grant", data=data, headers=ORIGIN).status_code == 303
    assert client.post("/approvals/grant", data=data, headers=ORIGIN).status_code == 303
    tokens = approvals.list_approvals(hub.log)
    assert len(tokens) == 1 and tokens[0].granted_by == HUMAN  # double submit granted once
    page = client.get("/approvals").text
    revoke = _form(page, "/approvals/revoke")
    response = client.post(
        "/approvals/revoke",
        data={**revoke, "token_id": tokens[0].token_id, "reason": "not needed"},
        headers=ORIGIN,
    )
    assert response.status_code == 303
    assert approvals.list_approvals(hub.log)[0].revoked_by == HUMAN


def _mcp(server: Any, tool: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
    async def call() -> dict[str, Any]:
        async with Client(server) as client:
            result = await client.call_tool(tool, args or {})
            assert result.is_error is False
            if result.structured_content is not None:
                return result.structured_content
            return json.loads(result.content[0].text)

    return asyncio.run(call())


def test_stop_from_the_console_shows_in_control_status_then_resume(
    hub: Hub, tmp_path: Path
) -> None:
    client = hub.client()
    reader = EventLog(tmp_path / "hub.db", read_only=True)
    server = build_server(reader, Registry(reader), ArtifactStore(hub.store_root, reader))
    assert _mcp(server, "control_status") == {"running": True, "stops": []}

    form = _form(client.get("/control").text, "/control/stop")
    stop = client.post("/control/stop", data={**form, "scope": "all", "reason": ""}, headers=ORIGIN)
    assert stop.status_code == 303

    status = _mcp(server, "control_status")
    assert status["running"] is False
    assert status["stops"][0]["scope"] == "all"
    assert status["stops"][0]["by"] == {"kind": "human", "id": "operator-1"}
    assert status["stops"][0]["reason"] == "Emergency stop from the console"
    with pytest.raises(control.Stopped) as stopped:
        control.assert_running(hub.log, project_id=hub.project)
    assert stopped.value.by == HUMAN
    assert "STOPPED: all" in client.get("/").text  # banner on every page

    page = client.get("/control").text
    resume = _form(page, "/control/resume")
    response = client.post(
        "/control/resume",
        data={**resume, "scope": "all", "reason": "checked the timeline"},
        headers=ORIGIN,
    )
    assert response.status_code == 303
    assert _mcp(server, "control_status") == {"running": True, "stops": []}
    control.assert_running(hub.log, project_id=hub.project)


def test_expired_lease_drops_off_the_console_board(hub: Hub) -> None:
    events: list[LeaseEvent] = []
    manager = LeaseManager(clock=lambda: 1000.0, sink=events.append)
    manager.acquire(Holder.agent("agent-1"), Scope.TIMELINE, ("demo-show", "Main"), ttl=60)
    wall = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    for event in events:
        record_lease_event(hub.log, event, "node-1", wall_offset=wall.timestamp() - 1000.0)
    held = hub.client(clock=lambda: wall + timedelta(seconds=59)).get("/leases").text
    assert "demo-show / Main" in held and "agent-1" in held
    gone = hub.client(clock=lambda: wall + timedelta(seconds=61)).get("/leases").text
    assert "demo-show / Main" not in gone and "No leases are held" in gone


# -- artifacts ---------------------------------------------------------------------


def test_artifact_range_requests(hub: Hub) -> None:
    client = hub.client()
    path = f"/artifacts/{hub.render}"
    full = client.get(path)
    assert full.status_code == 200 and full.content == RENDER
    assert full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"] == "video/mp4"
    cases = {
        "bytes=100-199": (100, 199),
        "bytes=-50": (974, 1023),
        "bytes=1000-": (1000, 1023),
        "bytes=1000-99999": (1000, 1023),
        "bytes=0-0": (0, 0),
    }
    for header, (start, end) in cases.items():
        part = client.get(path, headers={"Range": header})
        assert part.status_code == 206, header
        assert part.headers["content-range"] == f"bytes {start}-{end}/1024", header
        assert part.content == RENDER[start : end + 1], header
        assert part.headers["content-length"] == str(end - start + 1), header
    for header in ("bytes=1024-", "bytes=5000-6000", "bytes=-0"):
        bad = client.get(path, headers={"Range": header})
        assert bad.status_code == 416, header
        assert bad.headers["content-range"] == "bytes */1024"
    for header in ("bytes=5-1", "items=0-10", "bytes=0-1,5-6"):
        ignored = client.get(path, headers={"Range": header})
        assert ignored.status_code == 200 and ignored.content == RENDER, header
    head = client.head(path, headers={"Range": "bytes=10-19"})
    assert head.status_code == 206 and head.content == b""
    assert head.headers["content-length"] == "10"


def test_active_content_from_agents_is_served_as_a_sandboxed_download(hub: Hub) -> None:
    page = hub.store.put(XSS.encode(), actor=AGENT, mime="text/html")
    response = hub.client().get(f"/artifacts/{page}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in response.headers["content-security-policy"]


def test_invalid_artifact_ids_never_touch_the_filesystem_outside_the_store(
    hub: Hub, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top-secret-demo-value")
    store_root = str(hub.store_root.resolve())
    opened: list[str] = []
    recording = {"on": False}

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if recording["on"] and event == "open" and args and isinstance(args[0], (str, bytes)):
            opened.append(str(args[0]))

    sys.addaudithook(hook)  # audit hooks cannot be removed; this one is gated by the flag
    client = hub.client()
    attempts = [
        "../secret.txt",
        "..%2fsecret.txt",
        "%2e%2e%2fsecret.txt",
        "..%2f..%2f..%2fetc%2fpasswd",
        hub.render.upper(),
        hub.render[:63],
        hub.render + "0",
        hub.render[:62] + "..",
        hub.render[:63] + "%00",
        str(secret),
    ]
    recording["on"] = True
    try:
        responses = [(attempt, client.get(f"/artifacts/{attempt}")) for attempt in attempts]
        missing = client.get("/artifacts/" + "0" * 64)
    finally:
        recording["on"] = False
    for attempt, response in responses:
        assert response.status_code in (400, 404), attempt
        assert b"top-secret-demo-value" not in response.content, attempt
    assert missing.status_code == 404
    outside = [p for p in opened if not str(Path(p).resolve()).startswith(store_root)]
    assert outside == [], outside


# -- CLI -------------------------------------------------------------------------------


def test_console_cli_binds_loopback_by_default_and_refuses_remote(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    db, store = str(hub.root / "hub.db"), str(hub.store_root)
    args = ["hub", "console", "--db", db, "--artifacts", store, "--operator", "operator-1"]
    assert main(args) == 0
    assert calls[-1]["host"] == "127.0.0.1" and calls[-1]["proxy_headers"] is False
    assert main([*args, "--host", "0.0.0.0"]) == 2
    assert "allow-remote" in capsys.readouterr().err
    assert len(calls) == 1
    assert main([*args, "--host", "0.0.0.0", "--allow-remote"]) == 0
    assert calls[-1]["host"] == "0.0.0.0"
    assert main([*args[:-1], "<bad name>"]) == 2
