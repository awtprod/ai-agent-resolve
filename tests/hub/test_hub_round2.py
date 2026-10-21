"""Round-two hub tests: the review gate, withdrawals, forged events, console hardening, CLI.

Every test drives the real code over a real temporary SQLite log. The console is exercised
in-process through Starlette's ``TestClient`` (no port is bound) and ``uvicorn.run`` is
stubbed for the CLI tests.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from aar.cli import main
from aar.hub import approvals, control, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.common import iso
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.hub.graph import IllegalTransition, ProtectedNode, WorkGraph
from aar.hub.leaseboard import lease_board, record_lease_event
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

AGENT = Actor("agent", "agent-1")
OTHER_AGENT = Actor("agent", "agent-2")
HUMAN = Actor("human", "operator-1")
BASE = "http://127.0.0.1:8788"
ORIGIN = {"Origin": BASE}
JSON_ORIGIN = {"Origin": BASE, "Accept": "application/json"}
SECRET = b"0123456789abcdef0123456789abcdef"
RENDER = bytes(range(256)) * 4
STATUS_EVENT = "graph.node_status_changed"
# Superscript two, Arabic-Indic 12 and fullwidth 12: str.isdigit() accepts them, int() may not.
NON_ASCII_DIGITS = ["\u00b2", "\u0661\u0662", "\uff11\uff12"]


class Hub:
    """A hub with one project, one subject node, a render and two open reviews."""

    def __init__(self, tmp_path: Path) -> None:
        self.root = tmp_path
        self.db = tmp_path / "hub.db"
        self.log = EventLog(self.db)
        self.artifacts = tmp_path / "artifacts"
        self.store = ArtifactStore(self.artifacts, self.log)
        self.project = Registry(self.log).create_project("demo-show", HUMAN).id
        self.graph = WorkGraph(self.log, self.project)
        self.op = self.graph.add_node("op", AGENT, title="build").id
        self.render = self.store.put(RENDER, actor=AGENT, mime="video/mp4", project_id=self.project)
        self.review = self.new_review(self.op, "acme cut one")
        self.other = self.new_review(self.op, "acme cut two")

    def new_subject(self) -> str:
        return self.graph.add_node("op", AGENT, title="another build").id

    def new_review(self, subject: str, title: str, actor: Actor = AGENT) -> str:
        return reviews.request_review(
            self.log, self.store, self.project, subject, [self.render], actor, title=title, fps=24
        ).id

    def get(self, review_id: str) -> reviews.Review:
        found = reviews.get_review(self.log, self.project, review_id)
        assert found is not None
        return found

    def client(self, **kwargs: Any) -> TestClient:
        config = ConsoleConfig(operator="operator-1", csrf_secret=SECRET)
        app = build_app(self.log, self.store, config)
        kwargs.setdefault("client", ("127.0.0.1", 50000))
        return TestClient(app, base_url=BASE, follow_redirects=False, **kwargs)

    def count(self, event_type: str) -> int:
        return len(self.log.read(types=[event_type]))

    def notes_path(self, review_id: str) -> str:
        return f"/reviews/{self.project}/{review_id}/notes"

    def decide_path(self, review_id: str) -> str:
        return f"/reviews/{self.project}/{review_id}/decide"


@pytest.fixture
def hub(tmp_path: Path) -> Hub:
    return Hub(tmp_path)


def _form(html: str, action: str) -> dict[str, str]:
    """The hidden csrf and idem fields of the form that posts to ``action``."""
    start = html.index(f'action="{action}"')
    section = html[start : html.index("</form>", start)]
    csrf = re.search(r'name="csrf" value="([^"]*)"', section)
    idem = re.search(r'name="idem" value="([^"]*)"', section)
    assert csrf and idem, f"no form for {action}"
    return {"csrf": csrf.group(1), "idem": idem.group(1)}


def _tokens(client: TestClient, hub: Hub) -> dict[str, str]:
    """A valid csrf token and idem key, taken from an OPEN review's note form.

    The note form only renders on open reviews, and the csrf token is bound to the
    identity rather than the page, so these work for any write by the same operator.
    """
    page = client.get(f"/reviews/{hub.project}/{hub.review}").text
    return _form(page, hub.notes_path(hub.review))


def _fresh_idem(seed: int) -> str:
    return f"{seed:032x}"


def _error_code(response: Any) -> str:
    return str(response.json()["error"]["code"])


# -- A. the review gate --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("actor", "status"),
    [
        pytest.param(AGENT, "done", id="agent-done"),
        pytest.param(HUMAN, "done", id="human-done"),
        pytest.param(AGENT, "cancelled", id="requester-cancel"),
    ],
)
def test_generic_status_api_cannot_move_a_review(hub: Hub, actor: Actor, status: str) -> None:
    before = hub.log.latest_seq()

    with pytest.raises(ProtectedNode):
        WorkGraph(hub.log, hub.project).set_status(hub.review, status, actor)

    assert hub.log.latest_seq() == before
    review = hub.get(hub.review)
    assert review.pending and review.status == "active" and review.decision is None
    assert hub.review in [r.id for r in reviews.list_reviews(hub.log, pending_only=True)]


def test_an_agent_cannot_flip_a_decided_review(hub: Hub) -> None:
    reviews.decide(hub.log, hub.project, hub.review, "reject", "not yet", HUMAN)
    assert hub.get(hub.review).status == "failed"
    before = hub.log.latest_seq()

    # failed -> active is a legal transition, so only the review gate stops this one.
    with pytest.raises(ProtectedNode), hub.log.transaction() as txn:
        WorkGraph(hub.log, hub.project).set_status_in(txn, hub.review, "active", AGENT)
    with pytest.raises(ProtectedNode):
        WorkGraph(hub.log, hub.project).set_status(hub.review, "active", AGENT)
    # failed -> done is not a legal transition at all; the state machine refuses it first.
    with pytest.raises((IllegalTransition, ProtectedNode)), hub.log.transaction() as txn:
        WorkGraph(hub.log, hub.project).set_status_in(txn, hub.review, "done", AGENT)

    assert hub.log.latest_seq() == before
    review = hub.get(hub.review)
    assert review.status == "failed"
    assert review.decision is not None and review.decision.decision == "reject"


@pytest.mark.parametrize(
    ("actor", "target"),
    [
        pytest.param(AGENT, "done", id="requester-done"),
        pytest.param(AGENT, "failed", id="requester-failed"),
        pytest.param(OTHER_AGENT, "cancelled", id="stranger-cancel"),
    ],
)
def test_forged_review_status_event_is_skipped(hub: Hub, actor: Actor, target: str) -> None:
    # active -> done/failed/cancelled are all legal transitions, so only the review gate
    # (humans decide, only the requester withdraws) can be what skips these events.
    forged = hub.log.append(
        STATUS_EVENT, actor, {"node_id": hub.review, "to": target}, project_id=hub.project
    )

    review = hub.get(hub.review)
    assert review.status == "active" and review.pending and review.decision is None
    skipped = {item.event_id: item for item in WorkGraph(hub.log, hub.project).skipped()}
    assert forged.id in skipped
    assert "review" in skipped[forged.id].reason
    assert forged.id in {
        item.event_id for item in reviews.review_view(hub.log, hub.project, hub.review)[1]
    }

    decision = reviews.decide(hub.log, hub.project, hub.review, "approve", "looks good", HUMAN)
    assert decision.decision == "approve"
    assert hub.get(hub.review).status == "done"


def test_requester_withdraws_an_open_review(hub: Hub) -> None:
    reviews.withdraw(hub.log, hub.project, hub.review, "superseded", AGENT)

    review = hub.get(hub.review)
    assert review.status == "cancelled" and not review.pending and review.decision is None
    assert review.withdrawal is not None
    assert review.withdrawal.reason == "superseded" and review.withdrawal.by == AGENT
    (event,) = hub.log.read(types=["review.withdrawn"])
    assert event.actor == AGENT
    assert event.payload["reason"] == "superseded" and event.payload["review_id"] == hub.review
    pending = {r.id for r in reviews.list_reviews(hub.log, pending_only=True)}
    assert hub.review not in pending and hub.other in pending
    with pytest.raises(reviews.ReviewNotOpen):
        reviews.decide(hub.log, hub.project, hub.review, "approve", "too late", HUMAN)


def test_another_agent_cannot_withdraw_a_review(hub: Hub) -> None:
    before = hub.log.latest_seq()

    with pytest.raises(reviews.NotRequester):
        reviews.withdraw(hub.log, hub.project, hub.review, "not mine", OTHER_AGENT)

    assert hub.log.latest_seq() == before
    assert hub.count("review.withdrawn") == 0
    assert hub.get(hub.review).status == "active" and hub.get(hub.review).withdrawal is None


def test_withdraw_after_a_human_decision_is_refused(hub: Hub) -> None:
    reviews.decide(hub.log, hub.project, hub.review, "approve", "ship it", HUMAN)
    before = hub.log.latest_seq()

    with pytest.raises(reviews.ReviewNotOpen):
        reviews.withdraw(hub.log, hub.project, hub.review, "changed my mind", AGENT)

    assert hub.log.latest_seq() == before
    assert hub.count("review.withdrawn") == 0
    review = hub.get(hub.review)
    assert review.status == "done" and review.decision is not None


# -- B. reviews, smaller edges --------------------------------------------------------------


def test_note_on_a_decided_review_is_refused(hub: Hub) -> None:
    reviews.decide(hub.log, hub.project, hub.review, "reject", "redo it", HUMAN)
    before = hub.log.latest_seq()

    with pytest.raises(reviews.ReviewNotOpen):
        reviews.add_note(hub.log, hub.project, hub.review, 12, "one more thing", AGENT)

    assert hub.log.latest_seq() == before
    assert hub.get(hub.review).notes == ()


def test_console_note_on_a_decided_review_is_409(hub: Hub) -> None:
    reviews.decide(hub.log, hub.project, hub.review, "reject", "redo it", HUMAN)
    client = hub.client()
    # The decided review shows no note form, so the tokens come from the other, open one.
    tokens = _form(
        client.get(f"/reviews/{hub.project}/{hub.other}").text, hub.notes_path(hub.other)
    )
    before = hub.log.latest_seq()

    response = client.post(
        hub.notes_path(hub.review),
        data={**tokens, "frame": "3", "text": "too late"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 409
    assert _error_code(response) == "review_not_open"
    assert hub.log.latest_seq() == before
    assert hub.count("review.note_added") == 0


def test_cannot_approve_a_review_of_a_cancelled_subject(hub: Hub) -> None:
    subject = hub.new_subject()
    review_id = hub.new_review(subject, "acme cut three")
    WorkGraph(hub.log, hub.project).set_status(subject, "cancelled", AGENT)
    before = hub.log.latest_seq()

    with pytest.raises(reviews.SubjectCancelled, match=subject):
        reviews.decide(hub.log, hub.project, review_id, "approve", "fine", HUMAN)

    assert hub.log.latest_seq() == before
    assert hub.get(review_id).status == "active" and hub.get(review_id).decision is None
    rejection = reviews.decide(
        hub.log, hub.project, review_id, "reject", "subject was dropped", HUMAN
    )
    assert rejection.decision == "reject"
    assert hub.get(review_id).status == "failed"


def test_console_approve_of_a_cancelled_subject_is_409(hub: Hub) -> None:
    subject = hub.new_subject()
    review_id = hub.new_review(subject, "acme cut three")
    WorkGraph(hub.log, hub.project).set_status(subject, "cancelled", AGENT)
    client = hub.client()
    tokens = _tokens(client, hub)
    before = hub.log.latest_seq()

    response = client.post(
        hub.decide_path(review_id),
        data={**tokens, "decision": "approve", "reason": "fine"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 409
    assert _error_code(response) == "subject_cancelled"
    assert hub.log.latest_seq() == before
    assert hub.count("review.decided") == 0


# -- C. approvals ---------------------------------------------------------------------------


def test_forged_agent_revoke_does_not_revoke_a_token(hub: Hub) -> None:
    token = approvals.grant(hub.log, "db.switch", "demo-show", timedelta(hours=1), "go", HUMAN)
    now = datetime.now(UTC)
    forged = hub.log.append(
        "approval.revoked", AGENT, {"token_id": token.token_id, "reason": "x", "at": iso(now)}
    )

    current = approvals.get_approval(hub.log, token.token_id)
    assert current is not None and current.state(now) == "active"
    assert forged.id in {item.event_id for item in approvals.approvals_view(hub.log).skipped}
    consumed = approvals.consume(hub.log, token.token_id, "db.switch", "demo-show", AGENT)
    assert consumed.consumed_by == AGENT


# -- D. console input parsing and the 500 page -----------------------------------------------


@pytest.mark.parametrize("digits", NON_ASCII_DIGITS, ids=["superscript", "arabic", "fullwidth"])
def test_note_frame_with_non_ascii_digits_is_422(hub: Hub, digits: str) -> None:
    client = hub.client(raise_server_exceptions=False)
    tokens = _tokens(client, hub)
    before = hub.log.latest_seq()

    response = client.post(
        hub.notes_path(hub.review),
        data={**tokens, "frame": digits, "text": "hello"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 422
    assert _error_code(response) == "invalid_input"
    assert hub.log.latest_seq() == before


@pytest.mark.parametrize("digits", NON_ASCII_DIGITS, ids=["superscript", "arabic", "fullwidth"])
def test_grant_ttl_with_non_ascii_digits_is_422(hub: Hub, digits: str) -> None:
    client = hub.client(raise_server_exceptions=False)
    tokens = _tokens(client, hub)
    before = hub.log.latest_seq()

    response = client.post(
        "/approvals/grant",
        data={**tokens, "scope": "db.switch", "target": "demo-show", "ttl": digits, "reason": "go"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 422
    assert _error_code(response) == "invalid_input"
    assert hub.log.latest_seq() == before
    assert approvals.list_approvals(hub.log) == []


@pytest.mark.parametrize("digits", ["9" * 13, "9" * 5000], ids=["13-digits", "5000-digits"])
def test_note_frame_over_the_digit_cap_is_422(hub: Hub, digits: str) -> None:
    client = hub.client(raise_server_exceptions=False)
    tokens = _tokens(client, hub)
    before = hub.log.latest_seq()

    response = client.post(
        hub.notes_path(hub.review),
        data={**tokens, "frame": digits, "text": "hello"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 422
    assert _error_code(response) == "invalid_input"
    assert hub.log.latest_seq() == before


@pytest.mark.parametrize("digits", ["9" * 13, "9" * 5000], ids=["13-digits", "5000-digits"])
def test_grant_ttl_over_the_digit_cap_is_422(hub: Hub, digits: str) -> None:
    client = hub.client(raise_server_exceptions=False)
    tokens = _tokens(client, hub)
    before = hub.log.latest_seq()

    response = client.post(
        "/approvals/grant",
        data={**tokens, "scope": "db.switch", "target": "demo-show", "ttl": digits, "reason": "go"},
        headers=JSON_ORIGIN,
    )

    assert response.status_code == 422
    assert _error_code(response) == "invalid_input"
    assert hub.log.latest_seq() == before
    assert approvals.list_approvals(hub.log) == []


def test_unexpected_error_page_hides_the_exception(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("secret-detail /srv/private/path")

    monkeypatch.setattr("aar.hub.reviews.scan_reviews", boom)
    client = hub.client(raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="aar.hub.console"):
        page = client.get("/")
        api = client.get("/", headers={"Accept": "application/json"})

    assert page.status_code == 500
    assert "reference" in page.text
    for leak in ("secret-detail", "RuntimeError", "/srv/private"):
        assert leak not in page.text
    assert api.status_code == 500
    assert _error_code(api) == "internal_error"
    assert "secret-detail" not in api.json()["error"]["message"]
    assert "secret-detail" not in api.text
    # The detail is not lost: the traceback is in the log, under the page's reference.
    ref = re.search(r"reference ([0-9a-f]{8})", page.text)
    assert ref is not None
    assert any(
        ref.group(1) in record.getMessage() and record.exc_info is not None
        for record in caplog.records
    )


# -- E. security headers on every response --------------------------------------------------


_PAGE_STATUS = {
    "queue": 200,
    "review": 200,
    "leases": 200,
    "approvals": 200,
    "control": 200,
    "not-found": 404,
    "method-not-allowed": 405,
    "invalid-input": 422,
    "csrf-failed": 403,
    "too-large": 413,
}


def _fetch(hub: Hub, client: TestClient, kind: str) -> Any:
    if kind == "queue":
        return client.get("/")
    if kind == "review":
        return client.get(f"/reviews/{hub.project}/{hub.review}")
    if kind in ("leases", "approvals", "control"):
        return client.get(f"/{kind}")
    if kind == "not-found":
        return client.get("/nope")
    if kind == "method-not-allowed":
        return client.get("/control/stop")
    tokens = _tokens(client, hub)
    if kind == "invalid-input":
        data = {**tokens, "frame": "not-a-number", "text": "hello"}
        return client.post(hub.notes_path(hub.review), data=data, headers=ORIGIN)
    if kind == "csrf-failed":
        data = {**tokens, "csrf": "forged", "frame": "3", "text": "hello"}
        return client.post(hub.notes_path(hub.review), data=data, headers=ORIGIN)
    assert kind == "too-large"
    data = {**tokens, "scope": "all", "reason": "x" * (70 * 1024)}
    return client.post("/control/stop", data=data, headers=ORIGIN)


@pytest.mark.parametrize("kind", list(_PAGE_STATUS))
def test_every_page_and_error_page_carries_the_security_headers(hub: Hub, kind: str) -> None:
    response = _fetch(hub, hub.client(), kind)

    assert response.status_code == _PAGE_STATUS[kind]
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert "no-store" in response.headers["cache-control"]


# -- F. the request body cap ----------------------------------------------------------------


def test_oversized_body_is_413_and_nothing_is_stopped(hub: Hub) -> None:
    client = hub.client()
    page = client.get("/control").text
    big = _form(page, "/control/stop")
    before = hub.log.latest_seq()

    refused = client.post(
        "/control/stop",
        data={**big, "scope": "all", "reason": "x" * (70 * 1024)},
        headers=JSON_ORIGIN,
    )

    assert refused.status_code == 413
    assert _error_code(refused) == "body_too_large"
    assert control.status(hub.log).running is True
    assert hub.count("control.stopped") == 0
    assert hub.log.latest_seq() == before

    accepted = client.post(
        "/control/stop",
        data={**big, "idem": _fresh_idem(7), "scope": "all", "reason": "y" * 1000},
        headers=ORIGIN,
    )

    assert accepted.status_code == 303
    assert hub.count("control.stopped") == 1
    assert control.status(hub.log).running is False


# -- G. the lease board ---------------------------------------------------------------------


def test_late_grant_after_release_is_a_skipped_ghost_not_a_lease(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    ticks = {"now": 100.0}
    events: list[LeaseEvent] = []
    manager = LeaseManager(clock=lambda: ticks["now"], sink=events.append)
    holder = Holder.agent("agent-1")
    lease = manager.acquire(holder, Scope.PROJECT, ("demo-show",), ttl=3600)
    ticks["now"] = 160.0
    manager.release(lease.id, holder)
    assert [event.kind for event in events] == ["granted", "released"]
    granted, released = events
    wall = datetime.now(UTC)
    offset = wall.timestamp() - 160.0

    # The node's retry delivers the release first; the original grant arrives late.
    record_lease_event(log, released, "node-1", wall_offset=offset)
    late = record_lease_event(log, granted, "node-1", wall_offset=offset)

    board = lease_board(log, clock=lambda: wall + timedelta(seconds=1))
    assert board.leases == ()
    assert [item.event_id for item in board.skipped] == [late.id]
    assert "already ended" in board.skipped[0].reason


# -- H. the command line --------------------------------------------------------------------


def _console_argv(hub: Hub, *extra: str) -> list[str]:
    return ["hub", "console", "--db", str(hub.db), "--artifacts", str(hub.artifacts), *extra]


def _stub_uvicorn(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    return calls


def _warnings(err: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("WARNING:")]


def test_console_warns_when_it_trusts_tailscale_headers(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_uvicorn(monkeypatch)

    code = main(_console_argv(hub, "--operator", "operator-1", "--trust-tailscale-headers"))

    assert code == 0 and len(calls) == 1
    warnings = _warnings(capsys.readouterr().err)
    assert len(warnings) == 1 and "Tailscale-User-Login" in warnings[0]


def test_console_warns_when_it_binds_beyond_loopback(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_uvicorn(monkeypatch)

    code = main(
        _console_argv(hub, "--operator", "operator-1", "--host", "0.0.0.0", "--allow-remote")
    )

    assert code == 0 and [call["host"] for call in calls] == ["0.0.0.0"]
    warnings = _warnings(capsys.readouterr().err)
    assert len(warnings) == 1 and "0.0.0.0" in warnings[0]


def test_plain_loopback_console_prints_no_warning(
    hub: Hub, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls = _stub_uvicorn(monkeypatch)

    code = main(_console_argv(hub, "--operator", "operator-1"))

    assert code == 0 and [call["host"] for call in calls] == ["127.0.0.1"]
    assert "WARNING" not in capsys.readouterr().err


def test_review_list_on_a_directory_reports_database_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = tmp_path / "not-a-database"
    directory.mkdir()

    code = main(["hub", "review", "list", "--db", str(directory)])

    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    error = json.loads(captured.err)["error"]
    assert error["code"] == "database_unavailable"
    assert "(a directory)" in error["message"]
