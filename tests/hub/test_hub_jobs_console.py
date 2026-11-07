"""Read-only remote job visibility in the server-rendered hub console."""

from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from aar.hub import jobs
from aar.hub.artifacts import ArtifactStore
from aar.hub.console import ConsoleConfig, build_app
from aar.hub.events import Actor, EventLog
from aar.node.locator import ProjectLocator

BASE = "http://127.0.0.1:8788"
AGENT = Actor("agent", "editor")
PARAMS = {"steps": [{"op": "timeline.list", "params": {}}]}
PROJECT = ProjectLocator(("Shows",), "Comedy Hour", "resolve-show")
XSS = '<script>alert("jobs")</script>'


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def _client(log: EventLog, root: Path, clock: Clock) -> TestClient:
    app = build_app(log, ArtifactStore(root / "artifacts", log), ConsoleConfig(), clock=clock)
    return TestClient(app, base_url=BASE, client=("127.0.0.1", 50000))


def _submit(
    log: EventLog,
    clock: Clock,
    key: str,
    *,
    project: ProjectLocator = PROJECT,
    node: str = "edit-pc",
) -> jobs.HubJob:
    job, replayed = jobs.submit(
        log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=project,
        agent_id="editor",
        assigned_node=node,
        idempotency_key=key,
        clock=clock,
    )
    assert replayed is False
    return job


def test_empty_jobs_page_is_read_only_and_in_navigation(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    response = _client(log, tmp_path, Clock()).get("/jobs")

    assert response.status_code == 200
    assert '<a href="/jobs" class="" aria-current="page">Jobs</a>' in response.text
    assert "No remote jobs have been submitted" in response.text
    assert '<form method="get" action="/jobs"' in response.text
    assert 'method="post"' not in response.text
    assert "waiting_for_node" in response.text
    assert "not retried automatically" in response.text


def test_waiting_and_offline_state_survive_hub_restart(tmp_path: Path) -> None:
    clock = Clock()
    path = tmp_path / "hub.db"
    log = EventLog(path)
    job = _submit(log, clock, "submit")

    waiting = _client(log, tmp_path, clock).get("/jobs")
    assert waiting.status_code == 200
    assert job.job_id in waiting.text
    assert "Shows/Comedy Hour" in waiting.text
    assert "resolve-show" in waiting.text
    assert "edit-pc" in waiting.text
    assert "ops.sequence" in waiting.text
    assert "waiting_for_node" in waiting.text
    assert "10-20 12:00 UTC" in waiting.text

    claimed, replayed = jobs.claim(
        log,
        "edit-pc",
        ["resolve-show"],
        request_id="claim",
        lease_seconds=60,
        clock=clock,
    )
    assert claimed is not None and replayed is False
    log.close()

    restarted = EventLog(path, create=False)
    clock.advance(61)
    offline = _client(restarted, tmp_path, clock).get("/jobs")
    assert offline.status_code == 200
    assert job.job_id in offline.text
    assert "node_offline" in offline.text
    assert "not retried automatically" in offline.text


def test_jobs_page_paginates_and_preserves_filters(tmp_path: Path) -> None:
    clock = Clock()
    log = EventLog(tmp_path / "hub.db")
    submitted = [_submit(log, clock, f"submit-{index}") for index in range(26)]

    first = _client(log, tmp_path, clock).get("/jobs", params={"state": "waiting_for_node"})
    assert first.status_code == 200
    assert first.text.count('data-label="Job ID"') == 25
    assert submitted[0].job_id in first.text
    assert submitted[-1].job_id not in first.text
    assert 'href="/jobs?state=waiting_for_node&amp;page=2"' in first.text

    second = _client(log, tmp_path, clock).get(
        "/jobs", params={"state": "waiting_for_node", "page": "2"}
    )
    assert second.status_code == 200
    assert second.text.count('data-label="Job ID"') == 1
    assert submitted[-1].job_id in second.text
    assert 'href="/jobs?state=waiting_for_node"' in second.text


def test_empty_out_of_range_page_returns_to_first_page_with_filters(tmp_path: Path) -> None:
    clock = Clock()
    log = EventLog(tmp_path / "hub.db")
    job = _submit(log, clock, "submit")
    client = _client(log, tmp_path, clock)

    response = client.get("/jobs?state=waiting_for_node&page=999")

    assert response.status_code == 200
    assert "Showing 0&ndash;0 of 1 matching job." in response.text
    assert "No jobs are on this page." in response.text
    assert 'href="/jobs?state=waiting_for_node">First page</a>' in response.text
    assert 'data-label="Job ID"' not in response.text
    assert 'method="post"' not in response.text

    first = client.get("/jobs?state=waiting_for_node")
    assert first.status_code == 200
    assert job.job_id in first.text
    assert "Showing 1&ndash;1 of 1 matching job." in first.text


@pytest.mark.parametrize(
    "query",
    [
        "page=0",
        "page=-1",
        "page=nope",
        "page=1&page=2",
        "unknown=value",
        "state=%20waiting_for_node%20",
        "state=" + "x" * 129,
    ],
)
def test_jobs_page_rejects_malformed_queries(tmp_path: Path, query: str) -> None:
    log = EventLog(tmp_path / "hub.db")
    response = _client(log, tmp_path, Clock()).get(f"/jobs?{query}")

    assert response.status_code == 422
    assert "invalid_input" in response.text
    assert "internal_error" not in response.text


def test_corrupt_job_record_is_a_clear_non_500_response(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")
    log.append(jobs.EVENT_SUBMITTED, AGENT, {"job_id": "job_bad"})

    response = _client(log, tmp_path, Clock()).get("/jobs")

    assert response.status_code == 409
    assert "job_corrupt" in response.text
    assert "remote job queue is corrupt" in response.text
    assert "append-only job events" in response.text
    assert "internal_error" not in response.text


def test_job_values_and_active_filters_are_escaped(tmp_path: Path) -> None:
    clock = Clock()
    log = EventLog(tmp_path / "hub.db")
    project = ProjectLocator(("Shows <unsafe>",), XSS, "project<&>")
    node = "node<&>"
    job = _submit(log, clock, "unsafe", project=project, node=node)
    jobs.claim(log, node, ["project<&>"], request_id="claim", clock=clock)
    jobs.report(
        log,
        node,
        job.job_id,
        1,
        "waiting",
        {"waiting": {"kind": XSS}},
        request_id="waiting",
        clock=clock,
    )

    response = _client(log, tmp_path, clock).get("/jobs", params={"node": node})

    assert response.status_code == 200
    assert XSS not in response.text
    assert html.escape(XSS, quote=True) in response.text
    assert "Shows &lt;unsafe&gt;" in response.text
    assert "project&lt;&amp;&gt;" in response.text
    assert "node&lt;&amp;&gt;" in response.text
    assert 'value="node&lt;&amp;&gt;"' in response.text
