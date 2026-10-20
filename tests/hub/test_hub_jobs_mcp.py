"""Agent job submission/cancellation through the real MCP SDK client, without Resolve."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from mcp_types import CallToolResult

from aar.hub import jobs
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.mcp_server import build_server
from aar.hub.mcp_writes import READ_TOOL_NAMES, WRITE_TOOL_NAMES
from aar.hub.registry import Registry

PROJECT = {"folder_path": ["Shows"], "name": "Show", "project_id": "resolve-show"}
PARAMS = {"steps": [{"op": "timeline.list", "params": {}}]}
APPROVAL = "existing-human-approval-reference"


@pytest.fixture
def log(tmp_path: Path) -> Iterator[EventLog]:
    with_log = EventLog(tmp_path / "hub.db")
    try:
        yield with_log
    finally:
        with_log.close()


def _server(log: EventLog, tmp_path: Path, *, agent_writes: bool = True) -> Any:
    return build_server(
        log,
        Registry(log),
        ArtifactStore(tmp_path / "artifacts", log),
        agent_writes=agent_writes,
    )


def _run(server: Any, body: Any) -> Any:
    async def go() -> Any:
        async with asyncio.timeout(20), Client(server, read_timeout_seconds=5) as client:
            return await body(client)

    return asyncio.run(go())


def _request(**changes: Any) -> dict[str, Any]:
    return copy.deepcopy(
        {
            "agent_id": "editor",
            "idempotency_key": "submit-1",
            "kind": "ops.sequence",
            "params": PARAMS,
            "project": PROJECT,
            "assigned_node": "absent-edit-pc",
            "priority": 5,
            "approval_token": APPROVAL,
            **changes,
        }
    )


def _ok(result: CallToolResult) -> dict[str, Any]:
    assert result.is_error is False, result.content
    body = result.structured_content
    assert isinstance(body, dict)
    assert json.loads(result.content[0].text) == body
    return body


def _refusal(result: CallToolResult, code: str) -> dict[str, Any]:
    assert result.is_error is True
    body = result.structured_content
    assert isinstance(body, dict)
    assert body["ok"] is False
    error = body["error"]
    assert error["code"] == code, body
    assert set(error) == {"code", "message", "fix_hint", "details"}
    assert error["fix_hint"]
    assert json.loads(result.content[0].text) == body
    return error


def _summary(job: dict[str, Any]) -> None:
    assert not {"params", "approval_token", "report", "result"} & job.keys()
    assert APPROVAL not in json.dumps(job)
    assert len(json.dumps(job).encode()) < 24 * 1024


def test_tools_are_opt_in_and_have_no_human_authority(log: EventLog, tmp_path: Path) -> None:
    async def read_only(client: Client) -> set[str]:
        tools = (await client.list_tools()).tools
        assert all(tool.annotations.read_only_hint for tool in tools)
        for name, arguments in (
            ("job_submit", _request()),
            (
                "job_cancel",
                {"agent_id": "editor", "idempotency_key": "c", "job_id": "job_x", "reason": "x"},
            ),
        ):
            assert (await client.call_tool(name, arguments)).is_error is True
        return {tool.name for tool in tools}

    baseline = _run(_server(log, tmp_path, agent_writes=False), read_only)

    async def enabled(client: Client) -> None:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}
        assert set(tools) == baseline | set(WRITE_TOOL_NAMES) | set(READ_TOOL_NAMES)
        for name in ("job_submit", "job_cancel"):
            tool = tools[name]
            assert tool.annotations.read_only_hint is False
            assert tool.annotations.idempotent_hint is True
            assert {"agent_id", "idempotency_key"} <= set(tool.input_schema["required"])
        assert set(tools["job_submit"].input_schema["properties"]) == {
            "agent_id",
            "idempotency_key",
            "kind",
            "params",
            "project",
            "assigned_node",
            "priority",
            "approval_token",
        }
        assert set(tools["job_cancel"].input_schema["properties"]) == {
            "agent_id",
            "idempotency_key",
            "job_id",
            "reason",
        }
        for name in (
            "job_release",
            "job_reassign",
            "approval_grant",
            "approval_revoke",
            "control_resume",
        ):
            assert name not in tools
            assert (await client.call_tool(name, {})).is_error is True
        # Extra arguments cannot elevate the actor or choose a separate execution agent.
        result = await client.call_tool(
            "job_submit",
            _request(actor={"kind": "human", "id": "andrew"}, execution_agent_id="other"),
        )
        if not result.is_error:
            job = _ok(result)["job"]
            assert job["creator"] == {"kind": "agent", "id": "editor"}
            assert job["agent_id"] == "editor"

    _run(_server(log, tmp_path), enabled)
    assert all(event.actor.kind == "agent" for event in log.iter_events())
    assert all(event.type == jobs.EVENT_SUBMITTED for event in log.iter_events())


def test_submission_persists_without_node_and_replays_after_reopen(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "hub.db")

    async def submit(client: Client) -> dict[str, Any]:
        body = _ok(await client.call_tool("job_submit", _request()))
        assert set(body) == {"ok", "op", "replayed", "project_id", "job"}
        assert body["ok"] is True and body["op"] == "job.submit" and body["replayed"] is False
        assert body["project_id"] == PROJECT["project_id"]
        job = body["job"]
        _summary(job)
        assert job["state"] == "waiting_for_node" and job["claimed_node"] is None
        assert job["assigned_node"] == "absent-edit-pc" and job["priority"] == 5
        assert job["creator"] == {"kind": "agent", "id": "editor"}
        assert job["agent_id"] == "editor" and job["project"] == PROJECT
        return body

    try:
        original = _run(_server(log, tmp_path), submit)
    finally:
        log.close()
    reopened = EventLog(tmp_path / "hub.db")
    try:

        async def replay(client: Client) -> None:
            repeated = _ok(await client.call_tool("job_submit", _request()))
            assert repeated == {**original, "replayed": True}
            for change in (
                {"priority": 6},
                {"assigned_node": "different-pc"},
                {"params": {"steps": [{"op": "timeline.list", "params": {}, "name": "new"}]}},
                {"project": {**PROJECT, "project_id": "another-project"}},
                {"approval_token": "another-approval"},
            ):
                _refusal(
                    await client.call_tool("job_submit", _request(**change)),
                    "idempotency_conflict",
                )
            listing = _ok(await client.call_tool("jobs_list", {}))
            assert listing["jobs"] == [original["job"]] and listing["next_cursor"] is None
            _summary(listing["jobs"][0])

        _run(_server(reopened, tmp_path), replay)
        events = list(reopened.iter_events())
        assert len(events) == 1 and events[0].type == jobs.EVENT_SUBMITTED
        assert events[0].actor == Actor("agent", "editor")
        durable = jobs.fold(reopened)[original["job"]["job_id"]]
        assert durable.params["steps"][0]["op"] == "timeline.list"
        assert durable.target.to_dict() == PROJECT
        assert durable.approval_token == APPROVAL
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "claim_first, expected_state, terminal_state",
    [(False, "cancelled", "cancelled"), (True, "cancel_requested", None)],
    ids=["unclaimed", "claimed"],
)
def test_cancel_only_own_job_with_idempotent_replay(
    log: EventLog,
    tmp_path: Path,
    claim_first: bool,
    expected_state: str,
    terminal_state: str | None,
) -> None:
    async def body(client: Client) -> None:
        own = _ok(await client.call_tool("job_submit", _request()))["job"]
        if claim_first:
            claimed, _ = jobs.claim(
                log, "absent-edit-pc", [PROJECT["project_id"]], request_id="claim-own"
            )
            assert claimed is not None and claimed.job_id == own["job_id"]
        other = _ok(await client.call_tool("job_submit", _request(agent_id="other")))["job"]
        request = {
            "agent_id": "editor",
            "idempotency_key": "cancel-1",
            "job_id": other["job_id"],
            "reason": "superseded",
        }
        _refusal(await client.call_tool("job_cancel", request), "not_creator")
        cancelled = _ok(await client.call_tool("job_cancel", {**request, "job_id": own["job_id"]}))
        assert cancelled["ok"] is True and cancelled["op"] == "job.cancel"
        assert cancelled["replayed"] is False
        assert cancelled["job"]["state"] == expected_state
        assert cancelled["job"]["cancel_reason"] == "superseded"
        _summary(cancelled["job"])
        repeated = _ok(await client.call_tool("job_cancel", {**request, "job_id": own["job_id"]}))
        assert repeated == {**cancelled, "replayed": True}
        _refusal(
            await client.call_tool(
                "job_cancel", {**request, "job_id": own["job_id"], "reason": "different"}
            ),
            "idempotency_conflict",
        )
        _refusal(
            await client.call_tool("job_cancel", {**request, "job_id": "job_missing"}),
            "job_not_found",
        )
        durable = jobs.fold(log)
        assert durable[own["job_id"]].terminal_state == terminal_state
        assert durable[other["job_id"]].cancel_requested is False

    _run(_server(log, tmp_path), body)
    assert len(log.read(types=[jobs.EVENT_CANCEL])) == 1


@pytest.mark.parametrize(
    "change, field",
    [
        ({"project": {}}, "project"),
        ({"project": {"name": "Show"}}, "project"),
        ({"project": {**PROJECT, "project_id": None}}, "project"),
        ({"project": {**PROJECT, "project_id": " resolve-show "}}, "project.project_id"),
        ({"project": {**PROJECT, "project_id": "x" * 129}}, "project.project_id"),
        ({"project": {**PROJECT, "folder_path": "Shows"}}, "project.folder_path"),
        ({"project": {**PROJECT, "folder_path": [3]}}, "project"),
        ({"project": {**PROJECT, "extra": "no"}}, "project"),
        ({"params": {}}, "params"),
        ({"params": {"steps": []}}, "params"),
        ({"params": {"steps": [{"op": "not.an.op", "params": {}}]}}, "params"),
        ({"params": {"steps": [{"op": "timeline.list", "params": {"extra": 1}}]}}, "params"),
        # The SDK serializes Python NaN/Infinity as null. JSON text survives its
        # transport and is decoded by MCP's supported object pre-parser, exercising
        # the queue's finite-number guard before the kind model can normalize it.
        (
            {"params": json.dumps({"steps": [{"op": "timeline.list", "timeout": math.nan}]})},
            "params",
        ),
        (
            {"params": json.dumps({"steps": [{"op": "timeline.list", "timeout": math.inf}]})},
            "params",
        ),
        ({"params": {"extra": "x" * (256 * 1024)}}, "params"),
        ({"kind": "missing-kind"}, "params"),
        ({"agent_id": ""}, "agent_id"),
        ({"agent_id": "human:andrew"}, "agent_id"),
        ({"agent_id": "a" * 129}, "agent_id"),
        ({"idempotency_key": ""}, "idempotency_key"),
        ({"idempotency_key": "has spaces"}, "idempotency_key"),
        ({"idempotency_key": "k" * 129}, "idempotency_key"),
        ({"assigned_node": ""}, "assigned_node"),
        ({"priority": 1_000_001}, "priority"),
        ({"approval_token": "x" * 129}, "approval_token"),
    ],
)
def test_invalid_submission_has_structured_refusal_and_no_event(
    log: EventLog, tmp_path: Path, change: dict[str, Any], field: str
) -> None:
    async def body(client: Client) -> None:
        error = _refusal(await client.call_tool("job_submit", _request(**change)), "invalid_input")
        assert error["details"]["field"] == field

    _run(_server(log, tmp_path), body)
    assert list(log.iter_events()) == []


def test_depth_bound_and_error_internals_are_not_exposed(
    log: EventLog, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nested: Any = 1
    for _ in range(jobs.MAX_JSON_DEPTH + 1):
        nested = {"next": nested}

    async def invalid(client: Client) -> None:
        _refusal(
            await client.call_tool("job_submit", _request(params={"nested": nested})),
            "invalid_input",
        )
        result = await client.call_tool(
            "job_submit", _request(params={"steps": [], "secret": APPROVAL})
        )
        _refusal(result, "invalid_input")
        assert APPROVAL not in json.dumps(result.model_dump())

    _run(_server(log, tmp_path), invalid)
    assert list(log.iter_events()) == []

    def crash(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("private exception detail: " + APPROVAL)

    monkeypatch.setattr(jobs, "submit", crash)

    async def unexpected(client: Client) -> None:
        result = await client.call_tool("job_submit", _request())
        assert result.is_error is True
        assert "private exception detail" not in json.dumps(result.model_dump())
        assert APPROVAL not in json.dumps(result.model_dump())

    _run(_server(log, tmp_path), unexpected)
    assert list(log.iter_events()) == []


@pytest.mark.parametrize(
    "change",
    [{"params": []}, {"project": []}, {"priority": True}, {"priority": "5"}],
)
def test_sdk_rejects_wrong_types_without_writing(
    log: EventLog, tmp_path: Path, change: dict[str, Any]
) -> None:
    async def body(client: Client) -> None:
        assert (await client.call_tool("job_submit", _request(**change))).is_error is True

    _run(_server(log, tmp_path), body)
    assert list(log.iter_events()) == []


def test_invalid_cancellation_and_terminal_conflict(log: EventLog, tmp_path: Path) -> None:
    async def body(client: Client) -> None:
        job_id = _ok(await client.call_tool("job_submit", _request()))["job"]["job_id"]
        request = {
            "agent_id": "editor",
            "idempotency_key": "cancel",
            "job_id": job_id,
            "reason": "superseded",
        }
        for change, field in (
            ({"agent_id": "bad agent"}, "agent_id"),
            ({"idempotency_key": ""}, "idempotency_key"),
            ({"job_id": "not-a-job"}, "job_id"),
            ({"reason": ""}, "reason"),
            ({"reason": "x" * 1001}, "reason"),
        ):
            error = _refusal(
                await client.call_tool("job_cancel", {**request, **change}), "invalid_input"
            )
            assert error["details"]["field"] == field
        claimed, _ = jobs.claim(log, "absent-edit-pc", [PROJECT["project_id"]], request_id="claim")
        assert claimed is not None
        jobs.report(
            log, "absent-edit-pc", job_id, claimed.generation, "succeeded", {}, request_id="done"
        )
        _refusal(await client.call_tool("job_cancel", request), "job_conflict")

    _run(_server(log, tmp_path), body)
    assert log.read(types=[jobs.EVENT_CANCEL]) == []
