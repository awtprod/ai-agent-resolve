"""S19e: register a show that has no Resolve project (library, CLI and the MCP tool).

A source-less project is identified by its name: a second one with a name or alias that any
project already has is refused (``project_exists`` naming the existing id) unless it is a
retry of the same request. The Resolve source is attached later with ``project source``.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp.client import Client
from write_support import AGENT, HUMAN, Hub, make_hub

from aar.cli import main
from aar.hub import jobs, project_register
from aar.hub.mcp_server import build_server
from aar.hub.registry import DuplicateProjectName, ProjectSource, Registry
from aar.hub.writes import WriteError
from aar.node.hublink import LocalHubLink
from aar.node.locator import ProjectLocator

LOC = {"project_id": "uid-1", "name": "Fakes", "folder_path": ["Shows"]}
PARAMS = {"steps": [{"op": "timeline.list", "params": {}}]}


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    made = make_hub(tmp_path)  # prj_acme "acme-show" and prj_other "acme-other", no sources
    yield made
    made.log.close()


def _register(hub: Hub, key: str = "k1", **over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {"name": "Fakes", "idempotency_key": key, **over}
    actor = args.pop("actor", AGENT)
    return project_register.register_project(hub.log, actor, **args).to_dict()


def _refused(hub: Hub, code: str, **over: Any) -> WriteError:
    before = hub.counts()
    with pytest.raises(WriteError) as err:
        _register(hub, **over)
    assert err.value.code == code, err.value
    assert hub.counts() == before  # nothing written
    return err.value


# -- library -----------------------------------------------------------------------------------


def test_register_without_a_source_and_a_retry_replays(hub: Hub) -> None:
    first = _register(hub, aliases=["Fakes 5-23"])
    assert first["ok"] and first["replayed"] is False
    assert first["project"]["sources"] == [] and first["project"]["aliases"] == ["Fakes 5-23"]
    again = _register(hub, aliases=["Fakes 5-23"])
    assert again["replayed"] is True and again["project_id"] == first["project_id"]
    _refused(hub, "idempotency_conflict", name="Fakes 2")  # same key, other request
    assert len(Registry(hub.log).list()) == 3


@pytest.mark.parametrize(
    ("over", "taken"),
    [
        ({}, "Fakes"),  # same name
        ({"name": "Fakes 5-23"}, "Fakes 5-23"),  # the name is the first one's alias
        ({"name": "Other", "aliases": ["Fakes"]}, "Fakes"),  # an alias is its name
        ({"name": "acme-show"}, "acme-show"),  # a fixture project's name
    ],
)
def test_a_second_sourceless_project_with_a_taken_name_names_the_existing_id(
    hub: Hub, over: dict[str, Any], taken: str
) -> None:
    first = _register(hub, aliases=["Fakes 5-23"])["project_id"]
    err = _refused(hub, "project_exists", key="k2", **over)
    expected = "prj_acme" if taken == "acme-show" else first
    assert err.details == {"existing_id": expected, "name": taken}
    assert expected in err.message and expected in err.fix_hint


def test_a_sourced_project_holding_the_name_or_alias_refuses_a_sourceless_twin(hub: Hub) -> None:
    sourced = _register(
        hub, key="s", source_kind="local_db", locator=dict(LOC), aliases=["Fakes alt"]
    )["project_id"]
    assert _refused(hub, "project_exists", key="k2").details["existing_id"] == sourced
    assert _refused(hub, "project_exists", key="k3", name="Fakes alt").details["name"] == (
        "Fakes alt"
    )


def test_names_compare_exactly_as_the_registry_does(hub: Hub) -> None:
    _register(hub)
    # Not the same text, so not the same name (the registry and the node compare exactly).
    assert _register(hub, key="k2", name="fakes")["ok"]
    assert _register(hub, key="k3", name="Fakes ")["ok"]


def test_a_sourced_registration_is_still_identified_by_its_uid_only(hub: Hub) -> None:
    """Unchanged S17 behaviour, pinned: a sourced project may share a source-less one's name."""
    loose = _register(hub)["project_id"]
    sourced = _register(hub, key="s", source_kind="local_db", locator=dict(LOC))["project_id"]
    assert sourced != loose


def test_the_registry_itself_refuses_a_sourceless_twin(hub: Hub) -> None:
    registry = Registry(hub.log)
    first = registry.create_project("Show", HUMAN, idempotency_key="lib-1")
    assert registry.create_project("Show", HUMAN, idempotency_key="lib-1").id == first.id
    with pytest.raises(DuplicateProjectName) as err:
        registry.create_project("Show", HUMAN, idempotency_key="lib-2")
    assert (err.value.existing_id, err.value.name) == (first.id, "Show")
    with pytest.raises(DuplicateProjectName):
        registry.create_project("Show", HUMAN)
    assert [p.name for p in Registry(hub.log).list()].count("Show") == 1


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"source_kind": "local_db"}, "locator"),
        ({"locator": dict(LOC)}, "source_kind"),
        ({"host": "edit-pc"}, "host"),
    ],
)
def test_half_a_source_is_refused(hub: Hub, over: dict[str, Any], field: str) -> None:
    assert _refused(hub, "invalid_input", **over).details["field"] == field


# -- attaching the Resolve source later --------------------------------------------------------


def test_attaching_the_resolve_source_later_keeps_one_uid_one_project(hub: Hub) -> None:
    pid = _register(hub)["project_id"]
    locator = ProjectLocator(("Shows",), "Fakes", "uid-1")
    attached = project_register.add_source(
        hub.log,
        HUMAN,
        pid,
        source_kind="local_db",
        locator=dict(LOC),
        host="edit-pc",
        idempotency_key="src-1",
    )
    assert attached.project_id == pid and attached.replayed is False
    assert Registry(hub.log).get(pid).sources == [  # type: ignore[union-attr]
        ProjectSource("local_db", dict(LOC), "edit-pc")
    ]
    assert LocalHubLink(hub.log).hub_project_id(locator) == pid  # `aar node run` resolves it
    # a retry replays
    again = project_register.add_source(
        hub.log,
        HUMAN,
        pid,
        source_kind="local_db",
        locator=dict(LOC),
        host="edit-pc",
        idempotency_key="src-1",
    )
    assert again.replayed is True
    # the uid now belongs to pid: neither another project nor a new registration may take it
    with pytest.raises(WriteError) as err:
        project_register.add_source(
            hub.log,
            HUMAN,
            hub.project_id,
            source_kind="cloud_library",
            locator=dict(LOC),
            idempotency_key="src-2",
        )
    assert err.value.code == "project_exists" and err.value.details["existing_id"] == pid
    taken = _refused(
        hub, "project_exists", key="k9", name="Other", source_kind="local_db", locator=dict(LOC)
    )
    assert taken.details["existing_id"] == pid


def test_a_sourceless_project_cannot_take_a_uid_another_project_holds(hub: Hub) -> None:
    holder = _register(hub, key="s", name="Real", source_kind="local_db", locator=dict(LOC))
    pid = _register(hub)["project_id"]
    with pytest.raises(WriteError) as err:
        project_register.add_source(
            hub.log, HUMAN, pid, source_kind="local_db", locator=dict(LOC), idempotency_key="x"
        )
    assert err.value.code == "project_exists"
    assert err.value.details["existing_id"] == holder["project_id"]
    assert Registry(hub.log).get(pid).sources == []  # type: ignore[union-attr]


# -- job submission ----------------------------------------------------------------------------


def _submit(hub: Hub, project_id: str) -> tuple[jobs.HubJob, bool]:
    return jobs.submit(
        hub.log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=ProjectLocator(("Shows",), "Fakes", project_id),
        agent_id=AGENT.id,
        assigned_node="edit-pc",
        idempotency_key="job-1",
    )


def test_a_resolve_job_for_a_sourceless_project_is_refused_typed(hub: Hub) -> None:
    pid = _register(hub)["project_id"]
    with pytest.raises(jobs.NotAResolveProject) as err:
        _submit(hub, pid)
    assert err.value.code == "not_a_resolve_project"
    assert "no Resolve source" in err.value.message
    assert "aar hub project source" in err.value.fix_hint
    assert jobs.list_jobs(hub.log) == []
    # once the source is attached, the registry id is still not what a job names
    project_register.add_source(
        hub.log, HUMAN, pid, source_kind="local_db", locator=dict(LOC), idempotency_key="s"
    )
    with pytest.raises(jobs.NotAResolveProject) as err:
        _submit(hub, pid)
    assert "uid-1" in err.value.fix_hint
    job, replayed = _submit(hub, "uid-1")  # the Resolve uid is accepted
    assert replayed is False and job.target.project_id == "uid-1"


def test_the_job_refusal_is_typed_at_the_cli_and_the_tool(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    pid = _register(hub)["project_id"]
    project = {"folder_path": ["Shows"], "name": "Fakes", "project_id": pid}
    code = main(
        [
            "hub",
            "job",
            "submit",
            "--db",
            str(hub.root / "hub.db"),
            "--actor",
            "agent:agent-1",
            "--kind",
            "ops.sequence",
            "--params",
            json.dumps(PARAMS),
            "--project",
            json.dumps(project),
            "--agent-id",
            "agent-1",
            "--node",
            "edit-pc",
            "--idempotency-key",
            "cli-job",
        ]
    )
    err = json.loads(capsys.readouterr().err.strip().splitlines()[-1])
    assert code == 2 and err["error"]["code"] == "not_a_resolve_project"
    result = _call(
        hub,
        "job_submit",
        {
            "agent_id": "agent-1",
            "idempotency_key": "mcp-job",
            "kind": "ops.sequence",
            "params": PARAMS,
            "project": project,
            "assigned_node": "edit-pc",
        },
    )
    assert result.is_error is True
    error = result.structured_content["error"]
    assert error["code"] == "not_a_resolve_project"
    assert "aar hub project source" in error["fix_hint"]
    assert jobs.list_jobs(hub.log) == []


# -- CLI ---------------------------------------------------------------------------------------


def _cli(hub: Hub, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, Any]:
    code = main(["hub", "project", argv[0], "--db", str(hub.root / "hub.db"), *argv[1:]])
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    err = json.loads(captured.err.strip().splitlines()[-1]) if captured.err.strip() else None
    return code, out, err


ADD = ["add", "--actor", "agent:agent-1", "--name", "Fakes"]


def test_cli_add_without_a_source_list_show_and_attach(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _cli(hub, capsys, *ADD, "--idempotency-key", "c1")
    assert (code, err) == (0, None) and out["project"]["sources"] == []
    pid = out["project_id"]
    code, again, _ = _cli(hub, capsys, *ADD, "--idempotency-key", "c1")
    assert code == 0 and again["replayed"] is True and again["project_id"] == pid
    code, _, err = _cli(hub, capsys, *ADD, "--idempotency-key", "c2")
    assert code == 2 and err["error"]["code"] == "project_exists"
    assert err["error"]["details"]["existing_id"] == pid

    code, listing, _ = _cli(hub, capsys, "list")
    assert (
        code == 0
        and {"id": pid, "sources": []}.items()
        <= next(p for p in listing["projects"] if p["id"] == pid).items()
    )
    code, shown, _ = _cli(hub, capsys, "show", pid)
    assert code == 0 and shown["project"]["sources"] == []

    code, out, _ = _cli(
        hub,
        capsys,
        "source",
        pid,
        "--actor",
        "agent:agent-1",
        "--source-kind",
        "local_db",
        "--locator",
        json.dumps(LOC),
        "--idempotency-key",
        "c3",
    )
    assert code == 0 and out["project"]["sources"][0]["locator"] == LOC


@pytest.mark.parametrize(
    ("extra", "field"),
    [
        (["--source-kind", "local_db"], "locator"),
        (["--locator", json.dumps(LOC)], "source_kind"),
        (["--locator", "null"], "locator"),
        (["--host", "edit-pc"], "host"),
    ],
)
def test_cli_half_a_source_stays_an_error(
    hub: Hub, capsys: pytest.CaptureFixture[str], extra: list[str], field: str
) -> None:
    code, _, err = _cli(hub, capsys, *ADD, *extra, "--idempotency-key", "h1")
    assert code == 2 and err["error"]["code"] == "invalid_input"
    assert err["error"]["details"]["field"] == field
    assert len(Registry(hub.log).list()) == 2


def test_cli_source_still_requires_a_source(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    db = str(hub.root / "hub.db")
    argv = ["hub", "project", "source", "prj_acme", "--db", db, "--actor", "agent:a"]
    assert main([*argv, "--idempotency-key", "x"]) == 2
    assert "--source-kind, --locator" in capsys.readouterr().err


# -- MCP ---------------------------------------------------------------------------------------


def _call(hub: Hub, name: str, arguments: dict[str, Any]) -> Any:
    server = build_server(hub.log, Registry(hub.log), hub.store, agent_writes=True)

    async def body() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(body())


def test_project_register_tool_without_a_source_and_the_read_tools(hub: Hub) -> None:
    server = build_server(hub.log, Registry(hub.log), hub.store, agent_writes=True)

    async def schema() -> dict[str, Any]:
        async with Client(server) as client:
            tools = (await client.list_tools()).tools
            return next(t for t in tools if t.name == "project_register").input_schema

    assert set(asyncio.run(schema())["required"]) == {"agent_id", "idempotency_key", "name"}

    args = {"agent_id": "agent-1", "idempotency_key": "m1", "name": "Fakes"}
    ok = _call(hub, "project_register", args)
    assert ok.is_error is False, ok.content
    pid = ok.structured_content["project_id"]
    assert ok.structured_content["project"]["sources"] == []
    assert _call(hub, "project_register", args).structured_content["replayed"] is True

    dup = _call(hub, "project_register", {**args, "idempotency_key": "m2"})
    assert dup.is_error is True
    error = dup.structured_content["error"]
    assert error["code"] == "project_exists" and error["details"]["existing_id"] == pid

    half = _call(hub, "project_register", {**args, "idempotency_key": "m3", "locator": LOC})
    assert half.structured_content["error"]["code"] == "invalid_input"
    assert half.structured_content["error"]["details"]["field"] == "source_kind"

    got = _call(hub, "project_get", {"project_id": pid}).structured_content
    assert got["found"] is True and got["project"]["sources"] == []
    listed = _call(hub, "projects_list", {}).structured_content
    assert pid in [p["id"] for p in listed["projects"]]
    board = _call(hub, "project_board", {"project_id": pid})
    assert board.is_error is False and board.structured_content["leases"] == []
