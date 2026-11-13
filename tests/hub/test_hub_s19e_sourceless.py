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


def test_names_compare_nfc_normalised_and_case_sensitive(hub: Hub) -> None:
    composed = _register(hub, name="Caf\u00e9")["project_id"]
    err = _refused(hub, "project_exists", key="k2", name="Cafe\u0301")  # decomposed spelling
    assert err.details["existing_id"] == composed
    err = _refused(hub, "project_exists", key="k3", name="Other", aliases=["Cafe\u0301"])
    assert err.details["existing_id"] == composed
    assert _register(hub, key="k4", name="caf\u00e9")["ok"]  # case still matters, as on dev


@pytest.mark.parametrize(
    "bad",
    [
        " Fakes",
        "Fakes ",
        "Fakes\u00a0",  # a no-break space is whitespace too
        "Fa\u0085kes",  # C1 control
        "Fa\u200bkes",  # zero-width space
        "Fa\u200dkes",  # zero-width joiner
        "Fa\ufeffkes",  # zero-width no-break space (BOM)
        "Fa\u2028kes",  # line separator
    ],
)
def test_a_sourceless_name_or_alias_must_be_a_plain_show_name(hub: Hub, bad: str) -> None:
    assert _refused(hub, "invalid_input", name=bad).details["field"] == "name"
    assert _refused(hub, "invalid_input", aliases=[bad]).details["field"] == "alias"


def test_a_sourced_registration_keeps_its_name_rules(hub: Hub) -> None:
    """Unchanged from dev: a sourced project may carry surrounding whitespace."""
    out = _register(hub, name=" Fakes ", aliases=["x\u200by"], source_kind="local_db", locator=LOC)
    assert out["project"]["name"] == " Fakes " and out["project"]["aliases"] == ["x\u200by"]


def test_a_sourced_registration_cannot_take_a_sourceless_shows_name(hub: Hub) -> None:
    """S19e-f MED (review finding 8): registering the show's Resolve project under its name
    made a second project, splitting plans, reviews and Resolve jobs between two ids."""
    loose = _register(hub, aliases=["Fakes 5-23"])["project_id"]
    for n, name in enumerate(["Fakes", "Fakes 5-23"]):
        error = _refused(
            hub, "project_exists", key=f"s{n}", name=name, source_kind="local_db", locator=LOC
        )
        assert error.details == {"existing_id": loose, "name": name}
        assert f"`aar hub project source {loose}`" in error.fix_hint
        assert "project_source" in error.fix_hint
    error = _refused(  # an alias of the new project collides too
        hub, "project_exists", key="s2", name="X", aliases=["Fakes"], source_kind="local_db",
        locator=LOC,
    )  # fmt: skip
    assert error.details["existing_id"] == loose
    # the way through: attach the source to the show, which keeps its one id
    attached = project_register.add_source(
        hub.log, AGENT, loose, source_kind="local_db", locator=dict(LOC), idempotency_key="a1"
    )
    assert attached.project_id == loose
    assert [p.id for p in Registry(hub.log).list() if p.name == "Fakes"] == [loose]


def test_a_sourced_registration_retry_replays_after_a_sourceless_show_took_its_name(
    hub: Hub,
) -> None:
    """The name check runs after the retry check, as for a source-less registration."""
    first = _register(hub, key="s", source_kind="local_db", locator=dict(LOC))
    Registry(hub.log).create_project("Fakes", HUMAN)  # a library caller may still share it
    again = _register(hub, key="s", source_kind="local_db", locator=dict(LOC))
    assert again["replayed"] is True and again["project_id"] == first["project_id"]


def test_two_sourced_projects_may_still_share_a_name(hub: Hub) -> None:
    """Unchanged S17 behaviour: sourced projects are told apart by their Resolve ids."""
    first = _register(hub, key="s1", source_kind="local_db", locator=dict(LOC))["project_id"]
    other = {**LOC, "project_id": "uid-2"}
    second = _register(hub, key="s2", source_kind="local_db", locator=other)["project_id"]
    assert first != second


def test_the_registry_refuses_a_twin_only_when_asked(hub: Hub) -> None:
    registry = Registry(hub.log)
    first = registry.create_project("Show", HUMAN, idempotency_key="lib-1", unique_name=True)
    again = registry.create_project("Show", HUMAN, idempotency_key="lib-1", unique_name=True)
    assert again.id == first.id  # a retry replays before the name check
    with pytest.raises(DuplicateProjectName) as err:
        registry.create_project("Show", HUMAN, idempotency_key="lib-2", unique_name=True)
    assert (err.value.existing_id, err.value.name) == (first.id, "Show")
    with pytest.raises(DuplicateProjectName):
        registry.create_project("Show", HUMAN, unique_name=True)
    assert [p.name for p in Registry(hub.log).list()].count("Show") == 1


def test_the_library_default_is_unchanged(hub: Hub) -> None:
    """A library caller (not the agent-facing registration) may still share names and aliases."""
    registry = Registry(hub.log)
    first = registry.create_project("Show", HUMAN, aliases=["alias-0"])
    second = registry.create_project("Show", HUMAN, aliases=["alias-0"])
    assert first.id != second.id
    assert [p.name for p in Registry(hub.log).list()].count("Show") == 2


HALF = "give source_kind and locator together, or neither"


@pytest.mark.parametrize(
    ("over", "field", "says"),
    [
        ({"source_kind": "local_db"}, "locator", f"locator is missing: {HALF}"),
        ({"locator": dict(LOC)}, "source_kind", f"source_kind is missing: {HALF}"),
        ({"host": "edit-pc"}, "host", "host names where a source lives, and no source was given"),
    ],
)
def test_half_a_source_is_refused(hub: Hub, over: dict[str, Any], field: str, says: str) -> None:
    """The half-source refusal, not a bad-source one (build_source names the same field)."""
    err = _refused(hub, "invalid_input", **over)
    assert err.details["field"] == field
    assert err.message == says


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


UID = "8f3c2a10-1111-2222-3333-444455556666"


def _submit_key(hub: Hub, project_id: str, key: str) -> tuple[jobs.HubJob, bool]:
    return jobs.submit(
        hub.log,
        AGENT,
        kind="ops.sequence",
        params=PARAMS,
        project=ProjectLocator(("Shows",), "Show", project_id),
        agent_id=AGENT.id,
        assigned_node="edit-pc",
        idempotency_key=key,
    )


def test_a_registry_id_that_is_also_a_projects_resolve_uid_is_accepted(hub: Hub) -> None:
    """Probe p5: a sourced project registered with its own Resolve uid as its registry id."""
    loc = {"project_id": UID, "name": "Show", "folder_path": ["Shows"]}
    _register(hub, name="Show", source_kind="local_db", locator=loc, project_id=UID)
    job, replayed = _submit_key(hub, UID, "j1")
    assert replayed is False and job.target.project_id == UID


def test_a_retry_of_an_accepted_job_replays_after_the_id_became_a_registry_id(hub: Hub) -> None:
    """Probe p4: the job is accepted first, then a source-less project takes the uid as its id."""
    first, _ = _submit_key(hub, UID, "j1")
    _register(hub, name="Show", project_id=UID)
    again, replayed = _submit_key(hub, UID, "j1")  # the same request: replayed, not refused
    assert replayed is True and again.job_id == first.job_id
    with pytest.raises(jobs.NotAResolveProject):  # a new job for it is refused
        _submit_key(hub, UID, "j2")
    # once some project lists the text as its Resolve uid, it is a Resolve uid again
    loc = {"project_id": UID, "name": "Show", "folder_path": ["Shows"]}
    project_register.add_source(
        hub.log, HUMAN, UID, source_kind="local_db", locator=loc, idempotency_key="s1"
    )
    job, replayed = _submit_key(hub, UID, "j3")
    assert replayed is False and job.target.project_id == UID


def test_the_refusal_hint_names_only_other_uids(hub: Hub) -> None:
    pid = _register(hub, name="Show", source_kind="local_db", locator=dict(LOC))["project_id"]
    with pytest.raises(jobs.NotAResolveProject) as err:
        _submit_key(hub, pid, "j1")
    assert "uid-1" in err.value.fix_hint and pid not in err.value.fix_hint


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
    assert half.structured_content["error"]["message"] == f"source_kind is missing: {HALF}"

    got = _call(hub, "project_get", {"project_id": pid}).structured_content
    assert got["found"] is True and got["project"]["sources"] == []
    listed = _call(hub, "projects_list", {}).structured_content
    assert pid in [p["id"] for p in listed["projects"]]
    board = _call(hub, "project_board", {"project_id": pid})
    assert board.is_error is False and board.structured_content["leases"] == []


def test_project_source_tool_attaches_the_source_a_sourced_register_is_pointed_to(
    hub: Hub,
) -> None:
    """An MCP-only agent had no way to attach a source, so it registered the show twice."""
    base = {"agent_id": "agent-1", "name": "Fakes"}
    pid = _call(hub, "project_register", {**base, "idempotency_key": "m1"}).structured_content[
        "project_id"
    ]
    source = {"source_kind": "local_db", "locator": LOC}
    twin = _call(hub, "project_register", {**base, **source, "idempotency_key": "m2"})
    error = twin.structured_content["error"]
    assert twin.is_error is True and error["code"] == "project_exists"
    assert error["details"]["existing_id"] == pid and "project_source" in error["fix_hint"]

    args = {"agent_id": "agent-1", "idempotency_key": "m3", "project_id": pid, **source}
    ok = _call(hub, "project_source", args)
    assert ok.is_error is False, ok.content
    assert ok.structured_content["op"] == "project.source"
    assert ok.structured_content["project"]["sources"] == [
        {"kind": "local_db", "locator": LOC, "host": None}
    ]
    assert _call(hub, "project_source", args).structured_content["replayed"] is True
    assert [p.id for p in Registry(hub.log).list() if p.name == "Fakes"] == [pid]

    other = _call(hub, "project_source", {**args, "idempotency_key": "m4", "project_id": "prj_x"})
    assert other.structured_content["error"]["code"] == "unknown_project"
    taken = _call(
        hub, "project_source", {**args, "idempotency_key": "m5", "project_id": hub.project_id}
    )
    assert taken.structured_content["error"]["code"] == "project_exists"
    assert taken.structured_content["error"]["details"]["existing_id"] == pid


# -- rename and alias keep a source-less project's name its own (probe p1) ---------------------


def _change(hub: Hub, capsys: pytest.CaptureFixture[str], verb: str, *argv: str) -> Any:
    return _cli(hub, capsys, verb, *argv[:2], "--actor", "agent:agent-1", *argv[2:])


def _add(hub: Hub, capsys: pytest.CaptureFixture[str], name: str, *extra: str) -> str:
    code, out, err = _cli(
        hub, capsys, "add", "--actor", "agent:agent-1", "--name", name, *extra,
        "--idempotency-key", "add-" + name.encode().hex(),
    )  # fmt: skip
    assert code == 0, err
    return str(out["project_id"])


def test_cli_rename_and_alias_cannot_make_a_twin_of_a_sourceless_name(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    show = _add(hub, capsys, "Show")
    other = _add(hub, capsys, "Other")
    fourth = _add(hub, capsys, "Fourth")
    sourced = _add(hub, capsys, "Real", "--source-kind", "local_db", "--locator", json.dumps(LOC))
    before = hub.counts()
    for verb, pid in (
        ("rename", other),
        ("alias", fourth),
        ("rename", sourced),
        ("alias", sourced),
    ):
        code, _, err = _change(hub, capsys, verb, pid, "Show", "--idempotency-key", f"{verb}{pid}")
        assert code == 2 and err["error"]["code"] == "project_exists", (verb, pid, err)
        assert err["error"]["details"] == {"existing_id": show, "name": "Show"}
    # a source-less project may not take a name a sourced project holds either
    code, _, err = _change(hub, capsys, "rename", other, "Real", "--idempotency-key", "r-real")
    assert code == 2 and err["error"]["details"]["existing_id"] == sourced
    assert hub.counts() == before
    holders = [p for p in Registry(hub.log).list() if "Show" in (p.name, *p.aliases)]
    assert [p.id for p in holders] == [show]


def test_cli_rename_to_its_own_name_and_an_alias_it_has_work_as_before(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    show = _add(hub, capsys, "Show")
    assert _change(hub, capsys, "alias", show, "Show B", "--idempotency-key", "a1")[0] == 0
    assert _change(hub, capsys, "alias", show, "Show B", "--idempotency-key", "a2")[0] == 0
    assert _change(hub, capsys, "rename", show, "Show", "--idempotency-key", "r1")[0] == 0
    code, out, _ = _change(hub, capsys, "rename", show, "Show B", "--idempotency-key", "r2")
    assert code == 0 and out["project"]["name"] == "Show B"
    # a sourced project may still share a name with another sourced project, as on dev
    loc2 = {**LOC, "project_id": "uid-2"}
    one = _add(hub, capsys, "Real", "--source-kind", "local_db", "--locator", json.dumps(LOC))
    two = _add(hub, capsys, "Real 2", "--source-kind", "local_db", "--locator", json.dumps(loc2))
    assert _change(hub, capsys, "rename", two, "Real", "--idempotency-key", "r3")[0] == 0
    assert _change(hub, capsys, "alias", one, "Real 2", "--idempotency-key", "a3")[0] == 0


def test_the_library_rename_and_alias_are_unchanged_by_default(hub: Hub) -> None:
    registry = Registry(hub.log)
    show = registry.create_project("Show", HUMAN)
    other = registry.create_project("Other", HUMAN)
    assert registry.rename_project(other.id, "Show", HUMAN).name == "Show"
    assert registry.add_alias("prj_acme", "Show", HUMAN).aliases == ["Show"]
    with pytest.raises(DuplicateProjectName) as err:
        registry.add_alias("prj_other", "Show", HUMAN, unique_name=True)
    assert err.value.existing_id in {show.id, other.id, "prj_acme"}


@pytest.mark.parametrize("verb", ["rename", "alias"])
def test_cli_rename_and_alias_refuse_what_a_show_name_does_not_need(
    hub: Hub, capsys: pytest.CaptureFixture[str], verb: str
) -> None:
    show = _add(hub, capsys, "Caf\u00e9")
    other = _add(hub, capsys, "Other")
    before = hub.counts()
    for i, bad in enumerate(["Other ", "Ot\u200bher", "Ot\u0085her"]):
        code, _, err = _change(hub, capsys, verb, other, bad, "--idempotency-key", f"b{i}")
        assert code == 2 and err["error"]["code"] == "invalid_input", (bad, err)
    code, _, err = _change(hub, capsys, verb, other, "Cafe\u0301", "--idempotency-key", "nfd")
    assert code == 2 and err["error"]["details"]["existing_id"] == show
    assert hub.counts() == before
