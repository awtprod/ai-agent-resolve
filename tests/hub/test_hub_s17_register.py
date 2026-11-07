"""S17: register Resolve projects in the hub (library, CLI and the MCP tool)."""

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
from aar.hub import project_register
from aar.hub.events import Actor
from aar.hub.mcp_server import build_server
from aar.hub.registry import ProjectSource, Registry
from aar.hub.writes import WriteError
from aar.node.hublink import LocalHubLink
from aar.node.locator import ProjectLocator

LOC = {"project_id": "uid-1", "name": "Show", "folder_path": ["Shows", "2026"]}


@pytest.fixture
def hub(tmp_path: Path) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _register(hub: Hub, key: str = "k1", **over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "name": "Show",
        "source_kind": "local_db",
        "locator": dict(LOC),
        "idempotency_key": key,
    }
    args.update(over)
    actor = args.pop("actor", HUMAN)
    return project_register.register_project(hub.log, actor, **args).to_dict()


def _refused(hub: Hub, code: str, **over: Any) -> WriteError:
    with pytest.raises(WriteError) as err:
        _register(hub, **over)
    assert err.value.code == code, err.value
    return err.value


def test_register_creates_a_project_the_node_can_resolve(hub: Hub) -> None:
    out = _register(hub, aliases=["The Show"], host="edit-pc")
    assert out["ok"] and out["op"] == "project.register" and out["replayed"] is False
    stored = Registry(hub.log).get(out["project_id"])
    assert stored is not None and stored.name == "Show" and stored.aliases == ["The Show"]
    assert stored.sources == [ProjectSource("local_db", dict(LOC), "edit-pc")]
    link = LocalHubLink(hub.log)
    locator = ProjectLocator(("Shows", "2026"), "Show", "uid-1")
    assert link.hub_project_id(locator) == stored.id  # what `aar node run` needs


def test_a_retry_replays_and_a_changed_retry_conflicts(hub: Hub) -> None:
    first = _register(hub)
    again = _register(hub)
    assert again["replayed"] is True and again["project_id"] == first["project_id"]
    assert len(Registry(hub.log).list()) == 3  # two fixtures + one
    _refused(hub, "idempotency_conflict", name="Other name")


def test_second_registration_of_the_same_resolve_uid_names_the_existing_id(hub: Hub) -> None:
    first = _register(hub)
    err = _refused(hub, "project_exists", key="k2", name="Show again")
    assert err.details["existing_id"] == first["project_id"]
    assert err.details["resolve_uid"] == "uid-1"
    assert first["project_id"] in err.message
    # also when another folder path or kind carries the same uid
    _refused(hub, "project_exists", key="k3", locator={**LOC, "folder_path": []})
    _refused(hub, "project_exists", key="k4", source_kind="cloud_library")
    assert len(Registry(hub.log).list()) == 3


def test_a_drp_file_is_not_a_resolve_uid_and_may_repeat(hub: Hub) -> None:
    _register(hub, key="a", source_kind="drp_file", locator={"path": "/x/a.drp"})
    _register(hub, key="b", source_kind="drp_file", locator={"path": "/x/a.drp"})


@pytest.mark.parametrize(
    "locator",
    [
        {"name": "Show", "folder_path": []},
        {"project_id": "u", "folder_path": []},
        {"project_id": "u", "name": "Show"},
        {"project_id": "", "name": "Show", "folder_path": []},
        {"project_id": None, "name": "Show", "folder_path": []},
        {"project_id": 7, "name": "Show", "folder_path": []},
        {"project_id": "u", "name": "", "folder_path": []},
        {"project_id": "u", "name": 3, "folder_path": []},
        {"project_id": "u", "name": "Show", "folder_path": "Shows"},
        {"project_id": "u", "name": "Show", "folder_path": ["ok", ""]},
        {"project_id": "u", "name": "Show", "folder_path": ["ok", 1]},
        {"project_id": "u", "name": "Show", "folder_path": [], "extra": 1},
        ["not", "an", "object"],
        "text",
        None,
    ],
)
@pytest.mark.parametrize("kind", ["local_db", "cloud_library"])
def test_resolve_locators_are_validated_at_the_boundary(hub: Hub, locator: Any, kind: str) -> None:
    before = hub.counts()
    err = _refused(hub, "invalid_input", locator=locator, source_kind=kind)
    assert err.details["field"] == "locator"
    assert hub.counts() == before


@pytest.mark.parametrize(
    ("over", "field"),
    [
        ({"source_kind": "ftp"}, "source_kind"),
        ({"source_kind": None}, "source_kind"),
        ({"name": ""}, "name"),
        ({"name": 5}, "name"),
        ({"name": "x" * 501}, "name"),
        ({"host": ""}, "host"),
        ({"host": 3}, "host"),
        ({"aliases": "abc"}, "aliases"),
        ({"aliases": [""]}, "alias"),
        ({"aliases": [1]}, "alias"),
        ({"aliases": ["a"] * 65}, "aliases"),
        ({"project_id": ""}, "project_id"),
        ({"idempotency_key": "bad key!"}, "idempotency_key"),
    ],
)
def test_other_fields_are_validated(hub: Hub, over: dict[str, Any], field: str) -> None:
    before = hub.counts()
    err = _refused(hub, "invalid_input", **over)
    assert err.details["field"] == field
    assert hub.counts() == before


def test_a_node_may_not_register_but_a_human_and_an_agent_may(hub: Hub) -> None:
    _refused(hub, "actor_not_allowed", actor=Actor("node", "edit-pc"))
    assert _register(hub, key="h", actor=HUMAN)["ok"]
    out = project_register.register_project(
        hub.log,
        AGENT,
        name="Other",
        source_kind="local_db",
        locator={**LOC, "project_id": "uid-2"},
        idempotency_key="g",
    )
    assert out.replayed is False


def test_an_explicit_project_id_is_kept_and_not_reused(hub: Hub) -> None:
    out = _register(hub, project_id="prj_mine")
    assert out["project_id"] == "prj_mine"
    err = _refused(
        hub, "project_exists", key="z", project_id="prj_mine", locator={**LOC, "project_id": "u9"}
    )
    assert err.details["existing_id"] == "prj_mine"


def test_alias_rename_and_source_changes(hub: Hub) -> None:
    pid = _register(hub)["project_id"]
    log = hub.log
    assert project_register.add_alias(log, HUMAN, pid, "Alt", idempotency_key="a1").to_dict()[
        "project"
    ]["aliases"] == ["Alt"]
    renamed = project_register.rename(log, HUMAN, pid, "Renamed", idempotency_key="r1")
    assert renamed.body["project"]["name"] == "Renamed" and renamed.project_id == pid
    again = project_register.rename(log, HUMAN, pid, "Renamed", idempotency_key="r1")
    assert again.replayed is True
    added = project_register.add_source(
        log,
        HUMAN,
        pid,
        source_kind="drp_file",
        locator={"path": "/x/show.drp"},
        host="edit-pc",
        idempotency_key="s1",
    )
    assert len(added.body["project"]["sources"]) == 2
    with pytest.raises(WriteError) as err:
        project_register.add_alias(log, HUMAN, "prj_nope", "x", idempotency_key="a2")
    assert err.value.code == "unknown_project"
    with pytest.raises(WriteError) as err:
        project_register.rename(log, Actor("node", "n"), pid, "x", idempotency_key="r2")
    assert err.value.code == "actor_not_allowed"


def test_adding_a_source_with_another_projects_uid_is_refused(hub: Hub) -> None:
    first = _register(hub)["project_id"]
    with pytest.raises(WriteError) as err:
        project_register.add_source(
            hub.log,
            HUMAN,
            hub.project_id,
            source_kind="local_db",
            locator=dict(LOC),
            idempotency_key="s9",
        )
    assert err.value.code == "project_exists" and err.value.details["existing_id"] == first
    # the owner may add the same uid again (a no-op fold), and a retry replays
    project_register.add_source(
        hub.log, HUMAN, first, source_kind="local_db", locator=dict(LOC), idempotency_key="s10"
    )


# -- CLI ---------------------------------------------------------------------------------------


def _cli(hub: Hub, capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, Any, Any]:
    code = main(["hub", "project", argv[0], "--db", str(hub.root / "hub.db"), *argv[1:]])
    captured = capsys.readouterr()
    out = json.loads(captured.out) if captured.out.strip() else None
    err = json.loads(captured.err.strip().splitlines()[-1]) if captured.err.strip() else None
    return code, out, err


def _add_args(key: str = "k1", locator: str | None = None) -> list[str]:
    return [
        "add",
        "--actor",
        "human:andrew",
        "--name",
        "Show",
        "--source-kind",
        "local_db",
        "--locator",
        locator or json.dumps(LOC),
        "--host",
        "edit-pc",
        "--alias",
        "A1",
        "--alias",
        "A2",
        "--idempotency-key",
        key,
    ]


def test_cli_add_list_show_alias_rename_source(
    hub: Hub, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, err = _cli(hub, capsys, *_add_args())
    assert (code, err) == (0, None) and out["replayed"] is False
    pid = out["project_id"]
    assert out["project"]["aliases"] == ["A1", "A2"]
    assert out["project"]["sources"][0]["host"] == "edit-pc"
    code, again, _ = _cli(hub, capsys, *_add_args())
    assert code == 0 and again["replayed"] is True

    code, listing, _ = _cli(hub, capsys, "list")
    assert code == 0 and pid in [p["id"] for p in listing["projects"]]
    code, shown, _ = _cli(hub, capsys, "show", pid)
    assert code == 0 and shown["project"]["id"] == pid
    code, _, err = _cli(hub, capsys, "show", "prj_nope")
    assert code == 1 and err["error"]["code"] == "unknown_project"

    base = ["--actor", "human:andrew", "--idempotency-key"]
    code, out, _ = _cli(hub, capsys, "alias", pid, "Alt", *base, "a")
    assert code == 0 and "Alt" in out["project"]["aliases"]
    code, out, _ = _cli(hub, capsys, "rename", pid, "New", *base, "r")
    assert code == 0 and out["project"]["name"] == "New"
    code, out, _ = _cli(
        hub,
        capsys,
        "source",
        pid,
        "--source-kind",
        "drp_file",
        "--locator",
        '{"path": "/a.drp"}',
        *base,
        "s",
    )
    assert code == 0 and len(out["project"]["sources"]) == 2


def test_cli_refusals_are_typed_and_exit_2(hub: Hub, capsys: pytest.CaptureFixture[str]) -> None:
    code, out, _ = _cli(hub, capsys, *_add_args())
    assert code == 0
    code, _, err = _cli(hub, capsys, *_add_args("k2"))
    assert code == 2 and err["error"]["code"] == "project_exists"
    assert err["error"]["details"]["existing_id"] == out["project_id"]
    code, _, err = _cli(hub, capsys, *_add_args("k3", locator="{not json"))
    assert code == 2 and err["error"]["code"] == "invalid_input"
    code, _, err = _cli(hub, capsys, *_add_args("k4", locator='{"name": "x"}'))
    assert code == 2 and err["error"]["code"] == "invalid_input"
    args = _add_args("k5")
    args[args.index("human:andrew")] = "node:edit-pc"
    code, _, err = _cli(hub, capsys, *args)
    assert code == 2 and err["error"]["code"] == "actor_not_allowed"
    args[args.index("node:edit-pc")] = "nonsense"
    code, _, err = _cli(hub, capsys, *args)
    assert code == 2 and err["error"]["code"] == "invalid_actor"
    assert len(Registry(hub.log).list()) == 3


def test_cli_without_a_database_is_a_typed_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["hub", "project", "list", "--db", str(tmp_path / "missing.db")])
    assert code == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "database_unavailable"


# -- MCP ---------------------------------------------------------------------------------------


def _call(hub: Hub, name: str, arguments: dict[str, Any], **kw: Any) -> Any:
    server = build_server(hub.log, Registry(hub.log), hub.store, **kw)

    async def body() -> Any:
        async with Client(server) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(body())


def _tool_args(key: str = "m1", **over: Any) -> dict[str, Any]:
    args: dict[str, Any] = {
        "agent_id": "agent-1",
        "idempotency_key": key,
        "name": "Show",
        "source_kind": "local_db",
        "locator": dict(LOC),
    }
    args.update(over)
    return args


def test_project_register_tool_is_only_under_agent_writes(hub: Hub) -> None:
    server = build_server(hub.log, Registry(hub.log), hub.store)

    async def names() -> set[str]:
        async with Client(server) as client:
            return {t.name for t in (await client.list_tools()).tools}

    assert "project_register" not in asyncio.run(names())
    result = _call(hub, "project_register", _tool_args(), agent_writes=False)
    assert result.is_error is True
    assert len(Registry(hub.log).list()) == 2


def test_project_register_tool_registers_replays_and_refuses(hub: Hub) -> None:
    ok = _call(
        hub, "project_register", _tool_args(aliases=["Alt"], host="edit-pc"), agent_writes=True
    )
    assert ok.is_error is False, ok.content
    body = ok.structured_content
    assert body["op"] == "project.register" and body["replayed"] is False
    pid = body["project_id"]
    assert body["project"]["sources"][0]["locator"] == LOC
    stored = Registry(hub.log).get(pid)
    assert stored is not None and stored.created is not None
    assert [e.actor for e in hub.log.iter_events(types=["project.created"])][-1] == Actor(
        "agent", "agent-1"
    )

    again = _call(
        hub, "project_register", _tool_args(aliases=["Alt"], host="edit-pc"), agent_writes=True
    )
    assert again.structured_content["replayed"] is True

    dup = _call(hub, "project_register", _tool_args("m2"), agent_writes=True)
    assert dup.is_error is True
    error = dup.structured_content["error"]
    assert error["code"] == "project_exists" and error["details"]["existing_id"] == pid

    bad = _call(hub, "project_register", _tool_args("m3", locator={"name": "x"}), agent_writes=True)
    assert bad.is_error is True and bad.structured_content["error"]["code"] == "invalid_input"

    badagent = _call(hub, "project_register", _tool_args("m4", agent_id="a b"), agent_writes=True)
    assert badagent.structured_content["error"]["code"] == "invalid_input"
    assert len(Registry(hub.log).list()) == 3
