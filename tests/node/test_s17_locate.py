"""S17: ``project.locate`` and ``aar node locate`` read the open project without moving anything."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest

from aar.cli import main
from aar.node import cli as node_cli
from aar.node.gateway import UNVERIFIED_ROOT_REASON, GatewayError, ScriptApiGateway
from aar.node.ops import get_op

from .fake_resolve import FakeProject, FakeResolve
from .ops_support import CONFIG, OpsWorld

MOVES = {"GotoRootFolder", "GotoParentFolder", "OpenFolder", "LoadProject"}


def _fake(folder: tuple[str, ...], browse: tuple[str, ...] | None = None) -> FakeResolve:
    fake = FakeResolve({"Show": ["Main"], "Other": ["x"]}, current="Show", folder=folder)
    if browse is not None:
        fake.human_browse_project_manager(browse)
    return fake


def _no_render_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(self: Any) -> Any:
        raise AssertionError("page signal: render settings must not be read")

    monkeypatch.setattr(FakeProject, "GetCurrentRenderMode", boom)
    monkeypatch.setattr(FakeProject, "GetCurrentRenderFormatAndCodec", boom)


def test_locate_at_the_root_gives_the_full_locator(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _fake(())
    _no_render_reads(monkeypatch)
    where = ScriptApiGateway(fake).locate_open_project()
    project = fake.project("Show")
    assert (where.name, where.project_id) == ("Show", project.GetUniqueId())
    # the root lists the name, but a listing has no ids: a candidate, never a verified path
    assert where.folder_path is None and where.candidate_folder_path == ()
    assert where.folder_path_reason == UNVERIFIED_ROOT_REASON
    assert not MOVES & set(fake.call_names())


def test_locate_in_a_subfolder_does_not_walk_and_says_why(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake(("Shows", "2026"))
    _no_render_reads(monkeypatch)
    where = ScriptApiGateway(fake).locate_open_project()
    assert where.project_id == fake.project("Show").GetUniqueId()
    assert where.folder_path is None and where.pm_folder == "2026"
    assert where.folder_path_reason and "moves the folder" in where.folder_path_reason
    assert not MOVES & set(fake.call_names())
    assert fake.pm_path == ("Shows", "2026")


def test_locate_at_the_root_when_the_project_lives_in_a_subfolder() -> None:
    fake = _fake(("Shows",), browse=())
    where = ScriptApiGateway(fake).locate_open_project()
    assert where.folder_path is None and "subfolder" in (where.folder_path_reason or "")
    assert not MOVES & set(fake.call_names())


def test_a_root_namesake_does_not_verify_a_folder() -> None:
    """The open "Show" lives in Shows/2026; a different "Show" sits at the root. A name in the
    root listing says nothing about which project is open, so only a candidate is offered."""
    fake = _fake(("Shows", "2026"))
    open_project = fake.project("Show")
    fake.add_project("Show", ["other"], ())
    fake.current = open_project
    fake.human_browse_project_manager(())
    assert fake.current.folder == ("Shows", "2026")
    where = ScriptApiGateway(fake).locate_open_project()
    assert where.project_id == open_project.GetUniqueId()
    assert where.folder_path is None
    assert where.candidate_folder_path == ()
    assert where.folder_path_reason == UNVERIFIED_ROOT_REASON
    assert not MOVES & set(fake.call_names())


def test_locate_without_an_open_project_is_a_gateway_error() -> None:
    fake = FakeResolve({"Show": ["Main"]}, current="Show")
    fake.current = None
    with pytest.raises(GatewayError) as err:
        ScriptApiGateway(fake).locate_open_project()
    assert err.value.code == "no_project"


def test_project_locate_op_is_a_hot_read_op_through_the_broker() -> None:
    spec = get_op("project.locate")
    assert (spec.hot, spec.mutates) == (True, False)
    fake = _fake(())
    world = OpsWorld(fake, replace(CONFIG, snapshot_project_manager_path=False))
    result = world.ok("project.locate", {}, target=world.locator())
    assert result.project_id == fake.project("Show").GetUniqueId()
    assert result.folder_path is None and result.locator is None  # never verified from a name
    assert result.candidate_folder_path == () and result.pm_folder == ""
    assert result.folder_path_reason == UNVERIFIED_ROOT_REASON
    assert not MOVES & set(fake.call_names())


def test_project_locate_op_in_a_subfolder_returns_no_locator() -> None:
    fake = _fake(("Shows",))
    world = OpsWorld(fake, replace(CONFIG, snapshot_project_manager_path=False))
    result = world.ok("project.locate", {}, target=world.locator())
    assert result.folder_path is None and result.locator is None
    assert result.candidate_folder_path is None
    assert result.folder_path_reason and result.pm_folder == "Shows"


def test_node_locate_prints_the_candidate_but_no_verified_locator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _fake(())
    code = node_cli._cmd_locate(None, gateway_factory=lambda: ScriptApiGateway(fake))  # type: ignore[arg-type]
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["locator"] is None and out["folder_path"] is None
    assert out["candidate_folder_path"] == []
    assert out["folder_path_reason"] == UNVERIFIED_ROOT_REASON
    assert out["project_id"] == fake.project("Show").GetUniqueId()


def test_node_locate_in_a_subfolder_prints_no_locator_but_a_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = _fake(("Shows",))
    code = node_cli._cmd_locate(None, gateway_factory=lambda: ScriptApiGateway(fake))  # type: ignore[arg-type]
    out = json.loads(capsys.readouterr().out)
    assert code == 0 and out["locator"] is None and out["folder_path"] is None
    assert out["folder_path_reason"] and out["project_id"]
    assert out["candidate_folder_path"] is None


def test_node_locate_reports_an_unavailable_resolve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_resolve() -> Any:
        raise RuntimeError("aar node run requires Windows")

    assert node_cli._cmd_locate(None, gateway_factory=no_resolve) == 2  # type: ignore[arg-type]
    assert "requires Windows" in capsys.readouterr().err


def test_node_locate_is_registered_on_the_cli(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["node", "locate", "--help"]) == 0
    assert "without loading" in capsys.readouterr().out


@pytest.mark.parametrize(("name", "uid"), [("", "uid"), ("Show", ""), ("", "")])
def test_an_unidentified_open_project_is_refused(
    monkeypatch: pytest.MonkeyPatch, name: str, uid: str
) -> None:
    fake = _fake(())
    monkeypatch.setattr(FakeProject, "GetName", lambda self: name)
    monkeypatch.setattr(FakeProject, "GetUniqueId", lambda self: uid)
    with pytest.raises(GatewayError) as err:
        ScriptApiGateway(fake).locate_open_project()
    assert err.value.code == "project_unidentified"


@pytest.mark.parametrize("where", [(), ("Shows",), ("Shows", "2026")])
def test_locate_never_reads_the_render_getters(
    monkeypatch: pytest.MonkeyPatch, where: tuple[str, ...]
) -> None:
    called: list[str] = []
    for getter in ("GetCurrentRenderMode", "GetCurrentRenderFormatAndCodec"):
        monkeypatch.setattr(FakeProject, getter, lambda self, g=getter: called.append(g))
    fake = _fake(where)
    ScriptApiGateway(fake).locate_open_project()
    node_cli._cmd_locate(None, gateway_factory=lambda: ScriptApiGateway(fake))  # type: ignore[arg-type]
    assert called == []
