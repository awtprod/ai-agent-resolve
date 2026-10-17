"""``ScriptApiGateway`` against the fake Resolve objects."""

from __future__ import annotations

import pytest

from aar.node.gateway import (
    FolderRef,
    GatewayError,
    ResolveContext,
    ResolveGateway,
    ScriptApiGateway,
)

from .fake_resolve import FakeResolve


@pytest.fixture
def fake() -> FakeResolve:
    return FakeResolve({"A": ["A main", "A alt"], "B": ["B main"]}, current="A")


@pytest.fixture
def gw(fake: FakeResolve) -> ScriptApiGateway:
    return ScriptApiGateway(fake)


def test_gateway_satisfies_protocol(gw: ScriptApiGateway) -> None:
    assert isinstance(gw, ResolveGateway)


def test_get_context_reads_every_field_without_mutating(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    project = fake.project("A")
    project.SetCurrentTimeline(project.timeline("A alt"))
    project.timeline("A alt").timecode = "01:02:03:04"
    project.media_pool.current = project.media_pool.folder("Footage", "Day 2")
    fake.page = "color"
    fake.calls.clear()

    ctx = gw.get_context()

    assert ctx == ResolveContext(
        project="A",
        project_id=project.GetUniqueId(),
        timeline="A alt",
        timeline_id=project.timeline("A alt").GetUniqueId(),
        page="color",
        playhead="01:02:03:04",
        render_format="mp4",
        render_codec="H264",
        render_mode=1,
        folder=FolderRef("Day 2", project.media_pool.folder("Footage", "Day 2").GetUniqueId()),
        project_manager_folder="CCN",
        rendering=False,
    )
    assert fake.calls == []


def test_get_context_with_no_project_open(fake: FakeResolve, gw: ScriptApiGateway) -> None:
    fake.current = None
    ctx = gw.get_context()
    assert ctx.project is None
    assert ctx.timeline is None
    assert ctx.page == "edit"


def test_load_project_switches_and_is_a_noop_when_already_current(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    gw.load_project("A")
    assert "LoadProject" not in fake.call_names()
    gw.load_project("B")
    assert gw.get_context().project == "B"
    assert fake.call_names() == ["LoadProject"]


def test_load_project_unknown_raises_project_not_found(gw: ScriptApiGateway) -> None:
    with pytest.raises(GatewayError) as err:
        gw.load_project("Nope")
    assert err.value.code == "project_not_found"


def test_load_project_verifies_the_project_really_changed(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    fake.load_silently_fails.add("B")
    with pytest.raises(GatewayError) as err:
        gw.load_project("B")
    assert err.value.code == "project_load_failed"
    assert gw.get_context().project == "A"


def test_set_current_timeline_by_name_and_by_unique_id(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    gw.set_current_timeline("A alt")
    assert gw.get_context().timeline == "A alt"
    main_id = fake.project("A").timeline("A main").GetUniqueId()
    gw.set_current_timeline("renamed meanwhile", unique_id=main_id)
    assert gw.get_context().timeline_id == main_id


def test_set_current_timeline_missing_and_ambiguous(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    with pytest.raises(GatewayError) as err:
        gw.set_current_timeline("nope")
    assert err.value.code == "timeline_not_found"

    duplicate = FakeResolve({"X": ["dup", "dup"]})
    with pytest.raises(GatewayError) as err:
        ScriptApiGateway(duplicate).set_current_timeline("dup")
    assert err.value.code == "timeline_ambiguous"


def test_set_current_folder_finds_nested_folder_by_id(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    day1 = fake.project("A").media_pool.folder("Footage", "Day 1")
    gw.set_current_folder(FolderRef("Day 1", day1.GetUniqueId()))
    assert gw.get_context().folder == FolderRef("Day 1", day1.GetUniqueId())
    with pytest.raises(GatewayError) as err:
        gw.set_current_folder(FolderRef("Nowhere", "no-such-id"))
    assert err.value.code == "folder_not_found"


def test_open_page_validates_names(fake: FakeResolve, gw: ScriptApiGateway) -> None:
    gw.open_page("fairlight")
    assert fake.page == "fairlight"
    with pytest.raises(GatewayError) as err:
        gw.open_page("Edit")
    assert err.value.code == "page_invalid"


def test_set_playhead_rejects_bad_timecode(gw: ScriptApiGateway) -> None:
    gw.set_playhead("01:00:05:00")
    assert gw.get_context().playhead == "01:00:05:00"
    with pytest.raises(GatewayError) as err:
        gw.set_playhead("five seconds")
    assert err.value.code == "playhead_set_failed"


def test_render_preset_round_trip_restores_clobbered_deliver_settings(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    project = fake.project("A")
    project.SetRenderSettings({"TargetDir": "E:/Human", "CustomName": "cut v3"})
    before = project.deliver.copy()

    gw.save_render_preset("aar-restore-test")
    gw.load_render_preset("YouTube 1080p")
    assert project.deliver.settings["TargetDir"] == "C:/Exports"  # clobbered

    gw.load_render_preset("aar-restore-test")
    gw.delete_render_preset("aar-restore-test")
    assert project.deliver == before
    assert "aar-restore-test" not in project.GetRenderPresetList()
    with pytest.raises(GatewayError) as err:
        gw.delete_render_preset("aar-restore-test")
    assert err.value.code == "preset_delete_failed"


def test_save_project_failure_is_reported(fake: FakeResolve, gw: ScriptApiGateway) -> None:
    gw.save_project()
    assert fake.project("A").saves == 1
    fake.save_fails = True
    with pytest.raises(GatewayError) as err:
        gw.save_project()
    assert err.value.code == "save_failed"


def test_none_resolve_is_rejected() -> None:
    with pytest.raises(GatewayError) as err:
        ScriptApiGateway(None)
    assert err.value.code == "no_resolve"


def test_context_diff_ignores_render_state() -> None:
    a = ResolveContext(project="A", playhead="01:00:00:00", rendering=False)
    b = ResolveContext(project="A", playhead="01:00:01:00", rendering=True)
    assert a.diff(b) == {"playhead": ("01:00:00:00", "01:00:01:00")}
    assert a.same_position(ResolveContext(project="A", playhead="01:00:00:00", rendering=True))
