"""``ScriptApiGateway`` against the fake Resolve objects."""

from __future__ import annotations

import pytest

from aar.node.gateway import (
    FolderRef,
    GatewayError,
    PMFolderListing,
    ResolveContext,
    ResolveGateway,
    ScriptApiGateway,
)
from aar.node.locator import ProjectLocator

from .fake_resolve import FakeResolve


@pytest.fixture
def fake() -> FakeResolve:
    return FakeResolve({"A": ["A main", "A alt"], "B": ["B main"]}, current="A")


@pytest.fixture
def gw(fake: FakeResolve) -> ScriptApiGateway:
    return ScriptApiGateway(fake)


def loc(fake: FakeResolve, name: str) -> ProjectLocator:
    project = fake.project(name)
    return ProjectLocator(project.folder, name, project.GetUniqueId())


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
    gw.load_project(loc(fake, "A"))
    assert fake.calls == []
    gw.load_project(loc(fake, "B"))
    assert gw.get_context().project == "B"
    assert fake.calls == [("GotoRootFolder",), ("OpenFolder", "CCN"), ("LoadProject", "B")]


def test_load_project_unknown_raises_project_not_found(gw: ScriptApiGateway) -> None:
    with pytest.raises(GatewayError) as err:
        gw.load_project(ProjectLocator(("CCN",), "Nope"))
    assert err.value.code == "project_not_found"


def test_load_project_verifies_the_project_really_changed(
    fake: FakeResolve, gw: ScriptApiGateway
) -> None:
    fake.load_silently_fails.add("B")
    with pytest.raises(GatewayError) as err:
        gw.load_project(loc(fake, "B"))
    assert err.value.code == "project_load_failed"
    assert gw.get_context().project == "A"


# -- Project Manager folders (trap pm-folder-scoped-load) ---------------------------------------


@pytest.fixture
def library() -> FakeResolve:
    """Same-named projects in two folders; the human sits in one of them."""
    fake = FakeResolve(
        {"Desk": ["Desk main"]},
        current="Desk",
        folder=("Shows", "Season 2"),
    )
    fake.add_project("Episode 1", ["E1 main"], ("Shows", "Season 1"))
    fake.add_project("Episode 1", ["E1 main"], ("Shows", "Season 2"))
    fake.add_pm_folder(("Archive",))
    fake.calls.clear()
    return fake


def test_pm_path_at_the_root_reads_without_navigating(fake: FakeResolve) -> None:
    fake.human_browse_project_manager(())
    assert ScriptApiGateway(fake).get_project_manager_path() == ()
    assert fake.calls == []


def test_pm_path_walks_up_and_back_to_where_it_began(library: FakeResolve) -> None:
    gw = ScriptApiGateway(library)
    assert gw.get_project_manager_path() == ("Shows", "Season 2")
    assert library.pm_path == ("Shows", "Season 2")
    assert library.call_names() == [
        "GotoParentFolder",
        "GotoParentFolder",
        "OpenFolder",
        "OpenFolder",
    ]


def test_pm_path_reports_a_walk_that_cannot_get_back(library: FakeResolve) -> None:
    pm = library.GetProjectManager()
    real_open = pm.OpenFolder
    pm.OpenFolder = lambda name: False if name == "Season 2" else real_open(name)  # type: ignore[method-assign]
    with pytest.raises(GatewayError) as err:
        ScriptApiGateway(library).get_project_manager_path()
    assert err.value.code == "pm_open_failed"


def test_open_pm_path_and_listing(library: FakeResolve) -> None:
    gw = ScriptApiGateway(library)
    gw.open_project_manager_path(("Shows", "Season 1"))
    assert library.pm_path == ("Shows", "Season 1")
    assert gw.list_project_manager_folder() == PMFolderListing((), ("Episode 1",), ())
    gw.open_project_manager_path(())
    assert gw.list_project_manager_folder().folders == ("Archive", "Shows")
    with pytest.raises(GatewayError) as err:
        gw.open_project_manager_path(("Shows", "Season 9"))
    assert err.value.code == "pm_open_failed"


def test_walk_lists_every_folder_and_restores_the_path(library: FakeResolve) -> None:
    gw = ScriptApiGateway(library)
    listings = gw.walk_project_manager()
    assert [(x.path, x.projects) for x in listings] == [
        ((), ()),
        (("Archive",), ()),
        (("Shows",), ()),
        (("Shows", "Season 1"), ("Episode 1",)),
        (("Shows", "Season 2"), ("Desk", "Episode 1")),
    ]
    assert library.pm_path == ("Shows", "Season 2")
    assert gw.find_project_folders("Episode 1") == [("Shows", "Season 1"), ("Shows", "Season 2")]


def test_load_project_navigates_to_the_locator_folder_and_checks_the_id(
    library: FakeResolve,
) -> None:
    season1 = next(p for p in library.all_projects if p.folder == ("Shows", "Season 1"))
    gw = ScriptApiGateway(library)
    # The Project Manager sits in Season 2, which has its own "Episode 1": a load by name from
    # here would open the wrong one.
    gw.load_project(ProjectLocator(("Shows", "Season 1"), "Episode 1", season1.GetUniqueId()))
    ctx = gw.get_context(with_pm_path=True)
    assert ctx.project_id == season1.GetUniqueId()
    assert ctx.project_manager_path == ("Shows", "Season 1")


def test_load_project_refuses_a_project_whose_id_differs(library: FakeResolve) -> None:
    gw = ScriptApiGateway(library)
    with pytest.raises(GatewayError) as err:
        gw.load_project(ProjectLocator(("Shows", "Season 2"), "Episode 1", "proj-from-elsewhere"))
    assert err.value.code == "project_id_mismatch"


def test_locator_matches_by_id_when_it_has_one() -> None:
    by_id = ProjectLocator(("CCN",), "Show", "id-1")
    assert by_id.matches("Renamed show", "id-1")
    assert not by_id.matches("Show", "id-2")
    by_name = ProjectLocator(("CCN",), "Show")
    assert by_name.matches("Show", "anything")
    assert ProjectLocator(("a/b",), "c").key != ProjectLocator(("a",), "b/c").key
    assert ProjectLocator.from_dict(by_id.to_dict()) == by_id
    with pytest.raises(ValueError):
        ProjectLocator(("",), "Show")


def test_context_diff_compares_pm_paths_only_when_both_are_measured() -> None:
    a = ResolveContext(project="A", project_manager_path=("CCN",))
    assert a.diff(ResolveContext(project="A", project_manager_path=("Archive",))) == {
        "project_manager_path": (("CCN",), ("Archive",))
    }
    assert a.same_position(ResolveContext(project="A"))  # not measured is not a difference


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
