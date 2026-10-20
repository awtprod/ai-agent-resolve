"""S13h1 build ops over the fake: media.import, track.ensure, transition.add, item.set_color,
item.set_transform and title.place.

Each op: the happy path, ``present`` on a rerun with no mutating call, every error code, the
bounds of its params, and each fake fault on its write path (the op reports the stable code
and never claims success). The branch guard, the AST pins and the ops.sequence run are in
``test_ops_build_pins``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
from pydantic import ValidationError

from aar.node.ops import get_op
from aar.node.ops.build import IMPORT_GUARD_EXEMPTION, CdlSpec, same_lut
from aar.plan.model import Cdl

from .fake_resolve import FakeFolder, FakeMediaPoolItem, FakeNode, FakeTimelineItem
from .ops_support import CONFIG, Show, content_writes, human_settles

START = 108000
A_MOV = "X:/acme/media/a.mov"
B_WAV = "X:/acme/media/b.wav"
LUT_ROOT = "X:/acme/Support/LUT"
# The broker restores the human's media-pool folder when it gives the context back; the import
# tests run without that restore so they see only what the op itself does.
NO_RESTORE = dataclasses.replace(CONFIG, restore_delay_seconds=1e9)


def ref(item: FakeTimelineItem) -> dict[str, Any]:
    return {
        "track_type": item.track_type,
        "track": item.timeline.track_index_of(item),
        "item_id": item.GetUniqueId(),
    }


def calls(show: Show, name: str) -> list[tuple[Any, ...]]:
    return [c for c in show.fake.calls if c[0] == name]


def validate(op_name: str, params: dict[str, Any]) -> Any:
    return get_op(op_name).params.model_validate(params)


# -- media.import ------------------------------------------------------------------------------


def _import_world() -> Show:
    show = Show(config=NO_RESTORE)
    show.fake.edit_files = {
        A_MOV: {},
        B_WAV: {"kind": "audio"},
        "X:/acme/media/c.mov": {},
    }
    return show


def _bin(show: Show, *path: str) -> FakeFolder:
    return show.project.media_pool.folder(*path)


def test_import_creates_the_bin_imports_and_matches_by_file_path() -> None:
    show = _import_world()
    result = show.ok(
        "media.import",
        {"paths": [A_MOV, "x:\\ACME\\media\\B.WAV"], "bin_path": ["Footage", "Day 3"]},
    )
    assert result.bins_created == ["Footage/Day 3"]
    assert result.bin_path == ("Footage", "Day 3")
    day3 = _bin(show, "Footage", "Day 3")
    by_id = {c.GetUniqueId(): c for c in day3.clips}
    assert [m.status for m in result.media] == ["imported", "imported"]
    assert [m.path for m in result.media] == [A_MOV, "x:\\ACME\\media\\B.WAV"]
    # Each result names the item whose File Path is that file (as the filesystem spells it).
    assert [by_id[m.media.unique_id].props["File Path"] for m in result.media] == [A_MOV, B_WAV]
    assert [m.file_path for m in result.media] == [A_MOV, B_WAV]
    assert all(m.media.bin_path == ("Footage", "Day 3") for m in result.media)
    assert calls(show, "AddSubFolder") == [("AddSubFolder", "Footage", "Day 3")]
    assert calls(show, "ImportMedia") == [
        ("ImportMedia", [{"FilePath": A_MOV}, {"FilePath": "x:\\ACME\\media\\B.WAV"}])
    ]
    # The bin was made current for the import, then the folder that was current came back.
    assert [c[1] for c in calls(show, "SetCurrentFolder")] == ["Day 3", "Master"]
    assert show.project.media_pool.current is show.project.media_pool.GetRootFolder()
    assert [(n.code, n.trap_id) for n in result.notes] == [
        ("input_color_space_default", "rcm-input-colorspace")
    ]


def test_import_rerun_is_present_and_makes_no_call() -> None:
    show = _import_world()
    params = {"paths": [A_MOV, B_WAV], "bin_path": ["Footage", "Day 3"]}
    first = show.ok("media.import", params)
    show.fake.calls.clear()
    again = show.ok("media.import", params)
    assert [m.status for m in again.media] == ["present", "present"]
    assert [m.media.unique_id for m in again.media] == [m.media.unique_id for m in first.media]
    assert again.bins_created == [] and again.notes == []
    assert content_writes(show.fake) == []
    assert calls(show, "SetCurrentFolder") == []
    assert len(_bin(show, "Footage", "Day 3").clips) == 2


def test_import_matches_existing_items_case_insensitively_across_separators() -> None:
    show = _import_world()
    existing = show.project.media_pool.add_clip(("Footage", "Day 1"), "a", file_path=A_MOV)
    result = show.ok(
        "media.import", {"paths": ["x:\\ACME\\MEDIA\\A.MOV"], "bin_path": ["Footage", "Day 1"]}
    )
    assert [(m.status, m.media.unique_id) for m in result.media] == [
        ("present", existing.GetUniqueId())
    ]
    assert calls(show, "ImportMedia") == []


def test_import_restores_the_current_folder_after_success_and_failure() -> None:
    show = _import_world()
    pool = show.project.media_pool
    music = pool.folder("Music")
    pool.current = music
    human_settles(show)
    show.ok("media.import", {"paths": [A_MOV], "bin_path": ["Footage", "Day 2"]})
    assert pool.current is music
    error = show.fails(
        "media.import", {"paths": ["X:/acme/missing.mov", B_WAV], "bin_path": ["Footage", "Day 2"]}
    )
    assert error.code == "import_failed"
    assert pool.current is music
    assert [c[1] for c in calls(show, "SetCurrentFolder")] == ["Day 2", "Music"] * 2


def test_import_if_present_error_refuses_before_importing_anything() -> None:
    show = _import_world()
    existing = show.project.media_pool.add_clip(("Footage", "Day 2"), "a", file_path=A_MOV)
    before = dict(existing.props)
    error = show.fails(
        "media.import",
        {"paths": [A_MOV, B_WAV], "bin_path": ["Footage", "Day 2"], "if_present": "error"},
    )
    assert error.code == "media_present"
    assert [m["media"]["unique_id"] for m in error.detail["present"]] == [existing.GetUniqueId()]
    assert calls(show, "ImportMedia") == [] and calls(show, "SetCurrentFolder") == []
    assert existing.props == before
    assert len(_bin(show, "Footage", "Day 2").clips) == 1


def test_import_never_reimports_or_changes_an_existing_item() -> None:
    show = _import_world()
    pool = show.project.media_pool
    existing = pool.add_clip(
        ("Footage", "Day 2"), "a", file_path=A_MOV, input_color_space="Rec.709"
    )
    other = pool.add_clip(("Footage", "Day 2"), "kept")
    snapshots = {c.GetUniqueId(): dict(c.props) for c in (existing, other)}
    result = show.ok("media.import", {"paths": [A_MOV, B_WAV], "bin_path": ["Footage", "Day 2"]})
    assert [m.status for m in result.media] == ["present", "imported"]
    # Only the missing path went to ImportMedia; the existing items are untouched and kept.
    assert calls(show, "ImportMedia") == [("ImportMedia", [{"FilePath": B_WAV}])]
    folder = _bin(show, "Footage", "Day 2")
    assert existing in folder.clips and other in folder.clips and len(folder.clips) == 3
    assert {c.GetUniqueId(): dict(c.props) for c in (existing, other)} == snapshots
    assert [c for c in show.fake.calls if c[0] in ("SetClipProperty", "DeleteClips")] == []


def test_import_failed_names_the_missing_path_and_reports_the_rest_done() -> None:
    show = _import_world()
    # The missing path comes first: the item Resolve returns must be matched by its File
    # Path, not taken for the first path asked.
    error = show.fails(
        "media.import", {"paths": ["X:/acme/missing.mov", A_MOV], "bin_path": ["Footage", "Day 2"]}
    )
    assert error.code == "import_failed"
    assert error.detail["paths"] == ["X:/acme/missing.mov"]
    assert [(m["path"], m["status"]) for m in error.detail["completed"]] == [(A_MOV, "imported")]
    imported = error.detail["completed"][0]["media"]["unique_id"]
    assert {
        c.GetUniqueId(): c.props["File Path"] for c in _bin(show, "Footage", "Day 2").clips
    } == {imported: A_MOV}
    rerun = show.fails(
        "media.import", {"paths": ["X:/acme/missing.mov", A_MOV], "bin_path": ["Footage", "Day 2"]}
    )
    assert rerun.code == "import_failed"
    assert [(m["path"], m["status"]) for m in rerun.detail["completed"]] == [(A_MOV, "present")]


def test_import_partial_fault_is_import_failed_never_success() -> None:
    show = _import_world()
    show.fake.faults.add("ImportMedia:partial")
    error = show.fails("media.import", {"paths": [A_MOV, B_WAV], "bin_path": ["Footage", "Day 2"]})
    assert error.code == "import_failed" and error.detail["paths"] == [B_WAV]
    assert [m["path"] for m in error.detail["completed"]] == [A_MOV]


def test_import_duplicate_fault_is_unexpected_and_deletes_nothing() -> None:
    show = _import_world()
    show.fake.faults.add("ImportMedia:duplicate")
    error = show.fails("media.import", {"paths": [A_MOV, B_WAV], "bin_path": ["Footage", "Day 2"]})
    assert error.code == "import_unexpected"
    clips = _bin(show, "Footage", "Day 2").clips
    twins = [c.GetUniqueId() for c in clips if c.props["File Path"] == A_MOV]
    assert len(twins) == 2
    assert sorted(u["media_id"] for u in error.detail["unexpected"]) == sorted(twins)
    assert [m["path"] for m in error.detail["completed"]] == [B_WAV]
    assert len(clips) == 3 and calls(show, "DeleteClips") == []


def test_import_an_item_for_a_path_not_asked_for_is_unexpected() -> None:
    show = _import_world()
    pool = show.project.media_pool
    real = pool.ImportMedia

    def import_and_more(infos: list[Any]) -> list[Any]:
        made = list(real(infos))
        stray = FakeMediaPoolItem(show.fake, "s", file_path="X:/acme/media/stray.mov")
        pool.current.clips.append(stray)
        return [*made, stray]

    pool.ImportMedia = import_and_more  # type: ignore[method-assign]
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["Footage", "Day 2"]})
    assert error.code == "import_unexpected"
    assert [u["file_path"] for u in error.detail["unexpected"]] == ["X:/acme/media/stray.mov"]
    assert [m["path"] for m in error.detail["completed"]] == [A_MOV]


def test_import_an_item_that_is_not_in_the_bin_is_unverified() -> None:
    show = _import_world()
    pool = show.project.media_pool
    real = pool.ImportMedia

    def import_elsewhere(infos: list[Any]) -> list[Any]:
        made = list(real(infos))
        for clip in made:
            pool.current.clips.remove(clip)
            pool.GetRootFolder().clips.append(clip)
        return made

    pool.ImportMedia = import_elsewhere  # type: ignore[method-assign]
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["Footage", "Day 2"]})
    assert error.code == "import_unverified" and len(error.detail["media_ids"]) == 1


def test_import_missing_bin_without_create_bins_is_refused() -> None:
    show = _import_world()
    error = show.fails(
        "media.import",
        {"paths": [A_MOV], "bin_path": ["Footage", "Day 9"], "create_bins": False},
    )
    assert error.code == "bin_missing" and error.detail["bin_path"] == ["Footage", "Day 9"]
    assert content_writes(show.fake) == []


def test_import_bin_create_and_select_faults_are_reported() -> None:
    show = _import_world()
    show.fake.faults.add("AddSubFolder:fail")
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["New"]})
    assert error.code == "bin_create_failed" and calls(show, "ImportMedia") == []
    show.fake.faults = {"SetCurrentFolder:ignored"}
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["Footage", "Day 2"]})
    assert error.code == "bin_select_failed" and calls(show, "ImportMedia") == []


def test_a_bin_that_add_sub_folder_does_not_really_make_is_caught() -> None:
    show = _import_world()
    pool = show.project.media_pool
    pool.AddSubFolder = lambda folder, name: FakeFolder(name, "fold-detached")  # type: ignore[method-assign]
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["New"]})
    assert error.code == "bin_create_failed" and calls(show, "ImportMedia") == []


def test_import_ambiguous_bin_or_item_is_refused() -> None:
    show = _import_world()
    pool = show.project.media_pool
    pool.folder("Footage").subfolders.append(FakeFolder("Day 1", "fold-twin"))
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["Footage", "Day 1"]})
    assert error.code == "bin_ambiguous"
    pool.add_clip(("Music",), "a1", file_path=A_MOV)
    pool.add_clip(("Music",), "a2", file_path="x:\\acme\\media\\A.mov")
    error = show.fails("media.import", {"paths": [A_MOV], "bin_path": ["Music"]})
    assert error.code == "media_ambiguous" and len(error.detail["media_ids"]) == 2
    assert calls(show, "ImportMedia") == []


@pytest.mark.parametrize(
    "path",
    [
        "media/a.mov",
        "a.mov",
        "X:media\\a.mov",
        "/acme/media/a.mov",
        "X:/acme/../a.mov",
        "X:/acme/./a.mov",
        "X:/acme//a.mov",
        "X:/acme/media/",
        "X:/",
        "X:/acme/a\x00.mov",
        "X:/acme/a\t.mov",
        "X:/acme/a\n.mov",
        "\\\\?\\X:\\acme\\a.mov",
        "\\\\.\\pipe\\a",
        "\\\\host\\share",
        "\\\\host",
        "//host/share/a.mov",
        "X:/" + "a" * 1022,
        "",
    ],
)
def test_import_refuses_paths_that_are_not_absolute_edit_pc_files(path: str) -> None:
    with pytest.raises(ValidationError):
        validate("media.import", {"paths": [path], "bin_path": ["Footage"]})


@pytest.mark.parametrize(
    "path",
    ["X:/acme/media/a.mov", "x:\\acme\\media\\a.mov", "\\\\host\\share\\a.mov", "X:/" + "a" * 1021],
)
def test_import_accepts_drive_and_unc_paths(path: str) -> None:
    assert validate("media.import", {"paths": [path], "bin_path": ["Footage"]}).paths == [path]


@pytest.mark.parametrize(
    "params",
    [
        {"paths": [], "bin_path": ["Footage"]},
        {"paths": [f"X:/acme/{n}.mov" for n in range(51)], "bin_path": ["Footage"]},
        {"paths": [A_MOV, "x:\\ACME\\media\\a.MOV"], "bin_path": ["Footage"]},
        {"paths": [A_MOV], "bin_path": []},
        {"paths": [A_MOV], "bin_path": ["b"] * 9},
        {"paths": [A_MOV], "bin_path": ["  "]},
        {"paths": [A_MOV], "bin_path": ["b" * 129]},
        {"paths": [A_MOV], "bin_path": ["Footage"], "create_bins": "yes"},
        {"paths": [A_MOV], "bin_path": ["Footage"], "if_present": "replace"},
        {"paths": [A_MOV], "bin_path": ["Footage"], "timeline": {"unique_id": "x"}},
    ],
)
def test_import_params_are_bounded(params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        validate("media.import", params)


def test_import_list_bounds_accept_fifty_and_eight_deep() -> None:
    params = {"paths": [f"X:/acme/{n}.mov" for n in range(50)], "bin_path": ["b"] * 8}
    assert len(validate("media.import", params).paths) == 50


def test_import_guard_exemption_text_is_pinned() -> None:
    assert (
        get_op("media.import").guard_exemption
        == IMPORT_GUARD_EXEMPTION
        == (
            "additive: only adds media-pool items to the named bin; never alters, moves, relinks "
            "or deletes existing items or any timeline"
        )
    )


# -- track.ensure ------------------------------------------------------------------------------


def test_tracks_are_appended_named_and_read_back() -> None:
    show = Show()
    result = show.ok(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [
                {"type": "audio", "index": 3, "name": "A3 room", "audio_format": "mono"},
                {"type": "video", "index": 2, "name": "V2 titles", "enabled": False},
            ],
        },
    )
    a3, v2 = result.tracks
    assert (a3.type, a3.index, a3.created, a3.name, a3.audio_format) == (
        "audio", 3, True, "A3 room", "mono"
    )  # fmt: skip
    assert a3.changed == ["created", "name"] and not a3.locked and a3.enabled
    assert (v2.created, v2.name, v2.enabled, v2.audio_format) == (True, "V2 titles", False, None)
    # Appended only, with the subtype passed for audio; the fill-in track takes it too.
    assert calls(show, "AddTrack") == [
        ("AddTrack", "audio", "mono"),
        ("AddTrack", "audio", "mono"),
        ("AddTrack", "video", None),
    ]
    assert [(a.type, a.index, a.audio_format) for a in result.added] == [
        ("audio", 2, "mono"), ("audio", 3, "mono"), ("video", 2, None)
    ]  # fmt: skip
    assert [t.subtype for t in show.main.tracks["audio"]] == ["stereo", "mono", "mono"]
    assert show.main.tracks["video"][1].name == "V2 titles"
    assert not show.main.tracks["video"][1].enabled


def test_tracks_rerun_is_present_with_no_call() -> None:
    show = Show()
    params = {
        "timeline": show.ref("Main"),
        "tracks": [
            {"type": "audio", "index": 2, "name": "A2", "enabled": False, "locked": True},
            {"type": "video", "index": 1, "name": "Video 1"},
        ],
    }
    show.ok("track.ensure", params)
    show.fake.calls.clear()
    again = show.ok("track.ensure", params)
    assert [(t.created, t.changed) for t in again.tracks] == [(False, []), (False, [])]
    assert again.added == [] and content_writes(show.fake) == []


def test_a_lower_spec_keeps_its_own_format() -> None:
    show = Show()
    show.ok(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [
                {"type": "audio", "index": 3, "audio_format": "5.1"},
                {"type": "audio", "index": 2, "audio_format": "mono"},
            ],
        },
    )
    assert [t.subtype for t in show.main.tracks["audio"]] == ["stereo", "mono", "5.1"]


def test_an_existing_track_of_another_format_is_refused_before_any_write() -> None:
    show = Show()
    error = show.fails(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [
                {"type": "video", "index": 2},
                {"type": "audio", "index": 1, "audio_format": "mono", "name": "A1"},
            ],
        },
    )
    assert error.code == "track_format_mismatch"
    assert (error.detail["have"], error.detail["wanted"]) == ("stereo", "mono")
    assert content_writes(show.fake) == []


def test_a_locked_track_is_never_written_unless_the_spec_unlocks_it() -> None:
    show = Show()
    show.main.tracks["audio"][0].locked = True
    error = show.fails(
        "track.ensure",
        {"timeline": show.ref("Main"), "tracks": [{"type": "audio", "index": 1, "name": "x"}]},
    )
    assert error.code == "track_locked" and error.detail["fields"] == ["name"]
    error = show.fails(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [{"type": "audio", "index": 1, "enabled": False, "locked": True}],
        },
    )
    assert error.code == "track_locked" and error.detail["fields"] == ["enabled"]
    assert content_writes(show.fake) == []
    # Asking for locked=True with nothing else to change is fine (present).
    kept = show.ok(
        "track.ensure",
        {"timeline": show.ref("Main"), "tracks": [{"type": "audio", "index": 1, "locked": True}]},
    )
    assert kept.tracks[0].changed == [] and content_writes(show.fake) == []
    # locked=False in the same spec: unlock first, then write.
    result = show.ok(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [{"type": "audio", "index": 1, "name": "dialogue", "locked": False}],
        },
    )
    assert result.tracks[0].changed == ["locked", "name"]
    assert [c[0] for c in content_writes(show.fake)] == ["SetTrackLock", "SetTrackName"]
    assert show.main.tracks["audio"][0].name == "dialogue"


def test_a_locked_track_later_in_the_request_stops_it_before_any_write() -> None:
    show = Show()
    show.main.tracks["audio"][0].locked = True
    error = show.fails(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [
                {"type": "video", "index": 1, "name": "picture"},
                {"type": "audio", "index": 1, "name": "dialogue"},
            ],
        },
    )
    assert error.code == "track_locked" and content_writes(show.fake) == []


def test_a_lock_asked_for_is_set_after_the_other_writes() -> None:
    show = Show()
    result = show.ok(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [{"type": "video", "index": 1, "name": "picture", "locked": True}],
        },
    )
    assert result.tracks[0].changed == ["name", "locked"] and result.tracks[0].locked
    assert [c[0] for c in content_writes(show.fake)] == ["SetTrackName", "SetTrackLock"]


@pytest.mark.parametrize(
    ("fault", "spec", "field"),
    [
        ("SetTrackName:no-op", {"name": "x"}, "name"),
        ("SetTrackEnable:no-op", {"enabled": False}, "enabled"),
        ("SetTrackLock:no-op", {"locked": True}, "locked"),
    ],
)
def test_a_track_setter_that_does_not_stick_is_caught(
    fault: str, spec: dict[str, Any], field: str
) -> None:
    show = Show()
    show.fake.faults.add(fault)
    error = show.fails(
        "track.ensure",
        {"timeline": show.ref("Main"), "tracks": [{"type": "video", "index": 1, **spec}]},
    )
    assert error.code == "track_not_applied" and error.detail["field"] == field


def test_a_track_that_is_not_added_is_caught_with_the_tracks_done() -> None:
    show = Show()
    show.fake.faults.add("AddTrack:no-op")
    error = show.fails(
        "track.ensure",
        {
            "timeline": show.ref("Main"),
            "tracks": [{"type": "video", "index": 1, "name": "V1"}, {"type": "video", "index": 2}],
        },
    )
    assert error.code == "track_add_failed"
    assert [(t["index"], t["name"]) for t in error.detail["completed"]] == [(1, "V1")]


def test_a_refused_track_setter_is_track_set_failed() -> None:
    show = Show()
    show.main.tracks["video"][0].locked = True
    real = show.main.GetIsTrackLocked
    show.main.GetIsTrackLocked = lambda kind, index: False  # type: ignore[method-assign]
    error = show.fails(
        "track.ensure",
        {"timeline": show.ref("Main"), "tracks": [{"type": "video", "index": 1, "name": "x"}]},
    )
    show.main.GetIsTrackLocked = real  # type: ignore[method-assign]
    assert error.code == "track_set_failed" and error.detail["field"] == "name"


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "video", "index": 0},
        {"type": "video", "index": 17},
        {"type": "video", "index": True},
        {"type": "video", "index": 1, "name": "x" * 65},
        {"type": "video", "index": 1, "name": " "},
        {"type": "video", "index": 1, "audio_format": "mono"},
        {"type": "audio", "index": 1, "audio_format": "quad"},
        {"type": "audio", "index": 1, "enabled": "no"},
        {"type": "audio", "index": 1, "mute": True},
        {"type": "bus", "index": 1},
    ],
)
def test_track_specs_are_bounded(spec: dict[str, Any]) -> None:
    show = Show()
    with pytest.raises(ValidationError):
        validate("track.ensure", {"timeline": show.ref("Main"), "tracks": [spec]})


def test_track_lists_are_bounded_and_unique() -> None:
    show = Show()
    tl = show.ref("Main")
    many = [{"type": "audio", "index": n % 16 + 1} for n in range(51)]
    for tracks in ([], many, [{"type": "video", "index": 2}, {"type": "video", "index": 2}]):
        with pytest.raises(ValidationError):
            validate("track.ensure", {"timeline": tl, "tracks": tracks})
    assert validate("track.ensure", {"timeline": tl, "tracks": [{"type": "video", "index": 16}]})


# -- transition.add ----------------------------------------------------------------------------


def _cut(show: Show, *, a_right: int | None = None, b_left: int = 5000, kind: str = "video"):
    """Two clips of cam A meeting at START + 100 on track 1. ``a_right``: A's right handle."""
    clip = show.cam_a
    a_src = 1000 if a_right is None else clip.frames - 100 - a_right
    a = show.main.place(kind, 1, clip, START, 100, src_start=a_src)
    b = show.main.place(kind, 1, clip, START + 100, 100, src_start=b_left)
    return a, b


def _transitions(show: Show, kind: str = "video") -> list[FakeTimelineItem]:
    return [i for i in show.main.GetItemListInTrack(kind, 1) if i.item_type == "transition"]


def _tparams(show: Show, *targets: dict[str, Any]) -> dict[str, Any]:
    return {"timeline": show.ref("Main"), "targets": list(targets)}


def test_a_transition_is_added_at_the_edit_and_read_back() -> None:
    show = Show()
    a, _ = _cut(show)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    result = show.ok("transition.add", _tparams(show, target))
    (added,) = result.transitions
    (made,) = _transitions(show)
    assert (added.status, added.transition_id, added.duration) == ("added", made.GetUniqueId(), 12)
    assert (added.record_in, added.record_out) == (94, 106)
    assert (made.start, made.end) == (START + 94, START + 106)
    assert calls(show, "AddTransition") == [
        (
            "AddTransition",
            a.GetUniqueId(),
            {
                "type": "Cross Dissolve",
                "category": "simple",
                "position": "end",
                "alignment": "center",
                "duration": 12,
            },
        )
    ]
    assert [(h.side, h.needed) for h in added.handles] == [("outgoing", 6), ("incoming", 6)]


def test_a_transition_rerun_and_the_same_cut_from_the_other_side_are_present() -> None:
    show = Show()
    a, b = _cut(show)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    first = show.ok("transition.add", _tparams(show, target))
    show.fake.calls.clear()
    again = show.ok("transition.add", _tparams(show, target))
    other_side = {"item": ref(b), "position": "start", "alignment": "center", "frames": 12}
    from_b = show.ok("transition.add", _tparams(show, other_side))
    for result in (again, from_b):
        assert result.transitions[0].status == "present"
        assert result.transitions[0].transition_id == first.transitions[0].transition_id
    assert content_writes(show.fake) == [] and len(_transitions(show)) == 1


def test_a_transition_of_another_length_is_a_conflict_and_never_replaced() -> None:
    show = Show()
    a, _ = _cut(show)
    existing = show.main.place_transition("video", 1, START + 100, 8)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == "transition_conflict"
    assert error.detail["existing"] == [{"transition_id": existing.GetUniqueId(), "duration": 8}]
    assert content_writes(show.fake) == [] and _transitions(show) == [existing]


def test_short_handles_are_refused_before_the_call() -> None:
    show = Show()
    a, _ = _cut(show, a_right=4)
    target = {"item": ref(a), "position": "end", "alignment": "right", "frames": 10}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == "transition_needs_handles"
    assert (error.detail["needed"], error.detail["available"]) == (
        {"outgoing": 10},
        {"outgoing": 4},
    )
    assert "alignment" in error.detail["fix_hint"] and calls(show, "AddTransition") == []
    # Left-aligned, the same transition needs only the incoming clip's handle.
    target["alignment"] = "left"
    assert show.ok("transition.add", _tparams(show, target)).transitions[0].duration == 10


@pytest.mark.parametrize(
    ("mode", "code"), [("shorten", "transition_short"), ("refuse", "transition_refused")]
)
def test_the_handle_check_is_advisory_and_the_read_back_decides(mode: str, code: str) -> None:
    # An odd centred length: the advisory rule asks frames // 2 = 7 of each side, but the
    # fake (like Resolve may) puts 8 frames after the edit, past A's 7-frame handle.
    show = Show()
    show.fake.transition_short_handles = mode
    a, _ = _cut(show, a_right=7)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 15}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == code
    if mode == "shorten":
        assert (error.detail["measured"], error.detail["wanted"]) == (14, 15)
        (deleted,) = calls(show, "DeleteClips")
        assert len(deleted[1]) == 1
    assert _transitions(show) == []


def test_a_transition_refused_by_resolve_is_reported() -> None:
    show = Show()
    show.fake.faults.add("AddTransition:none")
    a, _ = _cut(show)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == "transition_refused" and _transitions(show) == []


@pytest.mark.parametrize(
    ("fault", "code"),
    [
        ("AddTransition:short", "transition_short"),
        ("AddTransition:misplaced", "transition_misplaced"),
    ],
)
def test_a_wrong_transition_is_deleted_again(fault: str, code: str) -> None:
    show = Show()
    show.fake.faults.add(fault)
    a, b = _cut(show)
    clips = {a.GetUniqueId(), b.GetUniqueId()}
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == code
    (deleted,) = calls(show, "DeleteClips")
    assert not set(deleted[1]) & clips  # only the new transition, never a clip
    assert _transitions(show) == [] and {i.GetUniqueId() for i in show.main.items("video")} == clips


def test_a_wrong_transition_that_cannot_be_deleted_is_left_behind_and_named() -> None:
    show = Show()
    show.fake.faults |= {"AddTransition:short", "DeleteClips:no-op"}
    a, _ = _cut(show)
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    error = show.fails("transition.add", _tparams(show, target))
    (left,) = _transitions(show)
    assert error.code == "transition_left_behind"
    assert error.detail["transition_id"] == left.GetUniqueId()
    assert (error.detail["cause"], error.detail["measured"]) == ("transition_short", 11)


def test_a_returned_item_that_is_not_a_new_transition_is_never_deleted() -> None:
    show = Show()
    a, _ = _cut(show)
    a.AddTransition = lambda options: a  # type: ignore[method-assign]
    target = {"item": ref(a), "position": "end", "alignment": "center", "frames": 12}
    error = show.fails("transition.add", _tparams(show, target))
    assert error.code == "transition_unverified" and calls(show, "DeleteClips") == []
    assert a in show.main.items("video")


def test_an_audio_crossfade_and_a_fade_from_black() -> None:
    show = Show()
    a, _ = _cut(show, kind="audio")
    picture = show.main.place("video", 1, show.cam_a, START, 100, src_start=0)
    result = show.ok(
        "transition.add",
        _tparams(
            show,
            {
                "item": ref(a),
                "position": "end",
                "type": "Cross Fade 0 dB",
                "category": "audio",
                "alignment": "center",
                "frames": 10,
            },
            # The first item, from black: right-aligned it needs no handle on either side.
            {"item": ref(picture), "position": "start", "alignment": "right", "frames": 15},
        ),
    )
    assert [(t.status, t.duration) for t in result.transitions] == [("added", 10), ("added", 15)]
    assert result.transitions[1].handles[0].side == "incoming"
    assert len(_transitions(show, "audio")) == 1 and len(_transitions(show)) == 1


def test_a_transition_target_must_be_a_clip() -> None:
    show = Show()
    _cut(show)
    existing = show.main.place_transition("video", 1, START + 100, 8)
    target = {"item": ref(existing), "position": "end", "alignment": "center", "frames": 8}
    assert show.fails("transition.add", _tparams(show, target)).code == "not_a_clip_item"


def test_transition_errors_carry_the_targets_done() -> None:
    show = Show()
    a, b = _cut(show)
    show.main.place("video", 1, show.cam_a, START + 200, 100, src_start=9000)
    real = b.AddTransition
    b.AddTransition = lambda options: None  # type: ignore[method-assign]
    error = show.fails(
        "transition.add",
        _tparams(
            show,
            {"item": ref(a), "position": "end", "alignment": "center", "frames": 12},
            {"item": ref(b), "position": "end", "alignment": "center", "frames": 12},
        ),
    )
    b.AddTransition = real  # type: ignore[method-assign]
    assert error.code == "transition_refused"
    assert [t["status"] for t in error.detail["completed"]] == ["added"]


@pytest.mark.parametrize(
    "patch",
    [
        {"frames": 0},
        {"frames": 241},
        {"frames": True},
        {"frames": 12.0},
        {"category": "fusion"},
        {"category": "ofx"},
        {"category": "audio"},
        {"alignment": "middle"},
        {"alignment": None},
        {"position": "both"},
        {"type": " "},
        {"type": "x" * 65},
        {"item": {"track_type": "subtitle", "track": 1, "item_id": "x"}},
        {"item": {"track_type": "audio", "track": 1, "item_id": "x"}},
        {"duration": 12},
    ],
)
def test_transition_targets_are_bounded(patch: dict[str, Any]) -> None:
    show = Show()
    target = {
        "item": {"track_type": "video", "track": 1, "item_id": "x"},
        "position": "end",
        "alignment": "center",
        "frames": 12,
        **patch,
    }
    target = {k: v for k, v in target.items() if v is not None}
    with pytest.raises(ValidationError):
        validate("transition.add", _tparams(show, target))


def test_transition_lists_are_bounded_and_unique() -> None:
    show = Show()
    one = {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "position": "end"}
    one |= {"alignment": "center", "frames": 12}
    for targets in (
        [],
        [one, one],
        [{**one, "item": {**one["item"], "item_id": str(n)}} for n in range(51)],
    ):
        with pytest.raises(ValidationError):
            validate("transition.add", {"timeline": show.ref("Main"), "targets": targets})
    assert validate("transition.add", _tparams(show, one, {**one, "position": "start"}))


# -- item.set_color ----------------------------------------------------------------------------


def _graded(show: Show, nodes: int = 1) -> FakeTimelineItem:
    item = show.main.place("video", 1, show.cam_a, START, 100)
    item.graph.nodes = [FakeNode() for _ in range(nodes)]
    return item


CDL = {"slope": [1.1, 1.0, 0.9], "offset": [0.01, -0.02, 0.0], "power": [1, 1, 1.05]}


def test_a_cdl_and_a_lut_are_set_and_the_lut_read_back() -> None:
    show = Show()
    item = _graded(show, nodes=2)
    lut = f"{LUT_ROOT}/Film/Look.cube"
    result = show.ok(
        "item.set_color",
        {
            "timeline": show.ref("Main"),
            "targets": [
                {"item": ref(item), "node": 2, "cdl": {**CDL, "saturation": 1.2}, "lut": lut}
            ],
        },
    )
    (applied,) = result.items
    sent = {
        "NodeIndex": 2,
        "Slope": "1.1 1 0.9",
        "Offset": "0.01 -0.02 0",
        "Power": "1 1 1.05",
        "Saturation": 1.2,
    }
    assert applied.cdl is not None and applied.cdl.model_dump() == {
        "verified": False,
        "reason": "no-cdl-getter-21.1",
        "sent": sent,
    }
    assert calls(show, "SetCDL") == [("SetCDL", item.GetUniqueId(), sent)]
    assert item.cdl_on(2) == {
        "Slope": (1.1, 1.0, 0.9),
        "Offset": (0.01, -0.02, 0.0),
        "Power": (1.0, 1.0, 1.05),
        "Saturation": 1.2,
    }
    assert item.cdl_on(1) is None
    # GetLUT reads back relative to the LUT folder; the op compares normalised suffixes.
    assert applied.lut is not None and applied.lut.status == "set"
    assert applied.lut.read_back == "Film\\Look.cube" == item.graph.nodes[1].lut
    assert (applied.status, applied.nodes, applied.page_bounced) == ("applied", 2, False)


def test_a_lut_already_on_the_node_is_present_with_no_call() -> None:
    show = Show()
    item = _graded(show)
    params = {
        "timeline": show.ref("Main"),
        "targets": [{"item": ref(item), "lut": f"{LUT_ROOT}/Film/Look.cube"}],
    }
    show.ok("item.set_color", params)
    show.fake.calls.clear()
    again = show.ok("item.set_color", params)
    assert again.items[0].status == "present" and again.items[0].lut.status == "present"
    assert content_writes(show.fake) == []
    relative = show.ok(
        "item.set_color",
        {"timeline": show.ref("Main"), "targets": [{"item": ref(item), "lut": "film/LOOK.cube"}]},
    )
    assert relative.items[0].status == "present" and content_writes(show.fake) == []


def test_a_cdl_is_written_again_on_a_rerun_because_it_cannot_be_read() -> None:
    show = Show()
    item = _graded(show)
    params = {"timeline": show.ref("Main"), "targets": [{"item": ref(item), "cdl": CDL}]}
    show.ok("item.set_color", params)
    show.fake.calls.clear()
    again = show.ok("item.set_color", params)
    assert again.items[0].status == "applied" and not again.items[0].cdl.verified
    assert [c[0] for c in content_writes(show.fake)] == ["SetCDL"]


def test_a_node_that_does_not_exist_is_refused() -> None:
    show = Show()
    item = _graded(show)
    error = show.fails(
        "item.set_color",
        {"timeline": show.ref("Main"), "targets": [{"item": ref(item), "node": 2, "cdl": CDL}]},
    )
    assert (error.code, error.detail["nodes"]) == ("node_out_of_range", 1)
    assert content_writes(show.fake) == []


def test_a_refused_cdl_is_cdl_refused_after_a_bounce() -> None:
    show = Show()
    show.fake.faults.add("SetCDL:false")
    item = _graded(show)
    error = show.fails(
        "item.set_color",
        {"timeline": show.ref("Main"), "targets": [{"item": ref(item), "cdl": CDL}]},
    )
    assert (error.code, error.detail["trap_id"]) == ("cdl_refused", "setproperty-after-color")
    assert len(calls(show, "SetCDL")) == 2 and item.cdl_on(1) is None


def test_a_lut_that_does_not_stick_or_is_unknown_is_caught() -> None:
    show = Show()
    item = _graded(show)
    show.fake.faults.add("SetLUT:no-op")
    params = {"timeline": show.ref("Main"), "targets": [{"item": ref(item), "lut": "Film/a.cube"}]}
    assert show.fails("item.set_color", params).code == "lut_not_applied"
    show.fake.faults.clear()
    show.fake.unknown_luts.add("Film/a.cube")
    assert show.fails("item.set_color", params).code == "lut_refused"


def test_colour_setters_bounce_the_page_when_they_jam() -> None:
    show = Show()
    item = _graded(show)
    show.fake.human_set_page("color")
    show.fake.human_set_page("edit")
    human_settles(show)
    result = show.ok(
        "item.set_color",
        {
            "timeline": show.ref("Main"),
            "targets": [{"item": ref(item), "cdl": CDL, "lut": "a.cube"}],
        },
    )
    assert result.items[0].page_bounced and item.graph.nodes[0].lut == "a.cube"
    assert [n.code for n in result.notes] == ["page_bounced"]
    assert show.fake.page == "edit"


def test_colour_errors_carry_the_items_done() -> None:
    show = Show()
    first, second = _graded(show), show.main.place("video", 1, show.cam_a, START + 100, 50)
    show.fake.unknown_luts.add("bad.cube")
    error = show.fails(
        "item.set_color",
        {
            "timeline": show.ref("Main"),
            "targets": [
                {"item": ref(first), "lut": "a.cube"},
                {"item": ref(second), "lut": "bad.cube"},
            ],
        },
    )
    assert error.code == "lut_refused"
    assert [i["item_id"] for i in error.detail["completed"]] == [first.GetUniqueId()]


@pytest.mark.parametrize(
    ("read_back", "wanted", "same"),
    [
        ("Film\\Look.cube", f"{LUT_ROOT}/Film/Look.cube", True),
        ("Film\\Look.cube", "x:\\ACME\\Support\\LUT\\film\\look.CUBE", True),
        ("Film\\Look.cube", "Film/Look.cube", True),
        ("Look.cube", "Film/Look.cube", False),
        ("Film\\Look.cube", "Look.cube", False),
        ("Film\\Look.cube", f"{LUT_ROOT}/Other/Look.cube", False),
        ("", "Look.cube", False),
        ("Film/Look.cube", "Film/Look2.cube", False),
    ],
)
def test_lut_paths_compare_by_normalised_suffix(read_back: str, wanted: str, same: bool) -> None:
    assert same_lut(read_back, wanted) is same


@pytest.mark.parametrize(
    "patch",
    [
        {"node": 0},
        {"node": 33},
        {"node": 1.0},
        {"cdl": None, "lut": None},
        {"lut": " "},
        {"lut": "a\n.cube"},
        {"lut": "a" * 1025},
        {"cdl": {"slope": [-0.1, 1, 1]}},
        {"cdl": {"power": [0, 1, 1]}},
        {"cdl": {"saturation": -1}},
        {"cdl": {"offset": [float("nan"), 0, 0]}},
        {"cdl": {"slope": [float("inf"), 1, 1]}},
        {"cdl": {"slope": [1, 1]}},
        {"cdl": {"slope": ["1", 1, 1]}},
        {"cdl": {"gamma": 1}},
        {"item": {"track_type": "audio", "track": 1, "item_id": "x"}},
    ],
)
def test_colour_targets_are_bounded(patch: dict[str, Any]) -> None:
    show = Show()
    target = {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "cdl": {}, **patch}
    target = {k: v for k, v in target.items() if v is not None}
    with pytest.raises(ValidationError):
        validate("item.set_color", {"timeline": show.ref("Main"), "targets": [target]})


@pytest.mark.parametrize(
    "cdl",
    [
        {},
        {"slope": [0, 0, 0]},
        {"slope": [2, 1, 0.5], "offset": [-1, 0, 1], "power": [0.5, 1, 2], "saturation": 0},
        {"slope": [-0.01, 1, 1]},
        {"power": [0, 1, 1]},
        {"power": [-1, 1, 1]},
        {"saturation": -0.5},
        {"offset": [float("inf"), 0, 0]},
        {"slope": [1, 1, 1, 1]},
    ],
)
def test_cdl_spec_has_the_bounds_of_the_plan_cdl(cdl: dict[str, Any]) -> None:
    assert set(CdlSpec.model_fields) == set(Cdl.model_fields)

    def accepts(model: Any) -> bool:
        try:
            model.model_validate(cdl)
        except ValidationError:
            return False
        return True

    assert accepts(CdlSpec) == accepts(Cdl)
