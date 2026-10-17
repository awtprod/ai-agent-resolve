"""Read ops against the fake: project.find, timeline.list, timeline.describe, media.find."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aar.node.locator import ProjectLocator
from aar.node.ops.raw import classify_media

from .fake_resolve import FakeMediaPoolItem, FakeResolve, FakeTrack
from .ops_support import CONFIG, OpsWorld, Show, content_writes

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "live_shapes_21_1.json").read_text(encoding="utf-8")
)
PM_NAVIGATION = {"GotoRootFolder", "GotoParentFolder", "OpenFolder"}

# -- project.find ----------------------------------------------------------------------------


def test_project_find_walks_the_tree_restores_it_and_ids_the_open_project() -> None:
    show = Show()
    show.fake.add_project("Old show", ["x"], ("Archive", "2024"))
    result = show.ok("project.find", {})

    found = {(p.locator.folder_path, p.locator.name): p for p in result.projects}
    assert set(found) == {
        (("Shows",), "Show"),
        (("Shows",), "Other"),
        (("Archive", "2024"), "Old show"),
    }
    mine = found[(("Shows",), "Show")]
    assert mine.is_open and mine.locator == show.locator()
    assert found[(("Shows",), "Other")].locator.project_id is None
    assert result.project_manager_path == ("Shows",)
    assert show.fake.pm_path == ("Shows",)
    assert result.folders_listed == 4


def test_project_find_leaves_ids_unset_when_the_open_name_is_ambiguous() -> None:
    show = Show()
    show.fake.add_project("Show", ["Main"], ("Archive",))
    result = show.ok("project.find", {"name": "Show"})
    assert sorted(p.locator.folder_path for p in result.projects) == [("Archive",), ("Shows",)]
    assert all(p.locator.project_id is None and not p.is_open for p in result.projects)
    assert [n.code for n in result.notes] == ["open_project_ambiguous"]
    assert result.notes[0].trap_id == "pm-folder-scoped-load"


def test_project_find_non_recursive_at_the_root_does_not_navigate() -> None:
    fake = FakeResolve({"Show": ["Main"]}, current="Show", folder=("Shows",))
    fake.add_project("Loose", ["x"], ())
    fake.human_browse_project_manager(())
    world = OpsWorld(fake, replace(CONFIG, snapshot_project_manager_path=False))
    result = world.ok("project.find", {"recursive": False})

    assert [(p.locator.folder_path, p.locator.name) for p in result.projects] == [((), "Loose")]
    assert result.folders_listed == 1
    assert not PM_NAVIGATION & set(fake.call_names())


# -- timeline.list -----------------------------------------------------------------------------


def test_timeline_list_reports_names_ids_and_frames() -> None:
    show = Show()
    show.main.place("video", 1, show.cam_a, 108000, 300)
    result = show.ok("timeline.list", {})
    assert [(t.name, t.is_current, t.duration) for t in result.timelines] == [
        ("Main", True, 300),
        ("Main [agent]", False, 0),
    ]
    assert result.timelines[0].unique_id == show.main.GetUniqueId()
    assert result.current is not None and result.current.name == "Main"
    assert result.project_id == show.project.GetUniqueId()


# -- timeline.describe ---------------------------------------------------------------------------


def test_describe_reports_items_gaps_markers_fades_and_notes() -> None:
    show = Show()
    pool = show.project.media_pool
    offline = pool.add_clip(("Footage",), "cam C", online="Offline", proxy="1920x1080")
    tl = show.main
    first = tl.place("video", 1, show.cam_a, 108000, 100, src_start=500)
    first.fades = {"FadeIn": 12.0, "FadeOut": 0.0}
    tl.place("video", 1, show.cam_b, 108100, 50, src_start=900)
    tl.place("video", 1, show.cam_a, 108200, 30, src_start=2000)
    tl.place("video", 2, offline, 108020, 40)
    tl.place("video", 2, show.multicam, 108100, 10)
    sound = tl.place("audio", 1, show.cam_a, 108000, 100, src_start=500)
    sound.props["AudioVolume"] = -3.5
    tl.AddMarker(10, "Red", "laugh", "big one", 1)
    show.fake.calls.clear()

    result = show.ok("timeline.describe", {"timeline": {"name": "Main"}})

    v1 = result.tracks[0]
    assert (v1.type, v1.index, v1.item_count) == ("video", 1, 3)
    assert [(i.record_in, i.record_out, i.source_start, i.source_end) for i in v1.items] == [
        (0, 100, 500, 600),
        (100, 150, 900, 950),
        (200, 230, 2000, 2030),
    ]
    assert v1.items[0].fade_in == 12 and v1.items[0].media_id == show.cam_a.GetUniqueId()
    assert [(g.record_in, g.record_out, g.frames) for g in v1.gaps] == [(150, 200, 50)]
    a1 = next(t for t in result.tracks if t.type == "audio")
    assert a1.items[0].audio_volume_db == -3.5 and a1.subtype == "stereo"
    assert [(m.frame, m.color, m.name, m.note) for m in result.markers] == [
        (10, "Red", "laugh", "big one")
    ]
    braw = next(m for m in result.media if m.media_id == show.cam_b.GetUniqueId())
    assert braw.is_braw and braw.has_video
    codes = {(n.code, n.trap_id) for n in result.notes}
    assert ("media_offline", "render-uses-original-media") in codes
    assert ("multicam_item", "multicam-angles") in codes
    assert result.timeline.start_frame == 108000 and result.timeline.fps == 29.97
    text = result.text
    assert 'timeline "Main"' in text and "V1 " in text and "A1 " in text
    assert "GAP" in text and "markers: 1" in text and "! media_offline" in text
    assert content_writes(show.fake) == []  # describing changed nothing


def _live_shaped_world() -> OpsWorld:
    """A fake timeline holding the items of the live probe, with their real return shapes."""
    fake = FakeResolve({"Live": ["Probe"]}, current="Live", folder=())
    project = fake.project("Live")
    shapes = FIXTURE["timeline"]
    project.settings.update(shapes["settings"])
    tl = project.timeline("Probe")
    tl.start_frame = shapes["GetStartFrame"]
    tl.tracks["video"] = [FakeTrack("V1"), FakeTrack("V2")]
    tl.tracks["audio"] = [FakeTrack("A1", "stereo"), FakeTrack("A2", "stereo")]
    kinds = {"Video + Audio": "av", "Compound": "compound", "Audio": "audio"}
    for row in FIXTURE["items"]:
        clip = FakeMediaPoolItem(fake, row["name"], kind=kinds[row["clip"]["Type"]])
        clip.props = {"Clip Name": row["name"], **row["clip"]}
        item = tl.place(
            row["track_type"],
            row["track"],
            clip,
            row["GetStart"],
            row["GetEnd"] - row["GetStart"],
            src_start=row["GetSourceStartFrame"],
        )
        item.src_end = row["GetSourceEndFrame"]
        item.fades = dict(row["GetFades"])
        item.props = dict(row["properties"])
    return OpsWorld(fake)


def test_describe_handles_the_live_return_shapes() -> None:
    world = _live_shaped_world()
    result = world.ok("timeline.describe", {}, target=world.locator("Live"))

    assert result.timeline.fps == 29.97 and result.timeline.drop_frame
    assert result.timeline.start_timecode  # a string, whatever the fake's form
    assert (result.timeline.width, result.timeline.height) == (1920, 1080)
    rows = [(t.type, t.index, i) for t in result.tracks for i in t.items]
    by_track = {(kind, index): item for kind, index, item in rows}
    clip = by_track[("video", 1)]
    assert (clip.record_in, clip.record_out, clip.duration) == (0, 186, 186)
    # Source readback is reported as Resolve gives it: here end - start is duration + 1.
    assert (clip.source_start, clip.source_end) == (2799, 2986)
    assert clip.fade_in == 15 and clip.audio_volume_db is None
    compound = by_track[("video", 2)]
    assert (compound.record_in, compound.record_out) == (90, 317)
    a1 = by_track[("audio", 1)]
    assert a1.fade_in == 6 and a1.audio_volume_db == pytest.approx(-4.2558, abs=1e-4)
    media = {m.name: m for m in result.media}
    assert media["clip-2"].is_braw and not media["clip-1"].is_braw
    assert media["clip-3"].frames is None and not media["clip-3"].has_video
    assert media["compound-1"].type == "Compound" and not media["compound-1"].is_still
    assert result.notes == []


@pytest.mark.parametrize("kind", FIXTURE["media_pool_clip_kinds"]["type_format_codec"])
def test_classifier_on_every_clip_kind_seen_live(kind: list[str]) -> None:
    clip_type, fmt, codec = kind
    facts = classify_media({"Type": clip_type, "Format": fmt, "Video Codec": codec, "FPS": 29.97})
    assert facts["is_braw"] == (fmt == "BRAW")
    assert facts["is_multicam"] == (clip_type == "Multicam")
    assert not facts["is_still"]
    assert facts["has_video"] == (clip_type != "Audio")


@pytest.mark.parametrize("pair", FIXTURE["media_pool_clip_kinds"]["type_extension"])
def test_no_live_file_extension_reads_as_a_still(pair: list[str]) -> None:
    clip_type, ext = pair
    facts = classify_media({"Type": clip_type, "File Name": f"x.{ext}" if ext else ""})
    assert not facts["is_still"]
    assert facts["is_braw"] == (ext == "braw")


# -- media.find -------------------------------------------------------------------------------


def _names(result: Any) -> list[str]:
    return sorted(m.name for m in result.media)


def test_media_find_is_non_recursive_unless_asked() -> None:
    show = Show()
    flat = show.ok("media.find", {"bin_path": ["Footage"]})
    assert _names(flat) == ["multicam", "title card"]
    deep = show.ok("media.find", {"bin_path": ["Footage"], "recursive": True})
    assert _names(deep) == ["cam 24p", "cam A", "cam B", "multicam", "title card"]
    day1 = next(m for m in deep.media if m.name == "cam B")
    assert day1.bin_path == ("Footage", "Day 1") and day1.is_braw
    still = next(m for m in deep.media if m.name == "title card")
    assert still.is_still


def test_media_find_by_name_or_path_and_limit() -> None:
    show = Show()
    by_name = show.ok("media.find", {"name": "cam A", "recursive": True})
    assert [m.media_id for m in by_name.media] == [show.cam_a.GetUniqueId()]
    by_path = show.ok("media.find", {"path_contains": "cam 24p", "recursive": True})
    assert _names(by_path) == ["cam 24p"] and by_path.media[0].fps == 24.0
    limited = show.ok("media.find", {"recursive": True, "limit": 2})
    assert len(limited.media) == 2 and limited.truncated
    error = show.fails("media.find", {"bin_path": ["Nope"]})
    assert error.code == "bin_not_found"


def test_read_ops_need_the_target_project_and_switch_to_it() -> None:
    show = Show()
    result = show.ok("timeline.list", {}, target=show.locator("Other"))
    assert result.project == "Other"
    assert show.gw.get_context().project == "Show"  # the broker restored the human's project
    assert isinstance(show.locator("Other"), ProjectLocator)


def test_transitions_are_listed_but_are_neither_gaps_nor_overlaps() -> None:
    show = Show()
    tl = show.main
    tl.place("audio", 1, show.cam_a, 108000, 286, src_start=10)
    tl.place("audio", 1, show.cam_a, 108286, 100, src_start=900)
    tl.place_transition("audio", 1, cut=108286, length=12)  # as read live: 280..292
    result = show.ok("timeline.describe", {"timeline": {"name": "Main"}, "track_types": ["audio"]})
    track = result.tracks[0]
    assert (track.item_count, track.transitions, track.overlaps) == (3, 1, 0)
    assert track.gaps == []
    transition = next(i for i in track.items if i.type == "transition")
    assert (transition.record_in, transition.record_out) == (280, 292)
    assert transition.source_start is None and transition.fade_in is None
    assert [n.code for n in result.notes] == []
