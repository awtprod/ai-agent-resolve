"""Write ops other than append: duplicate, gain, fades, grades, input colour space, markers."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from aar.node.ops import submit_op

from .fake_resolve import FakeNode, FakeTimelineItem
from .ops_support import AGENT, BRANCH, Show, human_settles, media

START = 108000


def ref(item: FakeTimelineItem) -> dict[str, Any]:
    return {
        "track_type": item.track_type,
        "track": item.timeline.track_index_of(item),
        "item_id": item.GetUniqueId(),
    }


def calls(show: Show, name: str) -> list[tuple[Any, ...]]:
    return [c for c in show.fake.calls if c[0] == name]


# -- timeline.duplicate ----------------------------------------------------------------------


def test_duplicate_makes_a_verified_copy_and_refuses_a_taken_name() -> None:
    show = Show()
    show.main.place("video", 1, show.cam_a, START, 100)
    result = show.ok("timeline.duplicate", {"source": {"name": "Main"}, "name": "Main v2"})
    assert result.created and result.timeline.name == "Main v2" and result.timeline_count == 3
    copy = show.project.timeline("Main v2")
    assert result.timeline.unique_id == copy.GetUniqueId()
    assert [(i.start, i.end) for i in copy.items("video")] == [(START, START + 100)]

    taken = show.fails("timeline.duplicate", {"source": {"name": "Main"}, "name": "Main v2"})
    assert taken.code == "timeline_exists"
    reused = show.ok(
        "timeline.duplicate", {"source": {"name": "Main"}, "name": "Main v2", "if_exists": "reuse"}
    )
    assert not reused.created and reused.timeline.unique_id == copy.GetUniqueId()
    assert len(calls(show, "DuplicateTimeline")) == 1


def test_duplicate_with_custom_resolution_keeps_the_colour_science() -> None:
    show = Show()
    result = show.ok(
        "timeline.duplicate",
        {
            "source": {"name": "Main"},
            "name": "Main 9x16",
            "resolution": {"width": 1080, "height": 1920},
        },
    )
    assert result.resolution is not None
    assert (result.resolution.width, result.resolution.height) == (1080, 1920)
    assert [(n.code, n.trap_id) for n in result.notes] == [
        ("colour_settings_restored", "custom-res-resets-colour-science")
    ]
    settings = show.project.timeline("Main 9x16").GetSettings()
    # The custom resolution reset these to plain davinciYRGB; the op set them back.
    assert settings["colorScienceMode"] == "davinciYRGBColorManagedv2"
    assert settings["outputDRT"] == "Luminance Mapping"
    assert all(c.actual == c.expected for c in result.colour)
    assert {c.key for c in result.colour} >= {"colorScienceMode", "colorSpaceTimeline", "inputDRT"}
    assert show.main.GetSettings().get("useCustomSettings") is None  # source untouched
    assert calls(show, "Project.SetSettings") == []


# -- item.set_audio_gain ---------------------------------------------------------------------


def test_gain_uses_audio_volume_and_reads_it_back() -> None:
    show = Show()
    sound = show.main.place("audio", 1, show.cam_a, START, 100)
    result = show.ok(
        "item.set_audio_gain",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(sound), "gain_db": -6.0}]},
    )
    change = result.changes[0]
    assert (change.before_db, change.after_db, change.page_bounced) == (0.0, -6.0, False)
    assert sound.props["AudioVolume"] == -6.0
    # trap volume-key: the only key sent is AudioVolume ("Volume" would be refused).
    assert [c[2] for c in calls(show, "SetProperties")] == [{"AudioVolume": -6.0}]


def test_gain_bounces_the_page_when_setters_jam_after_color() -> None:
    show = Show()
    sound = show.main.place("audio", 1, show.cam_a, START, 100)
    show.fake.human_set_page("color")
    show.fake.human_set_page("edit")  # property setters now return False
    human_settles(show)
    result = show.ok(
        "item.set_audio_gain",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(sound), "gain_db": -3.0}]},
    )
    assert result.changes[0].page_bounced and result.changes[0].after_db == -3.0
    assert [(n.code, n.trap_id) for n in result.notes] == [
        ("page_bounced", "setproperty-after-color")
    ]
    assert [c[1] for c in calls(show, "OpenPage")][:2] == ["media", "edit"]
    assert show.fake.page == "edit"


def test_gain_refuses_video_items() -> None:
    show = Show()
    picture = show.main.place("video", 1, show.cam_a, START, 100)
    error = show.fails(
        "item.set_audio_gain",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(picture), "gain_db": -3.0}]},
    )
    assert (error.code, error.detail["trap_id"]) == ("not_audio_item", "volume-key")
    assert calls(show, "SetProperties") == []


# -- item.set_fades --------------------------------------------------------------------------


def test_fades_are_set_read_back_and_idempotent() -> None:
    show = Show()
    item = show.main.place("video", 1, show.cam_a, START, 100)
    params = {"timeline": {"name": "Main"}, "targets": [{"item": ref(item), "fade_in": 10}]}
    first = show.ok("item.set_fades", params)
    change = first.changes[0]
    assert (change.before_in, change.after_in, change.after_out) == (0, 10, 0)
    show.ok("item.set_fades", params)
    assert len(calls(show, "SetFades")) == 1
    too_long = show.fails(
        "item.set_fades",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(item), "fade_out": 500}]},
    )
    assert too_long.code == "set_fades_failed"


# -- grades.copy -----------------------------------------------------------------------------


def _graded_world() -> tuple[Show, FakeTimelineItem, list[FakeTimelineItem]]:
    show = Show()
    reference = show.main.place("video", 1, show.cam_a, START, 100)
    reference.graph.nodes = [FakeNode("balance"), FakeNode("look", lut="Film/look.cube")]
    targets = [show.branch.place("video", 1, show.cam_a, START + i * 50, 50) for i in range(2)]
    targets[0].graph.nodes = [FakeNode(lut="Per clip/brighten.cube")]
    return show, reference, targets


def _grades_params(
    reference: FakeTimelineItem, targets: list[FakeTimelineItem], **kw: Any
) -> dict[str, Any]:
    return {
        "timeline": BRANCH,
        "groups": [
            {
                "camera": "A",
                "reference": ref(reference),
                "reference_timeline": {"name": "Main"},
                "targets": [ref(t) for t in targets],
            }
        ],
        **kw,
    }


def test_grades_copy_across_timelines_and_puts_per_clip_luts_back() -> None:
    show, reference, targets = _graded_world()
    result = show.ok("grades.copy", _grades_params(reference, targets))
    first, second = result.groups[0].targets
    assert result.groups[0].reference_nodes == 2
    assert (first.nodes_before, first.nodes_after) == (1, 2)
    assert first.luts_after == {1: "Per clip/brighten.cube", 2: "Film/look.cube"}
    assert [(r.node, r.after) for r in first.reapplied] == [(1, "Per clip/brighten.cube")]
    assert second.luts_after == {2: "Film/look.cube"} and second.reapplied == []


def test_without_reapplying_the_copy_wipes_per_clip_luts() -> None:
    show, reference, targets = _graded_world()
    result = show.ok("grades.copy", _grades_params(reference, targets, reapply_luts=False))
    assert result.groups[0].targets[0].luts_after == {2: "Film/look.cube"}


def test_grades_copy_fails_loudly_when_a_lut_cannot_go_back() -> None:
    show, reference, targets = _graded_world()
    targets[1].graph.nodes = [FakeNode(), FakeNode(), FakeNode(lut="Per clip/third.cube")]
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert (error.code, error.detail["trap_id"]) == ("lut_node_missing", "append-ungraded")

    show, reference, targets = _graded_world()
    show.fake.unknown_luts.add("Per clip/brighten.cube")
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert error.code == "lut_reapply_failed"


# -- clip.set_input_color_space ----------------------------------------------------------------


def test_input_colour_space_is_set_per_listed_item() -> None:
    show = Show()
    pool = show.project.media_pool
    graphic = pool.import_rec709(("Footage",), "lower third")
    assert graphic.props["Input Color Space"] == "Blackmagic Design Film Gen 5"  # the trap
    result = show.ok(
        "clip.set_input_color_space",
        {"items": [media(graphic)], "color_space": "Rec.709 (Scene)"},
    )
    change = result.changes[0]
    assert (change.before, change.after) == ("Blackmagic Design Film Gen 5", "Rec.709 (Scene)")
    assert graphic.props["Input Color Space"] == "Rec.709 (Scene)"
    assert calls(show, "Project.SetSettings") == [] and calls(show, "Timeline.SetSettings") == []


def test_input_colour_space_never_touches_braw() -> None:
    show = Show()
    graphic = show.project.media_pool.import_rec709(("Footage",), "lower third")
    error = show.fails(
        "clip.set_input_color_space",
        {"items": [media(graphic), media(show.cam_b)], "color_space": "Rec.709 (Scene)"},
    )
    assert (error.code, error.detail["trap_id"]) == ("braw_refused", "rcm-input-colorspace")
    assert error.detail["braw"] == [show.cam_b.GetUniqueId()]
    assert calls(show, "SetClipProperty") == []  # all or nothing: not even the non-BRAW one
    assert show.cam_b.props["Input Color Space"] == "Blackmagic Design Film Gen 5"


@pytest.mark.parametrize(
    "selector",
    [{"bin": "Footage"}, {"bin_path": ["Footage"], "recursive": True}, {"folder": "Footage"}],
)
def test_input_colour_space_refuses_bin_selectors(selector: dict[str, Any]) -> None:
    show = Show()
    with pytest.raises(ValidationError) as err:
        submit_op(
            show.broker,
            "clip.set_input_color_space",
            {"items": [media(show.cam_a)], "color_space": "Rec.709 (Scene)", **selector},
            target=show.locator(),
            holder=AGENT,
        )
    assert "rcm-input-colorspace" in str(err.value)


def test_input_colour_space_refuses_media_without_its_own_picture() -> None:
    show = Show()
    error = show.fails(
        "clip.set_input_color_space",
        {"items": [media(show.music)], "color_space": "Rec.709 (Scene)"},
    )
    assert error.code == "not_video_media"


# -- markers.add -----------------------------------------------------------------------------


def test_markers_are_added_read_back_and_rerun_safely() -> None:
    show = Show()
    params = {
        "timeline": {"name": "Main"},
        "markers": [
            {"frame": 30, "color": "Red", "name": "laugh", "note": "big"},
            {"frame": 90, "color": "Green", "duration": 5, "custom_data": "aar:review:1"},
        ],
    }
    first = show.ok("markers.add", params)
    assert [(m.frame, m.status, m.color, m.duration) for m in first.markers] == [
        (30, "added", "Red", 1),
        (90, "added", "Green", 5),
    ]
    assert first.markers[1].custom_data == "aar:review:1"
    again = show.ok("markers.add", params)
    assert [m.status for m in again.markers] == ["present", "present"]
    assert len(calls(show, "AddMarker")) == 2
    conflict = show.fails(
        "markers.add", {"timeline": {"name": "Main"}, "markers": [{"frame": 30, "name": "other"}]}
    )
    assert conflict.code == "marker_conflict"
    assert len(calls(show, "AddMarker")) == 2
