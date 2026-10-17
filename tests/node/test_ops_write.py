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
    assert too_long.code == "fades_too_long"
    assert len(calls(show, "SetFades")) == 1  # refused before calling Resolve


# -- grades.copy -----------------------------------------------------------------------------


def _graded_world() -> tuple[Show, FakeTimelineItem, list[FakeTimelineItem]]:
    show = Show()
    reference = show.main.place("video", 1, show.cam_a, START, 100)
    reference.graph.nodes = [
        FakeNode("balance", tools=("Primaries",)),
        FakeNode("look", lut="Film/look.cube"),
    ]
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


def test_a_lut_with_no_node_to_return_to_is_refused_before_anything_is_copied() -> None:
    show, reference, targets = _graded_world()
    targets[1].graph.nodes = [FakeNode(), FakeNode(), FakeNode(lut="Per clip/third.cube")]
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert (error.code, error.detail["trap_id"]) == ("lut_node_missing", "append-ungraded")
    assert error.detail["completed"] == []
    assert calls(show, "CopyGrades") == []  # the targets' grades were not wiped
    assert targets[0].graph.nodes[0].lut == "Per clip/brighten.cube"


def test_a_lut_resolve_does_not_know_fails_after_a_bounce_with_completed() -> None:
    show, reference, targets = _graded_world()
    show.fake.unknown_luts.add("Per clip/brighten.cube")
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert error.code == "lut_reapply_failed"
    assert error.detail["trap_id"] == "setproperty-after-color"
    assert "RefreshLUTList" in str(error.detail["fix_hint"])
    assert error.detail["completed"] == []


def test_an_ungraded_reference_is_refused() -> None:
    show, reference, targets = _graded_world()
    reference.graph.nodes = [FakeNode()]  # as appended: one empty node
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert (error.code, error.detail["trap_id"]) == ("reference_ungraded", "append-ungraded")
    assert calls(show, "CopyGrades") == []
    forced = show.ok(
        "grades.copy", _grades_params(reference, targets, require_graded_reference=False)
    )
    assert forced.groups[0].reference_nodes == 1


def test_grades_errors_carry_the_groups_already_done() -> None:
    show, reference, targets = _graded_world()
    second_ref = show.main.place("video", 1, show.cam_b, START + 200, 50)
    second_ref.graph.nodes = [FakeNode("b", tools=("Primaries",))]
    third = show.branch.place("video", 1, show.cam_b, START + 300, 50)
    params = _grades_params(reference, targets)
    params["groups"].append(
        {
            "camera": "B",
            "reference": ref(second_ref),
            "reference_timeline": {"name": "Main"},
            "targets": [ref(third)],
        }
    )
    show.fake.faults.add("CopyGrades:no-op")
    error = show.fails("grades.copy", params)
    assert error.code == "grade_unverified" and error.detail["completed"] == []
    show.fake.faults.clear()
    original = second_ref.CopyGrades
    second_ref.CopyGrades = lambda items: True  # type: ignore[method-assign]
    error = show.fails("grades.copy", params)
    second_ref.CopyGrades = original  # type: ignore[method-assign]
    assert error.code == "grade_unverified"
    assert [g["camera"] for g in error.detail["completed"]] == ["A"]


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


# -- setters that report success without applying (the read-backs matter) ----------------------


def test_gain_that_does_not_stick_is_caught() -> None:
    show = Show()
    sound = show.main.place("audio", 1, show.cam_a, START, 100)
    show.fake.faults.add("SetProperties:no-op")
    error = show.fails(
        "item.set_audio_gain",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(sound), "gain_db": -6.0}]},
    )
    assert (error.code, error.detail["trap_id"]) == ("gain_unverified", "volume-key")


def test_fades_that_do_not_stick_are_caught() -> None:
    show = Show()
    item = show.main.place("video", 1, show.cam_a, START, 100)
    show.fake.faults.add("SetFades:no-op")
    error = show.fails(
        "item.set_fades",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(item), "fade_in": 10}]},
    )
    assert error.code == "fades_unverified"


def test_a_grade_copy_that_copies_nothing_is_caught() -> None:
    show, reference, targets = _graded_world()
    targets[1].graph.nodes = [FakeNode("x"), FakeNode("y")]  # same node count as the reference
    show.fake.faults.add("CopyGrades:no-op")
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert (error.code, error.detail["trap_id"]) == ("grade_unverified", "append-ungraded")


def test_a_lut_that_does_not_stick_is_caught() -> None:
    show, reference, targets = _graded_world()
    show.fake.faults.add("SetLUT:no-op")
    error = show.fails("grades.copy", _grades_params(reference, targets))
    assert error.code == "lut_unverified"
    assert error.detail["completed"] == []


def test_a_colour_space_that_does_not_stick_is_caught() -> None:
    show = Show()
    graphic = show.project.media_pool.import_rec709(("Footage",), "lower third")
    show.fake.faults.add("SetClipProperty:no-op")
    error = show.fails(
        "clip.set_input_color_space", {"items": [media(graphic)], "color_space": "Rec.709 (Scene)"}
    )
    assert (error.code, error.detail["trap_id"]) == (
        "colour_space_unverified",
        "rcm-input-colorspace",
    )


def test_a_colour_space_already_set_is_not_written_again() -> None:
    show = Show()
    graphic = show.project.media_pool.import_rec709(("Footage",), "lower third")
    params = {"items": [media(graphic)], "color_space": "Rec.709 (Scene)"}
    show.ok("clip.set_input_color_space", params)
    again = show.ok("clip.set_input_color_space", params)
    assert len(calls(show, "SetClipProperty")) == 1
    assert again.changes[0].before == again.changes[0].after == "Rec.709 (Scene)"


def test_a_duplicate_that_is_not_in_the_project_is_caught() -> None:
    show = Show()
    show.fake.faults.add("DuplicateTimeline:detached")
    error = show.fails("timeline.duplicate", {"source": {"name": "Main"}, "name": "Main v2"})
    assert error.code == "duplicate_unverified"


# -- setters jammed after Color -> Edit: bounce and retry, or fail with the right hint ----------


def _after_color(show: Show) -> None:
    show.fake.human_set_page("color")
    show.fake.human_set_page("edit")
    human_settles(show)


def test_fades_bounce_the_page_when_the_setter_jams() -> None:
    show = Show()
    item = show.main.place("video", 1, show.cam_a, START, 100)
    _after_color(show)
    result = show.ok(
        "item.set_fades",
        {"timeline": {"name": "Main"}, "targets": [{"item": ref(item), "fade_in": 10}]},
    )
    assert result.changes[0].after_in == 10
    assert [(n.code, n.trap_id) for n in result.notes] == [
        ("page_bounced", "setproperty-after-color")
    ]


def test_grades_and_luts_bounce_the_page_when_the_setters_jam() -> None:
    show, reference, targets = _graded_world()
    _after_color(show)
    result = show.ok("grades.copy", _grades_params(reference, targets))
    assert result.groups[0].targets[0].luts_after[1] == "Per clip/brighten.cube"
    assert {n.trap_id for n in result.notes} == {"setproperty-after-color"}


@pytest.mark.parametrize("op_name", ["item.set_fades", "item.set_audio_gain", "grades.copy"])
def test_a_jam_a_bounce_cannot_clear_fails_with_the_trap_and_its_fix(op_name: str) -> None:
    show, reference, targets = _graded_world()
    sound = show.main.place("audio", 1, show.cam_a, START, 100)
    show.fake.faults.add("OpenPage:keeps-jam")
    _after_color(show)
    params = {
        "item.set_fades": {
            "timeline": {"name": "Main"},
            "targets": [{"item": ref(reference), "fade_in": 10}],
        },
        "item.set_audio_gain": {
            "timeline": {"name": "Main"},
            "targets": [{"item": ref(sound), "gain_db": -2.0}],
        },
        "grades.copy": _grades_params(reference, targets),
    }[op_name]
    error = show.fails(op_name, params)
    assert error.detail["trap_id"] == "setproperty-after-color"
    assert "open another page and back" in str(error.detail["fix_hint"])
    assert "fit inside" not in str(error.detail["fix_hint"])


# -- timeline.duplicate reuse never touches what it did not make (F6) ---------------------------


def test_reuse_never_picks_the_source_itself() -> None:
    show = Show()
    before = show.main.GetSettings()
    error = show.fails(
        "timeline.duplicate",
        {
            "source": {"name": "Main"},
            "name": "Main",
            "if_exists": "reuse",
            "resolution": {"width": 1080, "height": 1920},
        },
    )
    assert error.code == "reuse_is_source"
    assert show.main.GetSettings() == before
    assert calls(show, "Timeline.SetSettings") == []


def test_reuse_refuses_a_timeline_that_differs_and_leaves_it_alone() -> None:
    show = Show()
    before = show.branch.GetSettings()
    error = show.fails(
        "timeline.duplicate",
        {
            "source": {"name": "Main"},
            "name": "Main [agent]",
            "if_exists": "reuse",
            "resolution": {"width": 1080, "height": 1920},
        },
    )
    assert error.code == "reuse_mismatch"
    assert {m["key"] for m in error.detail["mismatches"]} >= {"timelineResolutionWidth"}
    assert show.branch.GetSettings() == before
    assert calls(show, "Timeline.SetSettings") == []


def test_reuse_picks_up_a_matching_earlier_attempt_unchanged() -> None:
    show = Show()
    params = {
        "source": {"name": "Main"},
        "name": "Main 9x16",
        "resolution": {"width": 1080, "height": 1920},
    }
    first = show.ok("timeline.duplicate", params)
    writes = len(calls(show, "Timeline.SetSettings"))
    again = show.ok("timeline.duplicate", {**params, "if_exists": "reuse"})
    assert not again.created and again.timeline == first.timeline
    assert again.resolution is not None and again.resolution.width == 1080
    assert len(calls(show, "Timeline.SetSettings")) == writes


# -- strict params (F7, F12) ------------------------------------------------------------------


@pytest.mark.parametrize("gain", ["3", True, None])
def test_gain_must_be_a_number(gain: Any) -> None:
    show = Show()
    target = {"item": {"track_type": "audio", "track": 1, "item_id": "x"}, "gain_db": gain}
    with pytest.raises(ValidationError):
        submit_op(
            show.broker,
            "item.set_audio_gain",
            {"targets": [target]},
            target=show.locator(),
            holder=AGENT,
        )


@pytest.mark.parametrize("frames", [True, 3.0, "3"])
def test_fade_lengths_must_be_whole_frames(frames: Any) -> None:
    show = Show()
    target = {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "fade_in": frames}
    with pytest.raises(ValidationError):
        submit_op(
            show.broker,
            "item.set_fades",
            {"targets": [target]},
            target=show.locator(),
            holder=AGENT,
        )


@pytest.mark.parametrize(
    ("op_name", "params"),
    [
        (
            "item.set_audio_gain",
            {
                "targets": [
                    {"item": {"track_type": "audio", "track": 1, "item_id": "x"}, "gain_db": -1.0},
                    {"item": {"track_type": "audio", "track": 1, "item_id": "x"}, "gain_db": -2.0},
                ]
            },
        ),
        (
            "item.set_fades",
            {
                "targets": [
                    {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "fade_in": 1},
                    {"item": {"track_type": "video", "track": 1, "item_id": "x"}, "fade_out": 2},
                ]
            },
        ),
        (
            "grades.copy",
            {
                "groups": [
                    {
                        "reference": {"track_type": "video", "track": 1, "item_id": "r"},
                        "targets": [
                            {"track_type": "video", "track": 1, "item_id": "t"},
                            {"track_type": "video", "track": 1, "item_id": "t"},
                        ],
                    }
                ]
            },
        ),
        (
            "grades.copy",
            {
                "groups": [
                    {
                        "reference": {"track_type": "video", "track": 1, "item_id": "r"},
                        "targets": [{"track_type": "video", "track": 1, "item_id": "r"}],
                    }
                ]
            },
        ),
        (
            "clip.set_input_color_space",
            {"items": [{"unique_id": "m"}, {"unique_id": "m"}], "color_space": "Rec.709 (Scene)"},
        ),
        ("markers.add", {"markers": [{"frame": 10, "name": "a"}, {"frame": 10, "name": "b"}]}),
    ],
)
def test_the_same_item_twice_in_one_request_is_refused(
    op_name: str, params: dict[str, Any]
) -> None:
    show = Show()
    with pytest.raises(ValidationError) as err:
        submit_op(show.broker, op_name, params, target=show.locator(), holder=AGENT)
    assert "listed" in str(err.value) or "both a reference and a target" in str(err.value)
    assert show.broker.pending() == []


def test_duplicate_marker_frames_are_refused_before_any_marker_is_added() -> None:
    show = Show()
    with pytest.raises(ValidationError):
        submit_op(
            show.broker,
            "markers.add",
            {
                "timeline": {"name": "Main"},
                "markers": [{"frame": 10, "name": "a"}, {"frame": 10, "name": "b"}, {"frame": 20}],
            },
            target=show.locator(),
            holder=AGENT,
        )
    assert calls(show, "AddMarker") == []


@pytest.mark.parametrize("space", [" Rec.709 (Scene)", "Rec.709 (Scene) ", "   "])
def test_colour_space_names_with_stray_spaces_are_refused(space: str) -> None:
    show = Show()
    with pytest.raises(ValidationError):
        submit_op(
            show.broker,
            "clip.set_input_color_space",
            {"items": [media(show.cam_a)], "color_space": space},
            target=show.locator(),
            holder=AGENT,
        )


def test_an_unknown_colour_space_is_refused_before_writing() -> None:
    show = Show()
    graphic = show.project.media_pool.import_rec709(("Footage",), "lower third")
    error = show.fails(
        "clip.set_input_color_space", {"items": [media(graphic)], "color_space": "Rec 709 Scene"}
    )
    assert (error.code, error.detail["trap_id"]) == ("color_space_unknown", "rcm-input-colorspace")
    assert "Rec.709 (Scene)" in error.detail["known"]
    assert calls(show, "SetClipProperty") == []
