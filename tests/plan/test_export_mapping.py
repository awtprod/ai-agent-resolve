"""FCPXML export mapping, one plan concept at a time, and every typed refusal."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Any

import pytest
from apply_builders import plan_data, seg, source
from export_support import FIXTURES, ccn, export, load_plan, permissive

from aar.plan.apply.errors import CompileError
from aar.plan.export import EXPORT_CODES, ExportError, build_layout, export_fcpxml
from aar.plan.export.fcpxml import number
from aar.plan.export.layout import transform_spec
from aar.plan.model import EditPlan, Segment
from aar.plan.profile import ProfileError, parse_profile


def _plan(**kwargs: Any) -> EditPlan:
    # The default source has a size, so its clips may move (reframe/transform need one).
    kwargs.setdefault("sources", [source(frame_size=[1920, 1080])])
    return EditPlan.model_validate(plan_data(**kwargs))


def _xml(plan: EditPlan, **kwargs: Any) -> ET.Element:
    return ET.fromstring(export(plan, **kwargs).fcpxml)


def _spine(root: ET.Element) -> list[ET.Element]:
    return list(root.find("library/event/project/sequence/spine"))  # type: ignore[arg-type]


def _refused(plan: EditPlan, code: str, **kwargs: Any) -> ExportError:
    with pytest.raises(ExportError) as caught:
        export(plan, **kwargs)
    assert caught.value.code == code, caught.value.as_dict()
    return caught.value


def _report(plan: EditPlan, **kwargs: Any) -> set[tuple[str, str]]:
    report = export(plan, **kwargs).report
    return {(i["feature"], i["location"]) for i in report["not_expressible"]}


# -- error codes -------------------------------------------------------------------------------


def test_export_codes_are_stable() -> None:
    assert sorted(EXPORT_CODES) == sorted(
        [
            "lint_errors",
            "fps_not_representable",
            "bad_timeline_start",
            "empty_timeline",
            "unknown_source",
            "zero_length_segment",
            "media_not_on_edit_pc",
            "source_size_required",
            "conflicting_source_durations",
            "pool_source_not_exportable",
            "bad_media_map",
            "outro_audio_track_missing",
            "duplicate_transition",
            "no_handle_for_transition",
            "transitions_overlap",
            "duplicate_keyframe_time",
            "non_finite_value",
            "unsafe_text",
            "fade_too_long",
            "item_outside_timeline",
            "self_check_failed",
            "output_exists",
            "output_conflict",
        ]
    )


def test_export_error_takes_only_its_own_codes() -> None:
    assert isinstance(ExportError("empty_timeline", "m", "f"), CompileError)
    with pytest.raises(ValueError, match="unknown ExportError code"):
        ExportError("template_too_large", "m", "f")  # a compile code, not an export one
    with pytest.raises(ValueError, match="unknown CompileError code"):
        CompileError("fps_not_representable", "m", "f")
    assert CompileError("lint_errors", "m", "f").as_dict()["code"] == "lint_errors"


# -- timing ------------------------------------------------------------------------------------


def test_long_timeline_has_no_drift() -> None:
    frames = 10 * 3600 * 24 + 17  # ten hours and 17 frames at 23.976
    plan = _plan(sources=[source(duration=frames + 10)], segments=[seg(a=0, b=frames)])
    root = _xml(plan)
    sequence = root.find("library/event/project/sequence")
    assert sequence is not None
    assert sequence.get("duration") == f"{frames * 1001}/24000s"
    assert _spine(root)[0].get("duration") == f"{frames * 1001}/24000s"


@pytest.mark.parametrize(
    ("fps", "start", "tc_start", "first"),
    [
        ({"num": 25, "den": 1}, "00:00:00:00", "0s", "0s"),
        # 1080000 frames * 1001/30000 s is exactly 36036 s, so it reduces to whole seconds.
        ({"num": 30000, "den": 1001}, "10:00:00:00", "36036s", "36036s"),
        ({"num": 30000, "den": 1001}, "00:00:00:01", "1001/30000s", "1001/30000s"),
        ({"num": 24, "den": 1}, "00:59:58:05", "86357/24s", "86357/24s"),
        ({"num": 50, "den": 1}, "01:00:00:00", "3600s", "3600s"),
    ],
)
def test_timeline_start_comes_from_the_profile(
    fps: dict[str, int], start: str, tc_start: str, first: str
) -> None:
    plan = _plan(fps=fps)
    root = _xml(plan, profile=permissive(start))
    sequence = root.find("library/event/project/sequence")
    assert sequence is not None and sequence.get("tcStart") == tc_start
    assert _spine(root)[0].get("offset") == first


def test_the_ccn_profile_starts_at_one_hour() -> None:
    assert ccn().export.timeline_start == "01:00:00:00"
    profile = parse_profile({"name": "x", "qa": ccn().qa.model_dump(), "rules": {}})
    assert profile.export.timeline_start == "01:00:00:00"  # older profiles still load


def test_refuses_a_rate_fcpxml_cannot_hold_exactly() -> None:
    error = _refused(_plan(fps={"num": 23976, "den": 1000}), "fps_not_representable")
    assert "24000/1001" in error.fix_hint


def test_refuses_a_bad_timeline_start() -> None:
    _refused(
        _plan(fps={"num": 24, "den": 1}), "bad_timeline_start", profile=permissive("00:00:00:24")
    )
    with pytest.raises(ProfileError):
        permissive("1:00:00:00")


# -- tracks and lanes --------------------------------------------------------------------------


def test_v2_and_v3_are_connected_lanes_over_a_v1_gap() -> None:
    plan = _plan(
        segments=[
            seg(1, a=0, b=48),
            seg(1, a=100, b=148, record_in=96),  # a 48-frame hole on V1
            seg(3, a=10, b=30, record_in=60),  # inside the hole
            seg(2, a=0, b=24, record_in=10),
        ]
    )
    spine = _spine(_xml(plan))
    assert [e.tag for e in spine] == ["asset-clip", "gap", "asset-clip"]
    gap = spine[1]
    # 01:00:00:00 = 86400 frames; the gap runs 48..96 and its local time is sequence time.
    assert gap.get("offset") == gap.get("start") == f"{(86400 + 48) * 1001}/24000s"
    v3 = gap.find("asset-clip")
    assert v3 is not None and v3.get("lane") == "2"
    assert v3.get("offset") == f"{(86400 + 60) * 1001}/24000s"
    v2 = spine[0].find("asset-clip")
    assert v2 is not None and v2.get("lane") == "1" and v2.get("offset") == "10010/24000s"


def test_track_gain_is_folded_into_clip_volume_and_mute_is_kept_audible() -> None:
    plan = _plan(
        segments=[seg(1, a=0, b=24, gain_db=-3.0), seg(2, a=0, b=24, record_in=0)],
        audio_tracks=[
            {"name": "A1", "gain_db": 1.5, "mute": True, "bus": "A5"},
            {"name": "A2", "gain_db": -2.0},
            {"name": "A5"},
        ],
    )
    result = export(plan)
    spine = _spine(ET.fromstring(result.fcpxml))
    assert spine[0].find("adjust-volume").get("amount") == "-1.5dB"  # type: ignore[union-attr]
    assert spine[0].find("asset-clip/adjust-volume").get("amount") == "-2dB"  # type: ignore[union-attr]
    assert spine[0].get("srcEnable") is None and spine[0].get("enabled") is None
    found = {(i["feature"], i["location"]) for i in result.report["not_expressible"]}
    assert ("audio_mute", "audio_tracks[0].mute") in found
    assert ("audio_bus", "audio_tracks[0].bus") in found
    assert ("audio_track_empty", "audio_tracks[2]") in found
    notes = {(i["feature"], i["location"]) for i in result.report["notes"]}
    assert ("audio_track_gain", "audio_tracks[0].gain_db") in notes


def test_outro_sound_follows_its_named_track() -> None:
    base = {
        "sources": [source("a"), source("o", duration=60)],
        "segments": [seg(1, "a", 0, 48)],
        "audio_tracks": [{"name": "A1"}, {"name": "A2"}, {"name": "Bus"}],
    }
    same = _plan(**base, outro={"source": "o", "src_out": 60, "audio_track": "A1"})
    spine = _spine(_xml(same))
    assert spine[1].get("srcEnable") is None and spine[1].find("asset-clip") is None
    assert ("outro_sound_track", "outro.audio_track") not in _report(same)
    # Round 3: the outro is always one clip with its own sound (Resolve ignores srcEnable); a
    # different planned track is reported to fix by hand.
    bus = _plan(**base, outro={"source": "o", "src_out": 60, "audio_track": "Bus"})
    spine = _spine(_xml(bus))
    assert "srcEnable" not in spine[1].attrib and spine[1].find("asset-clip") is None
    assert ("outro_sound_track", "outro.audio_track") in _report(bus)
    off_spine = _plan(**base, outro={"source": "o", "src_out": 60, "track": 2, "audio_track": "A1"})
    assert ("outro_sound_track", "outro.audio_track") in _report(off_spine)
    _refused(
        _plan(**{**base, "audio_tracks": []}, outro={"source": "o", "src_out": 60}),
        "outro_audio_track_missing",
    )


# -- transitions -------------------------------------------------------------------------------


def _join(out_dur: int | None, in_src: int, frames: int, video: bool = True) -> EditPlan:
    tx: dict[str, Any] = {"after": 0, "video": "dissolve" if video else "cut"}
    tx["video_frames"] = frames if video else 0
    tx["audio"] = {"type": "Cross Fade 0 dB", "frames": frames}
    return _plan(
        sources=[source("a", duration=out_dur), source("b")],
        segments=[seg(1, "a", 100, 148), seg(1, "b", in_src, in_src + 48)],
        transitions=[tx],
    )


@pytest.mark.parametrize(
    ("out_dur", "in_src", "alignment", "before"),
    [(1000, 100, "center", 5), (1000, 3, "right", 0), (None, 100, "left", 11)],
)
def test_transition_alignment_follows_the_handles(
    out_dur: int | None, in_src: int, alignment: str, before: int
) -> None:
    plan = _join(out_dur, in_src, 11)
    result = export(plan)
    assert [t["alignment"] for t in result.report["transitions"]] == [alignment]
    spine = _spine(ET.fromstring(result.fcpxml))
    a, transition, b = spine
    # The cut is at record 48; the window starts `before` frames ahead of it (Apple: offset and
    # duration give the transition's position and extent) and plays from the handles: the
    # clips keep their plan ranges and abut (round 3).
    assert transition.get("offset") == f"{(86400 + 48 - before) * 1001}/24000s"
    assert transition.get("duration") == "11011/24000s"
    assert a.get("duration") == f"{48 * 1001}/24000s"
    assert b.get("offset") == f"{(86400 + 48) * 1001}/24000s"
    assert b.get("start") == f"{in_src * 1001}/24000s"
    assert [c.tag for c in transition] == ["filter-video", "filter-audio"]


def test_transition_without_handles_is_refused() -> None:
    error = _refused(_join(148, 3, 11), "no_handle_for_transition")
    assert error.details["location"] == "transitions[0]"


def test_overlapping_transitions_are_refused() -> None:
    plan = _plan(
        segments=[seg(1, a=100, b=148), seg(1, a=200, b=210), seg(1, a=300, b=348)],
        transitions=[
            {"after": 0, "video": "dissolve", "video_frames": 12, "audio": {"frames": 12}},
            {"after": 1, "video": "dissolve", "video_frames": 12, "audio": {"frames": 12}},
        ],
    )
    _refused(plan, "transitions_overlap")


def test_two_transitions_on_one_join_are_refused() -> None:
    plan = _join(1000, 100, 4)
    data = plan.model_dump(mode="json")
    data["transitions"] = data["transitions"] * 2
    _refused(EditPlan.model_validate(data), "duplicate_transition")


def test_dissolve_carries_an_audio_crossfade_of_its_length() -> None:
    data = _join(1000, 100, 10).model_dump(mode="json")
    data["transitions"][0]["audio"] = None
    found = _report(EditPlan.model_validate(data))
    assert ("audio_crossfade_length", "transitions[0].audio") in found
    data["transitions"][0]["audio"] = {"type": "Cross Fade +3 dB", "frames": 6}
    found = _report(EditPlan.model_validate(data))
    assert ("audio_crossfade_length", "transitions[0].audio") in found
    assert ("audio_transition_type", "transitions[0].audio.type") in found


def test_non_dissolve_video_is_a_dissolve_placeholder() -> None:
    data = _join(1000, 100, 10).model_dump(mode="json")
    data["transitions"][0]["video"] = "wipe"
    assert ("transition_type", "transitions[0].video") in _report(EditPlan.model_validate(data))


def test_audio_only_crossfade_is_a_split_edit_not_a_transition() -> None:
    """Round 3: a transition element moved the picture cut in Resolve; only the J/L form is
    written."""
    a, b = _spine(_xml(_join(1000, 100, 8, video=False)))
    assert a.get("audioDuration") == f"{(48 + 4) * 1001}/24000s"
    assert b.get("audioStart") == f"{(100 - 4) * 1001}/24000s"


def test_off_spine_and_gapped_joins_are_reported_not_exported() -> None:
    on_v2 = _plan(
        segments=[seg(2, a=100, b=148), seg(2, a=200, b=248)],
        transitions=[{"after": 0, "audio": {"frames": 4}}],
    )
    result = export(on_v2)
    assert result.report["transitions"] == []
    assert ("transition", "transitions[0]") in {
        (i["feature"], i["location"]) for i in result.report["not_expressible"]
    }
    gapped = _plan(
        segments=[seg(1, a=100, b=148), seg(1, a=200, b=248, record_in=60)],
        transitions=[{"after": 0, "audio": {"frames": 4}}],
    )
    assert ("transition", "transitions[0]") in _report(gapped)


def test_head_after_a_gap_follows_the_gap() -> None:
    plan = _plan(
        segments=[seg(1, a=0, b=48, record_in=50)],
        head={"video": "dissolve", "video_frames": 10},
    )
    gap, transition, clip = _spine(_xml(plan))
    assert gap.tag == "gap" and gap.get("duration") == f"{50 * 1001}/24000s"  # 0..50, no overlap
    assert transition.get("offset") == clip.get("offset") == f"{(86400 + 50) * 1001}/24000s"


def test_head_off_the_spine_and_outro_off_the_spine_are_reported() -> None:
    plan = _plan(
        sources=[source("a"), source("o", duration=60)],
        segments=[seg(2, "a", 0, 48), seg(1, "a", 100, 148, record_in=10)],
        head={"video": "dissolve", "video_frames": 5},
        audio_tracks=[{"name": "A1"}, {"name": "A2"}],
        outro={
            "source": "o",
            "src_in": 20,
            "src_out": 60,
            "track": 2,
            "audio_track": "A2",
            "transition": {"video": "dissolve", "video_frames": 8, "audio": {"frames": 4}},
        },
    )
    found = _report(plan)
    assert ("head_transition", "head") in found
    assert ("outro_transition", "outro.transition") in found
    assert ("outro_audio_transition", "outro.transition.audio") in found


# -- transforms and reframes -------------------------------------------------------------------


def _segment(**kwargs: Any) -> Segment:
    return Segment.model_validate({"source": "a", "src_in": 0, "src_out": 48, **kwargs})


def test_static_transform_units() -> None:
    spec = transform_spec(
        _segment(transform={"zoom": 2.0, "pan": 108, "tilt": -54, "flip_v": True}),
        (1920, 1080),
        1920,
        1080,
    )
    assert spec.position == (10.0, -5.0)  # percent of the 1080 frame height, y up
    assert spec.scale == (2.0, -2.0)
    assert spec.keyframes == ()


def test_reframe_window_maths() -> None:
    keys = [{"time": 0, "x": 0.25, "y": 0.5, "zoom": 2.0}, {"time": 24, "x": 0.5, "zoom": 2.0}]
    spec = transform_spec(_segment(reframe=keys, transform={"zoom": 1.5}), (1920, 1080), 1920, 1080)
    first, second = spec.keyframes
    # Same aspect, so fill is 1:1. x 0.25 is 480 px left of centre; the total zoom 2 * 1.5 = 3
    # scales that offset, and moving the window left moves the picture right.
    assert math.isclose(first.position[0], 3 * 480 / 1080 * 100)
    assert first.position[1] == 0.0
    assert first.scale == second.scale == (3.0, 3.0)
    assert second.position == (0.0, 0.0)
    assert (first.time, second.time) == (0, 24)


@pytest.mark.parametrize(
    ("frame_size", "timeline", "key", "position"),
    [
        # A 16:9 source on a 9:16 short fills to 3413.33 x 1920; x 0.25 is 853.33 px left.
        ((1920, 1080), (1080, 1920), {"x": 0.25}, (0.25 * 1920 * (1920 / 1080) / 1920 * 100, 0.0)),
        # A face track's sampled size has the same aspect, so the same crop.
        ((640, 360), (1080, 1920), {"x": 0.25}, (0.25 * 1920 * (1920 / 1080) / 1920 * 100, 0.0)),
        # A 9:16 source on a 16:9 timeline fills to 1920 x 3413.33; y 0.25 is 853.33 px up.
        ((1080, 1920), (1920, 1080), {"y": 0.25}, (0.0, -0.25 * 1920 * (1920 / 1080) / 1080 * 100)),
        # A 4:3 source on 16:9 fills to 1920 x 1440 (the height is cropped), x 1.0 = 960 px right.
        ((1440, 1080), (1920, 1080), {"x": 1.0}, (-960 / 1080 * 100, 0.0)),
    ],
)
def test_reframe_is_in_source_units_after_fill(
    frame_size: tuple[int, int],
    timeline: tuple[int, int],
    key: dict[str, float],
    position: tuple[float, float],
) -> None:
    spec = transform_spec(_segment(reframe=[{"time": 0, **key}]), frame_size, *timeline)
    assert spec.position == pytest.approx(position, abs=1e-9)
    assert spec.scale == (1.0, 1.0)


def test_keyframes_are_clip_local_and_a_single_key_is_static() -> None:
    plan = _plan(
        segments=[
            seg(1, a=100, b=148, reframe=[{"time": 0}, {"time": 47, "x": 0.6}]),
            seg(1, a=0, b=48, reframe=[{"time": 5, "zoom": 1.5}]),
        ]
    )
    spine = _spine(_xml(plan))
    times = [
        k.get("time")
        for k in spine[0].findall(
            "adjust-transform/param[@name='scale']/keyframeAnimation/keyframe"
        )
    ]
    assert times == [f"{100 * 1001}/24000s", f"{147 * 1001}/24000s"]
    static = spine[1].find("adjust-transform")
    assert static is not None and static.get("scale") == "1.5 1.5" and len(static) == 0


def test_duplicate_reframe_times_are_refused() -> None:
    _refused(
        _plan(segments=[seg(reframe=[{"time": 3}, {"time": 3, "x": 0.6}])]),
        "duplicate_keyframe_time",
    )


def test_numbers_round_trip_exactly() -> None:
    for value in (-2.0, 0.1 + 0.2, 1e-7, -1e-300, 123456789.125, -0.0, 5e20):
        text = number(value)
        assert "e" not in text.lower()
        assert float(text) == value


# -- titles, markers, captions, colour ---------------------------------------------------------


def test_titles_and_markers() -> None:
    plan = _plan(
        segments=[seg(a=0, b=96)],
        titles=[{"lines": ["A & B", "<two>"], "record_in": 10, "record_out": 40}],
        markers=[{"record": 5, "name": "Hi", "note": "n", "duration": 3, "color": "Blue"}],
    )
    result = export(plan)
    clip = _spine(ET.fromstring(result.fcpxml))[0]
    title = clip.find("title")
    assert title is not None and title.find("text/text-style").text == "A & B\n<two>"  # type: ignore[union-attr]
    # Round 3: Resolve imports no FCPXML marker; none is written, each is listed with its TC.
    assert clip.find("marker") is None
    assert result.report["not_imported"]["markers"] == [
        {
            "location": "markers[0]",
            "timecode": "01:00:00:05",
            "record_frame": 5,
            "duration": 3,
            "name": "Hi",
            "color": "Blue",
            "note": "n",
        }
    ]
    items = {i["location"]: i for i in result.report["not_expressible"]}
    assert ("title_template", "titles[0]") in {(i["feature"], k) for k, i in items.items()}
    assert items["markers[0]"]["action"] == "add it by hand at 01:00:00:05"
    assert "Blue" in items["markers[0]"]["detail"] and "note 'n'" in items["markers[0]"]["detail"]


@pytest.mark.parametrize(
    "extra",
    [
        {"markers": [{"record": 24}]},
        {"markers": [{"record": 20, "duration": 5}]},
        {"titles": [{"lines": ["x"], "record_in": 10, "record_out": 25}]},
    ],
)
def test_items_past_the_end_are_refused(extra: dict[str, Any]) -> None:
    _refused(_plan(**extra), "item_outside_timeline")


def test_captions_and_colour_are_reported_not_exported() -> None:
    plan = _plan(
        sources=[
            source(
                "a",
                color={
                    "input_color_space": "Rec.709",
                    "cdl": {"saturation": 1.2},
                    "lut": "Show LUT",
                },
            )
        ],
        captions=[{"lines": ["hi"], "record_in": 0, "record_out": 10}],
    )
    result = export(plan)
    assert "caption" not in result.fcpxml
    found = {(i["feature"], i["location"]) for i in result.report["not_expressible"]}
    assert {
        ("captions", "captions"),
        ("input_color_space", "sources[0].color.input_color_space"),
        ("cdl", "sources[0].color.cdl"),
        ("lut", "sources[0].color.lut"),
    } <= found


# -- refusals ----------------------------------------------------------------------------------


def test_lint_errors_are_refused() -> None:
    plan = load_plan(FIXTURES / "short_broken.plan.json")
    with pytest.raises(ExportError) as caught:
        export_fcpxml(plan, ccn())
    assert caught.value.code == "lint_errors"
    assert caught.value.details["errors"]


def test_media_refusals() -> None:
    _refused(_plan(sources=[source(media="/srv/media/a.mov")]), "media_not_on_edit_pc")
    error = _refused(_plan(sources=[source(media="Ep1 Multicam")]), "pool_source_not_exportable")
    assert "camera file" in error.fix_hint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("label", "bad\x01label"),
        ("notes", "carriage\rreturn"),
        ("label", "￾"),
    ],
)
def test_text_xml_cannot_carry_is_refused(field: str, value: str) -> None:
    error = _refused(_plan(segments=[seg(**{field: value})]), "unsafe_text")
    assert error.details["location"] == f"segments[0].{field}"


def test_tabs_and_line_breaks_round_trip() -> None:
    result = export(_plan(segments=[seg(label="a\tb", notes="line 1\nline 2")]))
    assert result.report["self_check"]["ok"]


def test_non_finite_gain_and_zoom_are_refused() -> None:
    _refused(_plan(segments=[seg(gain_db=math.inf)]), "non_finite_value")
    _refused(_plan(segments=[seg(reframe=[{"time": 0, "zoom": math.inf}])]), "non_finite_value")


def test_fades_longer_than_the_clip_are_refused() -> None:
    _refused(_plan(head={"audio_fade_frames": 20}, end_fade_frames=5), "fade_too_long")


def test_empty_timeline_is_refused() -> None:
    _refused(_plan(segments=[]), "empty_timeline")


def test_layout_refuses_what_lint_would_catch() -> None:
    # export_fcpxml lints first; build_layout still refuses on its own.
    with pytest.raises(ExportError) as caught:
        build_layout(_plan(segments=[seg(src="ghost")]))
    assert caught.value.code == "unknown_source"
    with pytest.raises(ExportError) as caught:
        build_layout(_plan(segments=[seg(a=0, b=24), seg(a=10, b=10)]))
    assert caught.value.code == "zero_length_segment"


def test_self_check_failure_refuses_to_hand_out_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import aar.plan.export as export_module

    real = export_module.write_fcpxml
    monkeypatch.setattr(
        export_module,
        "write_fcpxml",
        lambda layout: real(layout).replace(
            'duration="24024/24000s"', 'duration="25025/24000s"', 1
        ),
    )
    _refused(_plan(), "self_check_failed")
