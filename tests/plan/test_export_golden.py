"""Golden FCPXML for the plan fixtures (CCN long-form, a reframed short, 29.97) in both audio
crossfade forms, the round-trip self-check on each, and the mapping pinned with hand-computed
values."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS, TRANSITION_GOLDENS, ccn, load_plan

from aar.cli import main
from aar.plan.export import export_fcpxml, verify_plan_fcpxml
from aar.plan.export.reader import read_fcpxml
from aar.plan.export.verify import window_from_file

#: (golden file stem, plan name, audio crossfade form)
GOLDENS = [(name, name, "split") for name in sorted(GOLDEN_PLANS)] + [
    (f"{name}.transition", name, "transition") for name in TRANSITION_GOLDENS
]
IDS = [stem for stem, _, _ in GOLDENS]


def _golden(stem: str, suffix: str) -> bytes:
    return (EXPORT_FIXTURES / f"{stem}{suffix}").read_bytes()


def _tree(stem: str) -> ET.Element:
    return ET.fromstring(_golden(stem, ".fcpxml"))


def _spine(stem: str) -> list[ET.Element]:
    return list(_tree(stem).find("library/event/project/sequence/spine"))  # type: ignore[arg-type]


def _report(stem: str) -> dict:  # type: ignore[type-arg]
    return json.loads(_golden(stem, ".report.json"))


def _features(stem: str) -> set[tuple[str, str]]:
    return {(i["feature"], i["location"]) for i in _report(stem)["not_expressible"]}


@pytest.mark.parametrize(("stem", "name", "form"), GOLDENS, ids=IDS)
def test_cli_export_matches_the_golden_bytes(
    stem: str, name: str, form: str, tmp_path: Path
) -> None:
    out = tmp_path / f"{stem}.fcpxml"
    argv = ["plan", "export", str(GOLDEN_PLANS[name]), "--out", str(out)]
    assert main([*argv, "--audio-crossfade", form]) == 0
    assert out.read_bytes() == _golden(stem, ".fcpxml")
    assert (tmp_path / f"{stem}.report.json").read_bytes() == _golden(stem, ".report.json")


@pytest.mark.parametrize(("stem", "name", "form"), GOLDENS, ids=IDS)
def test_library_export_matches_the_golden(stem: str, name: str, form: str) -> None:
    result = export_fcpxml(load_plan(GOLDEN_PLANS[name]), ccn(), audio_crossfade=form)  # type: ignore[arg-type]
    assert result.fcpxml.encode("utf-8") == _golden(stem, ".fcpxml")
    assert {**result.report, "file": f"{stem}.fcpxml"} == _report(stem)


@pytest.mark.parametrize(("stem", "name", "form"), GOLDENS, ids=IDS)
def test_golden_round_trips_with_the_self_check(stem: str, name: str, form: str) -> None:
    text = _golden(stem, ".fcpxml").decode("utf-8")
    assert read_fcpxml(text).problems == []
    result = verify_plan_fcpxml(text, load_plan(GOLDEN_PLANS[name]), ccn(), audio_crossfade=form)  # type: ignore[arg-type]
    assert result.ok, result.problems
    assert result.problems == []


@pytest.mark.parametrize(("stem", "name", "form"), GOLDENS, ids=IDS)
def test_a_golden_fails_the_other_form(stem: str, name: str, form: str) -> None:
    other = "transition" if form == "split" else "split"
    plan = load_plan(GOLDEN_PLANS[name])
    result = verify_plan_fcpxml(
        _golden(stem, ".fcpxml").decode(), plan, ccn(), audio_crossfade=other
    )  # type: ignore[arg-type]
    has_audio_only = any(t.audio and t.video == "cut" for t in plan.transitions)
    assert result.ok is not has_audio_only


def test_golden_self_check_counts() -> None:
    counts = {
        stem: verify_plan_fcpxml(
            _golden(stem, ".fcpxml").decode(),
            load_plan(GOLDEN_PLANS[name]),
            ccn(),
            audio_crossfade=form,  # type: ignore[arg-type]
        ).checked
        for stem, name, form in GOLDENS
    }
    common = {"titles": 2, "markers": 2, "assets": 4}
    assert counts == {
        "ccn_longform_v2": {"clips": 6, "transitions": 3, **common, "crop_windows": 1},
        "ccn_longform_v2.transition": {"clips": 6, "transitions": 3, **common, "crop_windows": 1},
        "short_reframe": {
            "clips": 2,
            "transitions": 0,
            "titles": 0,
            "markers": 1,
            "assets": 2,
            "crop_windows": 5,
        },
        "ccn_2997": {"clips": 6, "transitions": 3, **common, "crop_windows": 2},
        "ccn_2997.transition": {"clips": 6, "transitions": 3, **common, "crop_windows": 2},
    }


# -- the mapping, pinned (values computed by hand from the plan fixtures) ----------------------


def test_timing_is_exact_rational_from_01_00_00_00() -> None:
    root = _tree("ccn_longform_v2")
    fmt = root.find("resources/format")
    assert fmt is not None
    assert fmt.attrib == {
        "id": "r1",
        "name": "FFVideoFormat1080p2398",
        "frameDuration": "1001/24000s",
        "width": "1920",
        "height": "1080",
        "colorSpace": "1-1-1 (Rec. 709)",
    }
    sequence = root.find("library/event/project/sequence")
    assert sequence is not None
    # 01:00:00:00 at 23.976 = 86400 frames * 1001/24000 s; 426 frames long (306 + 120 outro).
    assert sequence.get("tcStart") == "86486400/24000s"
    assert sequence.get("duration") == "426426/24000s"
    assert sequence.get("tcFormat") == "NDF"
    seq2997 = _tree("ccn_2997").find("library/event/project/sequence")
    assert seq2997 is not None
    assert seq2997.get("tcStart") == "108108000/30000s"  # 108000 frames * 1001/30000 s
    fmt2997 = _tree("ccn_2997").find("resources/format")
    assert fmt2997 is not None and fmt2997.get("frameDuration") == "1001/30000s"
    vertical = _tree("short_reframe").find("resources/format")
    assert vertical is not None and "name" not in vertical.attrib  # no FCP name for 1080x1920


def test_head_from_black_is_a_dissolve_opening_the_spine() -> None:
    first, clip = _spine("ccn_longform_v2")[:2]
    assert first.tag == "transition"
    assert first.attrib == {
        "name": "Cross Dissolve",
        "offset": "86486400/24000s",
        "duration": "15015/24000s",
    }
    assert [c.tag for c in first] == ["filter-video"]  # the head's sound is a fade, not a mix
    assert clip.get("offset") == "86486400/24000s" and clip.get("start") == "0s"


def test_split_crossfade_keeps_the_picture_cut_and_overlaps_the_sound() -> None:
    spine = _spine("ccn_longform_v2")
    assert [e.tag for e in spine].count("transition") == 2  # head and outro only
    out_clip, in_clip = spine[3], spine[4]
    # Cut at record 216, 12 frames centred: 6 before, 6 after. The pictures meet at the cut.
    assert out_clip.get("offset") == f"{(86400 + 144) * 1001}/24000s"
    assert out_clip.get("duration") == f"{72 * 1001}/24000s"
    assert in_clip.get("offset") == f"{(86400 + 216) * 1001}/24000s"
    assert in_clip.get("start") == f"{30 * 1001}/24000s"
    # The outgoing sound runs 6 frames past the cut (L cut) and fades out over all 12 ...
    assert out_clip.get("audioStart") == f"{144 * 1001}/24000s"
    assert out_clip.get("audioDuration") == f"{78 * 1001}/24000s"
    fade_out = out_clip.find("adjust-volume/param/fadeOut")
    assert fade_out is not None and fade_out.get("duration") == "12012/24000s"
    # ... the incoming sound starts 6 frames before it (J cut) and fades in over all 12; it stops
    # at the outro (306) under the outro dissolve that lengthens its picture by 15.
    assert in_clip.get("audioStart") == f"{24 * 1001}/24000s"
    assert in_clip.get("audioDuration") == f"{96 * 1001}/24000s"  # 210..306
    assert in_clip.get("duration") == f"{105 * 1001}/24000s"  # 90 + 15 into the tail
    fade_in = in_clip.find("adjust-volume/param/fadeIn")
    assert fade_in is not None and fade_in.get("duration") == "12012/24000s"
    effects = [e.get("name") for e in _tree("ccn_longform_v2").iter("effect")]
    assert "Audio Crossfade" not in effects


def test_transition_form_crossfade_is_centred_on_the_cut() -> None:
    spine = _spine("ccn_longform_v2.transition")
    out_clip, transition, in_clip = spine[3], spine[4], spine[5]
    # Cut at record 216, 12 frames, 6 before and 6 after: window 210..222.
    assert transition.get("name") == "Audio Crossfade"
    assert [c.tag for c in transition] == ["filter-audio"]
    assert transition.get("offset") == f"{(86400 + 210) * 1001}/24000s"
    assert transition.get("duration") == "12012/24000s"
    assert out_clip.get("duration") == f"{78 * 1001}/24000s"  # 72 + 6 into the tail
    assert in_clip.get("offset") == transition.get("offset")
    assert in_clip.get("start") == f"{24 * 1001}/24000s"  # src_in 30 - 6
    assert in_clip.get("audioStart") == f"{24 * 1001}/24000s"
    assert in_clip.get("audioDuration") == f"{96 * 1001}/24000s"  # 210..306


def test_outro_dissolve_starts_on_the_cut_and_its_sound_is_on_a5() -> None:
    spine = _spine("ccn_longform_v2")
    transition, outro = spine[5], spine[6]
    # The outro starts at source frame 0 (no head handle): right alignment, 306..321.
    assert transition.get("offset") == f"{(86400 + 306) * 1001}/24000s"
    assert [c.tag for c in transition] == ["filter-video"]
    assert outro.get("srcEnable") == "video"
    sound = outro.find("asset-clip")
    assert sound is not None
    assert sound.get("lane") == "-4"  # A5
    assert sound.get("srcEnable") == "audio"
    volume = sound.find("adjust-volume")
    assert volume is not None and volume.get("amount") == "-1dB"  # A5's track gain
    fade = volume.find("param/fadeOut")
    assert fade is not None and fade.attrib == {"type": "linear", "duration": "2002/24000s"}


def test_gain_fades_conform_and_transform_on_content_clips() -> None:
    spine = _spine("ccn_longform_v2")
    first = spine[1]
    # A 1.1 punch-in on a 16:9 source on a 16:9 timeline: fill (moved), scale 1.1, no shift.
    assert first.find("adjust-conform").attrib == {"type": "fill"}  # type: ignore[union-attr]
    transform = first.find("adjust-transform")
    assert transform is not None
    assert transform.attrib == {"position": "0 0", "scale": "1.1 1.1", "rotation": "0"}
    assert [c.tag for c in first][:3] == ["adjust-conform", "adjust-transform", "adjust-volume"]
    fade_in = first.find("adjust-volume/param/fadeIn")
    assert fade_in is not None and fade_in.get("duration") == "24024/24000s"
    assert spine[3].find("adjust-volume").get("amount") == "-2dB"  # type: ignore[union-attr]
    assert spine[2].find("adjust-conform") is None  # unmoved, same aspect: Resolve's default
    # Muted A1 is exported audible: nothing disables the clips' sound.
    assert all(c.get("enabled", "1") == "1" for c in spine if c.tag == "asset-clip")
    assert not any(c.get("srcEnable") == "video" for c in spine[1:5] if c.tag == "asset-clip")
    note = _tree("ccn_longform_v2").find("library/event/project/sequence/note")
    assert note is not None and "audio_tracks[0].mute" in (note.text or "")


def test_titles_and_markers_sit_in_their_parent_clip() -> None:
    spine = _spine("ccn_longform_v2")
    title = spine[1].find("title")
    assert title is not None
    assert title.get("lane") == "1"  # one above the top content track
    assert title.get("offset") == "48048/24000s"  # record 48 in a clip starting at source 0
    assert title.find("text/text-style").text == "Jane Doe\nHead Judge"  # type: ignore[union-attr]
    marker = spine[3].find("marker")
    assert marker is not None
    assert marker.attrib == {
        "start": "144144/24000s",
        "duration": "1001/24000s",
        "value": "Act 2",
    }
    sting = spine[6].find("title")
    assert sting is not None and sting.get("offset") == "6006/24000s"


def test_connected_clip_flip_and_pan_tilt_at_2997() -> None:
    spine = _spine("ccn_2997")
    broll = spine[2].find("asset-clip")
    assert broll is not None
    assert broll.get("lane") == "1"  # video track 2
    assert broll.get("offset") == "400400/30000s"  # record 400 in a clip starting at source 300
    assert broll.get("start") == "100100/30000s"
    assert broll.find("adjust-conform") is not None
    assert broll.find("adjust-transform").get("scale") == "-1 1"  # type: ignore[union-attr]
    assert broll.find("adjust-volume").get("amount") == "-5dB"  # type: ignore[union-attr]
    transform = spine[3].find("adjust-transform")
    assert transform is not None
    # pan -40 px, tilt 12.5 px at 1080 high: -40/1080*100 %, 12.5/1080*100 %; zoom 1.25. With no
    # reframe the crop is centred, so rotation does not move it.
    x, y = (float(v) for v in transform.get("position", "").split())
    assert x == -40 / 1080 * 100 and y == 12.5 / 1080 * 100
    assert transform.get("scale") == "1.25 1.25" and transform.get("rotation") == "1.5"
    outro = spine[-1]
    assert outro.find("asset-clip").get("lane") == "-1"  # type: ignore[union-attr]  # A2


def test_reframe_of_a_16_9_source_on_a_9_16_timeline_fills_and_crops() -> None:
    """F1: the 1920x1080 camera file on the 1080x1920 short: fill makes it 1920 high, 3413.3
    wide; the reframe crops it in source-width units."""
    clip = _spine("short_reframe")[0]
    assert clip.find("adjust-conform").attrib == {"type": "fill"}  # type: ignore[union-attr]
    keys = clip.findall("adjust-transform/param[@name='position']/keyframeAnimation/keyframe")
    assert [k.get("time") for k in keys] == ["0s", "120120/24000s", "239239/24000s"]
    filled_width = 1920 * max(1080 / 1920, 1920 / 1080)  # 3413.33 px
    # Key 1: x 0.55, reframe zoom 1.05, static zoom 1.2: the centre moves left by
    # 1.26 * 0.05 * 3413.33 px, written as percent of the 1920 px frame height.
    x = float(keys[1].get("value", "").split()[0])
    assert x == pytest.approx(-(1.05 * 1.2) * (0.55 - 0.5) * filled_width / 1920 * 100, abs=1e-9)
    assert all(k.get("interp") == "linear" and k.get("curve") == "linear" for k in keys)
    scales = clip.findall("adjust-transform/param[@name='scale']/keyframeAnimation/keyframe")
    assert [s.get("value") for s in scales] == ["1.2 1.2", "1.26 1.26", "1.32 1.32"]
    # The crop window read back from the file: centre at 0.55 of 1920 px, full-fill height
    # (1080 px of source) over the zoom 1.26, at the timeline's 9:16 aspect.
    position = tuple(float(v) for v in keys[1].get("value", "").split())
    scale = tuple(float(v) for v in scales[1].get("value", "").split())
    window = window_from_file(position, scale, 0.0, (1920, 1080), 1080, 1920)  # type: ignore[arg-type]
    assert window == pytest.approx((0.55 * 1920, 540, 1080 / 1.26 * 9 / 16, 1080 / 1.26), abs=1e-6)
    second = _spine("short_reframe")[1]
    times = second.findall("adjust-transform/param[@name='position']/keyframeAnimation/keyframe")
    assert [k.get("time") for k in times] == ["240240/24000s", "479479/24000s"]  # src_in 240
    assert second.find("note").text == "wide reframed to the speaker"  # type: ignore[union-attr]


def test_media_src_is_a_percent_encoded_file_url() -> None:
    srcs = [r.get("src") for r in _tree("short_reframe").iter("media-rep")]
    assert srcs == [
        "file:///E:/CCN/Ep1/A001.braw",
        "file:///E:/CCN/Ep1/B%20001%20%28wide%29.braw",
    ]
    srcs = [r.get("src") for r in _tree("ccn_2997").iter("media-rep")]
    assert "file://nas/media/late/crowd.mov" in srcs
    assert "file:///E:/Show/Assets/Outro%202997.mov" in srcs  # a file: URL is used as is


def test_reports_list_what_fcpxml_cannot_carry() -> None:
    features = _features("ccn_longform_v2")
    assert ("audio_bus", "audio_tracks[0].bus") in features
    assert ("audio_mute", "audio_tracks[3].mute") in features
    assert ("audio_track_empty", "audio_tracks[5]") in features
    # A5's -1 dB reaches the outro on A5, not the A1-A4 content that feeds it (F9).
    assert {("bus_gain", f"audio_tracks[{i}].bus") for i in range(4)} <= features
    assert ("cdl", "sources[0].color.cdl") in features
    assert ("lut", "sources[0].color.lut") in features
    assert ("title_template", "titles[0]") in features
    assert ("title_template", "outro.sting") in features
    assert ("template", "template") in features
    assert ("split_audio_crossfade", "transitions[0]") not in features  # no element to drop
    assert ("split_audio_crossfade", "transitions[0]") in _features("ccn_longform_v2.transition")
    report = _report("ccn_longform_v2")
    assert report["audio_crossfade"] == "split"
    assert [t["form"] for t in report["transitions"]] == ["transition", "split", "transition"]
    assert [(t["track"], t["name"], t["lanes"]) for t in report["audio_tracks"]] == [
        ("A1", "A1", [0]),
        ("A5", "A5", [-4]),
    ]
    short = _features("short_reframe")
    assert ("captions", "captions") in short
    assert ("caption_style", "caption_style") in short
    assert ("reframe_interpolation", "segments[0].reframe[0..1]") in short
    assert not any(loc.startswith("segments[1].reframe") for _, loc in short)  # zoom fixed
    late = _report("ccn_2997")
    features = {(i["feature"], i["location"]) for i in late["not_expressible"]}
    assert ("input_color_space", "sources[1].color.input_color_space") in features
    assert [t["alignment"] for t in late["transitions"]] == ["right", "center", "right"]
    empty = [
        i
        for i in _report("ccn_longform_v2")["not_expressible"]
        if i["feature"] == "audio_track_empty"
    ]
    assert all("template" not in i["action"] for i in empty)
