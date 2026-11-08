"""Golden FCPXML for the plan fixtures (CCN long-form, a reframed short, 29.97), the round-trip
self-check on each, and the mapping pinned with hand-computed values (round 3: the layout the
Resolve 21.1 import test asked for)."""

from __future__ import annotations

import itertools
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS, ccn, load_plan

from aar.cli import main
from aar.plan.export import export_fcpxml, verify_plan_fcpxml
from aar.plan.export.reader import read_fcpxml
from aar.plan.export.verify import window_from_file

NAMES = sorted(GOLDEN_PLANS)


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


def _t(frames: int, den: int = 24000, step: int = 1001) -> str:
    return f"{frames * step}/{den}s"


@pytest.mark.parametrize("name", NAMES)
def test_cli_export_matches_the_golden_bytes(name: str, tmp_path: Path) -> None:
    out = tmp_path / f"{name}.fcpxml"
    assert main(["plan", "export", str(GOLDEN_PLANS[name]), "--out", str(out)]) == 0
    assert out.read_bytes() == _golden(name, ".fcpxml")
    assert (tmp_path / f"{name}.report.json").read_bytes() == _golden(name, ".report.json")


@pytest.mark.parametrize("name", NAMES)
def test_library_export_matches_the_golden(name: str) -> None:
    result = export_fcpxml(load_plan(GOLDEN_PLANS[name]), ccn())
    assert result.fcpxml.encode("utf-8") == _golden(name, ".fcpxml")
    assert {**result.report, "file": f"{name}.fcpxml"} == _report(name)


@pytest.mark.parametrize("name", NAMES)
def test_golden_round_trips_with_the_self_check(name: str) -> None:
    text = _golden(name, ".fcpxml").decode("utf-8")
    assert read_fcpxml(text).problems == []
    result = verify_plan_fcpxml(text, load_plan(GOLDEN_PLANS[name]), ccn())
    assert result.ok, result.problems
    assert result.problems == []


@pytest.mark.parametrize("name", NAMES)
def test_no_golden_holds_what_resolve_mishandles(name: str) -> None:
    text = _golden(name, ".fcpxml").decode("utf-8")
    assert "srcEnable" not in text and "<marker" not in text
    elements = [e for e in read_fcpxml(text).spine if e.kind != "transition"]
    assert all(a.raw_out == b.raw_in for a, b in itertools.pairwise(elements))


def test_golden_self_check_counts() -> None:
    counts = {
        name: verify_plan_fcpxml(
            _golden(name, ".fcpxml").decode(), load_plan(GOLDEN_PLANS[name]), ccn()
        ).checked
        for name in NAMES
    }
    assert counts == {
        "ccn_longform_v2": {
            "clips": 5,
            "transitions": 3,
            "titles": 2,
            "markers": 0,
            "assets": 4,
            "crop_windows": 1,
        },
        "short_reframe": {
            "clips": 2,
            "transitions": 0,
            "titles": 0,
            "markers": 0,
            "assets": 2,
            "crop_windows": 5,
        },
        "ccn_2997": {
            "clips": 5,
            "transitions": 3,
            "titles": 2,
            "markers": 0,
            "assets": 4,
            "crop_windows": 2,
        },
        # Round 4: the same plans with a start timecode check the same things.
        "tc_24": {
            "clips": 5,
            "transitions": 3,
            "titles": 2,
            "markers": 0,
            "assets": 4,
            "crop_windows": 1,
        },
        "tc_2997": {
            "clips": 5,
            "transitions": 3,
            "titles": 2,
            "markers": 0,
            "assets": 4,
            "crop_windows": 2,
        },
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
    seq2997 = _tree("ccn_2997").find("library/event/project/sequence")
    assert seq2997 is not None and seq2997.get("tcStart") == "108108000/30000s"
    vertical = _tree("short_reframe").find("resources/format")
    assert vertical is not None and "name" not in vertical.attrib  # no FCP name for 1080x1920


def test_spine_clips_abut_at_their_plan_ranges() -> None:
    """Round 3: no clip is lengthened for a transition (Resolve trimmed and duplicated)."""
    spine = _spine("ccn_longform_v2")
    assert [e.tag for e in spine] == [
        "transition",
        "asset-clip",
        "asset-clip",
        "asset-clip",
        "asset-clip",
        "transition",
        "asset-clip",
    ]
    clips = [e for e in spine if e.tag == "asset-clip"]
    # Plan record ranges 0-72, 72-144, 144-216, 216-306, outro 306-426; source ins 0, 72, 144,
    # 30, 0; lengths unchanged.
    assert [c.get("offset") for c in clips] == [_t(86400 + r) for r in (0, 72, 144, 216, 306)]
    assert [c.get("start") for c in clips] == ["0s", _t(72), _t(144), _t(30), "0s"]
    assert [c.get("duration") for c in clips] == [_t(n) for n in (72, 72, 72, 90, 120)]


def test_head_transition_starts_on_the_first_frame() -> None:
    head = _spine("ccn_longform_v2")[0]
    assert head.attrib == {
        "name": "Cross Dissolve",
        "offset": "86486400/24000s",
        "duration": "15015/24000s",
    }
    assert [c.tag for c in head] == ["filter-video"]  # the head's sound is a fade, not a mix


def test_outro_dissolve_starts_on_the_cut_from_the_tail_handle() -> None:
    """The outro starts at source frame 0 (no head handle): the 15-frame window starts on the
    cut at 306 and plays segment 4's media past its out (handle), segment 4 stays 90 frames."""
    spine = _spine("ccn_longform_v2")
    transition = spine[5]
    assert transition.get("offset") == _t(86400 + 306)
    assert transition.get("duration") == _t(15)
    assert spine[4].get("duration") == _t(90)
    assert spine[6].get("offset") == transition.get("offset")


def test_split_crossfade_keeps_the_picture_cut_and_overlaps_the_sound() -> None:
    spine = _spine("ccn_longform_v2")
    out_clip, in_clip = spine[3], spine[4]
    # Cut at 216, 12 frames centred: the outgoing sound runs 6 past (L cut) and fades out over
    # 12; the incoming starts 6 before (J cut) and fades in over 12.
    assert out_clip.get("audioStart") == _t(144) and out_clip.get("audioDuration") == _t(78)
    assert in_clip.get("audioStart") == _t(24) and in_clip.get("audioDuration") == _t(96)
    fade_out = out_clip.find("adjust-volume/param/fadeOut")
    fade_in = in_clip.find("adjust-volume/param/fadeIn")
    assert fade_out is not None and fade_out.get("duration") == _t(12)
    assert fade_in is not None and fade_in.get("duration") == _t(12)
    effects = [e.get("name") for e in _tree("ccn_longform_v2").iter("effect")]
    assert effects == ["Cross Dissolve", "Basic Title"]


def test_outro_is_one_clip_with_its_own_sound() -> None:
    """Round 3: no srcEnable split, no companion; its volume is the planned A5's -1 dB."""
    outro = _spine("ccn_longform_v2")[6]
    assert outro.find("asset-clip") is None and "srcEnable" not in outro.attrib
    volume = outro.find("adjust-volume")
    assert volume is not None and volume.get("amount") == "-1dB"
    fade = volume.find("param/fadeOut")
    assert fade is not None and fade.attrib == {"type": "linear", "duration": _t(2)}
    actions = {i["location"]: i["action"] for i in _report("ccn_longform_v2")["not_expressible"]}
    assert actions["outro.audio_track"] == (
        "outro sound is on the clip's own track; move it to the A5 bus by hand"
    )
    late = {i["location"]: i["action"] for i in _report("ccn_2997")["not_expressible"]}
    assert (
        late["outro.audio_track"] == "outro sound is on the clip's own track; move it to A2 by hand"
    )


def test_audio_mapping_is_dense_and_in_the_note() -> None:
    report = _report("ccn_longform_v2")
    assert [(r["track"], r["name"], r["imported"]) for r in report["audio_tracks"]] == [
        ("A1", "A1", ["A1"]),
        ("A5", "A5", ["A1"]),
    ]
    note = _tree("ccn_longform_v2").find("library/event/project/sequence/note")
    assert note is not None and note.text is not None
    assert (
        "audio tracks after import:\nplan A1 (A1) -> imported A1\nplan A5 (A5) -> imported A1"
        in (note.text)
    )
    late = _report("ccn_2997")
    assert [(r["track"], r["imported"]) for r in late["audio_tracks"]] == [
        ("A1", ["A1"]),
        ("A2", ["A1", "A2"]),  # the outro on V1 and the V2 b-roll's sound
    ]
    assert [(r["lane"], r["imported_video"]) for r in late["lanes"]] == [
        (2, "V3"),
        (1, "V2"),
        (0, "V1"),
    ]


def test_markers_are_listed_to_add_by_hand() -> None:
    report = _report("ccn_longform_v2")
    assert report["not_imported"]["heading"] == "not imported: add by hand"
    assert [(m["name"], m["timecode"], m["color"]) for m in report["not_imported"]["markers"]] == [
        ("Act 1", "01:00:00:00", "Green"),
        ("Act 2", "01:00:06:00", "Blue"),  # record 144 at 24 fps nominal
    ]
    late = _report("ccn_2997")["not_imported"]["markers"]
    assert [(m["timecode"], m["duration"], m["note"]) for m in late] == [
        ("01:00:00:00", 1, None),
        ("01:00:20:00", 30, "cut close to wide"),  # record 600 at 30 fps nominal
    ]


def test_import_instructions_are_in_the_report() -> None:
    steps = _report("short_reframe")["import"]
    assert steps[0].startswith("In Resolve use File > Import > Timeline")
    assert "'Use sizing information' and 'Automatically set project settings'" in steps[1]
    assert "Import source clips" in steps[2]


def test_gain_fades_conform_and_transform_on_content_clips() -> None:
    spine = _spine("ccn_longform_v2")
    first = spine[1]
    assert first.find("adjust-conform").attrib == {"type": "fill"}  # type: ignore[union-attr]
    transform = first.find("adjust-transform")
    assert transform is not None
    assert transform.attrib == {"position": "0 0", "scale": "1.1 1.1", "rotation": "0"}
    fade_in = first.find("adjust-volume/param/fadeIn")
    assert fade_in is not None and fade_in.get("duration") == _t(24)
    assert spine[3].find("adjust-volume").get("amount") == "-2dB"  # type: ignore[union-attr]
    assert spine[2].find("adjust-conform") is None
    note = _tree("ccn_longform_v2").find("library/event/project/sequence/note")
    assert note is not None and "audio_tracks[0].mute" in (note.text or "")


def test_titles_sit_in_their_parent_clip() -> None:
    spine = _spine("ccn_longform_v2")
    title = spine[1].find("title")
    assert title is not None
    assert title.get("lane") == "1"
    assert title.get("offset") == _t(48)
    assert title.find("text/text-style").text == "Jane Doe\nHead Judge"  # type: ignore[union-attr]
    sting = spine[6].find("title")
    assert sting is not None and sting.get("offset") == _t(6)


def test_connected_clip_flip_and_pan_tilt_at_2997() -> None:
    spine = _spine("ccn_2997")
    assert [e.tag for e in spine] == [
        "transition",
        "asset-clip",
        "asset-clip",
        "asset-clip",
        "transition",
        "asset-clip",
    ]
    broll = spine[2].find("asset-clip")
    assert broll is not None
    assert broll.get("lane") == "1"
    assert broll.get("offset") == _t(400, 30000)  # record 400 in a clip starting at source 300
    assert broll.find("adjust-transform").get("scale") == "-1 1"  # type: ignore[union-attr]
    assert broll.find("adjust-volume").get("amount") == "-5dB"  # type: ignore[union-attr]
    transform = spine[3].find("adjust-transform")
    assert transform is not None
    x, y = (float(v) for v in transform.get("position", "").split())
    assert x == -40 / 1080 * 100 and y == 12.5 / 1080 * 100
    assert spine[5].get("offset") == spine[4].get("offset") == _t(108000 + 900, 30000)
    assert spine[5].find("adjust-volume").get("amount") == "-2dB"  # type: ignore[union-attr]


def test_reframe_of_a_16_9_source_on_a_9_16_timeline_fills_and_crops() -> None:
    clip = _spine("short_reframe")[0]
    assert clip.find("adjust-conform").attrib == {"type": "fill"}  # type: ignore[union-attr]
    keys = clip.findall("adjust-transform/param[@name='position']/keyframeAnimation/keyframe")
    assert [k.get("time") for k in keys] == ["0s", "120120/24000s", "239239/24000s"]
    filled_width = 1920 * max(1080 / 1920, 1920 / 1080)
    x = float(keys[1].get("value", "").split()[0])
    assert x == pytest.approx(-(1.05 * 1.2) * (0.55 - 0.5) * filled_width / 1920 * 100, abs=1e-9)
    scales = clip.findall("adjust-transform/param[@name='scale']/keyframeAnimation/keyframe")
    position = tuple(float(v) for v in keys[1].get("value", "").split())
    scale = tuple(float(v) for v in scales[1].get("value", "").split())
    window = window_from_file(position, scale, 0.0, (1920, 1080), 1080, 1920)  # type: ignore[arg-type]
    assert window == pytest.approx((0.55 * 1920, 540, 1080 / 1.26 * 9 / 16, 1080 / 1.26), abs=1e-6)


def test_media_src_is_a_percent_encoded_file_url() -> None:
    srcs = [r.get("src") for r in _tree("short_reframe").iter("media-rep")]
    assert srcs == [
        "file:///E:/CCN/Ep1/A001.braw",
        "file:///E:/CCN/Ep1/B%20001%20%28wide%29.braw",
    ]
    srcs = [r.get("src") for r in _tree("ccn_2997").iter("media-rep")]
    assert "file://nas/media/late/crowd.mov" in srcs
    assert "file:///E:/Show/Assets/Outro%202997.mov" in srcs


def test_reports_list_what_fcpxml_cannot_carry() -> None:
    features = _features("ccn_longform_v2")
    assert ("audio_bus", "audio_tracks[0].bus") in features
    assert ("audio_mute", "audio_tracks[3].mute") in features
    assert ("audio_track_empty", "audio_tracks[5]") in features
    assert {("bus_gain", f"audio_tracks[{i}].bus") for i in range(4)} <= features
    assert ("cdl", "sources[0].color.cdl") in features
    assert ("title_template", "titles[0]") in features
    assert ("template", "template") in features
    assert ("outro_sound_track", "outro.audio_track") in features
    assert {("marker", "markers[0]"), ("marker", "markers[1]")} <= features
    assert ("split_audio_crossfade", "transitions[0]") not in features
    short = _features("short_reframe")
    assert ("captions", "captions") in short
    assert ("reframe_interpolation", "segments[0].reframe[0..1]") in short
    late = _report("ccn_2997")
    assert [t["alignment"] for t in late["transitions"]] == ["right", "center", "right"]
    assert [t["form"] for t in late["transitions"]] == ["transition", "split", "transition"]
