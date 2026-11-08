"""S18 round 4: the asset starts at its media's start timecode (Source.start_timecode), and every
time inside a clip counts from there: its start and audioStart, its keyframe times, and the offset
of every item anchored to it (titles, connected clips). The import-test rerun (2026-11-08) linked
every clip only with asset start = the file's timecode; a title whose offset was left un-shifted
landed an hour before the timeline start."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS, ccn, export, load_plan, permissive

from aar.cli import main
from aar.plan.export import ExportError, export_fcpxml, verify_plan_fcpxml
from aar.plan.export.reader import read_fcpxml
from aar.plan.export.timebase import Timebase
from aar.plan.model import EditPlan

TB24 = Timebase(Fraction(24))
FILM = Timebase(Fraction(24000, 1001))
NTSC = Timebase(Fraction(30000, 1001))
HOUR_24 = 86400  # 01:00:00:00 at 24 (and 23.976) fps
HOUR_2997 = 108000  # 01:00:00:00 at 29.97 non-drop


def _golden(name: str) -> str:
    return (EXPORT_FIXTURES / f"{name}.fcpxml").read_text(encoding="utf-8")


def _spine_clips(root: ET.Element) -> list[ET.Element]:
    spine = root.find("library/event/project/sequence/spine")
    assert spine is not None
    return [e for e in spine if e.tag == "asset-clip"]


def _with_timecode(name: str, timecode: int, **changes: Any) -> EditPlan:
    data = load_plan(GOLDEN_PLANS[name]).model_dump(mode="json")
    for source in data["sources"]:
        source["start_timecode"] = timecode
    data.update(changes)
    return EditPlan.model_validate(data)


# -- the goldens -------------------------------------------------------------------------------


def test_the_timecode_goldens_start_each_asset_at_01_00_00_00() -> None:
    tc24 = ET.fromstring(_golden("tc_24"))
    assert {a.get("start") for a in tc24.iter("asset")} == {"3600s"}
    tc2997 = ET.fromstring(_golden("tc_2997"))
    # 108000 frames of 1001/30000 s: 01:00:00:00 non-drop is 3603.6 s of media time.
    assert {a.get("start") for a in tc2997.iter("asset")} == {"108108000/30000s"}
    assert NTSC.time(HOUR_2997) == "108108000/30000s"


#: Per timecode golden, from its plan: the timebase and timecode; the spine clips' source ins;
#: the split crossfade's two clips (spine index) and their audioStart source frames; each title's
#: spine parent, its offset in source frames past the parent's start, and its record range.
GOLDENS: dict[str, dict[str, Any]] = {
    "tc_24": {
        "tb": TB24,
        "tc": HOUR_24,
        "src_ins": (0, 72, 144, 30, 0),
        "audio": ((2, 144), (3, 24)),  # cut at 216, 12 frames centred
        "titles": ((0, 48, (48, 120)), (4, 6, (312, 360))),  # lower third; sting on the outro
    },
    "tc_2997": {
        "tb": NTSC,
        "tc": HOUR_2997,
        "src_ins": (0, 300, 900, 0),
        "audio": ((1, 300), (2, 895)),  # cut at 600, 10 frames centred
        "titles": ((0, 30, (30, 150)), (3, 10, (910, 970))),
    },
}


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_clip_starts_and_audio_starts_count_from_the_timecode(name: str) -> None:
    case = GOLDENS[name]
    tb, tc = case["tb"], case["tc"]
    clips = _spine_clips(ET.fromstring(_golden(name)))
    assert [c.get("start") for c in clips] == [tb.time(tc + s) for s in case["src_ins"]]
    for index, source_frame in case["audio"]:
        assert clips[index].get("audioStart") == tb.time(tc + source_frame)


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_titles_are_anchored_in_their_parent_clips_timecode_time(name: str) -> None:
    """Addendum 1: each title's offset is its parent clip's start plus the frames into it, both
    in timecode-based time, so it resolves to the plan's record frames."""
    case = GOLDENS[name]
    tb, tc = case["tb"], case["tc"]
    clips = _spine_clips(ET.fromstring(_golden(name)))
    for parent, into, _ in case["titles"]:
        offsets = [t.get("offset") for t in clips[parent].findall("title")]
        assert offsets == [tb.time(tc + case["src_ins"][parent] + into)]
    decoded = read_fcpxml(_golden(name))
    assert decoded.problems == []
    assert [(t.record_in, t.record_in + t.frames) for t in decoded.titles] == [
        rng for _, _, rng in case["titles"]
    ]


def test_the_2997_connected_clip_is_anchored_in_its_parents_timecode_time() -> None:
    """Plan V2 b-roll at record 400 (source 100) under segment 1 (record 300, source 300)."""
    clips = _spine_clips(ET.fromstring(_golden("tc_2997")))
    [broll] = clips[1].findall("asset-clip")
    assert broll.get("offset") == NTSC.time(HOUR_2997 + 300 + 100)
    assert broll.get("start") == NTSC.time(HOUR_2997 + 100)


@pytest.mark.parametrize("name", sorted(GOLDENS))
def test_an_unshifted_title_offset_is_refused(name: str) -> None:
    """The hand-shifted test file moved each clip's start but not its titles' offsets: Resolve
    put the titles an hour early. The self-check refuses that file."""
    case = GOLDENS[name]
    tb, tc = case["tb"], case["tc"]
    root = ET.fromstring(_golden(name))
    for title in root.iter("title"):
        title.set("offset", tb.time(tb.frames(title.get("offset", "0s")) - tc))
    problems = verify_plan_fcpxml(
        ET.tostring(root, encoding="unicode"), load_plan(GOLDEN_PLANS[name]), ccn()
    ).problems
    assert any("is anchored outside its parent" in p for p in problems), problems
    first = case["titles"][0][2]
    assert any(f"titles[0] is not at its plan frames {first[0]}..{first[1]}" in p for p in problems)


def test_the_hand_shifted_clip_starts_alone_do_not_pass_either() -> None:
    """The other half of that file: clip starts at the timecode but the asset at 0s."""
    root = ET.fromstring(_golden("tc_24"))
    for asset in root.iter("asset"):
        asset.set("start", "0s")
    problems = verify_plan_fcpxml(
        ET.tostring(root, encoding="unicode"), load_plan(GOLDEN_PLANS["tc_24"]), ccn()
    ).problems
    assert any("not its media's start timecode 86400" in p for p in problems), problems


# -- built on the fly --------------------------------------------------------------------------


def test_a_connected_clip_is_anchored_in_its_parents_timecode_time() -> None:
    """A plan V2 clip at record 24 (source 0) over the first spine clip (record 0, source 0): its
    offset is the parent's start + 24, its own start the timecode + 0. A wrong offset fails."""
    data = load_plan(GOLDEN_PLANS["ccn_longform_v2"]).model_dump(mode="json")
    for source in data["sources"]:
        source["start_timecode"] = HOUR_24
    data["segments"].append(
        {"track": 2, "source": "camB", "src_in": 0, "src_out": 48, "record_in": 24}
    )
    plan = EditPlan.model_validate(data)
    result = export(plan, permissive())  # CCN lint: not chronological
    first = _spine_clips(ET.fromstring(result.fcpxml))[0]
    [connected] = first.findall("asset-clip")
    assert connected.get("lane") == "1"
    assert connected.get("offset") == FILM.time(HOUR_24 + 24)
    assert connected.get("start") == FILM.time(HOUR_24)
    assert verify_plan_fcpxml(result.fcpxml, plan, permissive()).ok
    root = ET.fromstring(result.fcpxml)
    shifted = _spine_clips(root)[0].find("asset-clip")
    assert shifted is not None
    shifted.set("offset", FILM.time(24))  # un-shifted, as in the hand-shifted file
    problems = verify_plan_fcpxml(
        ET.tostring(root, encoding="unicode"), plan, permissive()
    ).problems
    assert any("segment 4" in p and "is not in the file" in p for p in problems), problems


def test_keyframe_times_count_from_the_timecode() -> None:
    plan = _with_timecode("short_reframe", HOUR_24)
    result = export(plan, ccn())
    clips = _spine_clips(ET.fromstring(result.fcpxml))
    checked = 0
    for clip, seg in zip(clips, plan.segments, strict=True):
        times = [k.get("time") for k in clip.iter("keyframe")]
        if not seg.reframe or len(seg.reframe) < 2:
            continue
        want = [FILM.time(HOUR_24 + seg.src_in + key.time) for key in seg.reframe]
        assert times == want * 2  # position, then scale
        checked += 1
    assert checked >= 1
    assert verify_plan_fcpxml(result.fcpxml, plan, ccn()).ok


def test_an_unknown_timecode_is_reported_and_starts_at_zero() -> None:
    result = export(load_plan(GOLDEN_PLANS["ccn_longform_v2"]), ccn())
    assert {a.get("start") for a in ET.fromstring(result.fcpxml).iter("asset")} == {"0s"}
    notes = [n for n in result.report["notes"] if n["feature"] == "media_start_timecode"]
    assert [n["location"] for n in notes] == [
        "sources[0]",
        "sources[1]",
        "sources[2]",
        "sources[3]",
    ]
    assert notes[0]["detail"] == (
        "media start timecode unknown; Resolve links clips only if the file's timecode starts at "
        "00:00:00:00"
    )
    assert notes[0]["action"].startswith("set Source.start_timecode")
    timed = export(_with_timecode("ccn_longform_v2", HOUR_24), ccn()).report["notes"]
    assert not [n for n in timed if n["feature"] == "media_start_timecode"]


def _shared(first: int | None, second: int | None) -> EditPlan:
    """camA2 made the same file as camA (same duration), with the given timecodes."""
    data = load_plan(GOLDEN_PLANS["ccn_longform_v2"]).model_dump(mode="json")
    cam_a, cam_a2 = data["sources"][0], data["sources"][2]
    cam_a2["media"], cam_a2["duration"] = cam_a["media"], cam_a["duration"]
    cam_a["start_timecode"], cam_a2["start_timecode"] = first, second
    return EditPlan.model_validate(data)


def test_sources_sharing_a_file_must_agree_on_its_timecode() -> None:
    with pytest.raises(ExportError) as caught:
        export(_shared(HOUR_24, 0), ccn())
    assert caught.value.code == "conflicting_source_start_timecodes"
    assert caught.value.details["start_timecodes"] == [HOUR_24, 0]
    # One left unset takes the other's: the file has one timecode.
    result = export(_shared(None, HOUR_24), ccn())
    root = ET.fromstring(result.fcpxml)
    starts = {a.get("name"): a.get("start") for a in root.iter("asset")}
    assert sorted(starts.values()).count(FILM.time(HOUR_24)) == 1
    notes = [n for n in result.report["notes"] if n["feature"] == "media_start_timecode"]
    assert "sources[0]" not in {n["location"] for n in notes}


def test_a_clip_reading_before_the_asset_start_is_refused() -> None:
    """Bounds count from the asset's start: a clip start one frame before it reads before the
    media's first frame."""
    root = ET.fromstring(_golden("tc_24"))
    _spine_clips(root)[0].set("start", TB24.time(HOUR_24 - 1))
    problems = verify_plan_fcpxml(
        ET.tostring(root, encoding="unicode"), load_plan(GOLDEN_PLANS["tc_24"]), ccn()
    ).problems
    assert any("reads before its media's first frame (-1)" in p for p in problems), problems


# -- fix-check Q3: the timecode's own rate -----------------------------------------------------


def _at_rate(timecode: int, rate: Any, name: str = "tc_24") -> EditPlan:
    data = load_plan(GOLDEN_PLANS[name]).model_dump(mode="json")
    for source in data["sources"]:
        source["start_timecode"] = timecode
        source["start_timecode_rate"] = rate
    return EditPlan.model_validate(data)


def test_a_25_fps_file_at_01_00_00_00_in_a_24_fps_plan_starts_at_3600s() -> None:
    """90000 frames at 25 fps is 3600 s: the asset starts exactly there (it was 3750 s when the
    export counted the timecode at the plan's 24 fps)."""
    plan = _at_rate(90000, {"num": 25})
    result = export(plan, ccn())
    root = ET.fromstring(result.fcpxml)
    assert {a.get("start") for a in root.iter("asset")} == {"3600s"}
    assert _spine_clips(root)[1].get("start") == TB24.time(HOUR_24 + 72)  # plan frames after it
    assert verify_plan_fcpxml(result.fcpxml, plan, ccn()).ok
    # Unset, the rate falls back to the plan's: 90000 frames at 24 fps.
    fallback = export(_at_rate(90000, None), ccn())
    assert {a.get("start") for a in ET.fromstring(fallback.fcpxml).iter("asset")} == {"3750s"}


def test_the_self_check_converts_with_the_rate_too() -> None:
    plan = _at_rate(90000, {"num": 25})
    text = export(plan, ccn()).fcpxml
    # The file is right for rate 25; checked against a plan without the rate it is not.
    problems = verify_plan_fcpxml(text, _at_rate(90000, None), ccn()).problems
    assert any("not its media's start timecode 90000" in p for p in problems), problems


def test_a_timecode_off_the_plans_frame_grid_is_refused() -> None:
    """01:00:00;00 drop-frame at 29.97 is 3599.9964 s, between two frames of a 24 fps plan."""
    with pytest.raises(ExportError) as caught:
        export(_at_rate(107892, "30000/1001"), ccn())
    assert caught.value.code == "start_timecode_off_grid"


def test_sources_sharing_a_file_agree_by_time_not_frame_count() -> None:
    data = load_plan(GOLDEN_PLANS["tc_24"]).model_dump(mode="json")
    cam_a, cam_a2 = data["sources"][0], data["sources"][2]
    cam_a2["media"], cam_a2["duration"] = cam_a["media"], cam_a["duration"]
    cam_a["start_timecode"], cam_a["start_timecode_rate"] = 90000, {"num": 25}
    cam_a2["start_timecode"] = HOUR_24  # the same instant at the plan's 24 fps
    result = export(EditPlan.model_validate(data), ccn())
    assert {a.get("start") for a in ET.fromstring(result.fcpxml).iter("asset")} == {"3600s"}


# -- fix-check Q4: an unknown start timecode is refused by default -----------------------------


def test_an_unknown_start_timecode_is_refused_by_default() -> None:
    """BRAW, the CCN camera format, cannot be probed: without a start_timecode every such clip
    would import Media Offline, so the export refuses and names the sources."""
    with pytest.raises(ExportError) as caught:
        export_fcpxml(load_plan(GOLDEN_PLANS["ccn_longform_v2"]), ccn())
    error = caught.value
    assert error.code == "start_timecode_unknown"
    assert error.details["sources"] == ["camA", "camB", "camA2", "outroclip"]
    assert "'camA' (E:/CCN/Ep1/A001.braw)" in error.message
    assert "'Start TC' in Resolve or the camera metadata" in error.fix_hint
    assert "--allow-unknown-timecode" in error.fix_hint


def test_only_the_sources_without_one_are_named() -> None:
    data = load_plan(GOLDEN_PLANS["ccn_longform_v2"]).model_dump(mode="json")
    for source in data["sources"][:3]:
        source["start_timecode"] = HOUR_24
    with pytest.raises(ExportError) as caught:
        export_fcpxml(EditPlan.model_validate(data), ccn())
    assert caught.value.details["sources"] == ["outroclip"]


def test_the_flag_exports_anyway_with_the_asset_at_0s_and_the_note() -> None:
    result = export_fcpxml(
        load_plan(GOLDEN_PLANS["ccn_longform_v2"]), ccn(), allow_unknown_timecode=True
    )
    assert {a.get("start") for a in ET.fromstring(result.fcpxml).iter("asset")} == {"0s"}
    assert any(n["feature"] == "media_start_timecode" for n in result.report["notes"])
    assert result.fcpxml == _golden("ccn_longform_v2")


def test_set_explicitly_needs_no_flag() -> None:
    result = export_fcpxml(load_plan(GOLDEN_PLANS["tc_24"]), ccn())
    assert result.fcpxml == _golden("tc_24")


def test_the_cli_refuses_with_exit_2_and_the_flag_lets_it_through(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cut.fcpxml"
    argv = ["plan", "export", str(GOLDEN_PLANS["ccn_longform_v2"]), "--out", str(out)]
    assert main(argv) == 2
    err = capsys.readouterr().err
    assert "start_timecode_unknown" in err and "--allow-unknown-timecode" in err
    assert sorted(p.name for p in tmp_path.iterdir()) == []  # refused before writing anything
    assert main([*argv, "--allow-unknown-timecode"]) == 0
    assert out.read_text(encoding="utf-8") == _golden("ccn_longform_v2")
    # Verifying a file never refuses for it: the asset at 0s is compared.
    capsys.readouterr()
    assert main(["plan", "export", str(GOLDEN_PLANS["ccn_longform_v2"]), "--verify", str(out)]) == 0
