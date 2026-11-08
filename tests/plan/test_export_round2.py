"""S18 round 2: outro gain on every track combination (F2), a strict --verify (F3), split-edit
audio crossfades (F4), one alignment rule with the compiler (F7), URL edge cases (F8), the report
(F9), and the cases that pin the validator's surviving mutants (F10)."""

from __future__ import annotations

import itertools
import re
import xml.etree.ElementTree as ET
from typing import Any

import pytest
from apply_builders import plan_data, seg, source
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS, ccn, export, load_plan

from aar.cli import main
from aar.plan.apply import compile_finish
from aar.plan.apply.errors import CompileError
from aar.plan.export import ExportError, build_layout, media_map_from_mapping, verify_plan_fcpxml
from aar.plan.export.media import media_ref
from aar.plan.model import EditPlan, Source

SIZE = [1920, 1080]


def _plan(**kwargs: Any) -> EditPlan:
    kwargs.setdefault("sources", [source(frame_size=SIZE)])
    return EditPlan.model_validate(plan_data(**kwargs))


def _spine(text: str) -> list[ET.Element]:
    return list(ET.fromstring(text).find("library/event/project/sequence/spine"))  # type: ignore[arg-type]


def _refused(plan: EditPlan, code: str, **kwargs: Any) -> ExportError:
    with pytest.raises(ExportError) as caught:
        export(plan, **kwargs)
    assert caught.value.code == code, caught.value.as_dict()
    return caught.value


# -- F2: the outro's sound gain is the gain of the track it plays on ---------------------------

TRACK_GAINS = [-6.0, -2.5, 4.0]


@pytest.mark.parametrize(("picture", "sound"), list(itertools.product([1, 2, 3], [1, 2, 3])))
def test_outro_gain_for_every_picture_and_sound_track(picture: int, sound: int) -> None:
    plan = _plan(
        sources=[source("a", frame_size=SIZE), source("o", duration=60, frame_size=SIZE)],
        segments=[seg(1, "a", 0, 48)],
        audio_tracks=[{"name": f"A{i + 1}", "gain_db": g} for i, g in enumerate(TRACK_GAINS)],
        outro={"source": "o", "src_out": 60, "track": picture, "audio_track": f"A{sound}"},
    )
    result = export(plan)
    clips = [c for c in result.layout.clips if c.location.startswith("outro")]
    carrier = next(c for c in clips if c.enable != "video")
    assert carrier.audio_track == sound
    assert carrier.volume_db == TRACK_GAINS[sound - 1]
    # The file says the same: the clip carrying the outro's sound has that track's volume.
    root = ET.fromstring(result.fcpxml)
    outro_clips = [
        e
        for e in root.iter("asset-clip")
        if e.get("name") == "o.mov" and e.get("srcEnable") != "video"
    ]
    assert len(outro_clips) == 1
    volume = outro_clips[0].find("adjust-volume")
    assert volume is not None and volume.get("amount") == f"{TRACK_GAINS[sound - 1]:g}dB"
    used = {row["track"] for row in result.report["audio_tracks"]}
    assert f"A{sound}" in used
    assert result.report["self_check"]["ok"]


# -- F3: a strict --verify ---------------------------------------------------------------------

GOLDEN = (EXPORT_FIXTURES / "ccn_longform_v2.fcpxml").read_text(encoding="utf-8")
PLAN = load_plan(GOLDEN_PLANS["ccn_longform_v2"])


def _problems(text: str, form: str = "split") -> list[str]:
    result = verify_plan_fcpxml(text, PLAN, ccn(), audio_crossfade=form)  # type: ignore[arg-type]
    assert not result.ok
    return result.problems


def _tamper(old: str, new: str) -> str:
    assert GOLDEN.count(old) >= 1, old
    return GOLDEN.replace(old, new, 1)


def test_the_validators_cli_case_now_fails(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    """probes/cli/verify-tampered.txt: segment 1's clip disabled and its asset's start moved to
    3600s, a DTD-valid file that verified ok in round 1."""
    text = _tamper('<asset-clip ref="r5" offset', '<asset-clip ref="r5" enabled="0" offset')
    text = text.replace(
        '<asset id="r5" name="B001.braw" start="0s"',
        '<asset id="r5" name="B001.braw" start="3600s"',
        1,
    )
    assert text != GOLDEN and 'start="3600s"' in text
    path = tmp_path / "tampered.fcpxml"
    path.write_text(text, encoding="utf-8")
    code = main(["plan", "export", str(GOLDEN_PLANS["ccn_longform_v2"]), "--verify", str(path)])
    out = capsys.readouterr().out
    assert code == 1 and "MISMATCH" in out
    assert "attribute 'enabled', which aar's writer never emits" in out
    assert "<asset start='3600s'>: aar's writer only writes 0s" in out


@pytest.mark.parametrize(
    ("old", "new", "expect"),
    [
        # validator tamper-out.txt cases that passed in round 1
        (
            '<asset id="r7" name="Outro.mov" start="0s" duration="120120/24000s"',
            '<asset id="r7" name="Outro.mov" start="0s" duration="60060/24000s"',
            "past its asset's 60 frames",
        ),
        ('hasVideo="1" hasAudio="1">', 'hasVideo="1" hasAudio="0">', "<asset hasAudio='0'>"),
        (
            '<filter-video ref="r2" name="Cross Dissolve" />',
            '<filter-video ref="r2" name="Cross Dissolve" enabled="0" />',
            "<filter-video> has attribute 'enabled'",
        ),
        (
            'uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265"',
            'uid="FxPlug:00000000-0000-0000-0000-000000000000"',
            "effects",
        ),
        (
            'scale="1.1 1.1" rotation="0"',
            'scale="1.1 1.1" rotation="0" anchor="20 20"',
            "<adjust-transform> has attribute 'anchor'",
        ),
        ('<fadeIn type="linear"', '<fadeIn type="easeIn"', "<fadeIn type='easeIn'>"),
        ('tcFormat="NDF"', 'tcFormat="DF"', "<sequence tcFormat='DF'>"),
        ('audioRate="48k"', 'audioRate="44.1k"', "<sequence audioRate='44.1k'>"),
        (
            '<media-rep kind="original-media" src="file:///E:/CCN/Ep1/A001.braw" />',
            '<media-rep kind="original-media" src="file:///E:/CCN/Ep1/A001.braw" />'
            '<media-rep kind="proxy-media" src="file:///E:/CCN/Ep1/B001.braw" />',
            "must hold exactly one media-rep",
        ),
        # the conform, the format name and a stray clip lane
        ('<adjust-conform type="fill" />', "", "no adjust-conform fill"),
        (
            '<adjust-conform type="fill" />',
            '<adjust-conform type="fit" />',
            "<adjust-conform type='fit'>",
        ),
        ('name="FFVideoFormat1080p2398" ', 'name="FFVideoFormat1080p24" ', "format name"),
        (
            '<asset-clip ref="r5" offset',
            '<asset-clip ref="r5" lane="0" offset',
            "spine <asset-clip> has a lane",
        ),
    ],
)
def test_strict_verify_catches_what_round_1_missed(old: str, new: str, expect: str) -> None:
    problems = _problems(_tamper(old, new))
    assert any(expect in p for p in problems), problems


def test_metadata_and_foreign_elements_fail() -> None:
    problems = _problems(
        _tamper("</sequence>", "<metadata><md key='x' value='y' /></metadata></sequence>")
    )
    assert any("<metadata> is not something aar's writer emits" in p for p in problems), problems


def test_a_clip_reading_past_its_source_is_caught() -> None:
    """W05's failure mode: one frame past the end of the source duration the plan gives."""
    source_end = PLAN.source_map()["camA2"].duration
    assert source_end == 1000
    text = _tamper(
        'name="A002.mov" start="30030/24000s" duration="105105/24000s"',
        'name="A002.mov" start="30030/24000s" duration="1001000/24000s"',
    )
    problems = _problems(text)
    assert any("past the plan source's 1000 frames" in p for p in problems), problems


def test_a_wrong_tcstart_with_every_offset_shifted_is_caught() -> None:
    """W31: tcStart 10 frames later and every spine offset and gap start moved with it, so the
    spine itself is consistent; only tcStart says the timeline starts at the wrong timecode."""
    shift = 10 * 1001

    def bump(m: re.Match[str]) -> str:
        n = int(m.group(2))
        return f'{m.group(1)}="{n + shift}/24000s"' if n >= 86486400 else m.group(0)

    text = re.sub(r'(offset|start|tcStart)="(\d+)/24000s"', bump, GOLDEN)
    assert text != GOLDEN
    problems = _problems(text)
    assert any(p.startswith("tcStart is frame 86410") for p in problems), problems


def test_an_anchored_item_outside_its_parent_is_caught() -> None:
    """W27: the A5 outro sound anchored past the end of its 120-frame parent."""
    problems = _problems(_tamper('lane="-4" offset="0s"', 'lane="-4" offset="121121/24000s"'))
    assert any("anchored outside its parent" in p for p in problems), problems


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            '<effect id="r2" name="Cross Dissolve" '
            'uid="FxPlug:4731E73A-8DAC-4113-9A30-AE85B1761265" />',
            '<effect id="r2" name="Cross Dissolve" uid="FFAudioTransition" />',
        ),
        ('<effect id="r3" name="Basic Title"', '<effect id="r3" name="Other Title"'),
    ],
)
def test_effect_name_and_uid_are_checked(old: str, new: str) -> None:
    """W36: a transition or title pointing at another effect fails, by name or by uid."""
    problems = _problems(_tamper(old, new))
    assert any("effects" in p or "titles" in p for p in problems), problems


# -- F4: split-edit audio crossfades -----------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new", "expect"),
    [
        (
            'audioStart="144144/24000s" audioDuration="78078/24000s"',
            'audioStart="144144/24000s" audioDuration="77077/24000s"',
            "split crossfade",
        ),
        ('<fadeOut type="linear" duration="12012/24000s" />', "", "split crossfade"),
        (
            '<fadeIn type="linear" duration="12012/24000s" />',
            '<fadeIn type="linear" duration="11011/24000s" />',
            "split crossfade",
        ),
    ],
)
def test_split_crossfade_tampering_is_caught(old: str, new: str, expect: str) -> None:
    problems = _problems(_tamper(old, new))
    assert any(expect in p for p in problems), problems


def test_both_forms_hold_the_same_cut_and_media() -> None:
    """A split edit and a transition element describe the same 12-frame crossfade at 216."""
    plans = {form: export(PLAN, ccn(), audio_crossfade=form) for form in ("split", "transition")}
    for result in plans.values():
        (join,) = [t for t in result.layout.transitions if t.kind == "join"]
        assert (join.cut, join.frames, join.head_part, join.audio, join.video) == (
            216,
            12,
            6,
            True,
            False,
        )
    assert [t.form for t in plans["split"].layout.transitions] == [
        "transition",
        "split",
        "transition",
    ]
    split_spine = [e.tag for e in _spine(plans["split"].fcpxml)]
    assert split_spine.count("transition") == 2


def test_split_crossfade_fades_add_to_head_and_end_fades() -> None:
    plan = _plan(
        sources=[source("a", duration=1000, frame_size=SIZE)],
        segments=[seg(a=100, b=200), seg(a=300, b=400)],
        transitions=[{"after": 0, "audio": {"frames": 10}}],
        head={"audio_fade_frames": 8},
        end_fade_frames=3,
    )
    first, second = _spine(export(plan).fcpxml)
    assert first.find("adjust-volume/param/fadeIn").get("duration") == "8008/24000s"  # type: ignore[union-attr]
    assert first.find("adjust-volume/param/fadeOut").get("duration") == "10010/24000s"  # type: ignore[union-attr]
    assert second.find("adjust-volume/param/fadeIn").get("duration") == "10010/24000s"  # type: ignore[union-attr]
    assert second.find("adjust-volume/param/fadeOut").get("duration") == "3003/24000s"  # type: ignore[union-attr]


# -- F7: one alignment rule ---------------------------------------------------------------------


def _join_plan(frames: int, tail: int | None, head: int) -> EditPlan:
    out_src = source("a", media="E:/X/a.mov", duration=None if tail is None else 200 + tail)
    if tail is None:
        out_src.pop("duration")
    return EditPlan.model_validate(
        plan_data(
            sources=[out_src, source("b", media="E:/X/b.mov", duration=100000)],
            segments=[seg(1, "a", 100, 200), seg(1, "b", head, head + 100)],
            transitions=[
                {
                    "after": 0,
                    "video": "dissolve",
                    "video_frames": frames,
                    "audio": {"frames": frames},
                }
            ],
        )
    )


def _compiler_alignment(plan: EditPlan) -> str | None:
    try:
        comp = compile_finish(plan)
    except CompileError as exc:
        assert exc.code == "no_handle_for_transition"
        return None
    step = next(s for s in comp.steps if s.op == "transition.add")
    return next(t["alignment"] for t in step.params["targets"] if t["category"] == "simple")


@pytest.mark.parametrize("frames", range(1, 31))
def test_exporter_and_compiler_choose_the_same_alignment(frames: int) -> None:
    """Every length, odd ones included, over tails and heads around the thresholds."""
    edges = sorted(
        {
            0,
            frames // 2 - 1,
            frames // 2,
            frames // 2 + 1,
            frames - frames // 2,
            frames - 1,
            frames,
            frames + 1,
        }
        - {-1}
    )
    for tail, head in itertools.product([None, *edges], edges):
        plan = _join_plan(frames, tail, head)
        compiled = _compiler_alignment(plan)
        try:
            exported = next(t.alignment for t in build_layout(plan).transitions if t.kind == "join")
        except ExportError as exc:
            assert exc.code == "no_handle_for_transition"
            exported = None
        assert exported == compiled, (frames, tail, head)


def test_odd_centre_needs_the_extra_frame_on_the_tail() -> None:
    """W05: d=3 with a 1-frame tail and a long head is left-aligned, never centred (a centred
    window would read one frame past the outgoing source)."""
    plan = _join_plan(3, 1, 50)
    layout = build_layout(plan)
    (join,) = layout.transitions
    assert (join.alignment, join.head_part) == ("left", 3)
    result = export(plan)
    assert result.report["self_check"]["ok"]


def test_transitions_that_exactly_fill_a_clip_are_allowed() -> None:
    """W34: the incoming part after its cut (6) + the outgoing part before its cut (6) fill
    the 12-frame clip exactly."""
    plan = _plan(
        sources=[source("a", duration=1000, frame_size=SIZE)],
        segments=[seg(a=100, b=148), seg(a=200, b=212), seg(a=300, b=348)],
        transitions=[
            {"after": 0, "video": "dissolve", "video_frames": 12, "audio": {"frames": 12}},
            {"after": 1, "video": "dissolve", "video_frames": 12, "audio": {"frames": 12}},
        ],
    )
    result = export(plan)
    assert result.report["self_check"]["ok"]
    assert [t["alignment"] for t in result.report["transitions"]] == ["center", "center"]


def test_outro_sound_starts_at_the_cut_under_a_video_only_dissolve() -> None:
    """W10: outro picture and sound together on V1/A1 with a centred dissolve: the picture starts
    7 frames early under the dissolve, the sound only at the cut."""
    plan = _plan(
        sources=[
            source("a", duration=1000, frame_size=SIZE),
            source("o", duration=200, frame_size=SIZE),
        ],
        segments=[seg(1, "a", 0, 48)],
        audio_tracks=[{"name": "A1"}],
        outro={
            "source": "o",
            "src_in": 20,
            "src_out": 120,
            "audio_track": "A1",
            "transition": {"video": "dissolve", "video_frames": 15},
        },
    )
    result = export(plan)
    outro = _spine(result.fcpxml)[-1]
    assert outro.get("start") == f"{13 * 1001}/24000s"  # src_in 20 - 7 before the cut
    assert outro.get("audioStart") == f"{20 * 1001}/24000s"  # the sound starts at the cut
    assert outro.get("audioDuration") == f"{100 * 1001}/24000s"


def test_marker_notes_xml_cannot_carry_are_refused() -> None:
    """W33: the unsafe-text check covers marker notes too."""
    error = _refused(_plan(markers=[{"record": 0, "note": "bad\x07bell"}]), "unsafe_text")
    assert error.details["location"] == "markers[0].note"


# -- F1: framing needs the source size ---------------------------------------------------------


def test_a_moved_clip_without_a_source_size_is_refused() -> None:
    plan = _plan(sources=[source()], segments=[seg(reframe=[{"time": 0, "x": 0.4}])])
    error = _refused(plan, "source_size_required")
    assert error.details == {"location": "segments[0]", "source": "a"}
    assert "frame_size" in error.fix_hint


def test_an_unmoved_clip_of_another_aspect_is_filled() -> None:
    plan = _plan(sources=[source(frame_size=[1080, 1920])])
    clip = _spine(export(plan).fcpxml)[0]
    assert clip.find("adjust-conform").get("type") == "fill"  # type: ignore[union-attr]
    assert clip.find("adjust-transform") is None


def test_an_unknown_size_keeps_resolves_default_framing_and_says_so() -> None:
    result = export(_plan(sources=[source()]))
    assert _spine(result.fcpxml)[0].find("adjust-conform") is None
    assert ("frame_size", "sources[0].frame_size") in {
        (i["feature"], i["location"]) for i in result.report["notes"]
    }


def test_a_tampered_crop_is_caught_by_the_window_check() -> None:
    """The crop window is recomputed from the XML: a keyframe moved 1 % of the frame height
    is 19.2 px of the 1080x1920 frame, well past 0.5 source px."""
    text = (EXPORT_FIXTURES / "short_reframe.fcpxml").read_text(encoding="utf-8")
    plan = load_plan(GOLDEN_PLANS["short_reframe"])
    keys = re.findall(r'<keyframe time="120120/24000s" value="([^"]+)"', text)
    x, y = (float(v) for v in keys[0].split())
    bad = text.replace(f'value="{keys[0]}"', f'value="{x + 1} {y}"', 1)
    result = verify_plan_fcpxml(bad, plan, ccn())
    assert any("source px from the plan's" in p for p in result.problems), result.problems


# -- F8: file: URL and path edge cases ---------------------------------------------------------


@pytest.mark.parametrize(
    ("media", "reason"),
    [
        ("file://E:/a/b.mov", "drive-in-host"),
        ("file://user:pw@host/share/x.mov", "userinfo"),
        ("file://user@host/share/x.mov", "userinfo"),
        ("file://h%5Cx/share/y.mov", "encoded-host"),
        ("file://h%2Fx/share/y.mov", "encoded-host"),
        (r"E:\Media\clip.mov ", "trailing-space-or-dot"),
        (r"E:\Media\clip.mov.", "trailing-space-or-dot"),
        (r"E:\Media \clip.mov", "trailing-space-or-dot"),
        ("clip.mov", "relative"),
    ],
)
def test_url_and_path_edge_cases_are_refused(media: str, reason: str) -> None:
    with pytest.raises(ExportError) as caught:
        media_ref(Source(id="s", media=media))
    assert caught.value.code == "media_not_on_edit_pc"
    assert caught.value.details["reason"] == reason


@pytest.mark.parametrize("media", ["/srv/media", "/srv/media/"])
def test_a_media_map_root_is_a_directory_not_media(media: str) -> None:
    media_map = media_map_from_mapping({"/srv/media": "E:\\Media"})
    with pytest.raises(ExportError) as caught:
        media_ref(Source(id="s", media=media), media_map)
    assert caught.value.details["reason"] == "map-root"
    assert (
        media_ref(Source(id="s", media="/srv/media/a.mov"), media_map).windows_path
        == "E:\\Media\\a.mov"
    )
