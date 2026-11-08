"""Phase 3 (``compile_finish``): finishing ops, item resolution by (track, record_in), and the
typed refusals."""

from __future__ import annotations

from typing import Any

import pytest
from apply_builders import longform, longform_data, plan, plan_data, seg, source
from apply_support import synth_assemble, synth_prepare

from aar.node.ops.framework import TimelineIdentity
from aar.node.ops.write import AppendedRange
from aar.plan.apply import CompileError, compile_finish
from aar.plan.model import EditPlan

LF = longform()


def _finish(p: EditPlan, **kw: Any) -> Any:
    prep = synth_prepare(p)
    return compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep), **kw), prep


def _targets(comp: Any, op: str) -> list[dict[str, Any]]:
    return [t for s in comp.steps if s.op == op for t in s.params.get("targets", [])]


def _code(p: EditPlan, **kw: Any) -> str:
    with pytest.raises(CompileError) as ei:
        compile_finish(p, **kw)
    return ei.value.code


def test_resolved_all_intents_and_grouped_ops() -> None:
    comp, _ = _finish(LF)
    assert comp.phase == "finish"
    assert comp.unresolved == 0
    assert all(i.resolved for i in comp.intents)
    assert [s.op for s in comp.steps] == [
        "transition.add",
        "item.set_color",
        "item.set_transform",
        "item.set_audio_gain",
        "item.set_fades",
        "title.place",
        "markers.add",
    ]
    assert all(s.phase == "finish" for s in comp.steps)


def test_unresolved_without_results() -> None:
    comp = compile_finish(LF)
    assert comp.unresolved > 0
    assert any(not i.resolved for i in comp.intents)
    assert all(not i.resolved for i in comp.intents if i.kind in ("colour", "transform"))


def test_f6_input_colour_space_non_braw_only() -> None:
    d = longform_data()
    for s in d["sources"]:
        if s["id"] in ("camA", "camA2"):
            s["color"] = {**(s.get("color") or {}), "input_color_space": "Rec.709 Gamma 2.4"}
    p = EditPlan.model_validate(d)
    comp, _ = _finish(p)
    steps = [s for s in comp.steps if s.op == "clip.set_input_color_space"]
    assert len(steps) == 1
    params = steps[0].params
    assert params["color_space"] == "Rec.709 Gamma 2.4"
    assert [i["unique_id"] for i in params["items"]] == ["pool-camA2"]
    assert any("BRAW" in n for n in comp.notes)


def test_f6_braw_only_emits_no_step() -> None:
    d = longform_data()
    d["sources"][0]["color"]["input_color_space"] = "Rec.709 Gamma 2.4"
    comp, _ = _finish(EditPlan.model_validate(d))
    assert not [s for s in comp.steps if s.op == "clip.set_input_color_space"]
    assert any("BRAW" in n for n in comp.notes)


def test_f7_head_is_transition_from_black_not_fade() -> None:
    comp, _ = _finish(LF)
    head = [t for t in _targets(comp, "transition.add") if t["position"] == "start"]
    assert len(head) == 1
    assert head[0]["alignment"] == "right"
    assert head[0]["category"] == "simple"
    assert any(i.kind == "head_transition" for i in comp.intents)
    # the audio fade-in still uses a fade handle, but the video head does not
    assert not any(i.kind == "fade" and "video" in i.summary for i in comp.intents)


def test_f7_outro_transition_right_aligned() -> None:
    comp, _ = _finish(LF)
    outro = [i for i in comp.intents if i.kind == "outro_transition"]
    assert len(outro) == 1 and "right" in outro[0].summary
    ends = [t for t in _targets(comp, "transition.add") if t["position"] == "end"]
    assert any(t["category"] == "simple" and t["alignment"] == "right" for t in ends)


def test_f7_audio_crossfade_hits_every_audio_item() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    # give segment 2 (record 144) a second audio item
    app = asm.appends[0]
    new_ranges = []
    for r in app.ranges:
        if r.planned_record_in == 144 and r.video is not None and r.video.track == 1:
            extra = r.audio[0].model_copy(update={"item_id": "A1:144b"})
            r = r.model_copy(update={"audio": [*r.audio, extra]})
        new_ranges.append(r)
    asm = asm.model_copy(update={"appends": [app.model_copy(update={"ranges": new_ranges})]})
    comp = compile_finish(LF, prepare=prep, assemble=asm)
    xf = [i for i in comp.intents if i.kind == "audio_crossfade"]
    assert len(xf) == 2
    audio_targets = [t for t in _targets(comp, "transition.add") if t["category"] == "audio"]
    assert len(audio_targets) == 2
    assert {t["item"]["item_id"] for t in audio_targets} == {"A1:144", "A1:144b"}
    assert all(t["item"]["track_type"] == "audio" for t in audio_targets)


def test_f7_audio_crossfade_category_single_item() -> None:
    comp, _ = _finish(LF)
    audio = [t for t in _targets(comp, "transition.add") if t["category"] == "audio"]
    assert len(audio) == 1
    assert audio[0]["item"]["track_type"] == "audio"


def test_f7_no_handle_for_transition() -> None:
    p = plan(
        sources=[source("a", duration=100)],
        segments=[seg(a=40, b=60), seg(a=2, b=22)],
        transitions=[{"after": 0, "video": "dissolve", "video_frames": 100}],
    )
    assert _code(p) == "no_handle_for_transition"


def test_content_dissolve_center_when_both_handles_fit() -> None:
    p = plan(
        sources=[source("a", duration=2000)],
        segments=[seg(a=500, b=600), seg(a=700, b=800)],
        transitions=[{"after": 0, "video": "dissolve", "video_frames": 10}],
    )
    comp, _ = _finish(p)
    ts = _targets(comp, "transition.add")
    assert len(ts) == 1
    assert ts[0]["alignment"] == "center" and ts[0]["position"] == "end"


def test_f8_dropped_range_is_unmatched() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    app = asm.appends[0]
    # segment 1 (record 72) has no finishing op at all
    kept = [r for r in app.ranges if not (r.planned_record_in == 72 and r.video is not None)]
    assert len(kept) == len(app.ranges) - 1
    bad = asm.model_copy(update={"appends": [app.model_copy(update={"ranges": kept})]})
    assert _code(LF, prepare=prep, assemble=bad) == "segment_unmatched"


def test_f8_doubled_range_is_double_matched() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    app = asm.appends[0]
    dup: AppendedRange = app.ranges[1].model_copy()
    bad = asm.model_copy(
        update={"appends": [app.model_copy(update={"ranges": [*app.ranges, dup]})]}
    )
    assert _code(LF, prepare=prep, assemble=bad) == "segment_double_matched"


def test_f8_append_from_other_timeline_mismatch() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    app = asm.appends[0]
    other = TimelineIdentity(name=prep.branch.name, unique_id="tl-other")
    bad = asm.model_copy(update={"appends": [app, app.model_copy(update={"timeline": other})]})
    assert _code(LF, prepare=prep, assemble=bad) == "phase_result_mismatch"


def test_f8_prepare_assemble_branch_mismatch() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    other_prep = synth_prepare(LF, unique_id="tl-different")
    assert _code(LF, prepare=other_prep, assemble=asm) == "phase_result_mismatch"


def test_gain_out_of_range() -> None:
    p = plan(segments=[seg(gain_db=45.0)])
    assert _code(p) == "gain_out_of_range"
    comp, _ = _finish(plan(segments=[seg(gain_db=30.0)]))
    assert _targets(comp, "item.set_audio_gain")[0]["gain_db"] == 30.0


def test_f12_captions_not_applied() -> None:
    d = plan_data(
        captions=[{"lines": ["Hello"], "record_in": 0, "record_out": 10}],
    )
    comp, _ = _finish(EditPlan.model_validate(d))
    assert any("not applied by plan.apply" in n for n in comp.notes)
    assert [i.kind for i in comp.intents if i.kind == "captions"] == ["captions"]


def test_markers_carry_duration_and_note() -> None:
    d = plan_data(
        markers=[
            {"record": 5, "name": "Beat", "color": "Red", "note": "check audio", "duration": 7}
        ]
    )
    comp, _ = _finish(EditPlan.model_validate(d))
    step = next(s for s in comp.steps if s.op == "markers.add")
    m = step.params["markers"][0]
    assert (m["frame"], m["duration"], m["note"], m["name"], m["color"]) == (
        5,
        7,
        "check audio",
        "Beat",
        "Red",
    )


def test_fades_merge_into_one_target() -> None:
    d = plan_data(
        head={"video": "cut", "video_frames": 0, "audio_fade_frames": 10},
        end_fade_frames=5,
    )
    comp, _ = _finish(EditPlan.model_validate(d))
    fades = _targets(comp, "item.set_fades")
    assert len(fades) == 1
    assert fades[0]["fade_in"] == 10 and fades[0]["fade_out"] == 5


def test_transform_keeps_pan_and_tilt_distinct() -> None:
    d = plan_data(
        segments=[seg(transform={"zoom": 1.2, "pan": 40.0, "tilt": -15.0})],
    )
    comp, _ = _finish(EditPlan.model_validate(d))
    t = _targets(comp, "item.set_transform")[0]
    assert t["pan"] == 40.0 and t["tilt"] == -15.0 and t["zoom"] == 1.2


def test_title_with_five_lines_is_refused() -> None:
    # title.place allows at most 4 lines; a 5-line title is refused, not dropped (R8).
    p = plan(
        titles=[
            {
                "kind": "title",
                "lines": ["a", "b", "c", "d", "e"],
                "record_in": 0,
                "record_out": 10,
                "template": "T",
            }
        ]
    )
    prep = synth_prepare(p)
    assert _code(p, prepare=prep, assemble=synth_assemble(p, prep)) == "title_too_many_lines"


def test_idempotency_requires_full_source_range_identity() -> None:
    # An item at the right record_in but a different source range is NOT "present".
    from aar.node.ops.read import DescribeResult, ItemInfo, TimelineInfo, TrackInfo
    from aar.plan.apply import compile_assemble
    from aar.plan.apply.render import idempotency

    p = plan(segments=[seg(a=0, b=24), seg(a=48, b=72)])
    prep = synth_prepare(p)
    assemble = compile_assemble(p, prepare=prep)
    finish, _ = _finish(p)
    wrong = ItemInfo(
        index=0,
        item_id="x",
        name="n",
        type="Video",
        start=0,
        end=24,
        duration=24,
        record_in=0,
        record_out=24,
        source_start=999,
        source_end=1023,
        media_id="pool-a",
        fade_in=None,
        fade_out=None,
        audio_volume_db=None,
        enabled=True,
    )
    describe = DescribeResult(
        timeline=TimelineInfo(
            name=prep.branch.name,
            unique_id=prep.branch.unique_id,
            fps=24.0,
            drop_frame=False,
            start_frame=0,
            end_frame=100,
            duration=100,
            start_timecode="01:00:00:00",
            width=1920,
            height=1080,
        ),
        tracks=[
            TrackInfo(
                type="video",
                index=1,
                name="V1",
                enabled=True,
                locked=False,
                subtype="",
                item_count=1,
                transitions=0,
                truncated=False,
                items=[wrong],
                gaps=[],
                overlaps=0,
            )
        ],
        markers=[],
        media=[],
        notes=[],
        text="",
    )
    status = idempotency(p, assemble, finish, describe)
    # record_in 0 matches but the source range does not, so it is still "to add".
    assert status["appends"]["present"] == 0
    assert any("@ 0" in m for m in status["appends"]["missing"])
