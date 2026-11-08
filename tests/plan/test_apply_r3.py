"""Round-3 pins: media classification (R1), phase-result cross-checks (R2), the F1 gate
(R3: W15/W35/W37/W38/W36/W43), refuse-not-drop (R4), the idempotency/batch fixes (R5), unknown op
(R6), unknown-handle refusal (R7) and titles (R8). The job builder (R9 arity, N1-N3, N5, N7) is
pinned in test_apply_jobs.py; the round-4 compile-path checks (N4, N6, N8) in test_apply_r4.py."""

from __future__ import annotations

import pytest
from apply_builders import longform, longform_data, plan, seg, source
from apply_support import synth_assemble, synth_prepare, template_describe

from aar.plan.apply import (
    CompileError,
    build_submission,
    compile_assemble,
    compile_finish,
    compile_prepare,
    plan_hash,
    title_track,
)
from aar.plan.apply.inputs import classify_media
from aar.plan.apply.refs import find_unresolved
from aar.plan.model import EditPlan

LF = longform()


def _assemble_code(p: EditPlan, **kw) -> str:
    with pytest.raises(CompileError) as ei:
        compile_assemble(p, **kw)
    return ei.value.code


def _finish_code(p: EditPlan, **kw) -> str:
    with pytest.raises(CompileError) as ei:
        compile_finish(p, **kw)
    return ei.value.code


# -- R1: media classification ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("media", "kind"),
    [
        ("E:/CCN/A001.mov", "import"),
        ("X:\\CCN\\A001.braw", "import"),
        ("B001 multicam", "pool"),
        ("/srv/ccn/B001.mov", "error"),
        ("B001.mov", "error"),
        ("E:/CCN//B001.mov", "error"),
        ("E:B001.mov", "error"),
    ],
)
def test_media_classification(media: str, kind: str) -> None:
    assert classify_media(media) == kind


def test_non_edit_pc_path_is_refused() -> None:
    p = plan(sources=[source("a", media="/srv/ccn/B001.mov")], segments=[seg(src="a")])
    assert _assemble_code(p) == "media_not_on_edit_pc"
    with pytest.raises(CompileError) as ei:
        compile_prepare(p, template=template_describe(5, name=p.template))
    assert ei.value.code == "media_not_on_edit_pc"


def test_pool_name_source_gets_a_find_step_in_prepare() -> None:
    p = plan(
        sources=[source("a", media="E:/X/a.mov"), source("b", media="B001 multicam", camera="B")],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48)],
    )
    comp = compile_prepare(p, template=template_describe(5, name=p.template))
    finds = [s for s in comp.steps if s.name == "find:source:b"]
    assert len(finds) == 1
    assert finds[0].params["name"] == "B001 multicam"
    # N5: recursive over the WHOLE pool, not scoped to a camera bin.
    assert finds[0].params["bin_path"] == []
    assert finds[0].params["recursive"] is True


def test_w01_source_pool_id_only_from_prepare() -> None:
    # A source absent from the prepare result is Unresolved in assemble, never its own media text.
    p = plan(
        sources=[source("a", media="E:/X/a.mov"), source("b", media="B001 multicam", camera="B")],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48)],
    )
    prep = synth_prepare(p)
    prep = prep.model_copy(update={"sources": [s for s in prep.sources if s.source_id != "b"]})
    comp = compile_assemble(p, prepare=prep)
    # No resolved pool id anywhere equals the raw "B001 multicam"; it is Unresolved instead.
    for step in comp.steps:
        for rng in step.params.get("ranges", []):
            media = rng["media"]
            if "__unresolved__" not in media:
                assert media["unique_id"] != "B001 multicam"
    with pytest.raises(CompileError) as ei:
        build_submission(p, "assemble", comp.steps)
    assert ei.value.code == "unresolved_reference"


# -- R2: forged / stale phase results ----------------------------------------------------------


def test_forged_pool_id_is_a_path() -> None:
    good = synth_prepare(LF)
    bad = good.model_copy(
        update={
            "sources": [
                s.model_copy(
                    update={"media": s.media.model_copy(update={"unique_id": s.file_path})}
                )
                for s in good.sources
            ]
        }
    )
    assert _assemble_code(LF, prepare=bad) == "forged_phase_result"


def test_forged_file_path_mismatch() -> None:
    good = synth_prepare(LF)
    bad = good.model_copy(
        update={
            "sources": [
                s.model_copy(update={"file_path": "E:/Other/Z999.mov"})
                if s.source_id == "camA"
                else s
                for s in good.sources
            ]
        }
    )
    assert _assemble_code(LF, prepare=bad) == "forged_phase_result"


def test_duplicate_source_id() -> None:
    good = synth_prepare(LF)
    dup = good.sources[0].model_copy(
        update={"media": good.sources[0].media.model_copy(update={"unique_id": "pool-EVIL"})}
    )
    assert _assemble_code(
        LF, prepare=good.model_copy(update={"sources": [*good.sources, dup]})
    ) == ("duplicate_source")


def test_unknown_source_id_in_prepare() -> None:
    good = synth_prepare(LF)
    ghost = good.sources[0].model_copy(update={"source_id": "ghost"})
    assert (
        _assemble_code(LF, prepare=good.model_copy(update={"sources": [*good.sources, ghost]}))
        == "unknown_source"
    )


def test_forged_title_id_equal_to_template_name() -> None:
    good = synth_prepare(LF)
    bad = good.model_copy(
        update={
            "titles": [
                t.model_copy(update={"media": t.media.model_copy(update={"unique_id": t.template})})
                for t in good.titles
            ]
        }
    )
    assert _finish_code(LF, prepare=bad, assemble=synth_assemble(LF, good)) == "forged_phase_result"


def test_stale_prepare_sha() -> None:
    bad = synth_prepare(LF).model_copy(update={"plan_sha256": "0" * 64})
    assert _assemble_code(LF, prepare=bad) == "stale_phase_result"


def test_stale_assemble_segment_moved() -> None:
    # The assemble result is from an older plan: segment 1 changed source and range at the same
    # record_in. Matching by (track, record_in) alone would miss it; the range check catches it.
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)  # from LF
    d = longform_data()
    d["segments"][1]["source"] = "camA"
    d["segments"][1]["src_in"] = 500
    d["segments"][1]["src_out"] = 572
    newer = EditPlan.model_validate(d)
    newer_asm = asm.model_copy(update={"plan_sha256": plan_hash(newer)})
    assert _finish_code(newer, prepare=synth_prepare(newer), assemble=newer_asm) == (
        "stale_phase_result"
    )


def test_finish_requires_prepare() -> None:
    assert _finish_code(LF, assemble=synth_assemble(LF, synth_prepare(LF))) == (
        "missing_prepare_result"
    )


def test_finish_prepare_assemble_branch_mismatch() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, synth_prepare(LF, unique_id="tl-OTHER"))
    assert _finish_code(LF, prepare=prep, assemble=asm) == "phase_result_mismatch"


# -- R3: the F1 gate (W15, W35, W37, W38, W36, W43) --------------------------------------------


def test_w15_braw_recheck_uses_the_real_clip_path() -> None:
    # The plan media looks non-BRAW (a pool name), but the imported file is BRAW. The BRAW skip
    # must read the prepare file_path, not the plan text.
    p = plan(
        sources=[
            source(
                "a", media="B001 clip", camera="A", color={"input_color_space": "Rec.709 (Scene)"}
            )
        ],
        segments=[seg(src="a", a=0, b=24)],
    )
    prep = synth_prepare(p)
    prep = prep.model_copy(
        update={
            "sources": [
                s.model_copy(update={"file_path": "E:/CCN/B001.braw"}) for s in prep.sources
            ]
        }
    )
    comp = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    assert not [s for s in comp.steps if s.op == "clip.set_input_color_space"]
    assert any("BRAW" in n for n in comp.notes)


def test_w35_w37_w38_unresolved_scan_is_deep_and_covers_all_steps() -> None:
    # A marker nested in a list, not in the first step, must still be caught.
    p = plan(
        sources=[source("a", media="E:/X/a.mov"), source("b", media="B001 multicam", camera="B")],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48)],
    )
    prep = synth_prepare(p)
    prep = prep.model_copy(update={"sources": [s for s in prep.sources if s.source_id != "b"]})
    comp = compile_assemble(p, prepare=prep)
    # The unresolved media sits in ranges[...].media of an append step (a nested list).
    hits = [path for step in comp.steps for path, _ in find_unresolved(step.params)]
    assert any("ranges[" in h and h.endswith("media") for h in hits)
    with pytest.raises(CompileError) as ei:
        build_submission(p, "assemble", comp.steps)
    assert ei.value.code == "unresolved_reference"


def test_w36_key_depends_on_the_plan() -> None:
    steps = compile_prepare(LF, template=template_describe(5, name=LF.template)).steps
    key_lf = build_submission(LF, "prepare", steps, input_hashes=["h"]).idempotency_key
    d = longform_data()
    d["timeline"] = "A Different Timeline"
    other = EditPlan.model_validate(d)
    steps2 = compile_prepare(other, template=template_describe(5, name=other.template)).steps
    key_other = build_submission(other, "prepare", steps2, input_hashes=["h"]).idempotency_key
    assert key_lf != key_other


def test_w43_finish_key_depends_on_the_assemble_result() -> None:
    prep = synth_prepare(LF)
    steps = compile_finish(LF, prepare=prep, assemble=synth_assemble(LF, prep)).steps
    k1 = build_submission(LF, "finish", steps, input_hashes=["p", "asm1"]).idempotency_key
    k2 = build_submission(LF, "finish", steps, input_hashes=["p", "asm2"]).idempotency_key
    assert k1 != k2


# -- R4: refuse rather than drop ---------------------------------------------------------------


def test_outro_without_audio_track_is_refused() -> None:
    d = longform_data()
    d["outro"]["audio_track"] = None
    assert _assemble_code(EditPlan.model_validate(d)) == "outro_audio_track_missing"


def test_end_fade_without_audio_is_refused() -> None:
    # An outro with no audio_track and an end fade: no track to fade -> typed refusal, not a drop.
    d = longform_data()
    d["outro"]["audio_track"] = None
    # compile_finish hits the end fade (assemble refusal is phase 2); call finish directly.
    assert _finish_code(EditPlan.model_validate(d)) in (
        "end_fade_without_audio",
        "outro_audio_track_missing",
    )


# -- R6: unknown op ----------------------------------------------------------------------------


def test_unknown_op_has_its_own_code() -> None:
    from aar.plan.apply.model import Step

    bad = Step(phase="finish", op="does.not.exist", name="x", params={}, summary="")
    with pytest.raises(CompileError) as ei:
        build_submission(LF, "finish", [bad], input_hashes=["h"])
    assert ei.value.code == "unknown_op"


# -- R7: unknown handle -> no handle -----------------------------------------------------------


def test_unknown_duration_counts_as_no_handle() -> None:
    p = plan(
        sources=[source("a", media="E:/X/a.mov", duration=None)],
        segments=[seg(src="a", a=0, b=24), seg(src="a", a=2, b=26)],
        transitions=[{"after": 0, "video": "dissolve", "video_frames": 15}],
    )
    prep = synth_prepare(p)
    assert _finish_code(p, prepare=prep, assemble=synth_assemble(p, prep)) == (
        "no_handle_for_transition"
    )


# -- R8: title missing from prepare -> unresolved (W29) ----------------------------------------


def test_w29_title_missing_from_prepare_is_unresolved() -> None:
    prep = synth_prepare(LF)
    prep = prep.model_copy(update={"titles": []})
    comp = compile_finish(LF, prepare=prep, assemble=synth_assemble(LF, prep))
    assert find_unresolved([s.params for s in comp.steps if s.op == "title.place"]), (
        "a title missing from prepare must be Unresolved, not dropped"
    )


# -- validator-mutant kills (W07, W20, W23, W28, W31, W39, W40) --------------------------------


def test_w07_sting_only_plan_ensures_the_title_track() -> None:
    d = longform_data()
    d["titles"] = []  # keep the outro sting, drop the lower thirds
    p = EditPlan.model_validate(d)
    comp = compile_assemble(p, prepare=synth_prepare(p))
    tracks = comp.steps[0].params["tracks"]
    tt = title_track(p)
    assert any(t["type"] == "video" and t["index"] == tt for t in tracks)


def test_w20_centre_rounds_down() -> None:
    # 15-frame dissolve with exactly 7 frames of handle each side -> centre (7 = 15 // 2).
    p = plan(
        sources=[source("a", media="E:/X/a.mov", duration=31)],
        segments=[seg(src="a", a=0, b=24), seg(src="a", a=7, b=31)],
        transitions=[{"after": 0, "video": "dissolve", "video_frames": 15}],
    )
    prep = synth_prepare(p)
    comp = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    tx = next(s for s in comp.steps if s.op == "transition.add")
    simple = next(t for t in tx.params["targets"] if t["category"] == "simple")
    assert simple["alignment"] == "center"


def test_w23_outro_incoming_head_is_used() -> None:
    # Outro starts at source frame 10 with a long tail, so a 15-frame dissolve centres on the cut.
    d = longform_data()
    d["outro"]["src_in"] = 10
    d["outro"]["src_out"] = 100
    d["outro"]["transition"] = {"video": "dissolve", "video_frames": 15}
    p = EditPlan.model_validate(d)
    prep = synth_prepare(p)
    comp = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    outro_tx = next(
        t
        for s in comp.steps
        if s.op == "transition.add"
        for t in s.params["targets"]
        if t["category"] == "simple" and t["position"] == "end"
    )
    assert outro_tx["alignment"] == "center"


def test_w28_gain_below_minus_ninety_is_allowed() -> None:
    p = plan(segments=[seg(a=0, b=24, gain_db=-95.0)])
    prep = synth_prepare(p)
    comp = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    gains = [t for s in comp.steps if s.op == "item.set_audio_gain" for t in s.params["targets"]]
    assert gains and gains[0]["gain_db"] == -95.0


def test_w31_exactly_the_step_cap_is_allowed() -> None:
    # 4990 single-frame segments -> 499 append batches + 1 track.ensure = 500 steps (the cap).
    segs = [seg(src="a", a=i, b=i + 1) for i in range(4990)]
    p = plan(sources=[source("a", media="E:/X/a.mov", duration=5000)], segments=segs)
    comp = compile_assemble(p, prepare=synth_prepare(p))
    assert len(comp.steps) == 500  # allowed; the mutant (>=) would refuse


def _describe_one(
    record_in: int, src_in: int, src_out: int, media_id: str, track_type="video", index=1
):
    from aar.node.ops.read import DescribeResult, ItemInfo, TimelineInfo, TrackInfo

    item = ItemInfo(
        index=0,
        item_id="x",
        name="n",
        type="Video",
        start=record_in,
        end=record_in + 1,
        duration=1,
        record_in=record_in,
        record_out=record_in + 1,
        source_start=src_in,
        source_end=src_out,
        media_id=media_id,
        fade_in=None,
        fade_out=None,
        audio_volume_db=None,
        enabled=True,
    )
    return DescribeResult(
        timeline=TimelineInfo(
            name="b",
            unique_id="tl-b",
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
                type=track_type,
                index=index,
                name="T",
                enabled=True,
                locked=False,
                subtype="",
                item_count=1,
                transitions=0,
                truncated=False,
                items=[item],
                gaps=[],
                overlaps=0,
            )
        ],
        markers=[],
        media=[],
        notes=[],
        text="",
    )


def test_w39_idempotency_checks_media_id() -> None:
    from aar.plan.apply.render import idempotency

    p = plan(segments=[seg(src="a", a=0, b=24)])
    prep = synth_prepare(p)
    assemble = compile_assemble(p, prepare=prep)
    finish = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    # Right record_in and source range, but a different media id -> not present.
    describe = _describe_one(0, 0, 24, "pool-WRONG")
    status = idempotency(p, assemble, finish, describe)
    assert status["appends"]["present"] == 0


def test_w40_idempotency_titles_use_the_right_track() -> None:
    from aar.plan.apply.render import idempotency

    prep = synth_prepare(LF)
    assemble = compile_assemble(LF, prepare=prep)
    finish = compile_finish(LF, prepare=prep, assemble=synth_assemble(LF, prep))
    # A title present on the real title track reads as present (the mutant looks on +1).
    describe = _describe_one(48, 0, 1, "m", index=title_track(LF))
    status = idempotency(LF, assemble, finish, describe)
    assert status["titles"]["present"] >= 1


def test_x18_idempotency_keys_items_by_track_type() -> None:
    # R5: an A1 audio item with segment 0's record_in and source range must not mark the V1
    # append present (the mutant keys every track as video).
    from aar.plan.apply.render import idempotency

    p = plan(segments=[seg(src="a", a=0, b=24)])
    prep = synth_prepare(p)
    assemble = compile_assemble(p, prepare=prep)
    finish = compile_finish(p, prepare=prep, assemble=synth_assemble(p, prep))
    audio_only = _describe_one(0, 0, 24, "pool-a", track_type="audio", index=1)
    status = idempotency(p, assemble, finish, audio_only)
    assert status["appends"]["present"] == 0
    assert status["appends"]["missing"] == ["V1 @ 0 (segment 0)"]
    video = _describe_one(0, 0, 24, "pool-a", track_type="video", index=1)
    assert idempotency(p, assemble, finish, video)["appends"]["present"] == 1
