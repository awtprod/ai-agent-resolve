"""Round-4 pins on the compile path: pool-id shape on every envelope (N4), the whole-pool find for
pool-name sources (N5), the stale check failing closed and matching source frames to the read-back
tolerance (N6), one import per shared file (N7), record_out not compared (N8), plus the r3
survivors X03, X07, X08 and X12. The job builder is pinned in test_apply_jobs.py."""

from __future__ import annotations

import pytest
from apply_builders import longform, longform_data, plan, seg, source
from apply_support import synth_assemble, synth_prepare, template_describe

from aar.node.ops.framework import MediaRef
from aar.plan.apply import CompileError, compile_assemble, compile_finish, compile_prepare
from aar.plan.apply.inputs import (
    AssembleResult,
    FoundTitle,
    PrepareResult,
    classify_media,
    validate_assemble,
)
from aar.plan.model import EditPlan

LF = longform()


def _code(fn, *args, **kw) -> str:
    with pytest.raises(CompileError) as ei:
        fn(*args, **kw)
    return ei.value.code


def _pool_plan() -> EditPlan:
    d = longform_data()
    d["sources"][1]["media"] = "B001 multicam"  # camB: a CCN multicam clip, by pool name
    return EditPlan.model_validate(d)


def _with_source_id(prep: PrepareResult, source_id: str, unique_id: str) -> PrepareResult:
    sources = [
        s.model_copy(update={"media": MediaRef(unique_id=unique_id)})
        if s.source_id == source_id
        else s
        for s in prep.sources
    ]
    return prep.model_copy(update={"sources": sources})


# -- N4: a pool id is never a plan string and never path-shaped ---------------------------------


@pytest.mark.parametrize(
    "unique_id",
    [
        "B001 multicam",  # camB's own pool name (the W01 value)
        "camB",  # its source id
        "B",  # its camera
        "CCN Lower Third",  # a title template name
        "CCN Longform Template",  # the plan's template
        "uid:1",  # a colon anywhere
        "pool/1",
        "pool\\1",
    ],
)
def test_n4_pool_id_shape_is_refused_on_a_hand_written_envelope(unique_id: str) -> None:
    p = _pool_plan()
    bad = _with_source_id(synth_prepare(p), "camB", unique_id)
    assert _code(compile_assemble, p, prepare=bad) == "forged_phase_result"


def test_n4_an_ordinary_pool_id_is_accepted() -> None:
    p = _pool_plan()
    ok = _with_source_id(synth_prepare(p), "camB", "5f1c2e3a-0b9d-4c55-9a51-0d6c3a1f2b7e")
    assert compile_assemble(p, prepare=ok).phase == "assemble"


@pytest.mark.parametrize("unique_id", ["CCN Lower Third", "CCN Sting", "t:1", "t/1", "t\\1"])
def test_n4_title_id_shape_is_refused(unique_id: str) -> None:
    good = synth_prepare(LF)
    titles = [
        t.model_copy(update={"media": MediaRef(unique_id=unique_id)})
        if t.template == "CCN Lower Third"
        else t
        for t in good.titles
    ]
    bad = good.model_copy(update={"titles": titles})
    assert _code(compile_finish, LF, prepare=bad) == "forged_phase_result"


# -- X07 / X08: title templates in an envelope --------------------------------------------------


def test_x07_foreign_title_template_in_an_envelope_is_refused() -> None:
    good = synth_prepare(LF)
    extra = FoundTitle(template="Some Other Template", media=MediaRef(unique_id="title-x"))
    bad = good.model_copy(update={"titles": [*good.titles, extra]})
    assert _code(compile_finish, LF, prepare=bad) == "title_template_missing"


def test_x08_duplicate_title_template_in_an_envelope_is_refused() -> None:
    good = synth_prepare(LF)
    dup = good.titles[0].model_copy(update={"media": MediaRef(unique_id="title-dup")})
    bad = good.model_copy(update={"titles": [*good.titles, dup]})
    assert _code(compile_finish, LF, prepare=bad) == "title_template_ambiguous"


# -- X03: a backslash or drive-relative name is path-shaped, not a pool name --------------------


@pytest.mark.parametrize("media", ["CCN\\B001 multicam", "E:B001 multicam", "/srv/B001 multicam"])
def test_x03_path_shaped_names_are_not_pool_names(media: str) -> None:
    assert classify_media(media) == "error"
    p = plan(sources=[source("a", media=media)], segments=[seg(src="a")])
    assert _code(compile_prepare, p) == "media_not_on_edit_pc"


# -- X12: the one copy of the per-append timeline check -----------------------------------------


def test_x12_validate_assemble_refuses_an_append_from_another_timeline() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    other = asm.appends[0].timeline.model_copy(update={"unique_id": "tl-other"})
    moved = asm.appends[0].model_copy(update={"timeline": other})
    bad = asm.model_copy(update={"appends": [moved]})
    with pytest.raises(CompileError) as ei:
        validate_assemble(bad, LF)
    assert ei.value.code == "phase_result_mismatch"
    assert _code(compile_finish, LF, prepare=prep, assemble=bad) == "phase_result_mismatch"


# -- N5: a pool-name source is found over the whole pool; file imports keep their bin ----------


def test_n5_pool_source_find_is_recursive_over_the_whole_pool() -> None:
    p = _pool_plan()
    comp = compile_prepare(p, template=template_describe(10, name=p.template))
    find = next(s for s in comp.steps if s.name == "find:source:camB")
    assert find.op == "media.find"
    assert find.params["name"] == "B001 multicam"
    assert find.params["bin_path"] == []
    assert find.params["recursive"] is True
    assert find.params["limit"] == 200
    # the file imports keep their aar/<camera> bins
    imports = {tuple(s.params["bin_path"]) for s in comp.steps if s.op == "media.import"}
    assert imports == {("aar", "A"), ("aar", "A2"), ("aar", "X")}


def test_n5_title_find_is_recursive_over_the_whole_pool() -> None:
    comp = compile_prepare(LF, template=template_describe(10, name=LF.template))
    finds = [s for s in comp.steps if s.name.startswith("find:title:")]
    assert [s.params["name"] for s in finds] == ["CCN Lower Third", "CCN Sting"]
    assert all(s.params["bin_path"] == [] and s.params["recursive"] is True for s in finds)


# -- N7: two sources sharing one file are imported once -----------------------------------------


def test_n7_a_file_shared_across_cameras_is_imported_once_into_the_first_bin() -> None:
    p = plan(
        sources=[
            source("a", media="E:/X/shared.mov", camera="A"),
            source("b", media="e:\\x\\SHARED.mov", camera="B"),  # same path_key
            source("c", media="E:/X/c.mov", camera="B"),
        ],
        segments=[seg(src="a", a=0, b=24), seg(src="b", a=24, b=48), seg(src="c", a=0, b=24)],
    )
    comp = compile_prepare(p)
    imports = [
        (tuple(s.params["bin_path"]), s.params["paths"])
        for s in comp.steps
        if s.op == "media.import"
    ]
    assert imports == [(("aar", "A"), ["E:/X/shared.mov"]), (("aar", "B"), ["E:/X/c.mov"])]
    assert any("b (= a)" in note for note in comp.notes)


# -- N6: the stale check fails closed; source frames match to the read-back tolerance -----------


def _first_video(asm: AssembleResult, **update) -> AssembleResult:
    first = asm.appends[0]
    r0 = first.ranges[0]
    r0 = r0.model_copy(update={"video": r0.video.model_copy(update=update)})
    first = first.model_copy(update={"ranges": [r0, *first.ranges[1:]]})
    return asm.model_copy(update={"appends": [first, *asm.appends[1:]]})


@pytest.mark.parametrize("field", ["source_start", "source_end"])
def test_n6_unread_source_frame_fails_closed(field: str) -> None:
    prep = synth_prepare(LF)
    asm = _first_video(synth_assemble(LF, prep), **{field: None})
    assert _code(compile_finish, LF, prepare=prep, assemble=asm) == "unverified_phase_result"


@pytest.mark.parametrize(
    ("field", "delta"),
    [("source_start", 2), ("source_start", -2), ("source_end", 2), ("source_end", -2)],
)
def test_n6_source_two_frames_off_is_stale(field: str, delta: int) -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    honest = getattr(asm.appends[0].ranges[0].video, field)
    bad = _first_video(asm, **{field: honest + delta})
    assert _code(compile_finish, LF, prepare=prep, assemble=bad) == "stale_phase_result"


@pytest.mark.parametrize(
    ("field", "delta"),
    [("source_start", 1), ("source_start", -1), ("source_end", 1), ("source_end", -1)],
)
def test_n6_source_one_frame_off_is_the_readback_trap_not_stale(field: str, delta: int) -> None:
    # trap source-frame-readback-rounding: source frames read back up to 1 off on an honest
    # append (the op itself accepts that), so it must not read as stale.
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    honest = getattr(asm.appends[0].ranges[0].video, field)
    ok = _first_video(asm, **{field: honest + delta})
    assert compile_finish(LF, prepare=prep, assemble=ok).phase == "finish"


def test_n6_the_check_runs_on_every_segment_not_just_the_first() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    first = asm.appends[0]
    last_in = LF.record_ranges()[-1][0]  # the last content segment (not the outro)
    at = next(
        i
        for i, r in enumerate(first.ranges)
        if r.video is not None and r.planned_record_in == last_in
    )
    assert at > 0
    rng = first.ranges[at]
    unread = rng.model_copy(update={"video": rng.video.model_copy(update={"source_end": None})})
    ranges = [*first.ranges[:at], unread, *first.ranges[at + 1 :]]
    bad = asm.model_copy(update={"appends": [first.model_copy(update={"ranges": ranges})]})
    assert _code(compile_finish, LF, prepare=prep, assemble=bad) == "unverified_phase_result"


# -- N8: record_out is not compared (the op rate-converts it) -----------------------------------


def test_n8_mixed_rate_source_is_not_reported_stale() -> None:
    # A source at another rate than the timeline lands with a rate-converted length: the op's
    # planned_record_out / planned_frames and the item's record_out differ from the plan's
    # same-rate record_out, while record_in and the source range are the plan's.
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    first = asm.appends[0]
    r0 = first.ranges[0]
    assert r0.video is not None
    converted = r0.planned_frames * 2  # e.g. a 12 fps source on a 24 fps timeline
    v0 = r0.video.model_copy(
        update={
            "record_out": r0.video.record_in + converted,
            "end": r0.video.start + converted,
            "duration": converted,
        }
    )
    r0 = r0.model_copy(
        update={
            "video": v0,
            "planned_record_out": r0.planned_record_in + converted,
            "planned_frames": converted,
            "end_frame_used": r0.end_frame_used + converted,
        }
    )
    asm = asm.model_copy(
        update={"appends": [first.model_copy(update={"ranges": [r0, *first.ranges[1:]]})]}
    )
    assert compile_finish(LF, prepare=prep, assemble=asm).phase == "finish"


def test_n8_record_in_is_still_matched_exactly() -> None:
    prep = synth_prepare(LF)
    asm = synth_assemble(LF, prep)
    first = asm.appends[0]
    r0 = first.ranges[0].model_copy(
        update={"planned_record_in": first.ranges[0].planned_record_in + 1}
    )
    bad = asm.model_copy(
        update={"appends": [first.model_copy(update={"ranges": [r0, *first.ranges[1:]]})]}
    )
    assert _code(compile_finish, LF, prepare=prep, assemble=bad) == "segment_unmatched"
