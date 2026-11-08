"""Phase 1 (``compile_prepare``): branch, import, find title templates."""

from __future__ import annotations

import pytest
from apply_builders import longform, plan, plan_data, seg, source
from apply_support import template_describe

from aar.node.ops.read import DescribeResult, TrackInfo
from aar.plan.apply import CompileError, compile_prepare
from aar.plan.model import EditPlan

LF = longform()


def _with_audio_items(count: int, video: int) -> DescribeResult:
    base = template_describe(video, name=LF.template or "")
    audio = TrackInfo(
        type="audio",
        index=1,
        name="A1",
        enabled=True,
        locked=False,
        subtype="stereo",
        item_count=count,
        transitions=0,
        truncated=False,
        items=[],
        gaps=[],
        overlaps=0,
    )
    return base.model_copy(update={"tracks": [*base.tracks, audio]})


def test_op_order_branch_import_find() -> None:
    comp = compile_prepare(LF, template=template_describe(10, name=LF.template or ""))
    ops = [s.op for s in comp.steps]
    assert comp.phase == "prepare"
    assert ops[0] == "timeline.branch"
    n_import = sum(op == "media.import" for op in ops)
    n_find = sum(op == "media.find" for op in ops)
    # cameras A, B, A2, X -> four bins; titles: Lower Third + Sting -> two templates
    assert n_import == 4
    assert n_find == 2
    assert ops == ["timeline.branch"] + ["media.import"] * 4 + ["media.find"] * 2
    assert all(s.phase == "prepare" for s in comp.steps)


def test_branch_source_real_with_template_describe() -> None:
    comp = compile_prepare(
        LF, template=template_describe(5, name=LF.template or "", unique_id="U1")
    )
    branch = comp.steps[0]
    assert branch.params["source"]["unique_id"] == "U1"
    assert comp.unresolved == 0


def test_branch_source_unresolved_without_template() -> None:
    comp = compile_prepare(LF)
    assert comp.unresolved >= 1
    assert "__unresolved__" in comp.steps[0].params["source"]
    assert any("template size unchecked" in n for n in comp.notes)


def test_template_name_mismatch() -> None:
    with pytest.raises(CompileError) as ei:
        compile_prepare(LF, template=template_describe(3, name="Other"))
    assert ei.value.code == "template_name_mismatch"


def test_template_too_large_and_boundary() -> None:
    with pytest.raises(CompileError) as ei:
        compile_prepare(LF, template=template_describe(31, name=LF.template or ""))
    assert ei.value.code == "template_too_large"
    compile_prepare(LF, template=template_describe(30, name=LF.template or ""))


def test_template_count_sums_audio_items() -> None:
    compile_prepare(LF, template=_with_audio_items(10, 20))  # 30: allowed
    with pytest.raises(CompileError) as ei:
        compile_prepare(LF, template=_with_audio_items(11, 20))  # 31: refused
    assert ei.value.code == "template_too_large"
    assert ei.value.details["items"] == 31


def test_missing_template() -> None:
    p = EditPlan.model_validate(plan_data(template=None))
    with pytest.raises(CompileError) as ei:
        compile_prepare(p)
    assert ei.value.code == "missing_template"


def test_import_dedupes_same_path_and_camera() -> None:
    p = plan(
        sources=[source("a", "E:/X/same.mov"), source("b", "E:/X/same.mov")],
        segments=[seg(src="a"), seg(src="b", a=24, b=48)],
    )
    imports = [s for s in compile_prepare(p).steps if s.op == "media.import"]
    assert len(imports) == 1
    assert imports[0].params["paths"] == ["E:/X/same.mov"]


def test_import_bins_by_camera_and_sources_fallback() -> None:
    p = plan(
        sources=[
            source("a", "E:/X/a.mov", camera="A"),
            source("b", "E:/X/b.mov", camera=None),
        ],
        segments=[seg(src="a"), seg(src="b", a=0, b=24)],
    )
    imports = {
        tuple(s.params["bin_path"]) for s in compile_prepare(p).steps if s.op == "media.import"
    }
    assert imports == {("aar", "A"), ("aar", "sources")}


def test_fifty_files_in_one_bin_are_one_import_step() -> None:
    # media.import takes 1..50 paths; exactly 50 in a camera bin is a single step, 51 is two.
    srcs = [source(f"s{i}", media=f"E:/CCN/A{i:03d}.mov", camera="A") for i in range(50)]
    segs = [seg(src=f"s{i}", a=i * 2, b=i * 2 + 1) for i in range(50)]
    comp = compile_prepare(plan(segments=segs, sources=srcs))
    imports = [s for s in comp.steps if s.op == "media.import"]
    assert len(imports) == 1
    assert len(imports[0].params["paths"]) == 50

    srcs.append(source("s50", media="E:/CCN/A050.mov", camera="A"))
    segs.append(seg(src="s50", a=200, b=201))
    comp2 = compile_prepare(plan(segments=segs, sources=srcs))
    assert len([s for s in comp2.steps if s.op == "media.import"]) == 2
