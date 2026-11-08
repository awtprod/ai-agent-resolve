"""Phase 2 (``compile_assemble``): track.ensure then append_ranges batches."""

from __future__ import annotations

import json
import random
from typing import Any

import pytest
from apply_builders import longform, longform_data, plan, plan_data, seg, source
from apply_support import BRANCH_UID, synth_prepare

from aar.plan.apply import CompileError, compile_assemble
from aar.plan.apply.refs import find_unresolved
from aar.plan.model import EditPlan

LF = longform()


def _ranges(comp: Any) -> list[dict[str, Any]]:
    return [r for s in comp.steps if s.op == "timeline.append_ranges" for r in s.params["ranges"]]


def test_step_order_and_real_ids_with_prepare() -> None:
    comp = compile_assemble(LF, prepare=synth_prepare(LF))
    assert comp.phase == "assemble"
    assert comp.steps[0].op == "track.ensure"
    assert all(s.op == "timeline.append_ranges" for s in comp.steps[1:])
    assert comp.unresolved == 0
    for step in comp.steps:
        assert step.params["timeline"]["unique_id"] == BRANCH_UID
    ids = {r["media"]["unique_id"] for r in _ranges(comp)}
    assert ids <= {"pool-camA", "pool-camB", "pool-camA2", "pool-outroclip"}


def test_unresolved_without_prepare() -> None:
    comp = compile_assemble(LF)
    assert comp.unresolved > 0
    assert find_unresolved(comp.steps[0].params)  # timeline ref
    for r in _ranges(comp):
        assert find_unresolved(r["media"])


@pytest.mark.parametrize("batch", [1, 3, 10])
@pytest.mark.parametrize("seed", range(6))
def test_every_segment_appended_once_in_order(seed: int, batch: int) -> None:
    rng = random.Random(seed * 100 + batch)
    ntracks = rng.randint(1, 3)
    segments = []
    for _ in range(rng.randint(1, 25)):
        a = rng.randint(0, 500)
        segments.append(seg(track=rng.randint(1, ntracks), a=a, b=a + rng.randint(1, 60)))
    p = plan(segments=segments)
    comp = compile_assemble(p, prepare=synth_prepare(p), batch_size=batch)
    appends = [s for s in comp.steps if s.op == "timeline.append_ranges"]
    assert all(len(s.params["ranges"]) <= batch for s in appends)
    labels = [r["label"] for r in _ranges(comp)]
    assert sorted(labels) == sorted(f"segment {i}" for i in range(len(segments)))
    assert len(labels) == len(set(labels))
    assert sorted(pl.segment for pl in comp.placements) == list(range(len(segments)))
    by_track: dict[int, list[int]] = {}
    for r in _ranges(comp):
        by_track.setdefault(r["track"], []).append(r["record_in"])
    import itertools

    for ins in by_track.values():
        assert all(a < b for a, b in itertools.pairwise(ins))


@pytest.mark.parametrize("bad", [0, 11, -1])
def test_batch_size_invalid(bad: int) -> None:
    with pytest.raises(CompileError) as ei:
        compile_assemble(LF, batch_size=bad)
    assert ei.value.code == "batch_size_invalid"


def test_f4_outro_audio_on_named_track() -> None:
    comp = compile_assemble(LF, prepare=synth_prepare(LF))
    rngs = _ranges(comp)
    outro_audio = [r for r in rngs if r["media_type"] == "audio"]
    assert len(outro_audio) == 1
    assert outro_audio[0]["track"] == 5  # A5
    outro_video = [r for r in rngs if r["label"] == "outro"]
    assert len(outro_video) == 1
    assert outro_video[0]["media_type"] == "video"
    assert outro_video[0]["track"] == 1
    assert not [r for r in rngs if r["media_type"] == "audio" and r["track"] == 1]
    assert outro_audio[0]["record_in"] == outro_video[0]["record_in"] == LF.content_frames()


def test_f5_track_ensure_never_disables() -> None:
    comp = compile_assemble(LF, prepare=synth_prepare(LF))
    assert comp.steps[0].params["tracks"]
    assert all(t.get("enabled") is not False for t in comp.steps[0].params["tracks"])
    assert "false" not in json.dumps(comp.steps[0].params).lower()
    assert any("mute is template state" in w for w in comp.warnings)
    assert sum("mute is template state" in w for w in comp.warnings) == 4  # A1..A4 muted


def test_zero_length_segment() -> None:
    p = plan(segments=[seg(a=10, b=10)])
    with pytest.raises(CompileError) as ei:
        compile_assemble(p)
    assert ei.value.code == "zero_length_segment"


def test_title_track_out_of_range() -> None:
    segs = [seg(track=t, a=t * 100, b=t * 100 + 24) for t in range(1, 17)]
    data = plan_data(
        segments=segs,
        titles=[{"lines": ["Hi"], "record_in": 0, "record_out": 10, "template": "T"}],
    )
    with pytest.raises(CompileError) as ei:
        compile_assemble(EditPlan.model_validate(data))
    assert ei.value.code == "title_track_out_of_range"


def test_title_track_16_allowed() -> None:
    segs = [seg(track=t, a=t * 100, b=t * 100 + 24) for t in range(1, 16)]
    data = plan_data(
        segments=segs,
        titles=[{"lines": ["Hi"], "record_in": 0, "record_out": 10, "template": "T"}],
    )
    comp = compile_assemble(EditPlan.model_validate(data))
    assert comp.steps[0].op == "track.ensure"


def test_unknown_source() -> None:
    p = plan(segments=[seg(src="ghost")])
    with pytest.raises(CompileError) as ei:
        compile_assemble(p)
    assert ei.value.code == "unknown_source"


def test_longform_data_unchanged_by_builders() -> None:
    assert longform_data()["template"] == "CCN Longform Template"
    assert source("z")["media"].endswith("z.mov")
