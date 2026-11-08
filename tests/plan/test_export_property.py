"""Property test: random valid plans export, and reading the file back reproduces them exactly."""

from __future__ import annotations

import pytest
from export_support import (
    MEDIA_MAP,
    permissive,
    random_plan,
    reverify,
    structure_problems,
)

from aar.plan.export import ExportError, export_fcpxml, media_map_from_mapping
from aar.plan.export.reader import read_fcpxml
from aar.plan.model import EditPlan

SEEDS = range(300)


def _export(seed: int):
    data, start = random_plan(seed)
    plan = EditPlan.model_validate(data)
    return plan, export_fcpxml(
        plan,
        permissive(start),
        media_map=media_map_from_mapping(MEDIA_MAP),
        allow_unknown_timecode=True,  # the generator names no start timecodes
    )


@pytest.mark.parametrize("seed", SEEDS)
def test_random_plan_round_trips_exactly(seed: int) -> None:
    data, start = random_plan(seed)
    plan = EditPlan.model_validate(data)
    if plan.total_frames() == 0:
        with pytest.raises(ExportError) as caught:
            export_fcpxml(plan, permissive(start))
        assert caught.value.code == "empty_timeline"
        return
    _, result = _export(seed)
    assert result.report["self_check"] == {
        "ok": True,
        "problems": [],
        "checked": result.report["self_check"]["checked"],
    }
    assert structure_problems(result.fcpxml) == []
    # Never anything Resolve mishandles (import test, 2026-11-08): no srcEnable, no marker.
    assert "srcEnable" not in result.fcpxml and "<marker" not in result.fcpxml
    again = reverify(result)
    assert again.ok, again.problems
    # Independent of the verifier: the decoded file holds every plan segment at its plan record
    # range and source range (no clip is widened for a transition), and the spine's clips and
    # gaps abut, never overlap, from 0 to the plan's length.
    decoded = read_fcpxml(result.fcpxml)
    layout = result.layout
    ranges = plan.record_ranges()
    spans = {(c.lane, c.raw_in, c.raw_out, c.start) for c in decoded.clips}
    for index, seg in enumerate(plan.segments):
        row = (layout.lanes[seg.track], ranges[index][0], ranges[index][1], seg.src_in)
        assert row in spans, row
    elements = [e for e in decoded.spine if e.kind != "transition"]
    assert [e.raw_in for e in elements] == [0, *(e.raw_out for e in elements[:-1])]
    assert elements[-1].raw_out == layout.total_frames
    for k, entry in enumerate(decoded.spine):
        if entry.kind == "transition":  # always before the element it leads into
            assert decoded.spine[k + 1].kind == "clip"


@pytest.mark.parametrize("seed", range(0, 300, 37))
def test_random_export_is_deterministic(seed: int) -> None:
    if not random_plan(seed)[0]["segments"]:
        pytest.skip("no segments for this seed")
    _, first = _export(seed)
    _, second = _export(seed)
    assert first.fcpxml == second.fcpxml
    assert first.report == second.report


def test_the_generator_reaches_every_feature() -> None:
    """The random plans exercise what the exporter decides: all three alignments, gaps, V2+,
    audio-only and video transitions, outros on and off the spine, keyframes and titles."""
    seen: set[str] = set()
    for seed in SEEDS:
        data, _ = random_plan(seed)
        if EditPlan.model_validate(data).total_frames() == 0:
            continue
        _, result = _export(seed)
        layout = result.layout
        seen |= {f"align:{t.alignment}" for t in layout.transitions}
        seen |= {f"kind:{t.kind}" for t in layout.transitions}
        seen |= {"video-transition" for t in layout.transitions if t.video}
        seen |= {"audio-only-transition" for t in layout.transitions if not t.video}
        seen |= {"connected" for c in layout.clips if c.lane > 0}
        seen |= {"keyframes" for c in layout.clips if c.transform and c.transform.keyframes}
        seen |= {"title" for _ in layout.titles}
        seen |= {"marker" for _ in layout.markers}
        seen |= {f"rule:{m.rule}" for m in layout.media.values()}
        seen |= {"gap" for e in read_fcpxml(result.fcpxml).spine if e.kind == "gap"}
        seen |= {"split-audio" for c in layout.clips if "audioStart" in result.fcpxml}
        seen |= {f"form:{t.form}" for t in layout.transitions if not t.video}
        seen |= {"conform" for c in layout.clips if c.conform}
        seen |= {"conform-unmoved" for c in layout.clips if c.conform and c.transform is None}
        seen |= {"sparse-lanes" for t in layout.lanes if t != 1 and layout.lanes[t] != t - 1}
        seen |= {"outro-moved" for i in layout.not_expressible if i.feature == "outro_sound_track"}
    want = {
        "align:center",
        "align:right",
        "align:left",
        "kind:head",
        "kind:join",
        "kind:outro",
        "video-transition",
        "audio-only-transition",
        "connected",
        "keyframes",
        "title",
        "marker",
        "rule:windows-path",
        "rule:file-url",
        "rule:media-map",
        "gap",
        "split-audio",
        "form:split",
        "conform",
        "conform-unmoved",
        "sparse-lanes",
        "outro-moved",
    }
    assert want <= seen, want - seen
