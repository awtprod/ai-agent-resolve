"""Property test: random valid plans export, and reading the file back reproduces them exactly."""

from __future__ import annotations

import pytest
from export_support import MEDIA_MAP, permissive, random_plan, reverify, structure_problems

from aar.plan.export import ExportError, export_fcpxml, media_map_from_mapping
from aar.plan.export.reader import read_fcpxml
from aar.plan.model import EditPlan

SEEDS = range(300)


def _export(seed: int):
    data, start = random_plan(seed)
    plan = EditPlan.model_validate(data)
    return plan, export_fcpxml(plan, permissive(start), media_map=media_map_from_mapping(MEDIA_MAP))


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
    again = reverify(result)
    assert again.ok, again.problems
    # Independent of the verifier: the decoded file holds every plan segment and the spine
    # (with transition overlaps taken out) spans exactly the plan's length.
    decoded = read_fcpxml(result.fcpxml)
    layout = result.layout
    ranges = plan.record_ranges()
    spans = {(c.lane, c.raw_in, c.raw_out - c.raw_in) for c in decoded.clips}
    for index, seg in enumerate(plan.segments):
        if seg.track == 1:
            continue  # spine clips may be widened under transitions; the verifier decodes them
        assert (seg.track - 1, ranges[index][0], seg.length) in spans
    entries = decoded.spine
    overlap = sum(
        e.raw_out - e.raw_in for k, e in enumerate(entries) if e.kind == "transition" and k > 0
    )
    spine = sum(e.raw_out - e.raw_in for e in entries if e.kind != "transition")
    assert spine - overlap == layout.total_frames


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
        seen |= {"audio-only-clip" for c in layout.clips if c.enable == "audio"}
        seen |= {"keyframes" for c in layout.clips if c.transform and c.transform.keyframes}
        seen |= {"title" for _ in layout.titles}
        seen |= {"marker" for _ in layout.markers}
        seen |= {f"rule:{m.rule}" for m in layout.media.values()}
        seen |= {"gap" for e in read_fcpxml(result.fcpxml).spine if e.kind == "gap"}
        seen |= {"split-audio" for c in layout.clips if "audioStart" in result.fcpxml}
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
        "audio-only-clip",
        "keyframes",
        "title",
        "marker",
        "rule:windows-path",
        "rule:file-url",
        "rule:media-map",
        "gap",
        "split-audio",
    }
    assert want <= seen, want - seen
