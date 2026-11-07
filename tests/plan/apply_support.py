"""Helpers for the three-phase ``aar.plan.apply`` tests: synthesize the prepare and assemble
result files (and the template / branch describes) the compiler reads, without touching Resolve.
"""

from __future__ import annotations

from typing import Any

from aar.node.ops.framework import MediaRef, TimelineIdentity
from aar.node.ops.read import DescribeResult, ItemInfo, MarkerInfo, TimelineInfo, TrackInfo
from aar.node.ops.write import AppendedRange, AppendResult, PlacedItem
from aar.plan.apply import PrepareResult, compile_assemble
from aar.plan.apply.inputs import AssembleResult, FoundTitle, ImportedSource, plan_hash
from aar.plan.model import EditPlan

BRANCH_UID = "tl-branch-0001"


def _branch_name(plan: EditPlan) -> str:
    return f"{plan.template} [agent:ccn-bot v1]"


def synth_prepare(plan: EditPlan, *, unique_id: str = BRANCH_UID) -> PrepareResult:
    """Phase-1 result: the branch, one pool item per source, one pool item per title template."""
    sources = [
        ImportedSource(
            source_id=s.id,
            media=MediaRef(unique_id=f"pool-{s.id}", bin_path=None),
            file_path=s.media,
        )
        for s in plan.sources
    ]
    templates: list[str] = []
    for title in plan.titles:
        if title.template and title.template not in templates:
            templates.append(title.template)
    sting = plan.outro.sting if plan.outro is not None else None
    if sting is not None and sting.template and sting.template not in templates:
        templates.append(sting.template)
    titles = [
        FoundTitle(template=name, media=MediaRef(unique_id=f"title-{name}", bin_path=None))
        for name in templates
    ]
    return PrepareResult(
        plan_sha256=plan_hash(plan),
        job_id="job_prepare_test",
        branch=TimelineIdentity(name=_branch_name(plan), unique_id=unique_id),
        sources=sources,
        titles=titles,
    )


def _placed(
    track: int, rin: int, rout: int, item_id: str, ttype: str, src_in: int, src_out: int
) -> PlacedItem:
    return PlacedItem(
        item_id=item_id,
        track_type=ttype,
        track=track,
        start=rin,
        end=rout,
        duration=rout - rin,
        record_in=rin,
        record_out=rout,
        source_start=src_in,
        source_end=src_out,
    )


def synth_assemble(
    plan: EditPlan, prepare: PrepareResult, *, batch_size: int = 10
) -> AssembleResult:
    """Phase-2 result: a video and/or audio item for every range phase 2 would append."""
    comp = compile_assemble(plan, prepare=prepare, batch_size=batch_size)
    ranges: list[AppendedRange] = []
    index = 0
    for step in comp.steps:
        if step.op != "timeline.append_ranges":
            continue
        for rng in step.params["ranges"]:
            track, rin = rng["track"], rng["record_in"]
            si, so = rng["src_in"], rng["src_out"]
            frames = so - si
            rout = rin + frames
            mt = rng["media_type"]
            video = (
                _placed(track, rin, rout, f"V{track}:{rin}", "video", si, so)
                if mt != "audio"
                else None
            )
            audio = (
                [_placed(track, rin, rout, f"A{track}:{rin}", "audio", si, so)]
                if mt in ("av", "audio")
                else []
            )
            ranges.append(
                AppendedRange(
                    index=index,
                    status="appended",
                    planned_record_in=rin,
                    planned_record_out=rout,
                    planned_frames=frames,
                    end_frame_used=rout,
                    video=video,
                    audio=audio,
                    notes=[],
                )
            )
            index += 1
    result = AppendResult(
        timeline=prepare.branch,
        timeline_fps=24.0,
        cursor=0,
        next_cursor=None,
        done=True,
        ranges=ranges,
        gaps=[],
        notes=[],
    )
    return AssembleResult(
        plan_sha256=plan_hash(plan),
        job_id="job_assemble_test",
        branch=prepare.branch,
        appends=[result],
    )


def template_describe(
    item_count: int, *, name: str, unique_id: str = "tl-template-0001"
) -> DescribeResult:
    items = [
        ItemInfo(
            index=i,
            item_id=f"tpl{i}",
            name=f"item {i}",
            type="Video",
            start=i,
            end=i + 1,
            duration=1,
            record_in=i,
            record_out=i + 1,
            source_start=0,
            source_end=1,
            media_id=None,
            fade_in=None,
            fade_out=None,
            audio_volume_db=None,
            enabled=True,
        )
        for i in range(item_count)
    ]
    return DescribeResult(
        timeline=TimelineInfo(
            name=name,
            unique_id=unique_id,
            fps=24.0,
            drop_frame=False,
            start_frame=0,
            end_frame=item_count,
            duration=item_count,
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
                item_count=item_count,
                transitions=0,
                truncated=False,
                items=items,
                gaps=[],
                overlaps=0,
            )
        ],
        markers=[],
        media=[],
        notes=[],
        text="",
    )


def branch_describe(
    plan: EditPlan,
    prepare: PrepareResult,
    *,
    present_labels: set[str],
    marker_frames: list[int] | None = None,
) -> DescribeResult:
    """A describe of an agent branch holding, with full identity (record_in, source in/out,
    media id), only the appended ranges whose label is in ``present_labels``."""
    comp = compile_assemble(plan, prepare=prepare)
    by_track: dict[int, list[ItemInfo]] = {}
    idx = 0
    for step in comp.steps:
        if step.op != "timeline.append_ranges":
            continue
        for rng in step.params["ranges"]:
            if rng["label"] not in present_labels:
                continue
            rin = rng["record_in"]
            rout = rin + (rng["src_out"] - rng["src_in"])
            by_track.setdefault(rng["track"], []).append(
                ItemInfo(
                    index=idx,
                    item_id=f"b{idx}",
                    name=rng["label"],
                    type="Video",
                    start=rin,
                    end=rout,
                    duration=rout - rin,
                    record_in=rin,
                    record_out=rout,
                    source_start=rng["src_in"],
                    source_end=rng["src_out"],
                    media_id=rng["media"]["unique_id"],
                    fade_in=None,
                    fade_out=None,
                    audio_volume_db=None,
                    enabled=True,
                )
            )
            idx += 1
    tracks = [
        TrackInfo(
            type="video",
            index=t,
            name=f"V{t}",
            enabled=True,
            locked=False,
            subtype="",
            item_count=len(items),
            transitions=0,
            truncated=False,
            items=items,
            gaps=[],
            overlaps=0,
        )
        for t, items in sorted(by_track.items())
    ]
    markers = [
        MarkerInfo(frame=f, color="Blue", name="", note="", duration=1, custom_data="")
        for f in (marker_frames or [])
    ]
    return DescribeResult(
        timeline=TimelineInfo(
            name=prepare.branch.name,
            unique_id=prepare.branch.unique_id,
            fps=24.0,
            drop_frame=False,
            start_frame=0,
            end_frame=500,
            duration=500,
            start_timecode="01:00:00:00",
            width=1920,
            height=1080,
        ),
        tracks=tracks,
        markers=markers,
        media=[],
        notes=[],
        text="",
    )


def as_dict(obj: Any) -> Any:
    return obj.model_dump(mode="json")
