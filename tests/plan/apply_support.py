"""Helpers for the three-phase ``aar.plan.apply`` tests: synthesize the prepare and assemble
result files (and the template / branch describes) the compiler reads, without touching Resolve.
"""

from __future__ import annotations

from typing import Any

from aar.node.ops.build import ImportedMedia, ImportResult, TrackResult, TrackState, path_key
from aar.node.ops.collab import BranchResult
from aar.node.ops.framework import MediaRef, TimelineIdentity
from aar.node.ops.read import (
    DescribeResult,
    ItemInfo,
    MarkerInfo,
    MediaFindResult,
    MediaInfo,
    TimelineInfo,
    TrackInfo,
)
from aar.node.ops.write import AppendedRange, AppendResult, PlacedItem
from aar.plan.apply import PrepareResult, compile_assemble, compile_prepare
from aar.plan.apply.inputs import AssembleResult, FoundTitle, ImportedSource, plan_hash
from aar.plan.model import EditPlan

BRANCH_UID = "tl-branch-0001"
TEMPLATE_UID = "tl-template-0001"


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


# -- node-job builders (round 4): a well-formed job_result for phase_result_from_job -----------


def media_info(name: str, media_id: str, *, bin_path=("pool",), file_path: str = "") -> MediaInfo:
    return MediaInfo(
        media_id=media_id,
        name=name,
        type="Multicam",
        format="",
        video_codec="",
        file_name="",
        file_path=file_path,
        fps=24.0,
        frames=1000,
        is_braw=False,
        is_still=False,
        is_multicam=True,
        has_video=True,
        has_audio=True,
        online_status="Online",
        proxy="None",
        input_color_space="",
        bin_path=bin_path,
    )


def find_value(
    items: list[MediaInfo], *, bin_path=(), recursive: bool = True, truncated: bool = False
) -> dict:
    """A ``media.find`` result over ``bin_path`` (default: the whole pool, recursive)."""
    return MediaFindResult(
        bin_path=bin_path,
        recursive=recursive,
        scanned=len(items),
        truncated=truncated,
        media=items,
    ).model_dump(mode="json")


def prepare_job(plan: EditPlan, template: DescribeResult, *, uid: str = BRANCH_UID) -> dict:
    """A well-formed phase-1 job result: the compiled prepare steps, each with a synthesized op
    result -- the branch of ``template``; per import step one imported item per requested file
    (pool id ``pool-<owner source id>``); per find exactly one exact-name hit (``pool-<source id>``
    for a pool source, ``title-<n>`` for the n-th title template)."""
    owners = {path_key(s.media): s for s in reversed(plan.sources)}  # first source wins
    titles = 0
    steps = []
    for step in compile_prepare(plan, template=template).steps:
        value: dict
        if step.op == "timeline.branch":
            value = BranchResult(
                source=TimelineIdentity(
                    name=template.timeline.name, unique_id=template.timeline.unique_id
                ),
                timeline=TimelineIdentity(name=_branch_name(plan), unique_id=uid),
                agent_id="a",
                version=1,
                created=True,
                timeline_count=1,
                notes=[],
            ).model_dump(mode="json")
        elif step.op == "media.import":
            media = [
                ImportedMedia(
                    path=path,
                    status="imported",
                    media=MediaRef(unique_id=f"pool-{owners[path_key(path)].id}"),
                    file_path=path,
                )
                for path in step.params["paths"]
            ]
            value = ImportResult(
                bin_path=tuple(step.params["bin_path"]), bins_created=[], media=media, notes=[]
            ).model_dump(mode="json")
        elif step.name.startswith("find:source:"):
            name = step.params["name"]
            sid = step.name.removeprefix("find:source:")
            value = find_value([media_info(name, f"pool-{sid}", bin_path=("Night 8", "CCN"))])
        else:  # find:title:<template>
            value = find_value([media_info(step.params["name"], f"title-{titles}")])
            titles += 1
        steps.append({"op": step.op, "name": step.name, "params": step.params, "value": value})
    return {"job_id": "job_prepare_test", "steps": steps}


def job_step(job: dict, name: str) -> dict:
    """The step named ``name`` of a job result (mutable, for hostile-job tests)."""
    return next(s for s in job["steps"] if s["name"] == name)


def assemble_job(plan: EditPlan, prepare: PrepareResult, *, batch_size: int = 10) -> dict:
    """A well-formed phase-2 job result: track.ensure plus the append steps, each append carrying
    an AppendResult for its ranges (no branch step; the branch is in the prepare envelope)."""
    steps = []
    for step in compile_assemble(plan, prepare=prepare, batch_size=batch_size).steps:
        if step.op == "track.ensure":
            states = [
                TrackState(
                    type=t["type"],
                    index=t["index"],
                    created=False,
                    name=t.get("name") or f"{t['type'][0].upper()}{t['index']}",
                    enabled=True,
                    locked=False,
                    audio_format=None,
                    changed=[],
                )
                for t in step.params["tracks"]
            ]
            value = TrackResult(timeline=prepare.branch, tracks=states, added=[])
            steps.append(
                {
                    "op": step.op,
                    "name": step.name,
                    "params": step.params,
                    "value": value.model_dump(mode="json"),
                }
            )
            continue
        ranges = []
        for i, rng in enumerate(step.params["ranges"]):
            track, rin = rng["track"], rng["record_in"]
            si, so = rng["src_in"], rng["src_out"]
            rout = rin + (so - si)
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
                    index=i,
                    status="appended",
                    planned_record_in=rin,
                    planned_record_out=rout,
                    planned_frames=so - si,
                    end_frame_used=rout,
                    video=video,
                    audio=audio,
                    notes=[],
                )
            )
        value = AppendResult(
            timeline=prepare.branch,
            timeline_fps=24.0,
            cursor=0,
            next_cursor=None,
            done=True,
            ranges=ranges,
            gaps=[],
            notes=[],
        ).model_dump(mode="json")
        steps.append({"op": step.op, "name": step.name, "params": step.params, "value": value})
    return {"job_id": "job_assemble_test", "steps": steps}
