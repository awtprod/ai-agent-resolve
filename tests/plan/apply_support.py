"""Helpers for the ``aar.plan.apply`` tests: synthesize the phase-A results and the template /
branch describes that the compiler takes as inputs, without touching Resolve.

``synth_phase_a`` mirrors what a real phase-A run returns: one append result whose ranges carry
a video item and an audio item for each placement, with stable ``item_id``s derived from the
placement, so phase B can resolve them by ``(track, record_in)``.
"""

from __future__ import annotations

from aar.node.ops.framework import TimelineIdentity
from aar.node.ops.read import (
    DescribeResult,
    ItemInfo,
    MarkerInfo,
    TimelineInfo,
    TrackInfo,
)
from aar.node.ops.write import AppendedRange, AppendResult, PlacedItem
from aar.plan.apply.compile import BuildCompilation

BRANCH_NAME = "CCN Longform Template [agent:ccn-bot v1]"
BRANCH_UID = "tl-branch-0001"


def _placed(track: int, record_in: int, record_out: int, item_id: str, ttype: str) -> PlacedItem:
    return PlacedItem(
        item_id=item_id,
        track_type=ttype,  # type: ignore[arg-type]
        track=track,
        start=record_in,
        end=record_out,
        duration=record_out - record_in,
        record_in=record_in,
        record_out=record_out,
        source_start=0,
        source_end=record_out - record_in,
    )


def synth_phase_a(
    build: BuildCompilation, *, name: str = BRANCH_NAME, unique_id: str = BRANCH_UID
) -> list[AppendResult]:
    """One append result that places a video and an audio item for every range phase A appended
    (content segments *and* the outro), mirroring a real run."""
    ranges: list[AppendedRange] = []
    index = 0
    for step in build.steps:
        if step.op != "timeline.append_ranges":
            continue
        for rng in step.params["ranges"]:
            track = rng["track"]
            record_in = rng["record_in"]
            frames = rng["src_out"] - rng["src_in"]
            record_out = record_in + frames
            key = f"{track}:{record_in}"
            ranges.append(
                AppendedRange(
                    index=index,
                    status="appended",
                    planned_record_in=record_in,
                    planned_record_out=record_out,
                    planned_frames=frames,
                    end_frame_used=record_out,
                    video=_placed(track, record_in, record_out, f"V{key}", "video"),
                    audio=[_placed(track, record_in, record_out, f"A{key}", "audio")],
                    notes=[],
                )
            )
            index += 1
    return [
        AppendResult(
            timeline=TimelineIdentity(name=name, unique_id=unique_id),
            timeline_fps=24.0,
            cursor=0,
            next_cursor=None,
            done=True,
            ranges=ranges,
            gaps=[],
            notes=[],
        )
    ]


def template_describe(
    item_count: int, *, name: str = "CCN Longform Template", unique_id: str = "tl-template-0001"
) -> DescribeResult:
    """A describe of the template with ``item_count`` items spread on one video track."""
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
    present_record_ins: list[int],
    *,
    marker_frames: list[int] | None = None,
) -> DescribeResult:
    """A describe of an agent branch that already holds items at ``present_record_ins`` on video
    track 1, for the idempotency diff."""
    items = [
        ItemInfo(
            index=i,
            item_id=f"b{i}",
            name=f"item {i}",
            type="Video",
            start=r,
            end=r + 1,
            duration=1,
            record_in=r,
            record_out=r + 1,
            source_start=0,
            source_end=1,
            media_id=None,
            fade_in=None,
            fade_out=None,
            audio_volume_db=None,
            enabled=True,
        )
        for i, r in enumerate(present_record_ins)
    ]
    markers = [
        MarkerInfo(frame=f, color="Blue", name="", note="", duration=1, custom_data="")
        for f in (marker_frames or [])
    ]
    return DescribeResult(
        timeline=TimelineInfo(
            name=BRANCH_NAME,
            unique_id=BRANCH_UID,
            fps=24.0,
            drop_frame=False,
            start_frame=0,
            end_frame=500,
            duration=500,
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
                item_count=len(items),
                transitions=0,
                truncated=False,
                items=items,
                gaps=[],
                overlaps=0,
            )
        ],
        markers=markers,
        media=[],
        notes=[],
        text="",
    )
