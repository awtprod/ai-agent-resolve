"""Collaboration ops: agent branch timelines and review-note markers.

Three hot ops sit on top of the write-op library (DESIGN R3):

- ``timeline.branch`` duplicates a human's timeline to an agent branch
  ``<timeline> [agent:<id> v<n>]`` with the next free ``n``, built on
  ``timeline.duplicate``. It is idempotent: a re-run finds the branch it already made
  (``if_exists="reuse"``) instead of making another. Merging a branch back is a human
  step (Compare Timelines has no scripting API), so the result carries a handoff note.
- ``markers.mirror_review`` mirrors a hub review's timecoded notes to markers on the
  agent's branch. Each marker's ``customData`` carries the note id, so a re-run updates
  the marker in place (and prunes markers whose note is gone) rather than duplicating.
- ``markers.read_review`` maps the branch's markers back to notes: the ones it mirrored,
  and the ones a human added (for example Presentation comments), so the latter can flow
  back into the hub review as new notes.

Writing markers mutates a production project. The human-approval gate for that is S7's
(``aar.node.hublink`` with the hub's approval tokens); here these ops are classified
``mutates=True`` and exercised against the test fake only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from aar.node.branches import (
    branch_name,
    branch_version_for,
    note_marker_custom_data,
    parse_note_marker,
)
from aar.node.ops import raw
from aar.node.ops.framework import (
    Note,
    OpEnv,
    OpError,
    OpModel,
    TimelineIdentity,
    TimelineRef,
    op,
)
from aar.node.ops.timebase import converted_length
from aar.node.ops.write import DuplicateParams, MarkerColor, Resolution, timeline_duplicate

_HANDOFF = Note(
    code="merge_is_human",
    message="Merging the branch back into the human's timeline is a human step: use "
    "Compare Timelines (21.1). The scripting API cannot merge timelines.",
)


# -- timeline.branch -------------------------------------------------------------------------


class BranchParams(OpModel):
    source: TimelineRef = Field(description="The human timeline to branch from (needs unique_id).")
    agent_id: str = Field(min_length=1, description="The agent that owns the branch.")
    if_exists: Literal["reuse", "new"] = Field(
        default="reuse",
        description="'reuse' returns the agent's latest existing branch of the source "
        "(idempotent); 'new' always cuts the next version.",
    )
    resolution: Resolution | None = Field(
        default=None, description="Give the branch its own resolution (e.g. a vertical cut)."
    )

    @model_validator(mode="after")
    def _agent_id_clean(self) -> BranchParams:
        if "]" in self.agent_id or self.agent_id != self.agent_id.strip():
            raise ValueError("agent_id must not contain ']' or surrounding spaces")
        return self


class BranchResult(OpModel):
    source: TimelineIdentity
    timeline: TimelineIdentity
    agent_id: str
    version: int
    created: bool
    timeline_count: int
    notes: list[Note]


@op("timeline.branch", params=BranchParams, result=BranchResult, mutates=True)
def timeline_branch(env: OpEnv, params: BranchParams) -> BranchResult:
    """Branch a human timeline to an agent copy, cutting the next version or reusing one."""
    source = env.timeline(params.source)
    source_name = str(source.GetName())
    existing: dict[int, Any] = {}
    for timeline in env.timelines():
        version = branch_version_for(str(timeline.GetName()), source_name, params.agent_id)
        if version is not None:
            existing[version] = timeline

    if existing and params.if_exists == "reuse":
        version = max(existing)
        branch = existing[version]
        return BranchResult(
            source=env.identity(source),
            timeline=env.identity(branch),
            agent_id=params.agent_id,
            version=version,
            created=False,
            timeline_count=int(env.project().GetTimelineCount() or 0),
            notes=[
                Note(code="reused", message=f"branch v{version} already existed; unchanged"),
                _HANDOFF,
            ],
        )

    version = (max(existing) if existing else 0) + 1
    name = branch_name(source_name, params.agent_id, version)
    duplicated = timeline_duplicate(
        env, DuplicateParams(source=params.source, name=name, resolution=params.resolution)
    )
    return BranchResult(
        source=duplicated.source,
        timeline=duplicated.timeline,
        agent_id=params.agent_id,
        version=version,
        created=duplicated.created,
        timeline_count=duplicated.timeline_count,
        notes=[*duplicated.notes, _HANDOFF],
    )


# -- markers.mirror_review -------------------------------------------------------------------


class MirrorNote(OpModel):
    note_id: str = Field(min_length=1, description="The hub review note id.")
    frame: int = Field(strict=True, ge=0, description="0-based frame in the review render.")
    timecode: str = Field(default="", description="Display timecode label for the marker.")
    text: str = Field(default="", description="The note text, shown in the marker.")


class MirrorParams(OpModel):
    timeline: TimelineRef = Field(description="The agent branch (needs unique_id).")
    review_id: str = Field(min_length=1)
    notes: list[MirrorNote] = Field(max_length=1000)
    render_fps: float | None = Field(
        default=None,
        strict=True,
        gt=0,
        description="The render's frame rate, if it differs from the timeline's; note frames "
        "are converted to timeline frames.",
    )
    color: MarkerColor = "Cyan"
    prune: bool = Field(
        default=True,
        strict=True,
        description="Remove markers of this review whose note is no longer present.",
    )

    @model_validator(mode="after")
    def _check(self) -> MirrorParams:
        if ":" in self.review_id:
            raise ValueError("review_id must not contain ':'")
        ids = [note.note_id for note in self.notes]
        if len(ids) != len(set(ids)):
            raise ValueError("note_id values must be distinct")
        return self


class MirroredMarker(OpModel):
    note_id: str
    frame: int
    status: Literal["added", "present", "updated"]


class PrunedMarker(OpModel):
    note_id: str
    frame: int


class MirrorResult(OpModel):
    timeline: TimelineIdentity
    review_id: str
    markers: list[MirroredMarker]
    pruned: list[PrunedMarker]


def _timeline_fps(timeline: Any) -> float:
    fps = (timeline.GetSettings() or {}).get("timelineFrameRate")
    if fps is None:
        raise OpError(
            "timeline_fps_unknown",
            "timelineFrameRate is not set on the branch timeline",
            "check the timeline's settings",
        )
    return float(fps)


def _delete_by_custom_data(timeline: Any, custom_data: str, frame: int) -> None:
    if not timeline.DeleteMarkerByCustomData(custom_data):
        raise OpError(
            "marker_delete_failed",
            f"DeleteMarkerByCustomData({custom_data!r}) returned False",
            f"the marker was expected at frame {frame}; inspect markers with timeline.describe",
        )
    if frame in timeline.GetMarkers():
        read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(frame)
        if read is not None and read["custom_data"] == custom_data:
            raise OpError(
                "marker_delete_failed",
                f"marker {custom_data!r} is still at frame {frame} after deletion",
                "delete it by hand and re-run",
            )


@op("markers.mirror_review", params=MirrorParams, result=MirrorResult, mutates=True)
def markers_mirror_review(env: OpEnv, params: MirrorParams) -> MirrorResult:
    """Mirror a review's notes to branch markers, keyed by note id so a re-run updates."""
    timeline = env.timeline(params.timeline)
    tl_fps = _timeline_fps(timeline)

    desired: dict[str, tuple[int, str, str, str]] = {}
    for note in params.notes:
        frame = note.frame
        if params.render_fps is not None and params.render_fps != tl_fps:
            frame = converted_length(note.frame, params.render_fps, tl_fps)
        desired[note.note_id] = (
            frame,
            params.color,
            note.timecode,
            note.text,
        )

    existing = raw.read_markers(timeline)
    ours: dict[str, dict[str, Any]] = {}
    other_frames: set[int] = set()
    for marker in existing:
        parsed = parse_note_marker(marker["custom_data"])
        if parsed is not None and parsed[0] == params.review_id:
            ours[parsed[1]] = marker
        else:
            other_frames.add(marker["frame"])

    markers: list[MirroredMarker] = []
    pruned: list[PrunedMarker] = []
    occupied = set(other_frames)
    to_add: list[tuple[str, tuple[int, str, str, str]]] = []

    # Phase 1: keep unchanged markers, delete changed ones, prune notes that are gone.
    for note_id, want in desired.items():
        have = ours.get(note_id)
        custom_data = note_marker_custom_data(params.review_id, note_id)
        if (
            have is not None
            and (
                have["frame"],
                have["color"],
                have["name"],
                have["note"],
            )
            == want
        ):
            markers.append(MirroredMarker(note_id=note_id, frame=want[0], status="present"))
            occupied.add(have["frame"])
        else:
            if have is not None:
                _delete_by_custom_data(timeline, custom_data, have["frame"])
            to_add.append((note_id, want))
    for note_id, marker in ours.items():
        if note_id not in desired and params.prune:
            _delete_by_custom_data(
                timeline, note_marker_custom_data(params.review_id, note_id), marker["frame"]
            )
            pruned.append(PrunedMarker(note_id=note_id, frame=marker["frame"]))

    # Phase 2: add the new and the changed markers at their frames.
    for note_id, (frame, color, timecode, text) in to_add:
        status = "updated" if note_id in ours else "added"
        if frame in occupied:
            raise OpError(
                "marker_conflict",
                f"frame {frame} already has another marker; the note {note_id!r} cannot be "
                f"mirrored there",
                "move the note or remove the conflicting marker",
                note_id=note_id,
                frame=frame,
            )
        custom_data = note_marker_custom_data(params.review_id, note_id)
        if not timeline.AddMarker(frame, color, timecode, text, 1, custom_data):
            raise OpError(
                "add_marker_failed",
                f"AddMarker at frame {frame} for note {note_id!r} returned False",
                "the frame must lie inside the timeline and be free",
                note_id=note_id,
                frame=frame,
            )
        read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(frame)
        if read is None or read["custom_data"] != custom_data or read["note"] != text:
            raise OpError(
                "marker_unverified",
                f"the marker for note {note_id!r} reads back as {read}",
                "inspect markers with timeline.describe",
                note_id=note_id,
            )
        occupied.add(frame)
        markers.append(MirroredMarker(note_id=note_id, frame=frame, status=status))

    markers.sort(key=lambda m: m.frame)
    pruned.sort(key=lambda p: p.frame)
    return MirrorResult(
        timeline=env.identity(timeline),
        review_id=params.review_id,
        markers=markers,
        pruned=pruned,
    )


# -- markers.read_review ---------------------------------------------------------------------


class ReadReviewParams(OpModel):
    timeline: TimelineRef = Field(description="The agent branch to read markers from.")
    review_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ReadReviewParams:
        if ":" in self.review_id:
            raise ValueError("review_id must not contain ':'")
        return self


class MirroredRef(OpModel):
    note_id: str
    frame: int
    timecode: str
    text: str


class HumanMarker(OpModel):
    frame: int
    color: str
    text: str


class ReadReviewResult(OpModel):
    timeline: TimelineIdentity
    review_id: str
    mirrored: list[MirroredRef]
    human: list[HumanMarker]
    other: int
    """Count of markers this read ignored: notes of another review, or malformed note data."""


@op(
    "markers.read_review",
    params=ReadReviewParams,
    result=ReadReviewResult,
    mutates=False,
)
def markers_read_review(env: OpEnv, params: ReadReviewParams) -> ReadReviewResult:
    """Read a branch's markers: this review's mirrored notes, and a human's own markers."""
    timeline = env.timeline(params.timeline)
    mirrored: list[MirroredRef] = []
    human: list[HumanMarker] = []
    other = 0
    for marker in raw.read_markers(timeline):
        custom_data = marker["custom_data"]
        parsed = parse_note_marker(custom_data)
        if parsed is not None:
            if parsed[0] == params.review_id:
                mirrored.append(
                    MirroredRef(
                        note_id=parsed[1],
                        frame=marker["frame"],
                        timecode=marker["name"],
                        text=marker["note"],
                    )
                )
            else:
                other += 1  # a marker of another review
        elif custom_data.startswith("aar:note:"):
            other += 1  # our prefix but malformed: ignored, never fatal
        else:
            human.append(
                HumanMarker(
                    frame=marker["frame"],
                    color=marker["color"],
                    text=marker["note"] or marker["name"],
                )
            )
    return ReadReviewResult(
        timeline=env.identity(timeline),
        review_id=params.review_id,
        mirrored=mirrored,
        human=human,
        other=other,
    )
