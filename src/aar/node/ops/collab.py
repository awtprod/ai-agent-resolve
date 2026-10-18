"""Collaboration ops: agent branch timelines and review-note markers.

Three hot ops sit on top of the write-op library (DESIGN R3):

- ``timeline.branch`` duplicates a human's timeline to an agent branch
  ``<timeline> [agent:<id> v<n>]`` with the next free ``n``, built on
  ``timeline.duplicate``. The owner is the lease holder that submitted it, never a value
  the caller passes, and the branch is recorded in the node's owner registry by unique id
  so the branch-edit guard can check ownership. It is idempotent: a re-run finds the branch
  it already made. Merging a branch back is a human step (Compare Timelines has no scripting
  API), so the result carries a handoff note.
- ``markers.mirror_review`` mirrors a hub review's timecoded notes to markers on the
  agent's branch. Each marker's ``customData`` carries the note id, so a re-run updates in
  place and prunes markers whose note is gone, never duplicating. It pre-flights the whole
  plan (every add, delete and conflict, including human markers and duplicate custom data)
  before any mutation, adds a moved marker before deleting the old one, and reports exactly
  what was applied if a step fails.
- ``markers.read_review`` maps the branch's markers back to notes: the ones it mirrored,
  and the ones a human added (for example Presentation comments), so the latter flow back
  into the hub review as new notes.

Note bounds (text length, frame, fps, timecode) are the hub review's own, imported from
:mod:`aar.hub.reviews`. Writing markers mutates a production project; the human-approval
gate for that is S7's. Here these ops are ``mutates=True`` and tested against the fake only.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from aar.hub.common import clean_text
from aar.hub.reviews import _TIMECODE_RE as TIMECODE_RE
from aar.hub.reviews import MAX_FPS, MAX_FRAME, MAX_NOTE
from aar.node.branches import (
    branch_name,
    branch_version_for,
    note_marker_custom_data,
    parse_note_marker,
    valid_agent_id,
    valid_note_id,
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

MAX_NOTES = 1000
MAX_TOTAL_TEXT = 1_000_000  # total marker text bytes per mirror call

_HANDOFF = Note(
    code="merge_is_human",
    message="Merging the branch back into the human's timeline is a human step: use "
    "Compare Timelines (21.1). The scripting API cannot merge timelines.",
)


# -- timeline.branch -------------------------------------------------------------------------


class BranchParams(OpModel):
    source: TimelineRef = Field(description="The human timeline to branch from (needs unique_id).")
    if_exists: Literal["reuse", "new"] = Field(
        default="reuse",
        description="'reuse' returns the agent's latest existing branch of the source "
        "(idempotent); 'new' always cuts the next version.",
    )
    resolution: Resolution | None = Field(
        default=None, description="Give the branch its own resolution (e.g. a vertical cut)."
    )


class BranchResult(OpModel):
    source: TimelineIdentity
    timeline: TimelineIdentity
    agent_id: str
    version: int
    created: bool
    timeline_count: int
    notes: list[Note]


@op(
    "timeline.branch", params=BranchParams, result=BranchResult, mutates=True, creates_timeline=True
)
def timeline_branch(env: OpEnv, params: BranchParams) -> BranchResult:
    """Branch a human timeline to an agent copy owned by the submitting agent."""
    holder = env.holder
    if holder is None or holder.is_human:
        raise OpError(
            "no_agent_holder",
            "timeline.branch must run for an agent holder; the branch is owned by it",
            "submit the op as the agent that will own the branch",
        )
    agent_id = holder.id
    if not valid_agent_id(agent_id):
        raise OpError(
            "invalid_agent_id",
            f"the holder id {agent_id!r} is not a usable agent branch owner",
            "use an agent id of letters, digits, '.', '_' or '-', at most 128 characters",
        )

    source = env.timeline(params.source)
    source_name = str(source.GetName())
    source_id = str(source.GetUniqueId())
    existing: dict[int, Any] = {}
    for timeline in env.timelines():
        version = branch_version_for(str(timeline.GetName()), source_name, agent_id)
        if version is not None:
            existing[version] = timeline

    if existing and params.if_exists == "reuse":
        version = max(existing)
        branch = existing[version]
        self_notes = [
            Note(code="reused", message=f"branch v{version} already existed; unchanged"),
            _HANDOFF,
        ]
        self_created = False
        branch_identity = env.identity(branch)
        source_identity = env.identity(source)
        count = int(env.project().GetTimelineCount() or 0)
    else:
        version = (max(existing) if existing else 0) + 1
        name = branch_name(source_name, agent_id, version)
        duplicated = timeline_duplicate(
            env, DuplicateParams(source=params.source, name=name, resolution=params.resolution)
        )
        self_notes = [*duplicated.notes, _HANDOFF]
        self_created = duplicated.created
        branch_identity = duplicated.timeline
        source_identity = duplicated.source
        count = duplicated.timeline_count

    if env.branch_registry is not None:
        env.branch_registry.register(
            branch_identity.unique_id,
            source_timeline_id=source_id,
            agent_id=agent_id,
            version=version,
        )
    return BranchResult(
        source=source_identity,
        timeline=branch_identity,
        agent_id=agent_id,
        version=version,
        created=self_created,
        timeline_count=count,
        notes=self_notes,
    )


# -- markers.mirror_review -------------------------------------------------------------------


def _clean_note_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("text must be a string")
    return clean_text(value, "text", max_len=MAX_NOTE, required=False, multiline=True)


class MirrorNote(OpModel):
    note_id: str = Field(min_length=1, description="The hub review note id.")
    frame: int = Field(strict=True, ge=0, le=MAX_FRAME, description="0-based render frame.")
    timecode: str = Field(default="", description="Display timecode label for the marker.")
    text: str = Field(default="", description="The note text, shown in the marker.")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)
            if "text" in data:
                data["text"] = _clean_note_text(data["text"])
            timecode = data.get("timecode")
            if isinstance(timecode, str) and timecode and not TIMECODE_RE.fullmatch(timecode):
                raise ValueError(f"timecode must be HH:MM:SS:FF, got {timecode!r}")
        return data

    @model_validator(mode="after")
    def _ids(self) -> MirrorNote:
        if not valid_note_id(self.note_id):
            raise ValueError(f"note_id must be an id of letters/digits/._-: {self.note_id!r}")
        return self


class MirrorParams(OpModel):
    timeline: TimelineRef = Field(description="The agent branch (needs unique_id).")
    review_id: str = Field(min_length=1)
    notes: list[MirrorNote] = Field(max_length=MAX_NOTES)
    render_fps: float | None = Field(
        default=None,
        strict=True,
        gt=0,
        le=MAX_FPS,
        description="The render's frame rate, if it differs from the timeline's.",
    )
    color: MarkerColor = "Cyan"
    prune: bool = Field(default=True, strict=True)

    @model_validator(mode="after")
    def _check(self) -> MirrorParams:
        if not valid_note_id(self.review_id):
            raise ValueError(f"review_id must be an id of letters/digits/._-: {self.review_id!r}")
        ids = [note.note_id for note in self.notes]
        if len(ids) != len(set(ids)):
            raise ValueError("note_id values must be distinct")
        total = sum(len(note.text) for note in self.notes)
        if total > MAX_TOTAL_TEXT:
            raise ValueError(f"total marker text is {total} bytes, over the {MAX_TOTAL_TEXT} cap")
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


@op("markers.mirror_review", params=MirrorParams, result=MirrorResult, mutates=True)
def markers_mirror_review(env: OpEnv, params: MirrorParams) -> MirrorResult:
    """Mirror a review's notes to branch markers, keyed by note id so a re-run updates."""
    timeline = env.timeline(params.timeline)
    tl_fps = _timeline_fps(timeline)
    start = int(timeline.GetStartFrame())
    duration = int(timeline.GetEndFrame()) - start
    rid = params.review_id

    # Read the branch's markers once: which are ours (by note id), which belong to anyone else.
    ours: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    other_frames: dict[int, str] = {}
    for marker in raw.read_markers(timeline):
        parsed = parse_note_marker(marker["custom_data"])
        if parsed is not None and parsed[0] == rid:
            if parsed[1] in ours:
                duplicates.append(parsed[1])  # two markers claim the same note: never guess
            ours[parsed[1]] = marker
        else:
            other_frames[marker["frame"]] = marker["custom_data"]
    if duplicates:
        raise OpError(
            "duplicate_marker_custom_data",
            f"review {rid!r} has more than one marker for note(s) {sorted(set(duplicates))}",
            "remove the duplicate markers by hand, then re-run",
            note_ids=sorted(set(duplicates)),
        )

    # Resolve each note's timeline frame, in range. Nothing is mutated here.
    desired: dict[str, tuple[int, str, str, str]] = {}
    for note in params.notes:
        frame = note.frame
        if params.render_fps is not None and params.render_fps != tl_fps:
            frame = converted_length(note.frame, params.render_fps, tl_fps)
        if frame < 0 or frame > duration:
            raise OpError(
                "note_out_of_range",
                f"note {note.note_id!r} maps to frame {frame}, outside the branch [0, {duration}]",
                "re-anchor the note inside the review render, or render the full timeline",
                note_id=note.note_id,
                frame=frame,
            )
        desired[note.note_id] = (frame, params.color, note.timecode, note.text)

    # Pre-flight conflicts: the final markers must be one per frame. A human or other-review
    # marker is never moved, so a note landing on its frame is refused before any mutation.
    final: dict[int, str] = dict.fromkeys(other_frames, "a non-review marker")
    for note_id, (frame, *_rest) in desired.items():
        if frame in final:
            raise OpError(
                "marker_conflict",
                f"frame {frame} already holds {final[frame]}; note {note_id!r} cannot go there",
                "move the note to a free frame, or remove the conflicting marker",
                note_id=note_id,
                frame=frame,
            )
        final[frame] = f"note {note_id!r}"

    markers: list[MirroredMarker] = []
    pruned: list[PrunedMarker] = []
    try:
        for note_id, (frame, color, timecode, text) in desired.items():
            have = ours.get(note_id)
            custom_data = note_marker_custom_data(rid, note_id)
            if have is not None and (
                have["frame"],
                have["color"],
                have["name"],
                have["note"],
            ) == (frame, color, timecode, text):
                markers.append(MirroredMarker(note_id=note_id, frame=frame, status="present"))
                continue
            if have is None:
                _add(timeline, frame, color, timecode, text, custom_data, note_id)
            elif have["frame"] != frame:  # a move: add the new marker before deleting the old
                _add(timeline, frame, color, timecode, text, custom_data, note_id)
                _delete(timeline, custom_data, have["frame"])
            else:  # an in-place change: the frame is shared, so delete then add
                _delete(timeline, custom_data, have["frame"])
                _add(timeline, frame, color, timecode, text, custom_data, note_id)
            markers.append(
                MirroredMarker(note_id=note_id, frame=frame, status="updated" if have else "added")
            )
        for note_id, marker in ours.items():
            if note_id not in desired and params.prune:
                _delete(timeline, note_marker_custom_data(rid, note_id), marker["frame"])
                pruned.append(PrunedMarker(note_id=note_id, frame=marker["frame"]))
    except OpError as exc:
        raise _with_applied(exc, markers, pruned) from exc

    markers.sort(key=lambda m: m.frame)
    pruned.sort(key=lambda p: p.frame)
    return MirrorResult(
        timeline=env.identity(timeline), review_id=rid, markers=markers, pruned=pruned
    )


def _add(
    timeline: Any, frame: int, color: str, name: str, note: str, custom_data: str, note_id: str
) -> None:
    if not timeline.AddMarker(frame, color, name, note, 1, custom_data):
        raise OpError(
            "add_marker_failed",
            f"AddMarker at frame {frame} for note {note_id!r} returned False",
            "the frame must lie inside the timeline and be free",
            note_id=note_id,
            frame=frame,
        )
    read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(frame)
    if read is None or read["custom_data"] != custom_data or read["note"] != note:
        raise OpError(
            "marker_unverified",
            f"the marker for note {note_id!r} reads back as {read}",
            "inspect markers with timeline.describe",
            note_id=note_id,
        )


def _delete(timeline: Any, custom_data: str, frame: int) -> None:
    if not timeline.DeleteMarkerByCustomData(custom_data):
        raise OpError(
            "marker_delete_failed",
            f"DeleteMarkerByCustomData({custom_data!r}) returned False (expected at frame {frame})",
            "remove the marker by hand, then re-run",
        )
    read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(frame)
    if read is not None and read["custom_data"] == custom_data:
        raise OpError(
            "marker_delete_failed",
            f"marker {custom_data!r} is still at frame {frame} after deletion",
            "remove the marker by hand, then re-run",
        )


def _with_applied(
    exc: OpError, markers: list[MirroredMarker], pruned: list[PrunedMarker]
) -> OpError:
    if "applied" in exc.detail:
        return exc
    extra = {k: v for k, v in exc.detail.items() if k not in ("fix_hint", "trap_id")}
    return OpError(
        exc.code,
        exc.message,
        exc.fix_hint,
        exc.trap_id,
        **extra,
        applied={
            "markers": [m.model_dump() for m in markers],
            "pruned": [p.model_dump() for p in pruned],
        },
    )


# -- markers.read_review ---------------------------------------------------------------------


class ReadReviewParams(OpModel):
    timeline: TimelineRef = Field(description="The agent branch to read markers from.")
    review_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> ReadReviewParams:
        if not valid_note_id(self.review_id):
            raise ValueError(f"review_id must be an id of letters/digits/._-: {self.review_id!r}")
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
    """Markers this read ignored: notes of another review, or malformed note custom data."""


@op("markers.read_review", params=ReadReviewParams, result=ReadReviewResult, mutates=False)
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
