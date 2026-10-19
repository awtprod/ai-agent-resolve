"""Collaboration ops: agent branch timelines and review-note markers.

Three hot ops sit on top of the write-op library (DESIGN R3):

- ``timeline.branch`` duplicates a human's timeline to an agent branch
  ``<timeline> [agent:<id> v<n>]``, built on ``timeline.duplicate``. The owner is the lease
  holder that submitted it, never a value the caller passes, and the branch is recorded in
  the node's owner registry keyed by ``(project, timeline unique id)``. It is idempotent: a
  re-run reuses a branch only when the registry confirms this agent owns it, branched from
  this very source (by id) and at the resolution asked for (no resolution means the
  source's, as for ``timeline.duplicate``); a mere name match (a human timeline, or a copy
  with a fresh id) is never reused, it just takes a version number. Refusals that need no
  Resolve call (no agent holder, an unusable agent id, no registry) are made at submit. If
  the copy is made but a later step fails (its resolution will not set), the copy is
  deleted again, so the op leaves no unowned half-made branch behind and a retry cuts the
  same version. Merging a branch back is a human step, so the result carries a handoff
  note.
- ``markers.mirror_review`` mirrors a hub review's timecoded notes to markers on the agent's
  branch. Each marker's ``customData`` carries the note id, so a re-run updates in place and
  prunes markers whose note is gone, never duplicating. It pre-flights the whole plan (every
  add, delete and conflict, including human and other-review markers and duplicate custom
  data, and with ``prune`` off this review's markers that stay) before any mutation. A
  moved marker is added before the old one is deleted (by frame, after checking it is
  ours); where frames collide (a swap, an in-place change, a note onto a pruned note's
  frame) the delete goes first. If a step fails, the error reports exactly the marker
  writes that were applied.
- ``markers.read_review`` maps the branch's markers back to notes: the ones it mirrored, and
  the ones a human added, so the latter flow back into the hub review as new notes.

Note bounds (text length, frame, fps, timecode) are the hub review's own, imported from
:mod:`aar.hub.reviews`. Writing markers mutates a production project; the human-approval
gate for that is S7's. Here these ops are ``mutates=True`` and tested against the fake only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, model_validator

from aar.hub.common import clean_text
from aar.hub.reviews import MAX_FPS, MAX_FRAME, MAX_NOTE, is_timecode
from aar.node.branch_registry import BranchRegistry, OwnershipConflict
from aar.node.branches import (
    branch_name,
    branch_version_for,
    note_marker_custom_data,
    parse_note_marker,
    valid_agent_id,
    valid_note_id,
)
from aar.node.leases import Holder
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
        description="'reuse' returns the latest branch of this source that the node's "
        "registry says this agent owns, at the resolution asked for (idempotent), else cuts "
        "the next version; 'new' always cuts the next version.",
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


def _resolution_of(timeline: Any) -> tuple[int | None, int | None]:
    settings = timeline.GetSettings() or {}
    width = settings.get("timelineResolutionWidth")
    height = settings.get("timelineResolutionHeight")
    return (
        int(width) if width not in (None, "") else None,
        int(height) if height not in (None, "") else None,
    )


def _branch_refusal(
    params: BranchParams, holder: Holder | None, registry: BranchRegistry | None
) -> OpError | None:
    """Why ``timeline.branch`` cannot run for ``holder``, decided with no Resolve call.

    ``make_hot_op`` runs this at submit (the op's ``submit_check``), so a refused branch is
    never queued and never makes the broker load or save a project; the body runs it again.
    """
    if holder is None or holder.is_human:
        return OpError(
            "no_agent_holder",
            "timeline.branch must run for an agent holder; the branch is owned by it",
            "submit the op as the agent that will own the branch",
        )
    if not valid_agent_id(holder.id):
        return OpError(
            "invalid_agent_id",
            f"the holder id {holder.id!r} is not a usable agent branch owner",
            "use an agent id of letters, digits, '.', '_' or '-', at most 128 characters",
        )
    if registry is None:
        return OpError(
            "no_branch_registry",
            "timeline.branch needs the node's owner registry to record ownership",
            "wire the broker with a BranchRegistry (the node runtime provides one)",
        )
    return None


@op(
    "timeline.branch",
    params=BranchParams,
    result=BranchResult,
    mutates=True,
    creates_timeline=True,
    submit_check=_branch_refusal,
)
def timeline_branch(env: OpEnv, params: BranchParams) -> BranchResult:
    """Branch a human timeline to an agent copy owned by the submitting agent."""
    refusal = _branch_refusal(params, env.holder, env.branch_registry)
    if refusal is not None:
        raise refusal
    assert env.holder is not None and env.branch_registry is not None
    agent_id = env.holder.id
    registry = env.branch_registry
    project_id = env.target.project_id
    if project_id is None:  # make_hot_op refuses this at submit; never key ownership on None
        raise OpError(
            "project_id_required",
            "the target has no project_id, so branch ownership cannot be keyed",
            "take the id from project.find or the hub registry",
            "pm-folder-scoped-load",
        )

    source = env.timeline(params.source)
    source_name = str(source.GetName())
    source_id = str(source.GetUniqueId())
    # No resolution asked for means the source's, the rule timeline.duplicate(reuse) uses:
    # a vertical copy is never handed back for a plain branch.
    wanted = (
        _resolution_of(source)
        if params.resolution is None
        else (params.resolution.width, params.resolution.height)
    )
    # Names only number the versions. What may be reused is decided by the registry, by
    # unique id: a branch this agent owns, of this very source (by id), at the resolution
    # asked for. A human timeline or a copy that carries a branch name is never reused.
    taken: set[int] = set()
    reusable: dict[int, Any] = {}
    existing: set[str] = set()
    for timeline in env.timelines():
        uid = str(timeline.GetUniqueId())
        existing.add(uid)
        version = branch_version_for(str(timeline.GetName()), source_name, agent_id)
        if version is not None:
            taken.add(version)
        owner = registry.owner(project_id, uid)
        if owner is None or owner.agent_id != agent_id or owner.source_timeline_id != source_id:
            continue
        taken.add(owner.version)  # a renamed branch still holds its version
        if _resolution_of(timeline) == wanted:
            reusable[owner.version] = timeline

    if reusable and params.if_exists == "reuse":
        version = max(reusable)
        branch = reusable[version]
        return BranchResult(
            source=env.identity(source),
            timeline=env.identity(branch),
            agent_id=agent_id,
            version=version,
            created=False,
            timeline_count=int(env.project().GetTimelineCount() or 0),
            notes=[
                Note(code="reused", message=f"branch v{version} already existed; unchanged"),
                _HANDOFF,
            ],
        )

    version = max(taken, default=0) + 1
    name = branch_name(source_name, agent_id, version)
    try:
        duplicated = timeline_duplicate(
            env,
            DuplicateParams(
                source=params.source,
                name=name,
                resolution=params.resolution,
                if_exists="error",  # never pick up a same-named timeline as the new branch
            ),
        )
    except OpError as exc:
        raise _undo_half_made_branch(env, existing, name, exc) from exc
    try:
        registry.register(
            project_id,
            duplicated.timeline.unique_id,
            source_timeline_id=source_id,
            agent_id=agent_id,
            version=version,
        )
    except OwnershipConflict as exc:
        raise OpError(
            "branch_owner_conflict",
            f"the new timeline {duplicated.timeline.unique_id!r} is already registered: {exc}",
            "list timelines with timeline.list and branch again",
        ) from exc
    return BranchResult(
        source=duplicated.source,
        timeline=duplicated.timeline,
        agent_id=agent_id,
        version=version,
        created=duplicated.created,
        timeline_count=duplicated.timeline_count,
        notes=[*duplicated.notes, _HANDOFF],
    )


def _undo_half_made_branch(env: OpEnv, existing: set[str], name: str, exc: OpError) -> OpError:
    """Delete the copy a failed ``timeline.branch`` made, and say so on its error.

    The duplicate can be made and a later step still fail (its resolution will not set, or
    does not read back). That copy belongs to no one: it is not registered, so the guard
    lets no agent edit it, and a retry would cut the next version beside it, every time.
    Registering it instead would not help: it is not at the resolution asked for, so a
    retry could not reuse it either. So the op undoes its own half-made work and leaves the
    project as it found it; a retry then cuts the same version again.

    Only a timeline this very run made is touched: one whose unique id was not in the
    project before the duplicate and that carries the exact branch name asked for. If the
    delete fails, the copy is reported (``left_behind``) for a human to remove.
    """
    made = [
        timeline
        for timeline in env.timelines()
        if str(timeline.GetUniqueId()) not in existing and str(timeline.GetName()) == name
    ]
    if len(made) != 1:
        return exc  # nothing was made (or not unambiguously by us): nothing to undo
    copy = {"name": name, "unique_id": str(made[0].GetUniqueId())}
    try:
        env.media_pool().DeleteTimelines(made)
        gone = all(str(t.GetUniqueId()) != copy["unique_id"] for t in env.timelines())
    except Exception:  # a failed clean-up must never hide the original error
        gone = False
    extra = {k: v for k, v in exc.detail.items() if k not in ("fix_hint", "trap_id")}
    if gone:
        extra["removed"] = copy
    else:
        extra["left_behind"] = copy
    return OpError(exc.code, exc.message, exc.fix_hint, exc.trap_id, **extra)


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
            if isinstance(timecode, str) and timecode and not is_timecode(timecode):
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


@dataclass
class _Plan:
    """One note to (re)write: its target marker, and the marker it replaces, if any."""

    note_id: str
    frame: int
    color: str
    timecode: str
    text: str
    custom_data: str
    old_frame: int | None  # where this note's marker is now (a move, or in place), else None

    @property
    def status(self) -> Literal["added", "updated"]:
        return "added" if self.old_frame is None else "updated"


@dataclass
class _Applied:
    """Exactly the marker writes a mirror run made, in order, as measured from Resolve."""

    added: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"added": list(self.added), "deleted": list(self.deleted)}


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

    # Resolve each note's timeline frame, in range [0, duration). Nothing is mutated here.
    desired: dict[str, tuple[int, str, str, str]] = {}
    for note in params.notes:
        frame = note.frame
        if params.render_fps is not None and params.render_fps != tl_fps:
            frame = converted_length(note.frame, params.render_fps, tl_fps)
        if frame < 0 or frame >= duration:
            raise OpError(
                "note_out_of_range",
                f"note {note.note_id!r} maps to frame {frame}, outside the branch [0, {duration})",
                "re-anchor the note inside the review render, or render the full timeline",
                note_id=note.note_id,
                frame=frame,
            )
        desired[note.note_id] = (frame, params.color, note.timecode, note.text)

    # Pre-flight the final state, before any mutation: one marker per frame. A note may not
    # land on a human or other-review marker, on another note, nor (with prune off) on a
    # marker of this review that stays. The review's own markers that move or are pruned
    # free their frames, so they never conflict; the apply order below handles them.
    kept = {
        marker["frame"]: f"this review's kept marker for note {note_id!r}"
        for note_id, marker in ours.items()
        if note_id not in desired and not params.prune
    }
    final: dict[int, str] = {**dict.fromkeys(other_frames, "a non-review marker"), **kept}
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
    plans: list[_Plan] = []
    for note_id, (frame, color, timecode, text) in desired.items():
        have = ours.get(note_id)
        fields = (frame, color, timecode, text)
        if (
            have is not None
            and (have["frame"], have["color"], have["name"], have["note"]) == fields
        ):
            markers.append(MirroredMarker(note_id=note_id, frame=frame, status="present"))
            continue
        custom_data = note_marker_custom_data(rid, note_id)
        old_frame = None if have is None else int(have["frame"])
        plans.append(_Plan(note_id, frame, color, timecode, text, custom_data, old_frame))
    prunes = [
        PrunedMarker(note_id=note_id, frame=marker["frame"])
        for note_id, marker in ours.items()
        if note_id not in desired and params.prune
    ]

    applied = _Applied()
    try:
        _mirror_apply(timeline, rid, plans, prunes, applied)
    except OpError as exc:
        raise _with_applied(exc, applied) from exc
    markers.extend(MirroredMarker(note_id=p.note_id, frame=p.frame, status=p.status) for p in plans)
    return MirrorResult(
        timeline=env.identity(timeline),
        review_id=rid,
        markers=sorted(markers, key=lambda m: m.frame),
        pruned=sorted(prunes, key=lambda p: p.frame),
    )


def _mirror_apply(
    timeline: Any, rid: str, plans: list[_Plan], prunes: list[PrunedMarker], applied: _Applied
) -> None:
    """Apply a pre-flighted plan, never losing a marker that can be kept.

    A moved note's new marker is added *before* its old one is deleted. Where frames
    collide, deletes go first: a pruned marker sitting on a note's target frame is deleted
    before that note is added, and when every remaining note waits on a frame held by
    another remaining note's old marker (an in-place change, or a cycle such as a swap),
    one old marker is deleted to break the wait. Other prunes run last.
    """
    by_note = {plan.note_id: plan for plan in plans}
    # The frames this run will vacate, and whose marker (by note id) sits on each now.
    held: dict[int, str] = {p.old_frame: p.note_id for p in plans if p.old_frame is not None}
    targets = {plan.frame for plan in plans}
    for prune in prunes:
        if prune.frame in targets:
            _delete(timeline, rid, prune.note_id, prune.frame, applied)
    pending = list(plans)
    while pending:
        ready = [plan for plan in pending if plan.frame not in held]
        if not ready:
            # Every remaining target is held by a remaining note's old marker: delete one.
            blocker = by_note[held[pending[0].frame]]
            assert blocker.old_frame is not None
            _delete(timeline, rid, blocker.note_id, blocker.old_frame, applied)
            del held[blocker.old_frame]
            continue
        for plan in ready:
            _add(timeline, plan, applied)
            if plan.old_frame is not None and held.get(plan.old_frame) == plan.note_id:
                _delete(timeline, rid, plan.note_id, plan.old_frame, applied)
                del held[plan.old_frame]
            pending.remove(plan)
    for prune in prunes:
        if prune.frame not in targets:
            _delete(timeline, rid, prune.note_id, prune.frame, applied)


def _add(timeline: Any, plan: _Plan, applied: _Applied) -> None:
    """Add the note's marker; record it once Resolve shows it, then verify its fields."""
    ok = timeline.AddMarker(plan.frame, plan.color, plan.timecode, plan.text, 1, plan.custom_data)
    read = {m["frame"]: m for m in raw.read_markers(timeline)}.get(plan.frame)
    if read is None or read["custom_data"] != plan.custom_data:
        raise OpError(
            "add_marker_failed",
            f"AddMarker at frame {plan.frame} for note {plan.note_id!r} returned {ok} and the "
            f"frame holds {read}",
            "the frame must lie inside the timeline and be free; re-run to reconcile",
            note_id=plan.note_id,
            frame=plan.frame,
        )
    applied.added.append({"note_id": plan.note_id, "frame": plan.frame})
    if not ok or (read["color"], read["name"], read["note"]) != (
        plan.color,
        plan.timecode,
        plan.text,
    ):
        raise OpError(
            "marker_unverified",
            f"the marker for note {plan.note_id!r} reads back as {read}",
            "inspect markers with timeline.describe; re-run to reconcile",
            note_id=plan.note_id,
            frame=plan.frame,
        )


def _delete(timeline: Any, rid: str, note_id: str, frame: int, applied: _Applied) -> None:
    """Delete this note's marker at ``frame``, by frame: never by custom data, which deletes
    the *first* match and so would remove a just-added copy at an earlier frame. The frame is
    checked to hold our marker first, so another writer's marker is never removed."""
    custom_data = note_marker_custom_data(rid, note_id)
    if timeline.GetMarkerCustomData(frame) != custom_data:
        raise OpError(
            "marker_delete_failed",
            f"frame {frame} no longer holds marker {custom_data!r}",
            "another writer changed the marker; re-run to reconcile",
            note_id=note_id,
            frame=frame,
        )
    ok = timeline.DeleteMarkerAtFrame(frame)
    if timeline.GetMarkerCustomData(frame) == custom_data:
        raise OpError(
            "marker_delete_failed",
            f"DeleteMarkerAtFrame({frame}) returned {ok}; marker {custom_data!r} is still there",
            "remove the marker by hand, then re-run",
            note_id=note_id,
            frame=frame,
        )
    applied.deleted.append({"note_id": note_id, "frame": frame})


def _with_applied(exc: OpError, applied: _Applied) -> OpError:
    if "applied" in exc.detail:
        return exc
    extra = {k: v for k, v in exc.detail.items() if k not in ("fix_hint", "trap_id")}
    return OpError(
        exc.code, exc.message, exc.fix_hint, exc.trap_id, **extra, applied=applied.to_dict()
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
