"""Review requests, timecoded notes and human decisions, on the shared event log.

A review request is a ``review`` node in the project's work graph, linked to the node
it reviews by a ``reviews`` edge. Its data carries the subject, the render artifact
ids, and optionally the timeline name and frame rate. Notes and decisions are events
on the same log. The work graph stays the single projection of status; nothing here
keeps a second status store.

Status mapping, checked by the graph's state machine:

- requested: ``active``. The review is open for a human straight away, and the state
  machine has no ``pending -> done`` step, so a ``pending`` review could not be closed
  by one decision.
- approve: ``active -> done``.
- reject: ``active -> failed``.

A review node made with the generic graph API may still be ``pending`` or
``blocked``; deciding it activates it first, in the same transaction.

Only a human may decide, and only once. The decision event and the status change are
written in one ``BEGIN IMMEDIATE`` transaction, so two people deciding at once cannot
both win and a decision can never exist without its status (or the reverse).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from aar.hub.artifacts import ArtifactStore, InvalidArtifactId
from aar.hub.common import InvalidInput, actor_dict, clean_text, require_human
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)
from aar.hub.graph import EVENT_NODE_ADDED, IllegalTransition, WorkGraph
from aar.hub.registry import Registry

EVENT_NOTE_ADDED = "review.note_added"
EVENT_DECIDED = "review.decided"
_REVIEW_EVENT_TYPES = (EVENT_NOTE_ADDED, EVENT_DECIDED)

DECISIONS = ("approve", "reject")
DECISION_STATUS = {"approve": "done", "reject": "failed"}
OPEN_STATUSES = frozenset({"pending", "active", "blocked"})

MAX_TITLE = 200
MAX_NOTE = 4000
MAX_REASON = 2000
MAX_TIMELINE = 200
MAX_RENDERS = 16
MAX_FRAME = 10**9
MAX_FPS = 1000.0

_TIMECODE_RE = re.compile(r"^\d{2}:[0-5]\d:[0-5]\d[:;]\d{2}$")
_ARTIFACT_RE = re.compile(r"^[0-9a-f]{64}$")


class ReviewError(Exception):
    """Base class for review failures."""


class ProjectNotFound(ReviewError, LookupError):
    """The project id is not in the registry."""


class ReviewNotFound(ReviewError, LookupError):
    """No review node with this id exists in the project."""


class SubjectNotFound(ReviewError, LookupError):
    """The node a review is requested for does not exist in the project."""


class UnknownArtifact(ReviewError, LookupError):
    """A render artifact id has no content in the artifact store."""


class ReviewAlreadyDecided(ReviewError):
    """The review already has a decision. ``decision`` is the one that stands."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            f"review {decision.review_id} was already decided: {decision.decision} by "
            f"{decision.by.kind}:{decision.by.id} at {decision.ts}"
        )
        self.decision = decision


class ReviewNotOpen(ReviewError):
    """The review's status no longer allows a decision (for example it was cancelled)."""


@dataclass(frozen=True, slots=True)
class Note:
    """A note anchored at a frame of the render, with its display timecode."""

    id: str
    review_id: str
    frame: int
    timecode: str
    text: str
    author: Actor
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "review_id": self.review_id,
            "frame": self.frame,
            "timecode": self.timecode,
            "text": self.text,
            "author": actor_dict(self.author),
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True)
class Decision:
    """A human's approve or reject, with the reason."""

    event_id: str
    review_id: str
    decision: str
    reason: str
    by: Actor
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "review_id": self.review_id,
            "decision": self.decision,
            "reason": self.reason,
            "by": actor_dict(self.by),
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True)
class Review:
    """A review request as projected from the work graph plus its notes and decision."""

    id: str
    project_id: str
    title: str
    status: str
    subject: str | None
    renders: tuple[str, ...]
    timeline: str | None
    fps: float | None
    requested_by: str | None
    created: str | None
    updated: str | None
    notes: tuple[Note, ...]
    decision: Decision | None

    @property
    def pending(self) -> bool:
        """Whether the review is still open (its graph status is not terminal)."""
        return self.status in OPEN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "status": self.status,
            "pending": self.pending,
            "subject": self.subject,
            "renders": list(self.renders),
            "timeline": self.timeline,
            "fps": self.fps,
            "requested_by": self.requested_by,
            "created": self.created,
            "updated": self.updated,
            "notes": [note.to_dict() for note in self.notes],
            "decision": self.decision.to_dict() if self.decision else None,
        }


# -- validation --------------------------------------------------------------


def _check_fps(fps: Any) -> float | None:
    if fps is None:
        return None
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise InvalidInput("fps", f"must be a number, got {type(fps).__name__}")
    value = float(fps)
    if not math.isfinite(value) or value <= 0 or value > MAX_FPS:
        raise InvalidInput("fps", f"must be a positive frame rate up to {MAX_FPS:g}")
    return value


def _check_frame(frame: Any) -> int:
    if isinstance(frame, bool) or not isinstance(frame, int):
        raise InvalidInput("frame", f"must be an integer, got {type(frame).__name__}")
    if frame < 0 or frame > MAX_FRAME:
        raise InvalidInput("frame", f"must be between 0 and {MAX_FRAME}")
    return frame


def _check_artifact_id(value: Any) -> str:
    if not isinstance(value, str) or not _ARTIFACT_RE.fullmatch(value):
        raise InvalidInput("renders", f"artifact id must be 64 lowercase hex characters: {value!r}")
    return value


def frames_to_timecode(frame: int, fps: float) -> str:
    """Return the non-drop-frame timecode ``HH:MM:SS:FF`` of a 0-based frame.

    The timebase is the frame rate rounded to a whole number (23.976 counts in 24s),
    which is how Resolve labels non-drop-frame timecode.
    """
    frame = _check_frame(frame)
    base = round(_check_fps(fps) or 0)
    if base < 1:
        raise InvalidInput("fps", "must round to at least 1 frame per second")
    total_seconds, ff = divmod(frame, base)
    total_minutes, ss = divmod(total_seconds, 60)
    hh, mm = divmod(total_minutes, 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


# -- projection --------------------------------------------------------------


def _note_from_event(event: Event) -> Note:
    payload = event.payload
    return Note(
        id=event.id,
        review_id=payload["review_id"],
        frame=payload["frame"],
        timecode=payload["timecode"],
        text=payload["text"],
        author=event.actor,
        ts=event.ts,
    )


def _decision_from_event(event: Event) -> Decision:
    payload = event.payload
    return Decision(
        event_id=event.id,
        review_id=payload["review_id"],
        decision=payload["decision"],
        reason=payload["reason"],
        by=event.actor,
        ts=event.ts,
    )


def _renders_from(data: dict[str, Any]) -> tuple[str, ...]:
    """Return the well-formed render ids in a review node's data.

    Review nodes made with the generic graph API carry arbitrary data, so only ids
    that look like artifact ids are kept; nothing else is ever served.
    """
    renders = data.get("renders")
    if not isinstance(renders, list):
        return ()
    return tuple(r for r in renders if isinstance(r, str) and _ARTIFACT_RE.fullmatch(r))


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) else None


def _build_review(
    graph: WorkGraph,
    node_id: str,
    notes: list[Note],
    decision: Decision | None,
) -> Review | None:
    node = graph.node(node_id)
    if node is None or node.kind != "review":
        return None
    data = node.payload if isinstance(node.payload, dict) else {}
    fps = data.get("fps")
    try:
        fps_value = _check_fps(fps)
    except InvalidInput:
        fps_value = None
    return Review(
        id=node.id,
        project_id=graph.project_id,
        title=node.title or "(untitled review)",
        status=node.status,
        subject=_optional_str(data, "subject"),
        renders=_renders_from(data),
        timeline=_optional_str(data, "timeline"),
        fps=fps_value,
        requested_by=_optional_str(data, "requested_by"),
        created=node.created,
        updated=node.updated,
        notes=tuple(sorted(notes, key=lambda n: (n.frame, n.ts, n.id))),
        decision=decision,
    )


def _well_formed(event: Event) -> bool:
    """Whether a review event has the shape this module writes.

    The log is shared and anyone with the Python API can append any type, so the
    projection skips malformed events instead of failing every read.
    """
    payload = event.payload
    if event.project_id is None or not isinstance(payload.get("review_id"), str):
        return False
    if event.type == EVENT_NOTE_ADDED:
        frame = payload.get("frame")
        return (
            isinstance(frame, int)
            and not isinstance(frame, bool)
            and isinstance(payload.get("timecode"), str)
            and isinstance(payload.get("text"), str)
        )
    # Only a human's decision counts, even if an agent appended one straight to the log.
    return (
        event.actor.kind == "human"
        and payload.get("decision") in DECISIONS
        and isinstance(payload.get("reason"), str)
    )


def _review_events(
    log: EventLog, project_id: str | None
) -> tuple[dict[tuple[str, str], list[Note]], dict[tuple[str, str], Decision]]:
    notes: dict[tuple[str, str], list[Note]] = {}
    decisions: dict[tuple[str, str], Decision] = {}
    for event in log.iter_events(project_id=project_id, types=_REVIEW_EVENT_TYPES):
        if event.project_id is None or not _well_formed(event):
            continue
        key = (event.project_id, event.payload["review_id"])
        if event.type == EVENT_NOTE_ADDED:
            notes.setdefault(key, []).append(_note_from_event(event))
        elif event.type == EVENT_DECIDED and key not in decisions:
            # The first decision stands; decide() never writes a second one.
            decisions[key] = _decision_from_event(event)
    return notes, decisions


def get_review(log: EventLog, project_id: str, review_id: str) -> Review | None:
    """Return one review with its notes and decision, or ``None`` if there is no such review."""
    graph = WorkGraph(log, project_id)
    node = graph.node(review_id)
    if node is None or node.kind != "review":
        return None
    notes, decisions = _review_events(log, project_id)
    key = (project_id, review_id)
    return _build_review(graph, review_id, notes.get(key, []), decisions.get(key))


def list_reviews(
    log: EventLog, project_id: str | None = None, *, pending_only: bool = False
) -> list[Review]:
    """Return reviews (all projects, or one), oldest first; ``pending_only`` keeps open ones."""
    review_nodes: dict[str, list[str]] = {}
    for event in log.iter_events(project_id=project_id, types=[EVENT_NODE_ADDED]):
        if event.project_id is not None and event.payload.get("kind") == "review":
            review_nodes.setdefault(event.project_id, []).append(event.payload["node_id"])
    notes, decisions = _review_events(log, project_id)
    reviews: list[Review] = []
    for pid, node_ids in review_nodes.items():
        graph = WorkGraph(log, pid)
        for node_id in node_ids:
            key = (pid, node_id)
            review = _build_review(graph, node_id, notes.get(key, []), decisions.get(key))
            if review is not None and (review.pending or not pending_only):
                reviews.append(review)
    reviews.sort(key=lambda r: (r.created or "", r.id))
    return reviews


# -- mutations ---------------------------------------------------------------


def request_review(
    log: EventLog,
    store: ArtifactStore,
    project_id: str,
    subject_id: str,
    renders: list[str],
    actor: Actor,
    *,
    title: str,
    timeline: str | None = None,
    fps: float | None = None,
    idempotency_key: str | None = None,
) -> Review:
    """Open a review of ``subject_id`` with its render artifacts and return it.

    Every render must already be in ``store``. The review node and its ``reviews``
    edge are two graph mutations; a retry with the same ``idempotency_key`` completes
    a request that was interrupted between them.
    """
    title = clean_text(title, "title", max_len=MAX_TITLE)
    if timeline is not None:
        timeline = clean_text(timeline, "timeline", max_len=MAX_TIMELINE)
    fps_value = _check_fps(fps)
    if not isinstance(renders, (list, tuple)) or not renders:
        raise InvalidInput("renders", "at least one render artifact id is required")
    if len(renders) > MAX_RENDERS:
        raise InvalidInput("renders", f"at most {MAX_RENDERS} renders per review")
    render_ids = [_check_artifact_id(r) for r in renders]
    if len(set(render_ids)) != len(render_ids):
        raise InvalidInput("renders", "render artifact ids must be distinct")
    for artifact_id in render_ids:
        try:
            present = store.has(artifact_id)
        except InvalidArtifactId as exc:  # pragma: no cover - the regex above already checked
            raise InvalidInput("renders", str(exc)) from exc
        if not present:
            raise UnknownArtifact(f"render artifact {artifact_id} is not in the artifact store")

    if Registry(log).get(project_id) is None:
        raise ProjectNotFound(f"unknown project: {project_id!r}")
    graph = WorkGraph(log, project_id)
    if graph.node(subject_id) is None:
        raise SubjectNotFound(f"no node {subject_id!r} in project {project_id!r}")

    data = {
        "subject": subject_id,
        "renders": render_ids,
        "timeline": timeline,
        "fps": fps_value,
        "requested_by": f"{actor.kind}:{actor.id}",
    }
    node_key = None if idempotency_key is None else f"{idempotency_key}:node"
    edge_key = None if idempotency_key is None else f"{idempotency_key}:edge"
    node = graph.add_node(
        "review", actor, title=title, status="active", data=data, idempotency_key=node_key
    )
    graph.add_edge("reviews", node.id, subject_id, actor, idempotency_key=edge_key)
    review = get_review(log, project_id, node.id)
    if review is None:  # pragma: no cover - the node was just written
        raise ReviewNotFound(f"review {node.id} vanished after it was created")
    return review


def add_note(
    log: EventLog,
    project_id: str,
    review_id: str,
    frame: int,
    text: str,
    actor: Actor,
    *,
    timecode: str | None = None,
    idempotency_key: str | None = None,
) -> Note:
    """Anchor a note at ``frame`` of the review's render.

    ``frame`` is 0-based in the render (it is what the console seeks to). ``timecode``
    is the label shown beside it, for example the timeline's record timecode; when it
    is omitted it is derived from the frame and the review's frame rate.
    """
    frame = _check_frame(frame)
    text = clean_text(text, "text", max_len=MAX_NOTE, multiline=True)
    review = get_review(log, project_id, review_id)
    if review is None:
        raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
    if timecode is None:
        if review.fps is None:
            raise InvalidInput(
                "timecode", "is required because the review has no frame rate to derive it from"
            )
        timecode = frames_to_timecode(frame, review.fps)
    elif not isinstance(timecode, str) or not _TIMECODE_RE.fullmatch(timecode):
        raise InvalidInput("timecode", f"must look like HH:MM:SS:FF, got {timecode!r}")
    payload = {"review_id": review_id, "frame": frame, "timecode": timecode, "text": text}
    event = log.append(
        EVENT_NOTE_ADDED,
        actor,
        payload,
        project_id=project_id,
        idempotency_key=idempotency_key,
    )
    return _note_from_event(event)


def _existing_decision(txn: Transaction, project_id: str, review_id: str) -> Decision | None:
    for event in txn.read(0, project_id=project_id, types=[EVENT_DECIDED]):
        if event.payload.get("review_id") == review_id and _well_formed(event):
            return _decision_from_event(event)
    return None


def decide(
    log: EventLog,
    project_id: str,
    review_id: str,
    decision: str,
    reason: str,
    actor: Actor,
    *,
    idempotency_key: str | None = None,
) -> Decision:
    """Record a human's approve or reject and move the review's status, atomically.

    Raises :class:`~aar.hub.common.HumanRequired` for agents and nodes, and
    :class:`ReviewAlreadyDecided` if the review already has a decision. Retrying the
    same decision under the same ``idempotency_key`` returns the stored decision and
    writes nothing.
    """
    require_human(actor, "decide a review")
    if decision not in DECISIONS:
        raise InvalidInput("decision", f"must be one of {list(DECISIONS)}, got {decision!r}")
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    payload = {"review_id": review_id, "decision": decision, "reason": reason}
    with log.transaction() as txn:
        if idempotency_key is not None:
            existing = txn.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                match_or_conflict(existing, EVENT_DECIDED, payload, payload.keys())
                if existing.actor != actor or existing.project_id != project_id:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used for a "
                        "different request"
                    )
                return _decision_from_event(existing)
        # Built inside the transaction, so it sees every committed event and nothing
        # can change underneath it before the commit.
        graph = WorkGraph(log, project_id)
        node = graph.node(review_id)
        if node is None or node.kind != "review":
            raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
        prior = _existing_decision(txn, project_id, review_id)
        if prior is not None:
            raise ReviewAlreadyDecided(prior)
        try:
            if node.status in ("pending", "blocked"):
                graph.set_status_in(txn, review_id, "active", actor)
            graph.set_status_in(txn, review_id, DECISION_STATUS[decision], actor)
        except IllegalTransition as exc:
            raise ReviewNotOpen(
                f"review {review_id} is {node.status!r} and can no longer be decided"
            ) from exc
        event = txn.append(
            EVENT_DECIDED,
            actor,
            payload,
            project_id=project_id,
            idempotency_key=idempotency_key,
        )
    return _decision_from_event(event)
