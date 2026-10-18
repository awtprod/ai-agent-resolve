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
from aar.hub.common import (
    InvalidInput,
    SkippedEvent,
    actor_dict,
    clean_text,
    is_int,
    is_text,
    require_human,
)
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    match_or_conflict,
)
from aar.hub.graph import (
    EVENT_NODE_ADDED,
    GraphNode,
    IllegalTransition,
    ProtectedNode,
    WorkGraph,
)
from aar.hub.registry import Registry

EVENT_NOTE_ADDED = "review.note_added"
EVENT_DECIDED = "review.decided"
EVENT_WITHDRAWN = "review.withdrawn"
_REVIEW_EVENT_TYPES = (EVENT_NOTE_ADDED, EVENT_DECIDED, EVENT_WITHDRAWN)

DECISIONS = ("approve", "reject")
DECISION_STATUS = {"approve": "done", "reject": "failed"}
OPEN_STATUSES = frozenset({"pending", "active", "blocked"})

MAX_ID = 128
MAX_TITLE = 200
MAX_NOTE = 4000
MAX_REASON = 2000
MAX_TIMELINE = 200
MAX_RENDERS = 16
MAX_FRAME = 10**9
MAX_FPS = 1000.0

TIMECODE_RE = re.compile(r"^\d{2}:[0-5]\d:[0-5]\d[:;]\d{2}$")
"""A display timecode, ``HH:MM:SS:FF`` (``;`` before the frames for drop-frame)."""
_TIMECODE_RE = TIMECODE_RE  # kept: S11 imports the private name


def is_timecode(value: Any) -> bool:
    """Whether ``value`` is a display timecode string (``HH:MM:SS:FF``)."""
    return isinstance(value, str) and TIMECODE_RE.fullmatch(value) is not None


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
    """The review is closed (decided, withdrawn or cancelled), so it takes no more input."""


class SubjectCancelled(ReviewError):
    """The node under review was cancelled, so the review cannot be approved."""


class NotRequester(ReviewError, PermissionError):
    """Only the agent that requested a review (or a human) may withdraw it."""


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
class Withdrawal:
    """A review withdrawn by its requester (or a human) without a decision."""

    event_id: str
    review_id: str
    reason: str
    by: Actor
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "review_id": self.review_id,
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
    withdrawal: Withdrawal | None = None

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
            "withdrawal": self.withdrawal.to_dict() if self.withdrawal else None,
        }


# -- validation --------------------------------------------------------------


def _check_fps(fps: Any) -> float | None:
    if fps is None:
        return None
    if isinstance(fps, bool) or not isinstance(fps, (int, float)):
        raise InvalidInput("fps", f"must be a number, got {type(fps).__name__}")
    if isinstance(fps, int) and not 0 < fps <= MAX_FPS:
        raise InvalidInput("fps", f"must be a positive frame rate up to {MAX_FPS:g}")
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


def _note_problem(payload: Any) -> str | None:
    """Why a note payload is unusable, or ``None``. Shared by the writer and the fold."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("review_id"), MAX_ID):
        return "review_id is not an id"
    if not is_int(payload.get("frame"), 0, MAX_FRAME):
        return f"frame is not an integer between 0 and {MAX_FRAME}"
    timecode = payload.get("timecode")
    if not isinstance(timecode, str) or not _TIMECODE_RE.fullmatch(timecode):
        return "timecode is not HH:MM:SS:FF"
    if not is_text(payload.get("text"), MAX_NOTE):
        return f"text is not 1-{MAX_NOTE} characters"
    return None


def _decision_problem(actor: Actor, payload: Any) -> str | None:
    """Why a decision is unusable, or ``None``. Only a human's decision ever counts."""
    if actor.kind != "human":
        return f"decision by {actor.kind}:{actor.id}, not a human"
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("review_id"), MAX_ID):
        return "review_id is not an id"
    if payload.get("decision") not in DECISIONS:
        return "decision is not approve or reject"
    if not is_text(payload.get("reason"), MAX_REASON):
        return f"reason is not 1-{MAX_REASON} characters"
    return None


def _withdrawal_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("review_id"), MAX_ID):
        return "review_id is not an id"
    if not is_text(payload.get("reason"), MAX_REASON):
        return f"reason is not 1-{MAX_REASON} characters"
    return None


def _attribution_problem(graph: WorkGraph, event: Event) -> str | None:
    """Whether a decision or withdrawal was written with the status change it records.

    ``decide`` and ``withdraw`` append the review's status change and then this event
    in one transaction, so the genuine event is the very next one, by the same actor,
    and the status it implies is the one the graph holds. Anything else (appended
    straight to the log, or by someone else) would credit the wrong person and is
    skipped.
    """
    payload = event.payload
    decided = event.type == EVENT_DECIDED
    expected = DECISION_STATUS[payload["decision"]] if decided else "cancelled"
    change = graph.last_status_change(payload["review_id"])
    if change is None or change != (event.seq - 1, event.actor, expected):
        return (
            f"not written with the review's move to {expected!r}: it would credit "
            f"{event.actor.kind}:{event.actor.id} with an act the graph does not record"
        )
    return None


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


def _withdrawal_from_event(event: Event) -> Withdrawal:
    return Withdrawal(
        event_id=event.id,
        review_id=event.payload["review_id"],
        reason=event.payload["reason"],
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


def _optional_str(data: dict[str, Any], key: str, max_len: int = MAX_TITLE) -> str | None:
    value = data.get(key)
    return value if is_text(value, max_len) else None


def _build_review(
    graph: WorkGraph,
    node: GraphNode,
    notes: list[Note],
    decision: Decision | None,
    withdrawal: Withdrawal | None,
) -> Review:
    data = node.payload if isinstance(node.payload, dict) else {}
    try:
        fps_value = _check_fps(data.get("fps"))
    except InvalidInput:
        fps_value = None
    title = node.title if is_text(node.title, MAX_TITLE) else None
    return Review(
        id=node.id,
        project_id=graph.project_id,
        title=title or "(untitled review)",
        status=node.status,
        subject=_optional_str(data, "subject", MAX_ID),
        renders=_renders_from(data),
        timeline=_optional_str(data, "timeline", MAX_TIMELINE),
        fps=fps_value,
        requested_by=_optional_str(data, "requested_by", MAX_ID),
        created=node.created,
        updated=node.updated,
        notes=tuple(sorted(notes, key=lambda n: (n.frame, n.ts, n.id))),
        decision=decision,
        # A withdrawal is shown only when the graph agrees the review was cancelled.
        withdrawal=withdrawal if node.status == "cancelled" else None,
    )


@dataclass(frozen=True, slots=True)
class ReviewScan:
    """Reviews as projected from the log, plus the events that could not be used."""

    reviews: tuple[Review, ...]
    skipped: tuple[SkippedEvent, ...]


def _project_reviews(
    log: EventLog, project_id: str
) -> tuple[dict[str, Review], list[SkippedEvent]]:
    """Every review in one project, keyed by id, and the events skipped building them."""
    graph = WorkGraph(log, project_id)
    skipped = list(graph.skipped())
    review_nodes = {node.id: node for node in graph.nodes() if node.kind == "review"}
    notes: dict[str, list[Note]] = {}
    decisions: dict[str, Decision] = {}
    withdrawals: dict[str, Withdrawal] = {}
    for event in log.iter_events(project_id=project_id, types=_REVIEW_EVENT_TYPES):
        if event.type == EVENT_NOTE_ADDED:
            problem = _note_problem(event.payload)
        elif event.type == EVENT_DECIDED:
            problem = _decision_problem(event.actor, event.payload)
        else:
            problem = _withdrawal_problem(event.payload)
        review_id = event.payload.get("review_id") if problem is None else None
        if problem is None and review_id not in review_nodes:
            problem = f"no review {review_id!r} in this project"
        if problem is None and event.type == EVENT_DECIDED and review_id in decisions:
            problem = "a second decision; the first one stands"
        if problem is None and event.type != EVENT_NOTE_ADDED:
            problem = _attribution_problem(graph, event)
        if problem is not None:
            skipped.append(SkippedEvent.of(event, problem))
        elif event.type == EVENT_NOTE_ADDED:
            notes.setdefault(review_id, []).append(_note_from_event(event))
        elif event.type == EVENT_DECIDED:
            decisions[review_id] = _decision_from_event(event)
        else:
            withdrawals.setdefault(review_id, _withdrawal_from_event(event))
    reviews = {
        node_id: _build_review(
            graph, node, notes.get(node_id, []), decisions.get(node_id), withdrawals.get(node_id)
        )
        for node_id, node in review_nodes.items()
    }
    return reviews, skipped


def scan_reviews(
    log: EventLog, project_id: str | None = None, *, pending_only: bool = False
) -> ReviewScan:
    """Reviews (all projects, or one), oldest first, with the events skipped. Never raises
    on a malformed event: it is skipped and listed instead."""
    if project_id is not None:
        project_ids = [project_id]
    else:
        seen: dict[str, None] = {}
        for event in log.iter_events(types=[EVENT_NODE_ADDED]):
            if isinstance(event.project_id, str):
                seen.setdefault(event.project_id, None)
        project_ids = list(seen)
    found: list[Review] = []
    skipped: list[SkippedEvent] = []
    for pid in project_ids:
        reviews, problems = _project_reviews(log, pid)
        skipped.extend(problems)
        found.extend(r for r in reviews.values() if r.pending or not pending_only)
    found.sort(key=lambda r: (r.created or "", r.id))
    return ReviewScan(tuple(found), tuple(skipped))


def review_view(
    log: EventLog, project_id: str, review_id: str
) -> tuple[Review | None, tuple[SkippedEvent, ...]]:
    """One review (or ``None``) and the events skipped in its project."""
    reviews, skipped = _project_reviews(log, project_id)
    return reviews.get(review_id), tuple(skipped)


def get_review(log: EventLog, project_id: str, review_id: str) -> Review | None:
    """Return one review with its notes and decision, or ``None`` if there is no such review."""
    return review_view(log, project_id, review_id)[0]


def list_reviews(
    log: EventLog, project_id: str | None = None, *, pending_only: bool = False
) -> list[Review]:
    """Return reviews (all projects, or one), oldest first; ``pending_only`` keeps open ones."""
    return list(scan_reviews(log, project_id, pending_only=pending_only).reviews)


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
    if timecode is not None and (
        not isinstance(timecode, str) or not _TIMECODE_RE.fullmatch(timecode)
    ):
        raise InvalidInput("timecode", f"must look like HH:MM:SS:FF, got {timecode!r}")
    with log.transaction() as txn:
        if idempotency_key is not None:
            existing = txn.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                # A retry returns the stored note even if the review closed since.
                expected = {"review_id": review_id, "frame": frame, "text": text}
                compare = list(expected)
                if timecode is not None:
                    expected["timecode"] = timecode
                    compare.append("timecode")
                match_or_conflict(
                    existing,
                    EVENT_NOTE_ADDED,
                    expected,
                    compare,
                    project_id=project_id,
                    actor=actor,
                )
                return _note_from_event(existing)
        review = get_review(log, project_id, review_id)
        if review is None:
            raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
        _require_open(review)
        if timecode is None:
            if review.fps is None:
                raise InvalidInput(
                    "timecode", "is required because the review has no frame rate to derive it"
                )
            timecode = frames_to_timecode(frame, review.fps)
        payload = {"review_id": review_id, "frame": frame, "timecode": timecode, "text": text}
        _assert_valid(_note_problem(payload))
        event = txn.append(
            EVENT_NOTE_ADDED,
            actor,
            payload,
            project_id=project_id,
            idempotency_key=idempotency_key,
        )
    return _note_from_event(event)


def _assert_valid(problem: str | None) -> None:
    """Writers check their payload with the fold's own validator before appending."""
    if problem is not None:
        raise InvalidInput("payload", problem)


def _require_open(review: Review) -> None:
    if review.decision is not None:
        raise ReviewNotOpen(
            f"review {review.id} was already decided ({review.decision.decision} by "
            f"{review.decision.by.kind}:{review.decision.by.id}); it takes no more notes"
        )
    if not review.pending:
        raise ReviewNotOpen(f"review {review.id} is {review.status!r}; it takes no more input")


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
                match_or_conflict(
                    existing,
                    EVENT_DECIDED,
                    payload,
                    payload.keys(),
                    project_id=project_id,
                    actor=actor,
                )
                return _decision_from_event(existing)
        # Built inside the transaction, so it sees every committed event and nothing
        # can change underneath it before the commit.
        current = get_review(log, project_id, review_id)
        if current is None:
            raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
        if current.decision is not None:
            raise ReviewAlreadyDecided(current.decision)
        graph = WorkGraph(log, project_id)
        node = graph.node(review_id)
        if node is None:  # pragma: no cover - get_review found it in this transaction
            raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
        subject_id = node.payload.get("subject") if isinstance(node.payload, dict) else None
        subject = graph.node(subject_id) if isinstance(subject_id, str) else None
        if decision == "approve" and subject is not None and subject.status == "cancelled":
            raise SubjectCancelled(
                f"cannot approve review {review_id}: the node it reviews ({subject_id}) was "
                "cancelled. Reject the review instead, or have the agent request a new one."
            )
        _assert_valid(_decision_problem(actor, payload))
        try:
            if node.status in ("pending", "blocked"):
                graph.set_status_in(txn, review_id, "active", actor)
            graph.set_status_in(txn, review_id, DECISION_STATUS[decision], actor)
        except (IllegalTransition, ProtectedNode) as exc:
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


def withdraw(
    log: EventLog,
    project_id: str,
    review_id: str,
    reason: str,
    actor: Actor,
    *,
    idempotency_key: str | None = None,
) -> Withdrawal:
    """Withdraw an open review without a decision: it moves to ``cancelled``.

    Only the agent that requested the review, or a human, may withdraw it, and only
    while it is undecided. The ``review.withdrawn`` event (who, why) and the status
    change are written in one transaction.
    """
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    payload = {"review_id": review_id, "reason": reason}
    _assert_valid(_withdrawal_problem(payload))
    with log.transaction() as txn:
        if idempotency_key is not None:
            existing = txn.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                match_or_conflict(
                    existing,
                    EVENT_WITHDRAWN,
                    payload,
                    payload.keys(),
                    project_id=project_id,
                    actor=actor,
                )
                return _withdrawal_from_event(existing)
        review = get_review(log, project_id, review_id)
        if review is None:
            raise ReviewNotFound(f"no review {review_id!r} in project {project_id!r}")
        _require_open(review)
        graph = WorkGraph(log, project_id)
        try:
            graph.set_status_in(txn, review_id, "cancelled", actor)
        except ProtectedNode as exc:
            raise NotRequester(
                f"{actor.kind}:{actor.id} did not request review {review_id}; only its "
                "requester or a human may withdraw it"
            ) from exc
        event = txn.append(
            EVENT_WITHDRAWN,
            actor,
            payload,
            project_id=project_id,
            idempotency_key=idempotency_key,
        )
    return _withdrawal_from_event(event)
