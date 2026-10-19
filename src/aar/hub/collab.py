"""Collaboration reads and records on the hub side.

- :func:`project_board` narrows the lease board (:mod:`aar.hub.leaseboard`) to one project:
  the project and timeline leases on it, which humans are active, and whether any node has
  an unreconciled feed gap (so the view may be stale). It also reconciles a node's leases
  against that node's latest ``lease.snapshot``, so a lease whose release was dropped on an
  overflowing feed does not linger as a phantom.
- :func:`record_branch` records an agent branch timeline, linked to its source timeline and
  the owning agent; :func:`branches` projects them. Only a node may record one; the first
  node to record a ``(project, branch)`` owns that record, and only it may supersede it,
  never with a lower version.
- ``feed.gap`` is the event a node's feed emits when its outbox overflowed; a later
  ``lease.snapshot`` from the same node clears it. :func:`feed_gaps` returns the gaps that
  are still unreconciled.

The project id the board uses is the Resolve project's unique id (the lease key), not the
hub registry's ``prj_...`` id; the branch record carries both (the Resolve timeline ids and
the hub project id on its row).

No node runtime assembles the broker, feed and these recorders yet. Wiring the node's pump
to call ``NodeFeed.flush`` and to record each ``timeline.branch`` result here is the job of
whichever of S7 (the job runner) and S11 lands second; until then these are the hub-side
primitives the tests drive directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aar.hub.common import (
    Clock,
    InvalidInput,
    SkippedEvent,
    actor_dict,
    clean_text,
    is_int,
    is_text,
    is_ts,
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog
from aar.hub.leaseboard import (
    LEASE_EVENT_TYPES,
    MAX_LEASE_ID,
    MAX_NODE_ID,
    BoardLease,
    Presence,
    lease_board,
)
from aar.hub.registry import Registry

EVENT_BRANCH_RECORDED = "collab.branch_recorded"
EVENT_FEED_GAP = "feed.gap"
EVENT_LEASE_SNAPSHOT = "lease.snapshot"

MAX_ID = 128
MAX_NAME = 256
MAX_HANDOFF = 2000
MAX_VERSION = 10**6
MAX_SNAPSHOT_LEASES = 10_000
"""Lease ids one ``lease.snapshot`` may carry; a node with more does not reconcile."""
_PROJECT_SCOPES = frozenset({"project", "timeline"})


# -- feed gaps and lease snapshots -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeedGap:
    """A node's report that its feed dropped events, so its board history may be stale."""

    node_id: str
    dropped: int
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "dropped": self.dropped, "at": self.at}


@dataclass(frozen=True, slots=True)
class _Snapshot:
    """A node's latest ``lease.snapshot``: the ids of every lease it held, at log ``seq``."""

    seq: int
    held: frozenset[str]


def _gap_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_int(payload.get("dropped"), 1, 10**18):
        return "dropped is not a positive count"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    return None


def _snapshot_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_NODE_ID):
        return "node_id is not an id"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    ids = payload.get("lease_ids")
    if not isinstance(ids, list) or len(ids) > MAX_SNAPSHOT_LEASES:
        return f"lease_ids is not a list of at most {MAX_SNAPSHOT_LEASES} ids"
    if not all(is_text(lease_id, MAX_LEASE_ID) for lease_id in ids):
        return "lease_ids holds something that is not a lease id"
    return None


def _authored(event: Event) -> bool:
    """Whether ``event`` was written by the node its payload names."""
    return event.actor == Actor("node", event.payload["node_id"])


def _node_events(
    log: EventLog, event_type: str, problem: Callable[[Any], str | None]
) -> tuple[list[Event], list[SkippedEvent]]:
    """Well-formed ``event_type`` events written by the node they name; the rest, skipped."""
    good: list[Event] = []
    skipped: list[SkippedEvent] = []
    for event in log.iter_events(types=[event_type]):
        why = problem(event.payload)
        if why is None and not _authored(event):
            why = f"written by {event.actor.kind}:{event.actor.id}, not by the node it names"
        if why is None:
            good.append(event)
        else:
            skipped.append(SkippedEvent.of(event, why))
    return good, skipped


def _latest_snapshots(log: EventLog) -> tuple[dict[str, _Snapshot], list[SkippedEvent]]:
    """Each node's latest well-formed ``lease.snapshot``, by node id."""
    events, skipped = _node_events(log, EVENT_LEASE_SNAPSHOT, _snapshot_problem)
    latest = {
        event.payload["node_id"]: _Snapshot(event.seq, frozenset(event.payload["lease_ids"]))
        for event in events
    }
    return latest, skipped


def _gaps(
    log: EventLog, snapshots: dict[str, _Snapshot]
) -> tuple[list[FeedGap], list[SkippedEvent]]:
    gaps: dict[str, tuple[FeedGap, int]] = {}
    events, skipped = _node_events(log, EVENT_FEED_GAP, _gap_problem)
    for event in events:
        payload = event.payload
        gap = FeedGap(payload["node_id"], payload["dropped"], payload["at"])
        gaps[payload["node_id"]] = (gap, event.seq)
    active = [
        gap
        for node_id, (gap, seq) in gaps.items()
        if node_id not in snapshots or snapshots[node_id].seq < seq
    ]
    return sorted(active, key=lambda gap: gap.node_id), skipped


def feed_gaps(log: EventLog) -> list[FeedGap]:
    """Unreconciled feed gaps: each node's latest gap that no later snapshot has cleared.

    A node sends a ``lease.snapshot`` after a gap; once it lands, the board has that node's
    leases again (see :func:`project_board`), so the gap no longer makes the view stale.
    """
    snapshots, _ = _latest_snapshots(log)
    return _gaps(log, snapshots)[0]


def _last_held(log: EventLog) -> dict[tuple[str, str], int]:
    """The log seq of the latest grant or renewal of each ``(node id, lease id)``."""
    last: dict[tuple[str, str], int] = {}
    held_types = [LEASE_EVENT_TYPES["granted"], LEASE_EVENT_TYPES["renewed"]]
    for event in log.iter_events(types=held_types):
        payload = event.payload
        if (
            isinstance(payload, dict)
            and is_text(payload.get("node_id"), MAX_NODE_ID)
            and is_text(payload.get("lease_id"), MAX_LEASE_ID)
            and _authored(event)
        ):
            last[(payload["node_id"], payload["lease_id"])] = event.seq
    return last


# -- per-project lease + presence board ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectBoard:
    """The lease board narrowed to one project."""

    project_id: str
    leases: tuple[BoardLease, ...]
    presence: tuple[Presence, ...]
    gaps: tuple[FeedGap, ...]
    as_of: str
    skipped: tuple[SkippedEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "as_of": self.as_of,
            "stale": bool(self.gaps),
            "leases": [lease.to_dict() for lease in self.leases],
            "presence": [entry.to_dict() for entry in self.presence],
            "gaps": [gap.to_dict() for gap in self.gaps],
        }


def project_board(log: EventLog, project_id: str, *, clock: Clock = utcnow) -> ProjectBoard:
    """Who holds this project (its project and timeline leases), and which humans are active.

    ``project_id`` is the Resolve project's unique id, the first part of a project or
    timeline lease key. Instance leases belong to no project and are left out. Presence is
    node-level, so every active human is listed.

    A node's leases are reconciled against its latest ``lease.snapshot``: a lease last
    granted or renewed before the snapshot, and not in it, is no longer held (its release
    was dropped on an overflowing feed, or an earlier process of the node held it), so it
    is left out. ``gaps`` lists nodes whose feed dropped events and has not reconciled
    since, so the view may be stale.
    """
    board = lease_board(log, clock=clock)
    snapshots, snapshot_skips = _latest_snapshots(log)
    gaps, gap_skips = _gaps(log, snapshots)
    last_held = _last_held(log) if snapshots else {}

    def released(lease: BoardLease) -> bool:
        snapshot = snapshots.get(lease.node_id)
        return (
            snapshot is not None
            and lease.lease_id not in snapshot.held
            and last_held.get((lease.node_id, lease.lease_id), -1) < snapshot.seq
        )

    leases = tuple(
        lease
        for lease in board.leases
        if lease.scope in _PROJECT_SCOPES
        and lease.key
        and lease.key[0] == project_id
        and not released(lease)
    )
    return ProjectBoard(
        project_id,
        leases,
        board.presence,
        tuple(gaps),
        board.as_of,
        (*board.skipped, *snapshot_skips, *gap_skips),
    )


# -- agent branch records --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Branch:
    """An agent branch timeline, linked to its source timeline and the owning agent."""

    project_id: str | None
    source_timeline_id: str
    source_timeline_name: str
    agent_id: str
    branch_timeline_id: str
    branch_timeline_name: str
    version: int
    handoff: str
    recorded_by: Actor
    ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "source": {"id": self.source_timeline_id, "name": self.source_timeline_name},
            "agent_id": self.agent_id,
            "branch": {"id": self.branch_timeline_id, "name": self.branch_timeline_name},
            "version": self.version,
            "handoff": self.handoff,
            "recorded_by": actor_dict(self.recorded_by),
            "ts": self.ts,
        }


def record_branch(
    log: EventLog,
    project_id: str,
    actor: Actor,
    *,
    source_timeline_id: str,
    source_timeline_name: str,
    agent_id: str,
    branch_timeline_id: str,
    branch_timeline_name: str,
    version: int,
    handoff: str,
) -> Event:
    """Record one agent branch, for a known project, written by a node.

    The writer must be a node (only nodes report branches); the project must exist in the
    registry; names and the handoff are cleaned and bounded; ids and the version are
    bounded. The first node to record a ``(project, branch)`` owns its record: another
    node's record for it, or one with a lower version (a delayed retry), is refused.
    """
    if not isinstance(actor, Actor) or actor.kind != "node":
        raise InvalidInput("actor", "only a node may record a branch")
    project_id = clean_text(project_id, "project_id", max_len=MAX_ID)
    if Registry(log).get(project_id) is None:
        raise InvalidInput("project_id", f"unknown project: {project_id!r}")
    payload = {
        "project_id": project_id,
        "source_timeline_id": clean_text(source_timeline_id, "source_timeline_id", max_len=MAX_ID),
        "source_timeline_name": clean_text(
            source_timeline_name, "source_timeline_name", max_len=MAX_NAME
        ),
        "agent_id": clean_text(agent_id, "agent_id", max_len=MAX_ID),
        "branch_timeline_id": clean_text(branch_timeline_id, "branch_timeline_id", max_len=MAX_ID),
        "branch_timeline_name": clean_text(
            branch_timeline_name, "branch_timeline_name", max_len=MAX_NAME
        ),
        "version": version,
        "handoff": clean_text(
            handoff, "handoff", max_len=MAX_HANDOFF, required=False, multiline=True
        ),
    }
    if not is_int(version, 1, MAX_VERSION):
        raise InvalidInput("version", f"must be an integer between 1 and {MAX_VERSION}")
    current = _current_record(log, project_id, payload["branch_timeline_id"])
    if current is not None:
        problem = _supersede_problem(current, actor.id, version)
        if problem is not None:
            raise InvalidInput("branch_timeline_id", problem)
    return log.append(EVENT_BRANCH_RECORDED, actor, payload, project_id=project_id)


def _branch_problem(payload: Any) -> str | None:
    """Why a recorded branch payload is unusable, or ``None``. Used by the fold."""
    if not isinstance(payload, dict):
        return "payload is not an object"
    for field in ("source_timeline_id", "branch_timeline_id", "agent_id"):
        if not is_text(payload.get(field), MAX_ID):
            return f"{field} is not an id"
    for field in ("source_timeline_name", "branch_timeline_name"):
        if not is_text(payload.get(field), MAX_NAME):
            return f"{field} is not a name"
    if not is_int(payload.get("version"), 1, MAX_VERSION):
        return "version is not a positive integer"
    if not is_text(payload.get("handoff"), MAX_HANDOFF, allow_empty=True):
        return "handoff is not text"
    return None


def _branch_from_event(event: Event) -> Branch:
    payload = event.payload
    return Branch(
        project_id=event.project_id,
        source_timeline_id=payload["source_timeline_id"],
        source_timeline_name=payload["source_timeline_name"],
        agent_id=payload["agent_id"],
        branch_timeline_id=payload["branch_timeline_id"],
        branch_timeline_name=payload["branch_timeline_name"],
        version=payload["version"],
        handoff=payload.get("handoff", ""),
        recorded_by=event.actor,
        ts=event.ts,
    )


@dataclass(frozen=True, slots=True)
class BranchScan:
    branches: tuple[Branch, ...]
    skipped: tuple[SkippedEvent, ...]


def _supersede_problem(current: Branch, node_id: str, version: int) -> str | None:
    """Why ``node_id`` may not replace ``current`` with a record at ``version``, or ``None``.

    The node that first recorded a branch owns its record; only it may supersede the
    record, and never with a lower version (a delayed retry must not roll it back).
    """
    if node_id != current.recorded_by.id:
        return f"the branch is recorded by node {current.recorded_by.id!r}, not {node_id!r}"
    if version < current.version:
        return f"version {version} is behind the recorded version {current.version}"
    return None


_BranchState = dict[tuple[str | None, str], tuple[int, Branch]]
"""``(project id, branch timeline id)`` -> (seq of its first record, the current record)."""


def _fold(log: EventLog, project_id: str | None) -> tuple[_BranchState, list[SkippedEvent]]:
    state: _BranchState = {}
    skipped: list[SkippedEvent] = []
    for event in log.iter_events(project_id=project_id, types=[EVENT_BRANCH_RECORDED]):
        problem = _branch_problem(event.payload)
        if problem is None and event.actor.kind != "node":
            problem = f"recorded by {event.actor.kind}:{event.actor.id}, not a node"
        if problem is not None:
            skipped.append(SkippedEvent.of(event, problem))
            continue
        branch = _branch_from_event(event)
        key = (branch.project_id, branch.branch_timeline_id)
        first = state.get(key)
        if first is not None:
            problem = _supersede_problem(first[1], event.actor.id, branch.version)
            if problem is not None:
                skipped.append(SkippedEvent.of(event, problem))
                continue
        state[key] = (event.seq if first is None else first[0], branch)
    return state, skipped


def _current_record(log: EventLog, project_id: str, branch_timeline_id: str) -> Branch | None:
    state, _ = _fold(log, project_id)
    entry = state.get((project_id, branch_timeline_id))
    return entry[1] if entry is not None else None


def scan_branches(log: EventLog, project_id: str | None = None) -> BranchScan:
    """Every recorded branch (optionally for one project), oldest first, plus skipped events.

    Keyed by ``(project id, branch timeline id)`` so projects never collide. The first node
    to record a key owns it: a later record replaces it only when it comes from that same
    node with a version no lower than the recorded one (:func:`record_branch` refuses the
    others up front; the fold re-checks, for events written concurrently or by hand). A
    non-node writer, a cross-node overwrite, a rollback or a malformed event is skipped and
    listed.
    """
    state, skipped = _fold(log, project_id)
    ordered = sorted(state.items(), key=lambda item: (item[1][0], item[0]))
    return BranchScan(tuple(branch for _, (_, branch) in ordered), tuple(skipped))


def branches(log: EventLog, project_id: str | None = None) -> list[Branch]:
    """Recorded branches (all projects, or one), oldest first."""
    return list(scan_branches(log, project_id).branches)
