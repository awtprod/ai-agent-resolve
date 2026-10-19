"""Collaboration reads and records on the hub side.

- :func:`project_board` narrows the lease board (:mod:`aar.hub.leaseboard`) to one project:
  the project and timeline leases on it, which humans are active, and whether any node has
  an unreconciled feed gap (so the view may be stale). It also reconciles a node's leases
  against that node's latest ``lease.snapshot``, so a lease whose release was dropped on an
  overflowing feed does not linger as a phantom.
- :func:`record_branch` records an agent branch timeline, linked to its source timeline and
  the owning agent; :func:`branches` projects them. Only a node may record one; the first
  node to record a ``(project, branch)`` owns it, and only that node may supersede it, and
  only with a version that does not go backwards.
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
    iso,
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog
from aar.hub.leaseboard import BoardLease, Presence, lease_board
from aar.hub.registry import Registry

EVENT_BRANCH_RECORDED = "collab.branch_recorded"
EVENT_FEED_GAP = "feed.gap"
EVENT_LEASE_SNAPSHOT = "lease.snapshot"

MAX_ID = 128
MAX_NAME = 256
MAX_HANDOFF = 2000
MAX_VERSION = 10**6
_PROJECT_SCOPES = frozenset({"project", "timeline"})


# -- feed gaps and lease snapshots -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FeedGap:
    """A node reported it dropped feed events (its board history may be stale)."""

    node_id: str
    dropped: int
    at: str

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "dropped": self.dropped, "at": self.at}


@dataclass(frozen=True, slots=True)
class _Snapshot:
    at: str
    held: frozenset[str]
    seq: int


def _gap_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_ID):
        return "node_id is not an id"
    if not is_int(payload.get("dropped"), 1, 10**18):
        return "dropped is not a positive count"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    return None


def _snapshot_problem(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "payload is not an object"
    if not is_text(payload.get("node_id"), MAX_ID):
        return "node_id is not an id"
    if not is_ts(payload.get("at")):
        return "at is not a timestamp"
    if not isinstance(payload.get("leases"), list):
        return "leases is not a list"
    return None


def _latest_snapshots(log: EventLog) -> dict[str, _Snapshot]:
    """The latest ``lease.snapshot`` per node, by node id (node-authored only)."""
    latest: dict[str, _Snapshot] = {}
    for event in log.iter_events(types=[EVENT_LEASE_SNAPSHOT]):
        payload = event.payload
        if _snapshot_problem(payload) is not None:
            continue
        if event.actor != Actor("node", payload["node_id"]):
            continue
        held = frozenset(
            lease.get("lease_id")
            for lease in payload["leases"]
            if isinstance(lease, dict) and is_text(lease.get("lease_id"), MAX_ID)
        )
        latest[payload["node_id"]] = _Snapshot(payload["at"], held, event.seq)
    return latest


def feed_gaps(log: EventLog) -> list[FeedGap]:
    """Unreconciled feed gaps, by node id: the latest gap per node that no later snapshot
    has cleared. A ``lease.snapshot`` the node emits after a gap reconciles it."""
    gaps: dict[str, tuple[FeedGap, int]] = {}
    for event in log.iter_events(types=[EVENT_FEED_GAP]):
        payload = event.payload
        if _gap_problem(payload) is not None or event.actor != Actor("node", payload["node_id"]):
            continue
        gaps[payload["node_id"]] = (
            FeedGap(payload["node_id"], payload["dropped"], payload["at"]),
            event.seq,
        )
    snaps = _latest_snapshots(log)
    active = [
        gap
        for node_id, (gap, seq) in gaps.items()
        if node_id not in snaps or snaps[node_id].seq < seq
    ]
    return sorted(active, key=lambda gap: gap.node_id)


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
    node-level, so every active human is listed. A node's leases are reconciled against that
    node's latest ``lease.snapshot``: a lease that predates the snapshot and is not in it (a
    phantom left by a dropped release) is dropped. ``gaps`` flags nodes whose feed overflowed
    and has not yet reconciled.
    """
    board = lease_board(log, clock=clock)
    snaps = _latest_snapshots(log)
    leases = tuple(
        lease
        for lease in board.leases
        if lease.scope in _PROJECT_SCOPES
        and lease.key
        and lease.key[0] == project_id
        and not _superseded(lease, snaps.get(lease.node_id))
    )
    return ProjectBoard(
        project_id,
        leases,
        board.presence,
        tuple(feed_gaps(log)),
        board.as_of,
        board.skipped,
    )


def _superseded(lease: BoardLease, snapshot: _Snapshot | None) -> bool:
    """Whether ``snapshot`` shows ``lease`` is no longer held (acquired before it, absent)."""
    return (
        snapshot is not None
        and lease.since <= snapshot.at
        and lease.lease_id not in snapshot.held
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
    bounded. The first node to record a ``(project, branch)`` owns it; only that node may
    supersede it (see :func:`scan_branches`), and only with a version that never goes back.
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


@dataclass
class _Record:
    owner_node: str
    version: int
    seq: int
    branch: Branch


@dataclass(frozen=True, slots=True)
class BranchScan:
    branches: tuple[Branch, ...]
    skipped: tuple[SkippedEvent, ...]


def scan_branches(log: EventLog, project_id: str | None = None) -> BranchScan:
    """Every recorded branch (optionally for one project), oldest first, plus skipped events.

    Keyed by ``(project id, branch timeline id)`` so projects never collide. The first node
    to record a key owns it; a later record is applied only when it is from that same node
    and carries a version that is not lower than the one already recorded. A non-node actor,
    a cross-node overwrite, a version rollback, or a malformed event is skipped and listed.
    """
    state: dict[tuple[str | None, str], _Record] = {}
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
        current = state.get(key)
        if current is None:
            state[key] = _Record(event.actor.id, branch.version, event.seq, branch)
            continue
        if event.actor.id != current.owner_node:
            skipped.append(
                SkippedEvent.of(event, f"branch owned by node {current.owner_node!r}, not this one")
            )
            continue
        if branch.version < current.version:
            skipped.append(
                SkippedEvent.of(event, f"version {branch.version} is behind {current.version}")
            )
            continue
        current.version = branch.version
        current.branch = branch
    ordered = sorted(state.items(), key=lambda item: (item[1].seq, item[0]))
    return BranchScan(tuple(record.branch for _, record in ordered), tuple(skipped))


def branches(log: EventLog, project_id: str | None = None) -> list[Branch]:
    """Recorded branches (all projects, or one), oldest first."""
    return list(scan_branches(log, project_id).branches)
