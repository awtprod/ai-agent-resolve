"""Collaboration reads and records on the hub side.

Two things sit here, both projected from the shared event log:

- :func:`project_board` narrows the lease board (:mod:`aar.hub.leaseboard`) to one
  project: the project and timeline leases on it, plus which humans are active. It is the
  per-project view the console and an MCP tool show alongside the whole-fleet board.
- :func:`record_branch` records an agent branch timeline, linked to its source timeline
  and the agent that owns it; :func:`branches` projects them. The node reports a branch it
  made (``aar.node.ops.collab``) the same way it reports leases, through the feed.

The project id used by the board is the Resolve project's unique id (the lease key), not
the hub registry's ``prj_...`` id; the branch record carries both the Resolve timeline
ids and the hub project id on its row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aar.hub.common import (
    Clock,
    SkippedEvent,
    actor_dict,
    is_int,
    is_text,
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog
from aar.hub.leaseboard import BoardLease, Presence, lease_board

EVENT_BRANCH_RECORDED = "collab.branch_recorded"

MAX_ID = 128
MAX_NAME = 256
MAX_HANDOFF = 2000
MAX_VERSION = 10**6
_PROJECT_SCOPES = frozenset({"project", "timeline"})


# -- per-project lease + presence board ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProjectBoard:
    """The lease board narrowed to one project."""

    project_id: str
    leases: tuple[BoardLease, ...]
    presence: tuple[Presence, ...]
    as_of: str
    skipped: tuple[SkippedEvent, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "as_of": self.as_of,
            "leases": [lease.to_dict() for lease in self.leases],
            "presence": [entry.to_dict() for entry in self.presence],
        }


def project_board(log: EventLog, project_id: str, *, clock: Clock = utcnow) -> ProjectBoard:
    """Who holds this project (its project and timeline leases), and which humans are active.

    ``project_id`` is the Resolve project's unique id, which is the first part of a project
    or timeline lease key. Instance leases belong to no project and are left out. Presence
    is node-level, so every active human is listed: a human on that node may be on this
    project (the node presence event does not say which).
    """
    board = lease_board(log, clock=clock)
    leases = tuple(
        lease
        for lease in board.leases
        if lease.scope in _PROJECT_SCOPES and lease.key and lease.key[0] == project_id
    )
    return ProjectBoard(project_id, leases, board.presence, board.as_of, board.skipped)


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
    idempotency_key: str | None = None,
) -> Event:
    """Record one agent branch. Idempotent per branch timeline id, so a re-report is deduped."""
    payload = {
        "project_id": project_id,
        "source_timeline_id": source_timeline_id,
        "source_timeline_name": source_timeline_name,
        "agent_id": agent_id,
        "branch_timeline_id": branch_timeline_id,
        "branch_timeline_name": branch_timeline_name,
        "version": version,
        "handoff": handoff,
    }
    problem = _branch_problem(payload)
    if problem is not None:
        raise ValueError(f"cannot record branch: {problem}")
    key = idempotency_key or f"branch:{project_id}:{branch_timeline_id}"
    return log.append(
        EVENT_BRANCH_RECORDED, actor, payload, project_id=project_id, idempotency_key=key
    )


def _branch_problem(payload: Any) -> str | None:
    """Why a branch payload is unusable, or ``None``. Shared by the writer and the fold."""
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


def scan_branches(log: EventLog, project_id: str | None = None) -> BranchScan:
    """Every recorded branch (optionally for one project), oldest first, plus skipped events.

    Each branch timeline id appears once (the latest record for it wins); a malformed event
    is skipped and listed, never raised.
    """
    latest: dict[str, Branch] = {}
    order: dict[str, int] = {}
    skipped: list[SkippedEvent] = []
    for event in log.iter_events(project_id=project_id, types=[EVENT_BRANCH_RECORDED]):
        problem = _branch_problem(event.payload)
        if problem is not None:
            skipped.append(SkippedEvent.of(event, problem))
            continue
        branch = _branch_from_event(event)
        order.setdefault(branch.branch_timeline_id, event.seq)
        latest[branch.branch_timeline_id] = branch
    ordered = sorted(
        latest.values(), key=lambda b: (order[b.branch_timeline_id], b.branch_timeline_id)
    )
    return BranchScan(tuple(ordered), tuple(skipped))


def branches(log: EventLog, project_id: str | None = None) -> list[Branch]:
    """Recorded branches (all projects, or one), oldest first."""
    return list(scan_branches(log, project_id).branches)
