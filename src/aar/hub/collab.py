"""Collaboration reads and records on the hub side.

- :func:`project_board` narrows the lease board to one project: the project and timeline
  leases on it, which humans are active, and whether any node's view may be stale. It is a
  narrowing of :func:`aar.hub.leaseboard.lease_board`, the one reconciled board every view
  reads (the console's leases page and the MCP ``lease_board`` tool too), so the views
  cannot disagree: a node's leases are checked against its latest ``lease.snapshot`` there,
  so a lease whose release was dropped does not linger as a phantom.
- :func:`record_branch` records an agent branch timeline, linked to its source timeline and
  the owning agent; :func:`branches` projects them. Only a node may record one; the first
  node to record a ``(project, branch)`` owns that record, and only it may supersede it,
  never with a lower version and never for another agent.
- ``feed.gap`` is the event a node's feed emits when it dropped events; a later
  ``lease.snapshot`` from the same node clears it. :func:`feed_gaps` returns the gaps that
  are still unreconciled.

The project id the board uses is the Resolve project's unique id (the lease key), not the
hub registry's ``prj_...`` id; the branch record carries both (the Resolve timeline ids and
the hub project id on its row).

The node runtime, :class:`aar.node.runtime.NodeRuntime` (S7), assembles the broker, the feed
and these recorders. The wiring it does (tested in ``tests/node/test_s7_s11_integration.py``):

1. ``aar.node.ops``: import every op module, ``cloud, collab, read, render, write``, so all
   ops register; ``test_ops_framework.py``'s ``EXPECTED`` keeps both slices' ops.
2. One shared ``BranchRegistry`` per node, passed to every ``make_hot_op`` the job runner
   builds. Each job's ops run with the job's own agent as holder (``Holder.agent(<agent
   id>)``); a job without a valid agent id is refused at submit. ``job-runner:<node>`` is
   never an op holder.
3. The branch guard is on for every op, in every project, production or not. An op it
   refuses (shared project state, a human's timeline) waits for a ``project.write``
   approval on target ``<Resolve project id>#op:<op name>``
   (:func:`aar.node.jobs.runner.op_approval_target`); the job consumes it (audited as
   ``agent:<agent id>#<job id>``), then builds that one op with ``require_branch=False``,
   the only production path to it. The render ops carry a named ``guard_exemption``: they
   read the timeline and touch only the render jobs their own job created.
4. ``LeaseManager`` and ``NodeFeed`` are built by :func:`aar.node.feed.wire_feed`, bound
   late: the feed's ``lease_source`` is the lease table's ``active``, and both use the
   runtime's (monotonic) clock.
5. The job runner's pump calls ``PresenceReporter.sync()`` and then ``NodeFeed.flush()``
   on every tick. The feed backs off by itself while the hub is unreachable
   (``FlushReport.retry_at``), so that is cheap.
6. Each ``timeline.branch`` result a job gets goes to :func:`record_branch` (through
   ``aar.node.hublink.LocalHubLink.record_branch``) with the hub's ``prj_...`` id (the one
   registry project whose source carries the Resolve uid, else the one with its name; none
   or several is an alert on the job) and ``Actor("node", <the feed's node id>)``. A record
   the hub could not take is sent again on later pumps, and after a restart.
7. Exactly one live ``NodeFeed`` per node id: a process refuses a second runtime for an
   open node id (``NodeIdInUse``). The hub admits a new process of a node id and refuses
   the one it replaced (:class:`~aar.hub.leaseboard.FeedConflict`, and the node shows
   ``stale``); the refused feed stops, and its process must be stopped.
8. The remote transport (F-S11-1) runs :func:`~aar.hub.leaseboard.deliver_feed_event` on
   the hub and reports back to the node: "unreachable" as an ``OSError`` (the feed keeps
   the spec and backs off), "refused" as a ``ValueError`` (quarantined), and a
   ``FeedConflict`` as itself (the feed stops). It must keep a spec's idempotency key on
   every resend; the feed reuses keys, orders its events itself (``feed.seq``) and drains
   under lost acks (S11f), so a late or duplicate delivery is harmless.

The node's ``HubLink`` (approvals, stops, branch records) follows the same error contract
as the feed transport: ``OSError`` unreachable (the job waits, fail-closed), ``ValueError``
refused.
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
    utcnow,
)
from aar.hub.events import Actor, Event, EventLog
from aar.hub.leaseboard import (
    EVENT_FEED_GAP,
    EVENT_LEASE_SNAPSHOT,
    MAX_SNAPSHOT_LEASES,
    BoardLease,
    FeedGap,
    Presence,
    SupersededFeed,
    lease_board,
)
from aar.hub.registry import Registry

__all__ = [
    "EVENT_BRANCH_RECORDED",
    "EVENT_FEED_GAP",
    "EVENT_LEASE_SNAPSHOT",
    "MAX_SNAPSHOT_LEASES",
    "Branch",
    "BranchScan",
    "FeedGap",
    "ProjectBoard",
    "branches",
    "feed_gaps",
    "project_board",
    "record_branch",
    "scan_branches",
]

EVENT_BRANCH_RECORDED = "collab.branch_recorded"

MAX_ID = 128
MAX_NAME = 256
MAX_HANDOFF = 2000
MAX_VERSION = 10**6
_PROJECT_SCOPES = frozenset({"project", "timeline"})


# -- feed gaps and the per-project lease + presence board -------------------------------------


def feed_gaps(log: EventLog) -> list[FeedGap]:
    """Unreconciled feed gaps: each node's latest gap that no later snapshot has cleared.

    A node sends a ``lease.snapshot`` after a gap; once it lands, the board has that node's
    leases again (see :func:`aar.hub.leaseboard.lease_board`), so the gap no longer makes
    the view stale.
    """
    return list(lease_board(log).gaps)


@dataclass(frozen=True, slots=True)
class ProjectBoard:
    """The lease board narrowed to one project."""

    project_id: str
    leases: tuple[BoardLease, ...]
    presence: tuple[Presence, ...]
    gaps: tuple[FeedGap, ...]
    as_of: str
    skipped: tuple[SkippedEvent, ...] = ()
    superseded: tuple[SupersededFeed, ...] = ()
    stale: bool = False
    """The lease board's own verdict (:attr:`aar.hub.leaseboard.LeaseBoard.stale`)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "as_of": self.as_of,
            "stale": self.stale,
            "leases": [lease.to_dict() for lease in self.leases],
            "presence": [entry.to_dict() for entry in self.presence],
            "gaps": [gap.to_dict() for gap in self.gaps],
            "superseded": [entry.to_dict() for entry in self.superseded],
        }


def project_board(log: EventLog, project_id: str, *, clock: Clock = utcnow) -> ProjectBoard:
    """Who holds this project (its project and timeline leases), and which humans are active.

    ``project_id`` is the Resolve project's unique id, the first part of a project or
    timeline lease key. Instance leases belong to no project and are left out. Presence is
    node-level, so every active human is listed.

    It narrows :func:`aar.hub.leaseboard.lease_board`, the reconciled board every view
    reads, so it shows exactly what the console and the ``lease_board`` tool show: a node's
    leases checked against its latest ``lease.snapshot``, and ``gaps`` / ``superseded``
    naming the nodes whose view may be stale.
    """
    board = lease_board(log, clock=clock)
    leases = tuple(
        lease
        for lease in board.leases
        if lease.scope in _PROJECT_SCOPES and lease.key and lease.key[0] == project_id
    )
    return ProjectBoard(
        project_id,
        leases,
        board.presence,
        board.gaps,
        board.as_of,
        board.skipped,
        board.superseded,
        board.stale,
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
    node's record for it, one with a lower version (a delayed retry), or one naming another
    agent (at any version) is refused.
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
        problem = _supersede_problem(current, actor.id, payload["agent_id"], version)
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


def _supersede_problem(current: Branch, node_id: str, agent_id: str, version: int) -> str | None:
    """Why ``node_id`` may not replace ``current`` with ``agent_id``'s record at ``version``.

    The node that first recorded a branch owns its record; only it may supersede the
    record, never with a lower version (a delayed retry must not roll it back), and never
    for another agent: the owning agent is part of the branch's identity (a branch timeline
    belongs to the agent that made it, as on the node's registry), so even its own node
    cannot hand the record to another agent, at any version. ``None`` if it may.
    """
    if node_id != current.recorded_by.id:
        return f"the branch is recorded by node {current.recorded_by.id!r}, not {node_id!r}"
    if agent_id != current.agent_id:
        return f"the branch belongs to agent {current.agent_id!r}, not {agent_id!r}"
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
            problem = _supersede_problem(first[1], event.actor.id, branch.agent_id, branch.version)
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
    node, for the same agent, with a version no lower than the recorded one
    (:func:`record_branch` refuses the others up front; the fold re-checks, for events
    written concurrently or by hand). A non-node writer, a cross-node overwrite, an agent
    swap, a rollback or a malformed event is skipped and listed.
    """
    state, skipped = _fold(log, project_id)
    ordered = sorted(state.items(), key=lambda item: (item[1][0], item[0]))
    return BranchScan(tuple(branch for _, (_, branch) in ordered), tuple(skipped))


def branches(log: EventLog, project_id: str | None = None) -> list[Branch]:
    """Recorded branches (all projects, or one), oldest first."""
    return list(scan_branches(log, project_id).branches)
