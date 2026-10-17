"""Work graph: what is planned and in flight for one project.

Nodes are the units of work (``brief``, ``plan``, ``op``, ``artifact``, ``review``)
and carry a lifecycle status governed by a small state machine; illegal transitions
are rejected. Typed edges (``derives_from``, ``produces``, ``reviews``, ``blocks``)
connect them. Every mutation is an event on the shared log, and the live state is a
projection that is byte-for-byte rebuildable from that log.

Each mutation runs inside one ``BEGIN IMMEDIATE`` transaction: it first folds any
events another writer appended (so a stale projection cannot approve an illegal
transition), then validates, then appends. The catch-up read, the check and the
insert are therefore atomic.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from typing import Any

from aar.hub.common import SkippedEvent
from aar.hub.events import Actor, Event, EventLog, Transaction, match_or_conflict

NODE_KINDS = frozenset({"brief", "plan", "op", "artifact", "review"})
EDGE_TYPES = frozenset({"derives_from", "produces", "reviews", "blocks"})

# Lifecycle states and the transitions allowed between them. ``done`` and
# ``cancelled`` are terminal.
STATUSES = frozenset({"pending", "active", "blocked", "done", "failed", "cancelled"})
_TRANSITIONS: dict[str, frozenset[str]] = {
    "pending": frozenset({"active", "blocked", "cancelled"}),
    "active": frozenset({"blocked", "done", "failed", "cancelled"}),
    "blocked": frozenset({"active", "cancelled"}),
    "failed": frozenset({"active", "cancelled"}),
    "done": frozenset(),
    "cancelled": frozenset(),
}

# Edge endpoints that carry type meaning. An op produces an artifact; a review
# reviews something. Other edge types are unconstrained on purpose.
_EDGE_SRC_KINDS: dict[str, frozenset[str]] = {
    "produces": frozenset({"op"}),
    "reviews": frozenset({"review"}),
}
_EDGE_DST_KINDS: dict[str, frozenset[str]] = {
    "produces": frozenset({"artifact"}),
}

EVENT_NODE_ADDED = "graph.node_added"
EVENT_NODE_STATUS_CHANGED = "graph.node_status_changed"
EVENT_EDGE_ADDED = "graph.edge_added"

_GRAPH_TYPES = (EVENT_NODE_ADDED, EVENT_NODE_STATUS_CHANGED, EVENT_EDGE_ADDED)


class IllegalTransition(ValueError):
    """Raised when a status change is not permitted by the state machine."""


class DuplicateNodeId(ValueError):
    """Raised when an explicit node id is reused for a new node."""


class DuplicateEdgeId(ValueError):
    """Raised when an explicit edge id is reused for a new edge."""


class ProtectedNode(PermissionError):
    """Raised when a review node's status is changed outside the review gate.

    A review closes only through a human decision (``aar.hub.reviews.decide``) or
    its requester's withdrawal; the generic status API cannot finish or flip one.
    """


def new_node_id() -> str:
    """Return a fresh work-graph node id of the form ``nd_<hex>``."""
    return "nd_" + secrets.token_hex(8)


def new_edge_id() -> str:
    """Return a fresh work-graph edge id of the form ``edg_<hex>``."""
    return "edg_" + secrets.token_hex(8)


@dataclass
class GraphNode:
    """A unit of work with a lifecycle status."""

    id: str
    kind: str
    status: str
    title: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    created: str | None = None
    updated: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "title": self.title,
            "payload": self.payload,
            "created": self.created,
            "updated": self.updated,
        }


@dataclass
class Edge:
    """A typed relationship between two nodes."""

    id: str
    type: str
    src: str
    dst: str
    payload: dict[str, Any] = field(default_factory=dict)
    created: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "src": self.src,
            "dst": self.dst,
            "payload": self.payload,
            "created": self.created,
        }


@dataclass
class GraphState:
    """The projected state of one project's work graph.

    ``skipped`` lists events the projection could not use; ``creators`` records who
    added each node (the review gate lets a requester withdraw its own review).
    """

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)
    skipped: list[SkippedEvent] = field(default_factory=list)
    creators: dict[str, Actor] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }


def allowed_transitions(status: str) -> frozenset[str]:
    """Return the statuses reachable from ``status`` in one step."""
    return _TRANSITIONS.get(status, frozenset())


_REVIEW_CLOSED = frozenset({"done", "failed", "cancelled"})


def review_status_change_allowed(
    actor: Actor, creator: Actor | None, current: str, new_status: str
) -> bool:
    """Whether ``actor`` may move a ``review`` node from ``current`` to ``new_status``.

    A review is decided once: once closed (done, failed, cancelled) nobody reopens
    it. While open, humans decide; anyone else may only withdraw a review they
    requested themselves, by cancelling it.
    """
    if current in _REVIEW_CLOSED:
        return False
    if actor.kind == "human":
        return True
    return new_status == "cancelled" and creator is not None and actor == creator


def _is_id(value: Any) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 128


def _problem(state: GraphState, event: Event) -> str | None:
    """Why ``event`` cannot be folded into ``state``, or ``None`` if it can.

    The log is shared and append-only, so a malformed or forged event must be
    skipped rather than break every read of the graph.
    """
    payload = event.payload
    if not isinstance(payload, dict):
        return "payload is not an object"
    data = payload.get("data", {})
    if event.type == EVENT_NODE_ADDED:
        node_id = payload.get("node_id")
        if not _is_id(node_id):
            return "node_id is not a valid id"
        if node_id in state.nodes:
            return f"node {node_id} already exists"
        if payload.get("kind") not in NODE_KINDS:
            return "kind is not a node kind"
        if payload.get("status") not in STATUSES:
            return "status is not a status"
        title = payload.get("title")
        if title is not None and not isinstance(title, str):
            return "title is not text"
        return None if isinstance(data, dict) else "data is not an object"
    if event.type == EVENT_NODE_STATUS_CHANGED:
        node = state.nodes.get(payload.get("node_id")) if _is_id(payload.get("node_id")) else None
        if node is None:
            return "status change for an unknown node"
        to = payload.get("to")
        if to not in STATUSES or to not in allowed_transitions(node.status):
            return f"illegal transition from {node.status!r} to {to!r}"
        if node.kind == "review" and not review_status_change_allowed(
            event.actor, state.creators.get(node.id), node.status, to
        ):
            return f"review status set to {to!r} by {event.actor.kind}:{event.actor.id}"
        return None
    if event.type == EVENT_EDGE_ADDED:
        edge_id = payload.get("edge_id")
        if not _is_id(edge_id) or edge_id in state.edges:
            return "edge_id is missing or already used"
        edge_type = payload.get("type")
        if edge_type not in EDGE_TYPES:
            return "type is not an edge type"
        src = state.nodes.get(payload.get("src")) if _is_id(payload.get("src")) else None
        dst = state.nodes.get(payload.get("dst")) if _is_id(payload.get("dst")) else None
        if src is None or dst is None:
            return "edge endpoint is not a known node"
        if src.kind not in _EDGE_SRC_KINDS.get(edge_type, NODE_KINDS):
            return f"{edge_type} edge cannot start at a {src.kind} node"
        if dst.kind not in _EDGE_DST_KINDS.get(edge_type, NODE_KINDS):
            return f"{edge_type} edge cannot end at a {dst.kind} node"
        return None if isinstance(data, dict) else "data is not an object"
    return f"not a graph event type: {event.type!r}"


def _fold(state: GraphState, event: Event) -> None:
    """Fold one event into ``state``, or record it as skipped. Never raises."""
    problem = _problem(state, event)
    if problem is not None:
        state.skipped.append(SkippedEvent.of(event, problem))
        return
    if event.type == EVENT_NODE_ADDED:
        state.creators[event.payload["node_id"]] = event.actor
    _reduce(state, event.type, event.payload, event.ts)


def _reduce(state: GraphState, event_type: str, payload: dict[str, Any], ts: str) -> None:
    """Fold one graph event into ``state`` in place."""
    if event_type == EVENT_NODE_ADDED:
        node_id = payload["node_id"]
        state.nodes[node_id] = GraphNode(
            id=node_id,
            kind=payload["kind"],
            status=payload["status"],
            title=payload.get("title"),
            payload=payload.get("data", {}),
            created=ts,
            updated=ts,
        )
    elif event_type == EVENT_NODE_STATUS_CHANGED:
        node = state.nodes[payload["node_id"]]
        node.status = payload["to"]
        node.updated = ts
    elif event_type == EVENT_EDGE_ADDED:
        edge_id = payload["edge_id"]
        state.edges[edge_id] = Edge(
            id=edge_id,
            type=payload["type"],
            src=payload["src"],
            dst=payload["dst"],
            payload=payload.get("data", {}),
            created=ts,
        )


class WorkGraph:
    """The work graph for a single project, projected from the event log."""

    def __init__(self, log: EventLog, project_id: str) -> None:
        self._log = log
        self._project_id = project_id
        self._state = GraphState()
        self._last_seq = 0
        self.refresh()

    @property
    def project_id(self) -> str:
        return self._project_id

    # -- projection ---------------------------------------------------------

    def rebuild(self) -> GraphState:
        """Return a fresh projection built from the whole log (independent of live state)."""
        state = GraphState()
        for event in self._log.iter_events(project_id=self._project_id, types=_GRAPH_TYPES):
            _fold(state, event)
        return state

    def refresh(self) -> None:
        """Reload live state from the log.

        ``_last_seq`` comes from the events actually folded, never from a separate
        ``latest_seq()`` read that could jump past events appended since the rebuild.
        """
        state = GraphState()
        last_seq = 0
        for event in self._log.iter_events(project_id=self._project_id, types=_GRAPH_TYPES):
            _fold(state, event)
            last_seq = event.seq
        self._state = state
        self._last_seq = last_seq

    def snapshot(self) -> GraphState:
        """Return the current live projection."""
        return self._state

    def to_dict(self) -> dict[str, Any]:
        """Return the live graph as plain JSON-ready data."""
        return self._state.to_dict()

    # -- mutations ----------------------------------------------------------

    def add_node(
        self,
        kind: str,
        actor: Actor,
        *,
        title: str | None = None,
        status: str = "pending",
        data: dict[str, Any] | None = None,
        node_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> GraphNode:
        """Add a node and return it.

        An explicit ``node_id`` that already exists is rejected with
        :class:`DuplicateNodeId`, unless the same ``idempotency_key`` created it.
        """
        if kind not in NODE_KINDS:
            raise ValueError(f"node kind must be one of {sorted(NODE_KINDS)}, got {kind!r}")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}, got {status!r}")
        with self._log.transaction() as txn:
            self._catch_up(txn)
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    expected = {
                        "kind": kind,
                        "status": status,
                        "title": title,
                        "data": data or {},
                        "node_id": node_id,
                    }
                    compare = ["kind", "status", "title", "data"]
                    if node_id is not None:
                        compare.append("node_id")
                    match_or_conflict(
                        existing,
                        EVENT_NODE_ADDED,
                        expected,
                        compare,
                        project_id=self._project_id,
                        actor=actor,
                    )
                    return self._state.nodes[existing.payload["node_id"]]
            nid = node_id or new_node_id()
            if nid in self._state.nodes:
                raise DuplicateNodeId(f"node id already exists: {nid!r}")
            payload = {
                "node_id": nid,
                "kind": kind,
                "status": status,
                "title": title,
                "data": data or {},
            }
            event = txn.append(
                EVENT_NODE_ADDED,
                actor,
                payload,
                project_id=self._project_id,
                idempotency_key=idempotency_key,
            )
        self._apply(event)
        return self._state.nodes[nid]

    def set_status(
        self, node_id: str, new_status: str, actor: Actor, *, idempotency_key: str | None = None
    ) -> GraphNode:
        """Move a node to ``new_status``, rejecting transitions the state machine forbids.

        A retry with the same ``idempotency_key`` returns the existing node without
        re-checking the transition, so an idempotent retry never raises.
        """
        if new_status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}, got {new_status!r}")
        with self._log.transaction() as txn:
            self._catch_up(txn)
            node = self._state.nodes.get(node_id)
            if node is None:
                raise KeyError(f"unknown node: {node_id!r}")
            if node.kind == "review":
                raise ProtectedNode(
                    f"node {node_id} is a review: it closes only through a human decision or "
                    "its requester's withdrawal (aar.hub.reviews), not the generic status API"
                )
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    match_or_conflict(
                        existing,
                        EVENT_NODE_STATUS_CHANGED,
                        {"node_id": node_id, "to": new_status},
                        ["node_id", "to"],
                        project_id=self._project_id,
                        actor=actor,
                    )
                    return node
            if new_status not in allowed_transitions(node.status):
                raise IllegalTransition(
                    f"cannot move node {node_id} from {node.status!r} to {new_status!r}"
                )
            event = txn.append(
                EVENT_NODE_STATUS_CHANGED,
                actor,
                {"node_id": node_id, "to": new_status},
                project_id=self._project_id,
                idempotency_key=idempotency_key,
            )
        self._apply(event)
        return self._state.nodes[node_id]

    def add_edge(
        self,
        edge_type: str,
        src: str,
        dst: str,
        actor: Actor,
        *,
        data: dict[str, Any] | None = None,
        edge_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> Edge:
        """Connect two existing nodes with a typed edge."""
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"edge type must be one of {sorted(EDGE_TYPES)}, got {edge_type!r}")
        with self._log.transaction() as txn:
            self._catch_up(txn)
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    expected = {
                        "type": edge_type,
                        "src": src,
                        "dst": dst,
                        "data": data or {},
                        "edge_id": edge_id,
                    }
                    compare = ["type", "src", "dst", "data"]
                    if edge_id is not None:
                        compare.append("edge_id")
                    match_or_conflict(
                        existing,
                        EVENT_EDGE_ADDED,
                        expected,
                        compare,
                        project_id=self._project_id,
                        actor=actor,
                    )
                    return self._state.edges[existing.payload["edge_id"]]
            src_node = self._state.nodes.get(src)
            dst_node = self._state.nodes.get(dst)
            if src_node is None:
                raise KeyError(f"unknown src node: {src!r}")
            if dst_node is None:
                raise KeyError(f"unknown dst node: {dst!r}")
            allowed_src = _EDGE_SRC_KINDS.get(edge_type)
            if allowed_src is not None and src_node.kind not in allowed_src:
                raise ValueError(
                    f"{edge_type!r} edge must start at a node of kind {sorted(allowed_src)}, "
                    f"not {src_node.kind!r}"
                )
            allowed_dst = _EDGE_DST_KINDS.get(edge_type)
            if allowed_dst is not None and dst_node.kind not in allowed_dst:
                raise ValueError(
                    f"{edge_type!r} edge must end at a node of kind {sorted(allowed_dst)}, "
                    f"not {dst_node.kind!r}"
                )
            eid = edge_id or new_edge_id()
            if eid in self._state.edges:
                raise DuplicateEdgeId(f"edge id already exists: {eid!r}")
            payload = {
                "edge_id": eid,
                "type": edge_type,
                "src": src,
                "dst": dst,
                "data": data or {},
            }
            event = txn.append(
                EVENT_EDGE_ADDED,
                actor,
                payload,
                project_id=self._project_id,
                idempotency_key=idempotency_key,
            )
        self._apply(event)
        return self._state.edges[eid]

    # -- queries ------------------------------------------------------------

    def node(self, node_id: str) -> GraphNode | None:
        """Return the node with this id, or ``None``."""
        return self._state.nodes.get(node_id)

    def nodes(self) -> list[GraphNode]:
        """Return all nodes, ordered by creation."""
        return sorted(self._state.nodes.values(), key=lambda n: (n.created or "", n.id))

    def edges(self) -> list[Edge]:
        """Return all edges, ordered by creation."""
        return sorted(self._state.edges.values(), key=lambda e: (e.created or "", e.id))

    # -- composition --------------------------------------------------------

    def set_status_in(self, txn: Transaction, node_id: str, new_status: str, actor: Actor) -> Event:
        """Append a validated status change inside a caller-owned transaction.

        For callers that must change a node's status atomically with events of their
        own (a review decision moves its review node in the same ``BEGIN IMMEDIATE``
        transaction that records the decision). The transition is checked against the
        state as of ``txn``. Idempotency is the caller's job. Use a ``WorkGraph`` built
        for this one transaction: if the caller rolls back, its live state is stale.
        """
        if txn._log is not self._log:
            raise ValueError("transaction belongs to a different event log")
        if new_status not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}, got {new_status!r}")
        self._catch_up(txn)
        node = self._state.nodes.get(node_id)
        if node is None:
            raise KeyError(f"unknown node: {node_id!r}")
        if new_status not in allowed_transitions(node.status):
            raise IllegalTransition(
                f"cannot move node {node_id} from {node.status!r} to {new_status!r}"
            )
        if node.kind == "review" and not review_status_change_allowed(
            actor, self._state.creators.get(node_id), node.status, new_status
        ):
            raise ProtectedNode(
                f"{actor.kind}:{actor.id} may not move review {node_id} to {new_status!r}: "
                "only a human decides, and only the requester may withdraw"
            )
        event = txn.append(
            EVENT_NODE_STATUS_CHANGED,
            actor,
            {"node_id": node_id, "to": new_status},
            project_id=self._project_id,
        )
        self._apply(event)
        return event

    def skipped(self) -> tuple[SkippedEvent, ...]:
        """Events the live projection could not use (malformed or forged)."""
        return tuple(self._state.skipped)

    # -- internals ----------------------------------------------------------

    def _catch_up(self, txn: Transaction) -> None:
        for event in txn.read(self._last_seq, project_id=self._project_id, types=_GRAPH_TYPES):
            if event.seq > self._last_seq:
                _fold(self._state, event)
                self._last_seq = event.seq

    def _apply(self, event: Event) -> None:
        if event.seq > self._last_seq:
            _fold(self._state, event)
            self._last_seq = event.seq
