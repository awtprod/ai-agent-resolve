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
    """The projected state of one project's work graph."""

    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, Edge] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges.values()],
        }


def allowed_transitions(status: str) -> frozenset[str]:
    """Return the statuses reachable from ``status`` in one step."""
    return _TRANSITIONS.get(status, frozenset())


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
            _reduce(state, event.type, event.payload, event.ts)
        return state

    def refresh(self) -> None:
        """Reload live state from the log.

        ``_last_seq`` comes from the events actually folded, never from a separate
        ``latest_seq()`` read that could jump past events appended since the rebuild.
        """
        state = GraphState()
        last_seq = 0
        for event in self._log.iter_events(project_id=self._project_id, types=_GRAPH_TYPES):
            _reduce(state, event.type, event.payload, event.ts)
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
                    match_or_conflict(existing, EVENT_NODE_ADDED, expected, compare)
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
            if idempotency_key is not None:
                existing = txn.find_by_idempotency_key(idempotency_key)
                if existing is not None:
                    match_or_conflict(
                        existing,
                        EVENT_NODE_STATUS_CHANGED,
                        {"node_id": node_id, "to": new_status},
                        ["node_id", "to"],
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
                    match_or_conflict(existing, EVENT_EDGE_ADDED, expected, compare)
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

    # -- internals ----------------------------------------------------------

    def _catch_up(self, txn: Transaction) -> None:
        for event in txn.read(self._last_seq, project_id=self._project_id, types=_GRAPH_TYPES):
            if event.seq > self._last_seq:
                _reduce(self._state, event.type, event.payload, event.ts)
                self._last_seq = event.seq

    def _apply(self, event: Event) -> None:
        if event.seq > self._last_seq:
            _reduce(self._state, event.type, event.payload, event.ts)
            self._last_seq = event.seq
