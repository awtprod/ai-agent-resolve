"""Node enrolment and bearer-token authentication for the hub's node API (S15a).

A node on another host authenticates to ``aar hub nodeapi`` with an enrolled bearer token.
Identity is the token, not reachability (DESIGN S15 decision 3): the hub is tailnet-only and
the token is a second factor. The hub stores only ``sha256(token)``; the token itself is
printed once at enrolment (:func:`enroll`) and is never stored, logged or recoverable.

Enrolment is an append-only fold over two event types, so a revocation takes effect on the
very next request (no cache may outlive it) and the whole history is auditable:

- ``node.enrolled`` ``{node_id, token_sha256, at}`` -- a node gains a token.
- ``node.revoked`` ``{node_id, at, reason}`` -- its current token stops working.

The current token of a node is the ``token_sha256`` of its most recent ``node.enrolled``,
unless a later ``node.revoked`` cleared it. ``--rotate`` revokes the old token and enrols a
new one in one transaction, so the old token stops working the instant the new one is minted.
Lookups compare the presented token's hash to the stored hash with :func:`hmac.compare_digest`.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Iterable
from dataclasses import dataclass

from aar.hub.common import Clock, InvalidInput, clean_text, iso, utcnow
from aar.hub.events import Actor, Event, EventLog, Transaction
from aar.hub.leaseboard import MAX_NODE_ID

EVENT_ENROLLED = "node.enrolled"
EVENT_REVOKED = "node.revoked"
_NODE_TYPES = (EVENT_ENROLLED, EVENT_REVOKED)

TOKEN_BYTES = 32
"""Entropy for :func:`secrets.token_urlsafe` (decision 3); ~43 url-safe characters."""

HASH_PREFIX_LEN = 12
"""How many hex characters of a token's sha256 the listing shows (never the whole hash)."""

_SHA256_HEX_LEN = 64


class NodeError(ValueError):
    """A node-enrolment request the hub refused. ``code`` is a stable machine-readable string."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def clean_node_id(node_id: object) -> str:
    """Validate a node id with the existing node-id rules (:func:`aar.hub.common.clean_text`).

    Mirrors :func:`aar.node.feed._clean_node_id`: a non-empty, control-free string of at most
    :data:`MAX_NODE_ID` characters that is already stripped, so the enrolled id, the token's
    node id and the transport actor are byte-identical.
    """
    try:
        cleaned = clean_text(node_id, "node_id", max_len=MAX_NODE_ID)
    except InvalidInput as exc:
        raise NodeError("invalid_node_id", str(exc)) from exc
    if cleaned != node_id:
        raise NodeError(
            "invalid_node_id", f"node_id must have no surrounding whitespace: {node_id!r}"
        )
    return cleaned


def token_sha256(token: str) -> str:
    """The hex sha256 of a bearer token (the only form the hub ever stores)."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class NodeRecord:
    """One enrolled (or once-enrolled) node, for the listing. Never carries a token or a hash."""

    node_id: str
    enrolled_at: str | None
    revoked_at: str | None
    hash_prefix: str
    active: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "enrolled_at": self.enrolled_at,
            "revoked_at": self.revoked_at,
            "hash_prefix": self.hash_prefix,
            "active": self.active,
        }


def _fold(events: Iterable[Event]) -> dict[str, dict[str, object]]:
    """Current enrolment state per node id, in log order. Malformed events are skipped.

    ``sha`` is the most recent enrolment's hash (for display and, when ``active``, for the
    constant-time compare). ``active`` is false once a revocation followed that enrolment.
    """
    state: dict[str, dict[str, object]] = {}
    for event in events:
        payload = event.payload
        if not isinstance(payload, dict):
            continue
        node_id = payload.get("node_id")
        if not isinstance(node_id, str) or not node_id:
            continue
        if event.type == EVENT_ENROLLED:
            sha = payload.get("token_sha256")
            if not isinstance(sha, str) or len(sha) != _SHA256_HEX_LEN:
                continue
            state[node_id] = {
                "enrolled_at": payload.get("at"),
                "revoked_at": None,
                "sha": sha,
                "active": True,
            }
        elif node_id in state:  # EVENT_REVOKED, for a node we have seen enrolled
            state[node_id]["revoked_at"] = payload.get("at")
            state[node_id]["active"] = False
    return state


def enroll(
    log: EventLog,
    node_id: str,
    actor: Actor,
    *,
    rotate: bool = False,
    clock: Clock = utcnow,
) -> tuple[str, str]:
    """Enrol ``node_id`` and return ``(node_id, token)``. The token is shown once, here only.

    Only a human may enrol (``actor.kind == "human"``). A node that already has an active token
    is refused with ``already_enrolled`` unless ``rotate`` is set, in which case the old token is
    revoked and the new one enrolled in a single transaction. The token is generated with
    :func:`secrets.token_urlsafe` and only its sha256 is written; the token is never stored.
    """
    node_id = clean_node_id(node_id)
    if not isinstance(actor, Actor) or actor.kind != "human":
        kind = actor.kind if isinstance(actor, Actor) else type(actor).__name__
        raise NodeError("actor_not_human", f"only a human may enrol a node, not {kind}")
    token = secrets.token_urlsafe(TOKEN_BYTES)
    sha = token_sha256(token)
    now = iso(clock())
    with log.transaction() as txn:
        current = _fold(txn.read(types=_NODE_TYPES)).get(node_id)
        if current is not None and current["active"]:
            if not rotate:
                raise NodeError(
                    "already_enrolled",
                    f"node {node_id!r} is already enrolled; use --rotate to replace its token",
                )
            txn.append(EVENT_REVOKED, actor, {"node_id": node_id, "at": now, "reason": "rotate"})
        txn.append(EVENT_ENROLLED, actor, {"node_id": node_id, "token_sha256": sha, "at": now})
    return node_id, token


def revoke(
    log: EventLog,
    node_id: str,
    actor: Actor,
    *,
    reason: str = "revoke",
    clock: Clock = utcnow,
) -> None:
    """Revoke a node's current token. A node that is not enrolled is refused (``not_enrolled``)."""
    node_id = clean_node_id(node_id)
    if not isinstance(actor, Actor) or actor.kind != "human":
        kind = actor.kind if isinstance(actor, Actor) else type(actor).__name__
        raise NodeError("actor_not_human", f"only a human may revoke a node, not {kind}")
    now = iso(clock())
    with log.transaction() as txn:
        current = _fold(txn.read(types=_NODE_TYPES)).get(node_id)
        if current is None or not current["active"]:
            raise NodeError("not_enrolled", f"node {node_id!r} has no active token to revoke")
        txn.append(EVENT_REVOKED, actor, {"node_id": node_id, "at": now, "reason": reason})


def list_nodes(log: EventLog) -> list[NodeRecord]:
    """Every node ever enrolled, by node id, with enrol/revoke times and a hash prefix only."""
    state = _fold(log.iter_events(types=_NODE_TYPES))
    records = []
    for node_id in sorted(state):
        info = state[node_id]
        sha = info["sha"]
        records.append(
            NodeRecord(
                node_id=node_id,
                enrolled_at=info["enrolled_at"],  # type: ignore[arg-type]
                revoked_at=info["revoked_at"],  # type: ignore[arg-type]
                hash_prefix=sha[:HASH_PREFIX_LEN] if isinstance(sha, str) else "",
                active=bool(info["active"]),
            )
        )
    return records


def authenticate(log: EventLog | Transaction, token: object) -> str | None:
    """Return the node id a bearer token authenticates, or ``None``.

    Folds the enrolment events fresh on every call (no cache may outlive a revocation) and
    compares hashes with :func:`hmac.compare_digest`. A revoked or rotated token never matches.
    """
    if not isinstance(token, str) or not token:
        return None
    presented = token_sha256(token)
    events = (
        log.read(types=_NODE_TYPES)
        if isinstance(log, Transaction)
        else log.iter_events(types=_NODE_TYPES)
    )
    for node_id, info in _fold(events).items():
        sha = info["sha"]
        if info["active"] and isinstance(sha, str) and hmac.compare_digest(presented, sha):
            return node_id
    return None
