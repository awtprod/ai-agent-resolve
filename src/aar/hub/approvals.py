"""Human approval tokens (DESIGN section 5).

Switching databases, loading or writing a production project, creating or restoring
cloud projects, and forcing a switch past a human all need a human's approval,
recorded in the event log. A token is that approval: a human grants it for one scope
and one target, with a short lifetime and a reason. It can be revoked, and it can be
consumed exactly once.

Tokens are capabilities, not secrets: their ids appear in the event log and in the
read-only MCP listing. What makes one safe to see is that it is bound to a scope and
a target, expires, and is single-use, so using it can only do the one thing a human
approved.

``consume`` reads the token's history and appends the outcome inside one
``BEGIN IMMEDIATE`` transaction, so two processes can never both consume a token. As a
second, independent guard the consume event's idempotency key is unique per token,
so the database itself refuses a second consume row. Every step is an event, refused
consumes included.

This module is only the mechanism. Nothing here acts on Resolve.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from aar.hub.common import (
    Clock,
    InvalidInput,
    actor_dict,
    clean_text,
    iso,
    parse_ts,
    require_human,
    utcnow,
)
from aar.hub.events import (
    Actor,
    Event,
    EventLog,
    IdempotencyConflict,
    Transaction,
    match_or_conflict,
)

EVENT_GRANTED = "approval.granted"
EVENT_REVOKED = "approval.revoked"
EVENT_CONSUMED = "approval.consumed"
EVENT_CONSUME_REFUSED = "approval.consume_refused"
_TOKEN_TYPES = (EVENT_GRANTED, EVENT_REVOKED, EVENT_CONSUMED)

MIN_TTL = timedelta(seconds=1)
MAX_TTL = timedelta(hours=24)
MAX_TARGET = 200
MAX_REASON = 1000
MAX_TOKEN_ID = 64


class ApprovalScope(StrEnum):
    """What a token approves."""

    DB_SWITCH = "db.switch"
    PROJECT_LOAD = "project.load"
    PROJECT_WRITE = "project.write"
    CLOUD_CREATE = "cloud.create"
    CLOUD_RESTORE = "cloud.restore"
    SWITCH_FORCE = "switch.force"


class ApprovalError(Exception):
    """Base class for approval failures. ``code`` is a stable machine-readable reason."""

    code = "approval_error"

    def __init__(self, token_id: str, message: str) -> None:
        super().__init__(message)
        self.token_id = token_id


class ApprovalNotFound(ApprovalError, LookupError):
    code = "not_found"


class ApprovalExpired(ApprovalError):
    code = "expired"


class ApprovalRevoked(ApprovalError):
    code = "revoked"


class ApprovalAlreadyConsumed(ApprovalError):
    code = "already_consumed"


class ApprovalScopeMismatch(ApprovalError):
    code = "wrong_scope"


class ApprovalTargetMismatch(ApprovalError):
    code = "wrong_target"


@dataclass(frozen=True, slots=True)
class Approval:
    """A token's metadata and history, projected from its events."""

    token_id: str
    scope: ApprovalScope
    target: str
    reason: str
    granted_by: Actor
    granted_at: str
    expires_at: str
    revoked_by: Actor | None = None
    revoked_at: str | None = None
    revoke_reason: str | None = None
    consumed_by: Actor | None = None
    consumed_at: str | None = None

    def state(self, now: datetime) -> str:
        """Return ``consumed``, ``revoked``, ``expired`` or ``active`` as of ``now``."""
        if self.consumed_at is not None:
            return "consumed"
        if self.revoked_at is not None:
            return "revoked"
        if now >= parse_ts(self.expires_at):
            return "expired"
        return "active"

    def to_dict(self, now: datetime) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "scope": self.scope.value,
            "target": self.target,
            "reason": self.reason,
            "state": self.state(now),
            "granted_by": actor_dict(self.granted_by),
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revoked_by": actor_dict(self.revoked_by) if self.revoked_by else None,
            "revoked_at": self.revoked_at,
            "revoke_reason": self.revoke_reason,
            "consumed_by": actor_dict(self.consumed_by) if self.consumed_by else None,
            "consumed_at": self.consumed_at,
        }


# -- projection --------------------------------------------------------------


def _fold(events: list[Event]) -> dict[str, Approval]:
    tokens: dict[str, Approval] = {}
    for event in events:
        payload = event.payload
        token_id = payload.get("token_id")
        if not isinstance(token_id, str):
            continue
        if event.type == EVENT_GRANTED:
            # A token id is granted once, and only by a human: skip regrants, junk, and
            # grants an agent or node appended straight to the log, around grant().
            if token_id in tokens or event.actor.kind != "human":
                continue
            if not _grant_well_formed(payload):
                continue
            tokens[token_id] = Approval(
                token_id=token_id,
                scope=ApprovalScope(payload["scope"]),
                target=payload["target"],
                reason=payload["reason"],
                granted_by=event.actor,
                granted_at=payload["granted_at"],
                expires_at=payload["expires_at"],
            )
            continue
        token = tokens.get(token_id)
        if token is None:
            continue
        if event.type == EVENT_REVOKED and token.revoked_at is None:
            tokens[token_id] = replace(
                token,
                revoked_by=event.actor,
                revoked_at=payload.get("at", event.ts),
                revoke_reason=payload.get("reason"),
            )
        elif event.type == EVENT_CONSUMED and token.consumed_at is None:
            tokens[token_id] = replace(
                token, consumed_by=event.actor, consumed_at=payload.get("at", event.ts)
            )
    return tokens


def _grant_well_formed(payload: dict[str, Any]) -> bool:
    """Whether a grant event has the shape :func:`grant` writes (the log is shared)."""
    if payload.get("scope") not in {s.value for s in ApprovalScope}:
        return False
    fields = ("target", "reason", "granted_at", "expires_at")
    if not all(isinstance(payload.get(name), str) for name in fields):
        return False
    try:
        parse_ts(payload["expires_at"])
    except ValueError:
        return False
    return True


def _token_in(txn: Transaction, token_id: str) -> Approval | None:
    return _fold(txn.read(0, types=_TOKEN_TYPES)).get(token_id)


def list_approvals(log: EventLog) -> list[Approval]:
    """Return every token, newest grant first."""
    tokens = _fold(list(log.iter_events(types=_TOKEN_TYPES)))
    return sorted(tokens.values(), key=lambda t: (t.granted_at, t.token_id), reverse=True)


def get_approval(log: EventLog, token_id: str) -> Approval | None:
    """Return one token, or ``None``."""
    return _fold(list(log.iter_events(types=_TOKEN_TYPES))).get(token_id)


# -- mutations ---------------------------------------------------------------


def _ttl(ttl: timedelta | float | int) -> timedelta:
    if isinstance(ttl, bool):
        raise InvalidInput("ttl", "must be a duration")
    if isinstance(ttl, (int, float)):
        ttl = timedelta(seconds=ttl)
    if not isinstance(ttl, timedelta):
        raise InvalidInput("ttl", f"must be a duration, got {type(ttl).__name__}")
    if ttl < MIN_TTL or ttl > MAX_TTL:
        raise InvalidInput("ttl", f"must be between {MIN_TTL} and {MAX_TTL}")
    return ttl


def _scope(scope: ApprovalScope | str) -> ApprovalScope:
    try:
        return ApprovalScope(scope)
    except ValueError as exc:
        allowed = [s.value for s in ApprovalScope]
        raise InvalidInput("scope", f"must be one of {allowed}, got {scope!r}") from exc


def _token_id(token_id: Any) -> str:
    return clean_text(token_id, "token_id", max_len=MAX_TOKEN_ID)


def grant(
    log: EventLog,
    scope: ApprovalScope | str,
    target: str,
    ttl: timedelta | float,
    reason: str,
    by: Actor,
    *,
    idempotency_key: str | None = None,
    clock: Clock = utcnow,
) -> Approval:
    """Grant a single-use token for ``scope`` on ``target`` that expires after ``ttl``.

    Only a human may grant. A retry under the same ``idempotency_key`` returns the
    token the first call granted.
    """
    require_human(by, "grant an approval token")
    scope = _scope(scope)
    target = clean_text(target, "target", max_len=MAX_TARGET)
    ttl = _ttl(ttl)
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    request = {
        "scope": scope.value,
        "target": target,
        "ttl_seconds": ttl.total_seconds(),
        "reason": reason,
    }
    with log.transaction() as txn:
        if idempotency_key is not None:
            existing = txn.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                match_or_conflict(existing, EVENT_GRANTED, request, request.keys())
                if existing.actor != by:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used by another actor"
                    )
                token = _token_in(txn, existing.payload["token_id"])
                if token is None:  # pragma: no cover - the grant event is the token
                    raise ApprovalNotFound(existing.payload["token_id"], "granted token vanished")
                return token
        now = clock()
        token_id = "apv_" + secrets.token_hex(16)
        payload = {
            **request,
            "token_id": token_id,
            "granted_at": iso(now),
            "expires_at": iso(now + ttl),
        }
        txn.append(EVENT_GRANTED, by, payload, idempotency_key=idempotency_key)
        token = _token_in(txn, token_id)
    if token is None:  # pragma: no cover - just appended
        raise ApprovalNotFound(token_id, "granted token vanished")
    return token


def revoke(
    log: EventLog,
    token_id: str,
    reason: str,
    by: Actor,
    *,
    idempotency_key: str | None = None,
    clock: Clock = utcnow,
) -> Approval:
    """Revoke a token that has not been consumed or revoked. Only a human may revoke."""
    require_human(by, "revoke an approval token")
    token_id = _token_id(token_id)
    reason = clean_text(reason, "reason", max_len=MAX_REASON, multiline=True)
    request = {"token_id": token_id, "reason": reason}
    with log.transaction() as txn:
        if idempotency_key is not None:
            existing = txn.find_by_idempotency_key(idempotency_key)
            if existing is not None:
                match_or_conflict(existing, EVENT_REVOKED, request, request.keys())
                if existing.actor != by:
                    raise IdempotencyConflict(
                        f"idempotency key {idempotency_key!r} was already used by another actor"
                    )
                token = _token_in(txn, token_id)
                if token is None:  # pragma: no cover - a revoke implies a grant
                    raise ApprovalNotFound(token_id, f"no approval token {token_id!r}")
                return token
        token = _token_in(txn, token_id)
        if token is None:
            raise ApprovalNotFound(token_id, f"no approval token {token_id!r}")
        if token.consumed_at is not None:
            raise ApprovalAlreadyConsumed(
                token_id, f"token {token_id} was already consumed at {token.consumed_at}"
            )
        if token.revoked_at is not None:
            raise ApprovalRevoked(
                token_id, f"token {token_id} was already revoked at {token.revoked_at}"
            )
        txn.append(
            EVENT_REVOKED,
            by,
            {**request, "at": iso(clock())},
            idempotency_key=idempotency_key,
        )
        token = _token_in(txn, token_id)
    if token is None:  # pragma: no cover - checked above
        raise ApprovalNotFound(token_id, f"no approval token {token_id!r}")
    return token


def _refusal(
    token: Approval | None,
    token_id: str,
    scope: ApprovalScope,
    target: str,
    now: datetime,
) -> ApprovalError | None:
    if token is None:
        return ApprovalNotFound(token_id, f"no approval token {token_id!r}")
    if token.consumed_at is not None:
        who = token.consumed_by
        by = f" by {who.kind}:{who.id}" if who else ""
        return ApprovalAlreadyConsumed(
            token_id, f"token {token_id} was already consumed{by} at {token.consumed_at}"
        )
    if token.revoked_at is not None:
        return ApprovalRevoked(token_id, f"token {token_id} was revoked at {token.revoked_at}")
    if now >= parse_ts(token.expires_at):
        return ApprovalExpired(token_id, f"token {token_id} expired at {token.expires_at}")
    if token.scope is not scope:
        return ApprovalScopeMismatch(
            token_id, f"token {token_id} approves {token.scope.value!r}, not {scope.value!r}"
        )
    if token.target != target:
        return ApprovalTargetMismatch(
            token_id, f"token {token_id} approves target {token.target!r}, not {target!r}"
        )
    return None


def consume(
    log: EventLog,
    token_id: str,
    scope: ApprovalScope | str,
    target: str,
    actor: Actor,
    *,
    clock: Clock = utcnow,
) -> Approval:
    """Use a token once for ``scope`` on ``target``, atomically, and return it.

    Raises :class:`ApprovalNotFound`, :class:`ApprovalExpired`,
    :class:`ApprovalRevoked`, :class:`ApprovalAlreadyConsumed`,
    :class:`ApprovalScopeMismatch` or :class:`ApprovalTargetMismatch`. A refused
    attempt is itself recorded as an ``approval.consume_refused`` event.
    """
    token_id = _token_id(token_id)
    scope = _scope(scope)
    target = clean_text(target, "target", max_len=MAX_TARGET)
    with log.transaction() as txn:
        now = clock()
        token = _token_in(txn, token_id)
        refusal = _refusal(token, token_id, scope, target, now)
        if refusal is None:
            try:
                txn.append(
                    EVENT_CONSUMED,
                    actor,
                    {
                        "token_id": token_id,
                        "scope": scope.value,
                        "target": target,
                        "at": iso(now),
                        "attempt": secrets.token_hex(8),
                    },
                    # One consume row per token, enforced by the database's unique key.
                    idempotency_key=f"approval-consume:{token_id}",
                )
            except IdempotencyConflict:
                refusal = ApprovalAlreadyConsumed(
                    token_id, f"token {token_id} was already consumed"
                )
        if refusal is not None:
            txn.append(
                EVENT_CONSUME_REFUSED,
                actor,
                {
                    "token_id": token_id,
                    "scope": scope.value,
                    "target": target,
                    "code": refusal.code,
                    "at": iso(now),
                },
            )
        else:
            token = _token_in(txn, token_id)
    if refusal is not None:
        raise refusal
    if token is None:  # pragma: no cover - consumed above
        raise ApprovalNotFound(token_id, f"no approval token {token_id!r}")
    return token
