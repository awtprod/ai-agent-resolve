"""Leases over the shared Resolve instance, its projects and their timelines.

Scopes:

- ``instance``: the right to switch the instance's current project or timeline.
  Key: ``(instance_id,)``.
- ``project``: exclusive use of a project. Key: ``(project_id,)``, the project's
  ``Project.GetUniqueId()``; never its name, which changes on rename and is shared by
  same-named projects in other folders.
- ``timeline``: exclusive use of one timeline. Key: ``(project_id, timeline_id)``, with the
  timeline's ``Timeline.GetUniqueId()``.

Conflict rules: two leases conflict when they have the same scope and key, or when one is a
project lease and the other a timeline lease in that project. Leases held by the same holder
never conflict with each other.

Precedence: a human always wins. A human acquiring over an agent's lease revokes the agent's
lease. An agent can never take or preempt a human's lease, and agents never preempt each
other. A human only displaces another human with ``force=True`` (the console's override).
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum

Clock = Callable[[], float]
"""Monotonic seconds. Injected so tests never sleep."""

_ENDED_MEMORY = 1024


class Scope(StrEnum):
    INSTANCE = "instance"
    PROJECT = "project"
    TIMELINE = "timeline"


_KEY_ARITY = {Scope.INSTANCE: 1, Scope.PROJECT: 1, Scope.TIMELINE: 2}


class HolderKind(StrEnum):
    HUMAN = "human"
    AGENT = "agent"


@dataclass(frozen=True)
class Holder:
    kind: HolderKind
    id: str

    @classmethod
    def human(cls, holder_id: str) -> Holder:
        return cls(HolderKind.HUMAN, holder_id)

    @classmethod
    def agent(cls, holder_id: str) -> Holder:
        return cls(HolderKind.AGENT, holder_id)

    @property
    def is_human(self) -> bool:
        return self.kind is HolderKind.HUMAN

    def __str__(self) -> str:
        return f"{self.kind.value}:{self.id}"


@dataclass(frozen=True)
class Lease:
    id: str
    scope: Scope
    key: tuple[str, ...]
    holder: Holder
    acquired_at: float
    expires_at: float | None
    ttl: float | None

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at

    def describe(self) -> str:
        return f"{self.scope.value} lease on {'/'.join(self.key)} held by {self.holder}"


@dataclass(frozen=True)
class LeaseEvent:
    """Emitted on grant, renew, release, expiry and preemption."""

    kind: str
    at: float
    lease: Lease
    by: Holder | None = None


class LeaseError(Exception):
    """Base class for lease failures."""


class LeaseConflict(LeaseError):
    def __init__(self, scope: Scope, key: tuple[str, ...], blocking: tuple[Lease, ...]) -> None:
        detail = "; ".join(lease.describe() for lease in blocking)
        super().__init__(f"cannot lease {scope.value} {'/'.join(key)}: {detail}")
        self.scope = scope
        self.key = key
        self.blocking = blocking


class LeaseNotHeld(LeaseError):
    def __init__(self, lease_id: str, reason: str) -> None:
        super().__init__(f"lease {lease_id} is not held: {reason}")
        self.lease_id = lease_id
        self.reason = reason


def instance_key(instance_id: str = "default") -> tuple[str]:
    return (instance_id,)


def project_key(project_id: str) -> tuple[str]:
    """Key of a project lease: the project's unique id."""
    return (project_id,)


def timeline_key(project_id: str, timeline_id: str) -> tuple[str, str]:
    """Key of a timeline lease: the project's and the timeline's unique ids."""
    return (project_id, timeline_id)


def leases_conflict(a_scope: Scope, a_key: tuple[str, ...], b: Lease) -> bool:
    """Whether a lease on ``(a_scope, a_key)`` conflicts with lease ``b`` (holders ignored)."""
    if a_scope is b.scope:
        return a_key == b.key
    pair = {a_scope, b.scope}
    if pair == {Scope.PROJECT, Scope.TIMELINE}:
        return a_key[0] == b.key[0]
    return False


class LeaseManager:
    """In-memory lease table with lazy expiry. Not persisted; the hub mirrors it later."""

    def __init__(
        self,
        clock: Clock = time.monotonic,
        sink: Callable[[LeaseEvent], None] | None = None,
    ) -> None:
        self._clock = clock
        self._sink = sink
        self._leases: dict[str, Lease] = {}
        self._ended: OrderedDict[str, str] = OrderedDict()

    # -- internals -------------------------------------------------------------------------

    def _emit(self, kind: str, at: float, lease: Lease, by: Holder | None = None) -> None:
        if self._sink is not None:
            self._sink(LeaseEvent(kind, at, lease, by))

    def _end(self, lease: Lease, reason: str) -> None:
        del self._leases[lease.id]
        self._ended[lease.id] = reason
        while len(self._ended) > _ENDED_MEMORY:
            self._ended.popitem(last=False)

    def _purge(self, now: float) -> None:
        for lease in [lease for lease in self._leases.values() if lease.expired(now)]:
            self._end(lease, "expired")
            self._emit("expired", now, lease)

    @staticmethod
    def _validate(scope: Scope, key: tuple[str, ...]) -> None:
        if len(key) != _KEY_ARITY[scope] or not all(isinstance(k, str) and k for k in key):
            raise ValueError(f"{scope.value} lease key must be {_KEY_ARITY[scope]} non-empty str")

    @staticmethod
    def _expiry(holder: Holder, ttl: float | None, now: float) -> float | None:
        if ttl is None:
            if not holder.is_human:
                raise ValueError("agent leases need a positive ttl")
            return None
        if ttl <= 0:
            raise ValueError("ttl must be positive")
        return now + ttl

    def _held(self, lease_id: str, holder: Holder, now: float) -> Lease:
        self._purge(now)
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotHeld(lease_id, self._ended.get(lease_id, "unknown lease"))
        if lease.holder != holder:
            raise LeaseNotHeld(lease_id, f"held by {lease.holder}, not {holder}")
        return lease

    # -- queries ---------------------------------------------------------------------------

    def active(self) -> list[Lease]:
        """Unexpired leases, oldest first."""
        self._purge(self._clock())
        return sorted(self._leases.values(), key=lambda lease: (lease.acquired_at, lease.id))

    def get(self, lease_id: str) -> Lease | None:
        self._purge(self._clock())
        return self._leases.get(lease_id)

    def conflicts_for(self, holder: Holder, scope: Scope, key: tuple[str, ...]) -> list[Lease]:
        """Leases held by others that would conflict with ``holder`` leasing ``(scope, key)``."""
        self._validate(scope, key)
        return [
            lease
            for lease in self.active()
            if lease.holder != holder and leases_conflict(scope, key, lease)
        ]

    # -- mutations -------------------------------------------------------------------------

    def acquire(
        self,
        holder: Holder,
        scope: Scope,
        key: tuple[str, ...],
        ttl: float | None,
        *,
        force: bool = False,
    ) -> Lease:
        """Grant a lease, renewing it if ``holder`` already has this exact one.

        Raises ``LeaseConflict`` when a conflicting lease cannot be displaced.
        """
        self._validate(scope, key)
        if force and not holder.is_human:
            raise ValueError("only a human can force a lease")
        now = self._clock()
        expires_at = self._expiry(holder, ttl, now)
        self._purge(now)

        for lease in self._leases.values():
            if lease.holder == holder and lease.scope is scope and lease.key == key:
                renewed = replace(lease, expires_at=expires_at, ttl=ttl)
                self._leases[lease.id] = renewed
                self._emit("renewed", now, renewed)
                return renewed

        blocking = self.conflicts_for(holder, scope, key)
        humans = tuple(lease for lease in blocking if lease.holder.is_human)
        agents = tuple(lease for lease in blocking if not lease.holder.is_human)
        if not holder.is_human and blocking:
            raise LeaseConflict(scope, key, tuple(blocking))
        if humans and not force:
            raise LeaseConflict(scope, key, humans)
        # Only humans get here; check everything before revoking anything.
        for lease in (*agents, *humans):
            self._end(lease, f"preempted by {holder}")
            self._emit("preempted", now, lease, by=holder)

        lease = Lease(uuid.uuid4().hex, scope, key, holder, now, expires_at, ttl)
        self._leases[lease.id] = lease
        self._emit("granted", now, lease)
        return lease

    def renew(self, lease_id: str, holder: Holder, ttl: float | None = None) -> Lease:
        """Extend a lease. ``ttl`` defaults to the lease's original duration."""
        now = self._clock()
        lease = self._held(lease_id, holder, now)
        if ttl is None:
            ttl = lease.ttl
        renewed = replace(lease, expires_at=self._expiry(holder, ttl, now), ttl=ttl)
        self._leases[lease_id] = renewed
        self._emit("renewed", now, renewed)
        return renewed

    def release(self, lease_id: str, holder: Holder) -> None:
        """Give a lease back. Raises ``LeaseNotHeld`` if expired, preempted or not yours."""
        now = self._clock()
        lease = self._held(lease_id, holder, now)
        self._end(lease, "released")
        self._emit("released", now, lease)
