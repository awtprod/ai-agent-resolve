"""Agent-facing collaboration reads on the node side.

When a hot op is refused or delayed because another holder has a lease, or because a
human is working, the agent needs to be told *who* holds *what* and *since when*, as
structured fields and as readable text (DESIGN R3, R5). :func:`contention` builds that
from the same lease table and presence guard the broker consults, so its answer matches
why the broker is actually holding back. :func:`explain_op_contention` is the
convenience that works it out for a named op and its parameters.

This reads the live lease table and presence guard; it never goes through the broker
queue (a read op would be blocked by the very lease it is trying to explain).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from aar.node.leases import Holder, LeaseManager, Scope, project_key, timeline_key
from aar.node.locator import ProjectLocator
from aar.node.ops.framework import OpModel, _lease_timeline, get_op
from aar.node.presence import PresenceGuard

# PresenceGuard.may_switch codes that mean a human is positively present.
_HUMAN_PRESENT_CODES = frozenset({"human_context_change", "human_input"})


@dataclass(frozen=True, slots=True)
class HeldLease:
    """One lease held by another party that blocks the work."""

    scope: str
    key: tuple[str, ...]
    holder_kind: str
    holder_id: str
    held_seconds: float
    expires_in_seconds: float | None
    since: str | None
    """ISO wall-clock time the lease was acquired, when a wall offset was supplied."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "key": list(self.key),
            "holder": {"kind": self.holder_kind, "id": self.holder_id},
            "held_seconds": self.held_seconds,
            "expires_in_seconds": self.expires_in_seconds,
            "since": self.since,
        }


@dataclass(frozen=True, slots=True)
class Contention:
    """Why the work is (or is not) held back, as structured fields plus readable text."""

    blocked: bool
    code: str
    leases: tuple[HeldLease, ...]
    human_active: bool
    presence_reason: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocked": self.blocked,
            "code": self.code,
            "leases": [lease.to_dict() for lease in self.leases],
            "human_active": self.human_active,
            "presence_reason": self.presence_reason,
            "text": self.text,
        }


def _blocking_leases(
    leases: LeaseManager, holder: Holder, project_id: str, timeline_ids: Sequence[str]
) -> list[Any]:
    """Leases held by others that stop ``holder`` acting, exactly as the broker decides.

    An op naming timelines needs each of those timelines; an op naming none acts on
    whichever timeline is current, so it needs the whole project.
    """
    tids = [tid for tid in dict.fromkeys(timeline_ids) if tid]
    if not tids:
        return leases.conflicts_for(holder, Scope.PROJECT, project_key(project_id))
    out: list[Any] = []
    seen: set[str] = set()
    for tid in tids:
        for lease in leases.conflicts_for(holder, Scope.TIMELINE, timeline_key(project_id, tid)):
            if lease.id not in seen:
                seen.add(lease.id)
                out.append(lease)
    return out


def _held(lease: Any, now: float, wall_offset: float | None) -> HeldLease:
    since = None
    if wall_offset is not None:
        since = datetime.fromtimestamp(lease.acquired_at + wall_offset, UTC).isoformat()
    return HeldLease(
        scope=lease.scope.value,
        key=tuple(lease.key),
        holder_kind=lease.holder.kind.value,
        holder_id=lease.holder.id,
        held_seconds=max(0.0, now - lease.acquired_at),
        expires_in_seconds=(None if lease.expires_at is None else max(0.0, lease.expires_at - now)),
        since=since,
    )


def _lease_line(held: HeldLease, now: float, wall_offset: float | None) -> str:
    who = f"{held.holder_kind}:{held.holder_id}"
    key = "/".join(held.key)
    if wall_offset is not None and held.since is not None:
        when = datetime.fromtimestamp((now - held.held_seconds) + wall_offset, UTC).strftime(
            "%H:%M"
        )
        since = f"since {when}"
    else:
        since = f"held for {held.held_seconds:.0f}s"
    return f"{held.scope} {key} is held by {who} ({since})"


def contention(
    leases: LeaseManager,
    presence: PresenceGuard,
    *,
    holder: Holder,
    project_id: str,
    timeline_ids: Sequence[str] = (),
    now: float,
    wall_offset: float | None = None,
) -> Contention:
    """Describe who blocks ``holder`` from working on ``project_id`` (its ``timeline_ids``).

    ``wall_offset`` is Unix time minus the node's monotonic clock (as the feed uses); with
    it the readable text names an actual clock time rather than a relative duration.
    """
    blocking = _blocking_leases(leases, holder, project_id, timeline_ids)
    held = tuple(_held(lease, now, wall_offset) for lease in blocking)
    decision = presence.may_switch(now)
    presence_blocks = not decision.allowed
    human_active = decision.code in _HUMAN_PRESENT_CODES
    blocked = bool(held) or presence_blocks

    has_human_lease = any(lease.holder.is_human for lease in blocking)
    if held and has_human_lease:
        code = "held_by_human"
    elif held:
        code = "held_by_agent"
    elif human_active:
        code = "human_active"
    elif presence_blocks:
        code = decision.code
    else:
        code = "clear"

    lines = [_lease_line(entry, now, wall_offset) for entry in held]
    if presence_blocks:
        lines.append(f"a human is active on this node: {decision.reason}")
    text = "; ".join(lines) if lines else "no lease or presence blocks this work"
    return Contention(blocked, code, held, human_active, decision.reason, text)


def explain_op_contention(
    leases: LeaseManager,
    presence: PresenceGuard,
    name: str,
    params: OpModel | dict[str, Any],
    *,
    target: ProjectLocator,
    holder: Holder,
    now: float,
    wall_offset: float | None = None,
) -> Contention:
    """Describe the contention a hot op ``name`` with ``params`` would meet right now.

    It leases the same identities the broker would (the op's timeline ids, or the whole
    project when it names none), so the answer matches whether the op can run.
    """
    if target.project_id is None:
        raise ValueError("target needs a project_id: contention is decided by identity")
    spec = get_op(name)
    checked = params if isinstance(params, spec.params) else spec.params.model_validate(params)
    _, timeline_id, extra = _lease_timeline(spec, checked)
    timeline_ids = tuple(tid for tid in (timeline_id, *extra) if tid)
    return contention(
        leases,
        presence,
        holder=holder,
        project_id=target.project_id,
        timeline_ids=timeline_ids,
        now=now,
        wall_offset=wall_offset,
    )
