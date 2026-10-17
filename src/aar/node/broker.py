"""Session broker: a per-project hot-op queue in front of the one Resolve instance.

Hot ops need Resolve's current project (and often its current timeline or page). The broker
time-slices them:

- One queue per project. The current project is preferred, and a project keeps the instance
  for a batch of up to ``batch_max_ops`` ops or ``batch_max_seconds``, so Resolve switches
  projects as rarely as possible. When the batch quota runs out and other projects are
  waiting, the next batch goes to another project. An op that has waited ``max_wait_seconds``
  ends the running batch at the next op boundary and goes next (bounded starvation).
- Projects are addressed by ``ProjectLocator`` (Project Manager folder path, name, and the
  stable ``project_id`` when known). Queues are keyed by ``locator.key``; leases by project
  name, which is conservative: same-named projects in different folders share a lease.
- Before it acts in or switches away from a context it did not create, the broker asks the
  ``PresenceGuard``, checks leases, and snapshots the human's context: the full Deliver
  settings (saved into a temporary render preset), the full Project Manager path, and where
  the human's project lives. When the queue drains it restores that context (navigating the
  Project Manager back before loading), but never while the human is active again; if the
  human has changed the context in the meantime, the human has taken over and the restore is
  abandoned.
- Before every op it asserts the expected project (by ``project_id`` when the locator has
  one, and timeline and page when given), re-selecting once on mismatch. An op is never run
  against the wrong project; it fails with a structured ``OpError`` instead.
- An op body that raises ``HotOpFailure`` (the op library's ``OpError`` does) fails with that
  failure's code and detail instead of the generic ``op_raised``.
- Each op gets its own ``OpResult``. A failing op does not abort the batch unless it is
  flagged ``abort_batch_on_failure``; then the rest of the batch from the same holder is
  cancelled and the batch ends.

Every decision is emitted to an injectable sink as a ``BrokerEvent``. Kinds: ``submitted``,
``skipped``, ``deferred``, ``snapshot``, ``switch``, ``switch_failed``, ``batch_started``,
``batch_yielded``, ``batch_ended``, ``batch_aborted``, ``reselect``, ``timeline_selected``,
``page_opened``, ``op_started``, ``op_succeeded``, ``op_failed``, ``op_cancelled``,
``restore_pending``, ``restore_deferred``, ``restored``, ``restore_abandoned``,
``lease_lost``. Repeated identical ``deferred``/``skipped``/``restore_*`` notices are
emitted once, not on every tick.

The broker is a single-worker state machine driven by ``tick()``; it never sleeps. A runner
(slice S7) calls ``tick()`` on a timer. ``submit``/``cancel``/``result`` may be called from
other threads.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum

from aar.node.gateway import VALID_PAGES, GatewayError, ResolveContext, ResolveGateway
from aar.node.leases import (
    Clock,
    Holder,
    Lease,
    LeaseConflict,
    LeaseManager,
    LeaseNotHeld,
    Scope,
    instance_key,
    project_key,
    timeline_key,
)
from aar.node.locator import ProjectLocator
from aar.node.presence import Decision, PresenceGuard

OpFn = Callable[[ResolveGateway], object]
"""An op body. It receives the gateway (``ScriptApiGateway.resolve`` gives raw API access)."""


class HotOpFailure(Exception):
    """An op body's structured failure: the broker records its ``code`` and ``detail``."""

    def __init__(self, code: str, message: str, detail: Mapping[str, object] | None = None) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.detail: Mapping[str, object] = dict(detail or {})


@dataclass(frozen=True)
class HotOp:
    project: ProjectLocator
    fn: OpFn
    holder: Holder
    timeline: str | None = None
    page: str | None = None
    priority: int = 0
    abort_batch_on_failure: bool = False
    label: str = ""
    submitted_at: float | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


class OpStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class OpError:
    code: str
    message: str
    detail: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OpResult:
    op_id: str
    status: OpStatus
    finished_at: float
    value: object = None
    error: OpError | None = None
    started_at: float | None = None
    context: ResolveContext | None = None
    """The asserted context the op ran in (``None`` if it never ran)."""


@dataclass(frozen=True)
class BrokerEvent:
    kind: str
    at: float
    data: Mapping[str, object]


EventSink = Callable[[BrokerEvent], None]


@dataclass(frozen=True)
class HumanSnapshot:
    context: ResolveContext
    """The human's context; ``project_manager_path`` is measured when the config allows."""
    taken_at: float
    deliver_preset: str | None
    warnings: tuple[str, ...] = ()
    project: ProjectLocator | None = None
    """Where the human's open project lives, if it could be located."""


@dataclass(frozen=True)
class RestoreReport:
    ok: bool
    snapshot: HumanSnapshot
    final: ResolveContext
    errors: Mapping[str, str]
    mismatches: Mapping[str, tuple[object, object]]


class TickKind(StrEnum):
    IDLE = "idle"
    RAN = "ran"
    DEFERRED = "deferred"
    SWITCH_FAILED = "switch_failed"
    RESTORE_PENDING = "restore_pending"
    RESTORE_DEFERRED = "restore_deferred"
    RESTORED = "restored"


_NO_PROGRESS = frozenset(
    {TickKind.IDLE, TickKind.DEFERRED, TickKind.RESTORE_PENDING, TickKind.RESTORE_DEFERRED}
)


@dataclass(frozen=True)
class TickOutcome:
    kind: TickKind
    project: str | None = None
    ran: tuple[str, ...] = ()
    code: str = ""
    reason: str = ""


@dataclass(frozen=True)
class BrokerConfig:
    batch_max_ops: int = 8
    batch_max_seconds: float = 30.0
    max_wait_seconds: float = 300.0
    restore_delay_seconds: float = 15.0
    """How long the queue must stay empty before the human's context is restored."""
    instance_id: str = "default"
    broker_id: str = "broker"
    lease_ttl: float = 120.0
    save_before_switch: bool = True
    """Save the current project before loading another, so neither side loses work."""
    snapshot_deliver_preset: bool = True
    snapshot_project_manager_path: bool = True
    """Measure the human's Project Manager path (a walk up to the root and back) so it can be
    restored, and locate the human's project for the restore."""
    preset_prefix: str = "aar-restore-"

    def __post_init__(self) -> None:
        if self.batch_max_ops < 1:
            raise ValueError("batch_max_ops must be at least 1")
        if self.batch_max_seconds <= 0 or self.max_wait_seconds <= 0 or self.lease_ttl <= 0:
            raise ValueError("batch_max_seconds, max_wait_seconds and lease_ttl must be positive")
        if self.restore_delay_seconds < 0:
            raise ValueError("restore_delay_seconds must not be negative")


class Broker:
    def __init__(
        self,
        gateway: ResolveGateway,
        *,
        leases: LeaseManager,
        presence: PresenceGuard,
        clock: Clock = time.monotonic,
        sink: EventSink | None = None,
        config: BrokerConfig | None = None,
    ) -> None:
        self._gw = gateway
        self._leases = leases
        self._presence = presence
        self._clock = clock
        self._sink = sink
        self.config = config or BrokerConfig()
        self.holder = Holder.agent(self.config.broker_id)
        self._lock = threading.RLock()
        self._queues: dict[str, list[HotOp]] = {}
        self._seq: dict[str, int] = {}
        self._counter = itertools.count()
        self._results: dict[str, OpResult] = {}
        self._snapshot: HumanSnapshot | None = None
        self._control_since: float | None = None
        self._instance_lease: Lease | None = None
        self._last_turn: tuple[str, bool] | None = None  # (project key, used its whole quota)
        self._last_served: dict[str, float] = {}
        self._drained_since: float | None = None
        self._notices: dict[str, str] = {}
        self.last_restore: RestoreReport | None = None

    # -- public API ------------------------------------------------------------------------

    @property
    def in_control(self) -> bool:
        """True while the broker holds a borrowed human context that it still has to restore."""
        return self._snapshot is not None

    @property
    def snapshot(self) -> HumanSnapshot | None:
        return self._snapshot

    def submit(self, op: HotOp) -> str:
        if not isinstance(op.project, ProjectLocator):
            raise ValueError("a hot op needs a ProjectLocator")
        if op.page is not None and op.page not in VALID_PAGES:
            raise ValueError(f"page {op.page!r} is not one of {sorted(VALID_PAGES)}")
        if op.submitted_at is None:
            op = replace(op, submitted_at=self._clock())
        with self._lock:
            if op.id in self._seq or op.id in self._results:
                raise ValueError(f"duplicate op id {op.id}")
            self._queues.setdefault(op.project.key, []).append(op)
            self._seq[op.id] = next(self._counter)
        self._emit(
            "submitted",
            op_id=op.id,
            project=op.project.key,
            project_id=op.project.project_id,
            timeline=op.timeline,
            page=op.page,
            holder=str(op.holder),
            priority=op.priority,
            label=op.label,
        )
        return op.id

    def cancel(self, op_id: str, reason: str = "cancelled by request") -> bool:
        with self._lock:
            op = self._find(op_id)
            if op is None:
                return False
        self._finish_unrun(op, OpStatus.CANCELLED, OpError("cancelled", reason))
        return True

    def result(self, op_id: str) -> OpResult | None:
        with self._lock:
            return self._results.get(op_id)

    def pending(self, project: str | None = None) -> list[HotOp]:
        """Queued ops in run order (per project), optionally for one project (a locator key)."""
        with self._lock:
            projects = [project] if project is not None else sorted(self._queues)
            return [op for p in projects for op in self._ordered(p)]

    def tick(self) -> TickOutcome:
        """Make one scheduling decision: run a batch, defer, restore, or nothing."""
        now = self._clock()
        ctx = self._gw.get_context()
        self._presence.observe(ctx, now)
        self._check_takeover(now)

        runnable = self._runnable(notify=True)
        if not runnable:
            return self._idle_or_restore(ctx, now)
        self._drained_since = None
        self._notices.pop("restore", None)

        project, why = self._select(runnable, self._current_key(runnable, ctx), now)
        target = runnable[project][0].project
        decision = self._gate(ctx, target, now)
        if decision.allowed:
            decision = self._take_control(ctx, now)
        if not decision.allowed:
            self._notice("defer", "deferred", decision, project=project)
            return TickOutcome(
                TickKind.DEFERRED, project, code=decision.code, reason=decision.reason
            )
        self._notices.pop("defer", None)

        if not _is_open(target, ctx):
            failed = self._switch(ctx, target)
            if failed is not None:
                return failed
        return self._run_batch(project, why)

    def run_until_idle(self, max_ticks: int = 1000) -> list[TickOutcome]:
        """Tick until a tick makes no progress (idle, deferred or waiting to restore)."""
        outcomes: list[TickOutcome] = []
        for _ in range(max_ticks):
            outcome = self.tick()
            outcomes.append(outcome)
            if outcome.kind in _NO_PROGRESS:
                return outcomes
        raise RuntimeError(f"broker still busy after {max_ticks} ticks")

    # -- events ----------------------------------------------------------------------------

    def _emit(self, kind: str, **data: object) -> None:
        if self._sink is not None:
            self._sink(BrokerEvent(kind, self._clock(), data))

    def _notice(self, key: str, kind: str, decision: Decision, **data: object) -> None:
        """Emit a repeated decision only when its code changes."""
        if self._notices.get(key) == decision.code:
            return
        self._notices[key] = decision.code
        self._emit(kind, code=decision.code, reason=decision.reason, **data)

    # -- queue -----------------------------------------------------------------------------

    def _order_key(self, op: HotOp) -> tuple[int, float, int]:
        assert op.submitted_at is not None
        return (-op.priority, op.submitted_at, self._seq[op.id])

    def _ordered(self, project: str) -> list[HotOp]:
        with self._lock:
            return sorted(self._queues.get(project, []), key=self._order_key)

    def _find(self, op_id: str) -> HotOp | None:
        with self._lock:
            for ops in self._queues.values():
                for op in ops:
                    if op.id == op_id:
                        return op
        return None

    def _remove(self, op: HotOp) -> bool:
        key = op.project.key
        with self._lock:
            ops = self._queues.get(key, [])
            if op not in ops:
                return False
            ops.remove(op)
            if not ops:
                del self._queues[key]
            del self._seq[op.id]
            self._notices.pop(f"skip:{op.id}", None)
            return True

    def _record(self, result: OpResult) -> None:
        with self._lock:
            self._results[result.op_id] = result

    def _finish_unrun(self, op: HotOp, status: OpStatus, error: OpError) -> OpResult | None:
        """Take an op off the queue without running it, recording why."""
        if not self._remove(op):
            return None
        result = OpResult(op.id, status, self._clock(), error=error)
        self._record(result)
        self._emit(
            "op_cancelled" if status is OpStatus.CANCELLED else "op_failed",
            op_id=op.id,
            project=op.project.key,
            holder=str(op.holder),
            code=error.code,
            message=error.message,
            detail=dict(error.detail),
        )
        return result

    # -- leases and scheduling ---------------------------------------------------------------

    def _block_reason(self, op: HotOp) -> str | None:
        """Why another holder's lease stops ``op`` from running now, if it does.

        An op that names a timeline needs that timeline (and no one else's project lease). An
        op that names none may act on whichever timeline is current, so it needs the whole
        project: any other holder's project or timeline lease in it blocks the op. This is
        decided from leases alone, never from which timeline happens to be current, so a
        blocked op cannot make the broker switch projects only to find it cannot run.
        """
        name = op.project.name
        if op.timeline is None:
            blocking = self._leases.conflicts_for(op.holder, Scope.PROJECT, project_key(name))
        else:
            blocking = self._leases.conflicts_for(
                op.holder, Scope.TIMELINE, timeline_key(name, op.timeline)
            )
        return blocking[0].describe() if blocking else None

    def _runnable(self, *, notify: bool) -> dict[str, list[HotOp]]:
        with self._lock:
            projects = sorted(self._queues)
        runnable: dict[str, list[HotOp]] = {}
        for project in projects:
            ops = []
            for op in self._ordered(project):
                reason = self._block_reason(op)
                if reason is None:
                    ops.append(op)
                elif notify:
                    self._notice(
                        f"skip:{op.id}",
                        "skipped",
                        Decision(False, f"lease:{reason}", reason),
                        op_id=op.id,
                        project=op.project.key,
                    )
            if ops:
                runnable[project] = ops
        return runnable

    @staticmethod
    def _current_key(runnable: dict[str, list[HotOp]], ctx: ResolveContext) -> str | None:
        """The queue key of the project that is open now, if any runnable queue targets it."""
        for key, ops in runnable.items():
            if _is_open(ops[0].project, ctx):
                return key
        return None

    def _waiting_since(self, project: str, ops: list[HotOp]) -> float:
        """Since when ``project`` has gone unserved: its oldest op, or its last op if later."""
        oldest = min(op.submitted_at or 0.0 for op in ops)
        return max(oldest, self._last_served.get(project, oldest))

    def _select(
        self, runnable: dict[str, list[HotOp]], current: str | None, now: float
    ) -> tuple[str, str]:
        waiting = {p: self._waiting_since(p, ops) for p, ops in runnable.items()}
        starved = [p for p in runnable if now - waiting[p] >= self.config.max_wait_seconds]
        if starved:
            return min(starved, key=lambda p: (waiting[p], p)), "starved"
        just_used = None
        if self._last_turn is not None and self._last_turn[1] and len(runnable) > 1:
            just_used = self._last_turn[0]
        if current in runnable and current != just_used:
            return current, "current"
        candidates = [p for p in runnable if p != just_used]
        best = min(
            candidates,
            key=lambda p: (-max(op.priority for op in runnable[p]), waiting[p], p),
        )
        return best, "next"

    def _other_starved(self, project: str, now: float) -> str | None:
        runnable = self._runnable(notify=False)
        for other, ops in sorted(runnable.items()):
            waited = now - self._waiting_since(other, ops)
            if other != project and waited >= self.config.max_wait_seconds:
                return other
        return None

    def _gate(self, ctx: ResolveContext, target: ProjectLocator | None, now: float) -> Decision:
        """May the broker act now, moving to ``target`` if it is not the current project?"""
        decision = self._presence.may_switch(now)
        if not decision.allowed:
            return decision
        blocking = self._leases.conflicts_for(
            self.holder, Scope.INSTANCE, instance_key(self.config.instance_id)
        )
        if blocking:
            return Decision(False, "instance_leased", blocking[0].describe())
        if target is not None and not _is_open(target, ctx):
            if ctx.rendering:
                return Decision(
                    False, "render_in_progress", f"project {ctx.project!r} is rendering"
                )
            if ctx.project is not None:
                for lease in self._leases.active():
                    if (
                        lease.holder.is_human
                        and lease.scope in (Scope.PROJECT, Scope.TIMELINE)
                        and lease.key[0] == ctx.project
                    ):
                        return Decision(False, "human_lease_on_current", lease.describe())
        return Decision(True, "ok", "")

    # -- control of the instance --------------------------------------------------------------

    def _hold_instance_lease(self) -> Decision:
        key = instance_key(self.config.instance_id)
        if self._instance_lease is not None:
            try:
                self._instance_lease = self._leases.renew(self._instance_lease.id, self.holder)
                return Decision(True, "ok", "")
            except LeaseNotHeld as exc:
                self._emit("lease_lost", lease_id=exc.lease_id, reason=exc.reason)
                self._instance_lease = None
        try:
            self._instance_lease = self._leases.acquire(
                self.holder, Scope.INSTANCE, key, self.config.lease_ttl
            )
        except LeaseConflict as exc:
            return Decision(False, "instance_leased", str(exc))
        return Decision(True, "ok", "")

    def _release_instance_lease(self) -> None:
        lease, self._instance_lease = self._instance_lease, None
        if lease is None:
            return
        try:
            self._leases.release(lease.id, self.holder)
        except LeaseNotHeld as exc:
            self._emit("lease_lost", lease_id=exc.lease_id, reason=exc.reason)

    def _take_control(self, ctx: ResolveContext, now: float) -> Decision:
        decision = self._hold_instance_lease()
        if not decision.allowed or self._snapshot is not None:
            return decision
        preset = None
        warnings: list[str] = []
        if self.config.snapshot_deliver_preset and ctx.project is not None:
            name = f"{self.config.preset_prefix}{uuid.uuid4().hex[:12]}"
            try:
                self._gw.save_render_preset(name)
                preset = name
            except GatewayError as exc:
                warnings.append(f"Deliver settings not snapshotted: {exc}")
        human_project: ProjectLocator | None = None
        if self.config.snapshot_project_manager_path:
            pm_path, human_project = self._snapshot_project_manager(ctx, warnings)
            ctx = replace(ctx, project_manager_path=pm_path)
            # The walk ends where it began; note it so it can never read as a human change.
            self._presence.note_broker_context(self._gw.get_context(), now)
        self._snapshot = HumanSnapshot(ctx, now, preset, tuple(warnings), human_project)
        self._control_since = now
        self._emit(
            "snapshot",
            context=asdict(ctx),
            deliver_preset=preset,
            project_locator=human_project.to_dict() if human_project is not None else None,
            warnings=warnings,
        )
        return decision

    def _snapshot_project_manager(
        self, ctx: ResolveContext, warnings: list[str]
    ) -> tuple[tuple[str, ...] | None, ProjectLocator | None]:
        """Measure the human's Project Manager path and locate the human's open project.

        The project is looked for in the human's current Project Manager folder first; if it
        is not there (the human browsed elsewhere after opening it), the whole tree is
        searched. Only a single match is trusted. Failures become snapshot warnings.
        """
        try:
            path = self._gw.get_project_manager_path()
        except GatewayError as exc:
            warnings.append(f"Project Manager path not measured: {exc}")
            return None, None
        if ctx.project is None:
            return path, None
        try:
            if ctx.project in self._gw.list_project_manager_folder().projects:
                return path, ProjectLocator(path, ctx.project, ctx.project_id)
            folders = self._gw.find_project_folders(ctx.project)
        except GatewayError as exc:
            warnings.append(f"human project not located: {exc}")
            return path, None
        if len(folders) == 1:
            return path, ProjectLocator(folders[0], ctx.project, ctx.project_id)
        where = ", ".join("/".join(f) or "<root>" for f in folders) or "nowhere"
        warnings.append(
            f"human project {ctx.project!r} is not in the current Project Manager folder and "
            f"was found in {len(folders)} folder(s) ({where}); it cannot be reloaded on restore"
        )
        return path, None

    def _drop_control(self) -> None:
        self._snapshot = None
        self._control_since = None
        self._drained_since = None
        self._last_turn = None
        self._release_instance_lease()

    def _check_takeover(self, now: float) -> bool:
        """If a human changed the context while the broker held it, hand it over. True if so."""
        if self._snapshot is None or self._control_since is None:
            return False
        if not self._presence.human_changed_since(self._control_since):
            return False
        snapshot = self._snapshot
        self._emit(
            "restore_abandoned",
            reason="a human changed the context while the broker held it",
            changed={k: [str(a), str(b)] for k, (a, b) in self._presence.last_human_change.items()},
            snapshot=asdict(snapshot.context),
            deliver_preset_kept=snapshot.deliver_preset,
        )
        self._drop_control()
        return True

    def _switch(self, ctx: ResolveContext, target: ProjectLocator) -> TickOutcome | None:
        key = target.key
        if self.config.save_before_switch and ctx.project is not None:
            try:
                self._gw.save_project()
            except GatewayError as exc:
                self._emit(
                    "switch_failed",
                    source=ctx.project,
                    target=key,
                    code=exc.code,
                    message=exc.message,
                )
                return TickOutcome(TickKind.DEFERRED, key, code=exc.code, reason=exc.message)
        try:
            self._gw.load_project(target)
        except GatewayError as exc:
            self._note_broker_context()
            self._emit(
                "switch_failed",
                source=ctx.project,
                target=key,
                code=exc.code,
                message=exc.message,
            )
            error = OpError(
                "project_unavailable",
                f"could not open project {key!r}: {exc.message}",
                {
                    "gateway_code": exc.code,
                    "expected": target.name,
                    "expected_path": list(target.folder_path),
                    "expected_id": target.project_id,
                },
            )
            failed = [
                op.id
                for op in self._ordered(key)
                if self._finish_unrun(op, OpStatus.FAILED, error) is not None
            ]
            return TickOutcome(TickKind.SWITCH_FAILED, key, tuple(failed), exc.code, exc.message)
        self._note_broker_context()
        self._emit("switch", source=ctx.project, target=key)
        return None

    def _note_broker_context(self) -> ResolveContext:
        ctx = self._gw.get_context()
        self._presence.note_broker_context(ctx, self._clock())
        return ctx

    # -- batches and ops -----------------------------------------------------------------------

    def _run_batch(self, project: str, why: str) -> TickOutcome:
        cfg = self.config
        start = self._clock()
        batch = self._runnable(notify=False).get(project, [])[: cfg.batch_max_ops]
        self._emit("batch_started", project=project, why=why, ops=[op.id for op in batch])
        ran: list[str] = []
        exhausted = len(batch) >= cfg.batch_max_ops
        stop = ""
        for i, op in enumerate(batch):
            if self._find(op.id) is None:  # cancelled meanwhile
                continue
            if i > 0:
                now = self._clock()
                if now - start >= cfg.batch_max_seconds:
                    exhausted, stop = True, "time"
                    break
                ctx = self._gw.get_context()
                self._presence.observe(ctx, now)
                if self._check_takeover(now):
                    stop = "human_took_over"
                    break
                starved = self._other_starved(project, now)
                if starved is not None:
                    self._emit("batch_yielded", project=project, to=starved)
                    exhausted, stop = True, "starved"
                    break
                decision = self._gate(ctx, op.project, now)
                if decision.allowed:
                    decision = self._hold_instance_lease()
                if not decision.allowed:
                    self._notice("defer", "deferred", decision, project=project)
                    stop = decision.code
                    break
            result = self._run_op(op)
            if result is None:
                continue
            ran.append(op.id)
            if result.status is OpStatus.FAILED and op.abort_batch_on_failure:
                error = OpError(
                    "batch_aborted",
                    f"op {op.id} failed and was flagged abort_batch_on_failure",
                    {"failed_op": op.id},
                )
                cancelled = [
                    later.id
                    for later in batch[i + 1 :]
                    if later.holder == op.holder
                    and self._finish_unrun(later, OpStatus.CANCELLED, error) is not None
                ]
                self._emit("batch_aborted", project=project, failed_op=op.id, cancelled=cancelled)
                stop = "aborted"
                break
        self._last_turn = (project, exhausted)
        self._emit("batch_ended", project=project, ran=ran, exhausted=exhausted, stop=stop)
        if not ran:
            return TickOutcome(
                TickKind.DEFERRED,
                project,
                code=stop or "blocked",
                reason="no op in the batch could run",
            )
        return TickOutcome(TickKind.RAN, project, tuple(ran), code=stop)

    def _fail(self, op: HotOp, code: str, message: str, **detail: object) -> OpResult | None:
        return self._finish_unrun(op, OpStatus.FAILED, OpError(code, message, detail))

    def _run_op(self, op: HotOp) -> OpResult | None:
        """Assert the context, then run ``op``. ``None``: lease-blocked, so it stays queued."""
        gw = self._gw
        want = op.project
        ctx = gw.get_context()
        if not _is_open(want, ctx):
            self._emit("reselect", op_id=op.id, expected=want.key, actual=ctx.project)
            try:
                gw.load_project(want)
            except GatewayError as exc:
                self._note_broker_context()
                return self._fail(
                    op,
                    "wrong_project",
                    f"expected project {want.key!r}, found {ctx.project!r}; re-select failed",
                    expected=want.name,
                    expected_id=want.project_id,
                    actual=ctx.project,
                    actual_id=ctx.project_id,
                    gateway_code=exc.code,
                    gateway_message=exc.message,
                )
            ctx = self._note_broker_context()
        if op.timeline is not None and ctx.timeline != op.timeline:
            try:
                gw.set_current_timeline(op.timeline)
            except GatewayError as exc:
                self._note_broker_context()
                return self._fail(
                    op,
                    "timeline_unavailable",
                    exc.message,
                    expected=op.timeline,
                    actual=ctx.timeline,
                    gateway_code=exc.code,
                )
            self._emit("timeline_selected", op_id=op.id, source=ctx.timeline, target=op.timeline)
        if op.page is not None and ctx.page != op.page:
            try:
                gw.open_page(op.page)
            except GatewayError as exc:
                self._note_broker_context()
                return self._fail(
                    op, "page_unavailable", exc.message, expected=op.page, gateway_code=exc.code
                )
            self._emit("page_opened", op_id=op.id, source=ctx.page, target=op.page)

        ctx = self._note_broker_context()
        if not _is_open(want, ctx):
            return self._fail(
                op,
                "wrong_project",
                f"expected project {want.key!r}, found {ctx.project!r}",
                expected=want.name,
                expected_id=want.project_id,
                actual=ctx.project,
                actual_id=ctx.project_id,
            )
        if op.timeline is not None and ctx.timeline != op.timeline:
            return self._fail(
                op,
                "wrong_timeline",
                f"expected timeline {op.timeline!r}, found {ctx.timeline!r}",
                expected=op.timeline,
                actual=ctx.timeline,
            )
        if op.page is not None and ctx.page != op.page:
            return self._fail(
                op,
                "wrong_page",
                f"expected page {op.page!r}, found {ctx.page!r}",
                expected=op.page,
                actual=ctx.page,
            )
        reason = self._block_reason(op)  # a lease may have been taken since the batch began
        if reason is not None:
            self._notice(
                f"skip:{op.id}",
                "skipped",
                Decision(False, f"lease:{reason}", reason),
                op_id=op.id,
                project=want.key,
            )
            return None

        if not self._remove(op):
            return None  # cancelled from another thread after the checks
        started = self._clock()
        self._emit(
            "op_started",
            op_id=op.id,
            project=want.key,
            timeline=ctx.timeline,
            holder=str(op.holder),
            label=op.label,
        )
        value: object = None
        error: OpError | None = None
        try:
            value = op.fn(gw)
        except HotOpFailure as exc:
            error = OpError(exc.code, exc.message, dict(exc.detail))
        except Exception as exc:
            error = OpError(
                "op_raised",
                f"{type(exc).__name__}: {exc}",
                {"exception": type(exc).__name__, "traceback": traceback.format_exc()},
            )
        finished = self._clock()
        self._last_served[want.key] = finished
        self._note_broker_context()
        status = OpStatus.OK if error is None else OpStatus.FAILED
        result = OpResult(op.id, status, finished, value, error, started, ctx)
        self._record(result)
        if error is None:
            self._emit("op_succeeded", op_id=op.id, project=want.key, holder=str(op.holder))
        else:
            self._emit(
                "op_failed",
                op_id=op.id,
                project=want.key,
                holder=str(op.holder),
                code=error.code,
                message=error.message,
                detail=dict(error.detail),
            )
        return result

    # -- restore -------------------------------------------------------------------------------

    def _idle_or_restore(self, ctx: ResolveContext, now: float) -> TickOutcome:
        snapshot = self._snapshot
        if snapshot is None:
            return TickOutcome(TickKind.IDLE)
        if self._drained_since is None:
            self._drained_since = now
        waited = now - self._drained_since
        if waited < self.config.restore_delay_seconds:
            decision = Decision(
                False,
                "restore_delay",
                f"queue empty for {waited:.1f}s of {self.config.restore_delay_seconds}s",
            )
            self._notice("restore", "restore_pending", decision)
            return TickOutcome(TickKind.RESTORE_PENDING, code=decision.code, reason=decision.reason)
        decision = self._gate(ctx, _human_target(snapshot), now)
        if not decision.allowed:
            self._notice("restore", "restore_deferred", decision)
            return TickOutcome(
                TickKind.RESTORE_DEFERRED, code=decision.code, reason=decision.reason
            )
        self._notices.pop("restore", None)
        report = self._restore(snapshot, ctx)
        return TickOutcome(
            TickKind.RESTORED,
            snapshot.project.key if snapshot.project is not None else snapshot.context.project,
            code="ok" if report.ok else "partial",
            reason=""
            if report.ok
            else f"errors={dict(report.errors)} mismatches={sorted(report.mismatches)}",
        )

    def _restore(self, snapshot: HumanSnapshot, ctx: ResolveContext) -> RestoreReport:
        gw, want = self._gw, snapshot.context
        errors: dict[str, str] = {}

        def step(name: str, action: Callable[[], None]) -> bool:
            try:
                action()
            except GatewayError as exc:
                errors[name] = f"{exc.code}: {exc.message}"
                return False
            return True

        def load_human_project() -> None:
            if snapshot.project is None:
                raise GatewayError(
                    "human_project_unlocated",
                    f"the snapshot does not say which Project Manager folder holds "
                    f"{want.project!r}; see the snapshot warnings",
                )
            gw.load_project(snapshot.project)

        project_ok = True
        target = _human_target(snapshot)
        if target is not None and not _is_open(target, ctx):
            if self.config.save_before_switch and ctx.project is not None:
                project_ok = step("save", gw.save_project)
            project_ok = project_ok and step("project", load_human_project)
        if project_ok:  # never touch the rest of the context in the wrong project
            pm_path = want.project_manager_path
            if pm_path is not None:
                step("project_manager_path", lambda: gw.open_project_manager_path(pm_path))
            if want.timeline is not None:
                step("timeline", lambda: gw.set_current_timeline(want.timeline, want.timeline_id))
            if want.folder is not None:
                step("folder", lambda: gw.set_current_folder(want.folder))
            preset = snapshot.deliver_preset
            if preset is not None and step("deliver_preset", lambda: gw.load_render_preset(preset)):
                step("deliver_preset_cleanup", lambda: gw.delete_render_preset(preset))
            if want.render_format is not None and want.render_codec is not None:
                step(
                    "render_format",
                    lambda: gw.set_render_format_and_codec(want.render_format, want.render_codec),
                )
            if want.render_mode is not None:
                step("render_mode", lambda: gw.set_render_mode(want.render_mode))
            if want.page is not None:
                step("page", lambda: gw.open_page(want.page))
            if want.playhead is not None and want.timeline is not None:
                step("playhead", lambda: gw.set_playhead(want.playhead))

        final = gw.get_context(with_pm_path=want.project_manager_path is not None)
        self._presence.note_broker_context(final, self._clock())
        mismatches = want.diff(final)
        report = RestoreReport(not errors and not mismatches, snapshot, final, errors, mismatches)
        self.last_restore = report
        self._emit(
            "restored",
            ok=report.ok,
            project=snapshot.project.key if snapshot.project is not None else want.project,
            errors=dict(errors),
            mismatches={k: [str(a), str(b)] for k, (a, b) in mismatches.items()},
            deliver_preset=snapshot.deliver_preset,
        )
        self._drop_control()
        return report


def _is_open(target: ProjectLocator, ctx: ResolveContext) -> bool:
    """Is ``target`` the project open in ``ctx``? By id when the locator has one."""
    return ctx.project is not None and target.matches(ctx.project, ctx.project_id)


def _human_target(snapshot: HumanSnapshot) -> ProjectLocator | None:
    """The locator to compare the open project against when restoring the human's context."""
    want = snapshot.context
    if want.project is None:
        return None
    if snapshot.project is not None:
        return snapshot.project
    # Not located: still recognisable by id (or name), just not reloadable.
    return ProjectLocator((), want.project, want.project_id)
