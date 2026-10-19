"""The node job runner: drives checkpointed, resumable jobs over the session broker.

The runner owns no durable job state; the :class:`~aar.node.jobs.store.JobStore` does, scoped
to this node and guarded by a single-writer lock, so two runners can never double-execute a
job. A fresh runner opened on the same store resumes every job from its last checkpoint.

A pump has three phases, in this order, so a stop is honoured before anything runs:

1. **Stop check** (DESIGN R5, §5): a job whose project or node is stopped is paused at its
   checkpoint as ``waiting`` and its queued op is cancelled, BEFORE the broker runs anything.
2. **Run the broker** off the pump thread. Exactly one drain runs at a time (concurrent pumps
   share it), and it ticks the broker on every pump -- also when the queue is empty, which is
   how the broker restores the human's project, timeline and page once the agents are done.
   The pump waits for the drain up to ``drain_budget`` seconds (``None``: until it finishes),
   so a hung gateway call cannot block the pump and the watchdog can still stall the job. A
   broker crash on the drain thread is logged and recorded on the affected jobs.
3. **Advance** each job: collect its op's result (a result that came back is never dropped, not
   even when a stop landed while the op ran), then submit its next op; then the watchdog
   stalls any job whose op or step ran past its (pausable, persisted) deadline.

Every op the runner submits carries a broker *precheck* (the emergency stop and "does the job
still want this op"), asked right before the op runs, so a stop takes effect between ops even
inside one broker batch; and a *journal* (:mod:`aar.node.ops.journal`) through which a render
build records each side effect on the job's checkpoint as it happens, so a crash, a stop or a
cancel can never duplicate or silently orphan a render.

Gates around a step: the emergency stop (above), the F15 approval gate (the first mutating op
against a production project consumes a human token at the moment of the write; the consume is
bound to the job and verified against the hub), and a per-project render lock (one agent
render per Resolve project at a time).
"""

from __future__ import annotations

import functools
import logging
import math
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from aar.hub.approvals import ApprovalError, ApprovalScope
from aar.hub.control import Stopped
from aar.hub.events import Actor
from aar.node.branch_registry import BranchRegistry
from aar.node.branches import valid_agent_id
from aar.node.broker import Broker, HotOp, HotOpFailure, OpResult, OpStatus
from aar.node.broker import OpError as BrokerOpError
from aar.node.hublink import HubLink
from aar.node.leases import Clock, Holder
from aar.node.locator import ProjectLocator
from aar.node.ops import get_op, make_hot_op
from aar.node.ops.collab import BranchResult
from aar.node.ops.framework import BRANCH_GUARD_CODES, OpError
from aar.node.ops.journal import op_journal

from .kinds import get_kind
from .model import (
    TERMINAL_STATES,
    Done,
    Fail,
    JobState,
    OpOutcome,
    Pause,
    Step,
    StepContext,
    StructuredError,
    Submit,
    WaitReason,
)
from .render import live_render_id
from .store import JobRecord, JobStore
from .watchdog import Watchdog, WatchdogConfig

if TYPE_CHECKING:
    from aar.node.feed import NodeFeed, PresenceReporter

_log = logging.getLogger(__name__)

_APPROVAL_SCOPE = ApprovalScope.PROJECT_WRITE
_DRAIN_BUDGET = 0.5
"""Real seconds a pump waits for the broker to finish before leaving a hung op to the watchdog."""
DEFAULT_MAX_RESTARTS = 5
"""Node restarts that may interrupt one running step before the job is stalled (P5b)."""
DEADLINE_PERSIST_EVERY = 5.0
"""Clock seconds between saves of a running step's remaining deadline when nothing else saves
it, so a restart loses at most this much of the countdown (P5b)."""

_drain_delay = 0.0
"""TEST-ONLY: seconds every drain sleeps before it ticks. Set above ``_DRAIN_BUDGET`` to prove
no test depends on the drain finishing inside the wall-clock budget (``AAR_TEST_DRAIN_DELAY``,
see ``tests/node/conftest.py``). Always 0 in production."""

_NOT_RUN = frozenset({"emergency_stop", "job_inactive", "project_stalled", "hub_unreachable"})
"""Precheck refusals: the op never ran, so the step re-issues it later (not a failure)."""

_JOURNALED_EVENTS = frozenset({"added", "started", "deleted", "orphaned", "restore_failed"})
"""Journal entries that also become job events (a handful per build: bounded)."""

_NOT_A_STATE = frozenset(
    {"build_started", "snapshot_recovered", "snapshot_deleted", "restore_failed"}
)
"""Journal entries that record a fact without moving the build's ``state``."""

_PROJECT_APPROVAL = "project"
"""The approval key of the F15 production approval (target: the project id)."""

_BRANCH_OP = "timeline.branch"
"""The op whose results the runner records on the hub (S11's ``record_branch``)."""

_RENDER_ENDED = frozenset({"render_failed", "render_job_unknown"})
"""Failure codes after which the job's render is known to be over (no orphan alert)."""


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """When a mutating op needs a human approval token.

    ``live`` is whether this node runs against a real, production Resolve; it defaults to
    ``True`` so every real project counts as production until the D3 sandbox approval is given
    (fail closed: an unknown project id is production too). The test fake gateway is not
    production, so tests set ``live=False``. ``sandbox_exempt`` is a frozenset of exact project
    ids a human has marked as a sandbox; it defaults to none.
    """

    live: bool = True
    sandbox_exempt: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if isinstance(self.sandbox_exempt, str) or not all(
            isinstance(x, str) for x in self.sandbox_exempt
        ):
            raise TypeError("sandbox_exempt must be a set of exact project ids, not a string")

    def is_production(self, project_id: str | None) -> bool:
        if not self.live:
            return False
        if project_id is None:
            return True  # fail closed: an unidentified project counts as production
        return project_id not in self.sandbox_exempt


@dataclass(frozen=True, slots=True)
class PumpOutcome:
    """What one pump did. ``active`` counts jobs that can still make progress on their own."""

    active: int
    running: int
    waiting: int
    stalled: int


@dataclass(frozen=True, slots=True)
class JobView:
    """An agent-facing snapshot of a job: measured values, structured error, waiting info."""

    job_id: str
    kind: str
    node_id: str
    state: str
    target: dict[str, Any]
    current_step: int
    steps_total: int
    step_name: str | None
    priority: int
    results: list[dict[str, Any]]
    measured: dict[str, Any]
    error: dict[str, Any] | None
    waiting: dict[str, Any] | None
    stall: dict[str, Any] | None
    approval_token: str | None
    approval_consumed: bool
    approvals: dict[str, str]
    """Consumed approvals: ``"project"`` (F15) and one ``"op:..."`` per guarded op (P4)."""
    agent_id: str | None
    created_at: float
    updated_at: float
    events: list[dict[str, Any]]
    alerts: list[dict[str, Any]]
    """Things a human must look at (an orphaned render, Deliver not restored ...)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Runtime:
    """Per-job state that lives only in this process (timers, the pending op, built steps)."""

    steps: list[Step]
    params: BaseModel
    pending_op_id: str | None = None
    op_submitted_at: float | None = None
    deadline_anchor: float | None = None
    """Clock time the current step's countdown last (re)started; ``None`` while paused."""
    op_outcome: OpOutcome | None = None
    quarantined: bool = False
    broker_error: dict[str, Any] | None = None
    """The last broker crash seen while this job's op was outstanding (the stall's cause)."""
    persisted_at: float | None = None
    """Clock time the remaining deadline was last saved while running."""
    pending_op_name: str | None = None
    """The name of the op ``pending_op_id`` runs (to record ``timeline.branch`` results)."""
    verify_pending: bool = False
    """The approvals could not be verified on load (hub unreachable): verify before running."""
    branch_records: list[dict[str, Any]] | None = None
    """``timeline.branch`` results not yet recorded on the hub (it was unreachable)."""


class JobRunner:
    """Submits, drives, checkpoints and reports node jobs over one broker."""

    def __init__(
        self,
        broker: Broker,
        store: JobStore,
        hublink: HubLink,
        *,
        node_id: str,
        branch_registry: BranchRegistry,
        feed: NodeFeed,
        presence_reporter: PresenceReporter,
        clock: Clock = time.monotonic,
        wall_clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        approval: ApprovalPolicy | None = None,
        watchdog: Watchdog | None = None,
        drain_budget: float | None = _DRAIN_BUDGET,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
    ) -> None:
        """Build a runner. ``branch_registry``, ``feed`` and ``presence_reporter`` are the
        node's single shared instances (S11 wiring, P5); :class:`aar.node.runtime.NodeRuntime`
        assembles them. Every op runs as its job's agent (P2); the runner has no holder."""
        if not node_id:
            raise ValueError("a job runner needs a non-empty node_id")
        if not isinstance(branch_registry, BranchRegistry):
            raise TypeError("a job runner needs the node's BranchRegistry")
        if feed.node_id != node_id:
            raise ValueError(
                f"the feed reports for node {feed.node_id!r}, the runner is node {node_id!r}: "
                "the hub records must name one node"
            )
        if drain_budget is not None and not (math.isfinite(drain_budget) and drain_budget > 0):
            raise ValueError("drain_budget must be a positive number of seconds, or None")
        if max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        self.broker = broker
        self.store = store
        self.hublink = hublink
        self.node_id = node_id
        self.clock = clock
        self.wall_clock = wall_clock
        self.actor = Actor("node", node_id)
        """The node actor: it records branches on the hub (the feed's node id, P5)."""
        self.branch_registry = branch_registry
        self.feed = feed
        self.presence_reporter = presence_reporter
        self.policy = approval or ApprovalPolicy()
        self.watchdog = watchdog or Watchdog(WatchdogConfig())
        self.max_restarts = max_restarts
        self._drain_budget = drain_budget
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._runtime: dict[str, _Runtime] = {}
        self._stalled_projects: set[str] = set()
        self._render_inflight: dict[str, str] = {}  # project_id -> job_id holding the render slot
        self._abandoned: set[str] = set()  # op ids a terminal job dropped while they ran
        self._late: dict[str, str] = {}  # abandoned op id -> job id, until its result lands
        self._drain_lock = threading.Lock()
        self._drain_thread: threading.Thread | None = None
        self._drain_done: threading.Event | None = None
        self._drain_errors: list[dict[str, Any]] = []
        self._hub_sync_lock = threading.Lock()
        self.last_flush = None
        """The last ``FlushReport`` of the node's feed."""
        self._closed = False
        self._load()

    # -- startup ---------------------------------------------------------------------------

    def _load(self) -> None:
        now = self.clock()
        for record in self.store.load_all(self.node_id):
            self._jobs[record.job_id] = record
            if not record.readonly:
                try:
                    steps = self._build_steps(record)
                    params = self._parse_params(record)
                except Exception as exc:  # a forged or future row must not brick the runner
                    record.readonly = True
                    record.state = JobState.FAILED
                    record.error = StructuredError(
                        "quarantined_bad_record",
                        f"job record could not be loaded: {type(exc).__name__}: {exc}",
                        "the row is malformed or from a newer version; remove or migrate it",
                    )
            if record.readonly:
                self._runtime[record.job_id] = _Runtime([], _Empty(), quarantined=True)
                self._note_quarantine(record, now)
                continue
            rt = _Runtime(steps=steps, params=params)
            self._runtime[record.job_id] = rt
            if record.agent_id is None or not valid_agent_id(record.agent_id):
                if record.state not in TERMINAL_STATES:
                    self._fail(
                        record,
                        StructuredError(
                            "no_agent_id",
                            f"job {record.job_id} has no valid agent id ({record.agent_id!r}); "
                            "its ops would have no agent to run as",
                            "resubmit the job with submit(..., agent_id=<the agent's id>)",
                        ),
                        now,
                    )
                continue
            self._requeue_branch_records(record, rt)
            try:
                self._verify_approval(record, now)
            except OSError as exc:  # the hub is unreachable: verify before the job runs
                rt.verify_pending = True
                _log.warning("job %s: approvals not verified on load: %s", record.job_id, exc)
            if record.state is JobState.RUNNING:
                rt.deadline_anchor = now  # the persisted remaining deadline runs on
                self._count_restart(record, now)
            if record.state is JobState.STALLED and record.target.project_id is not None:
                self._stalled_projects.add(record.target.project_id)
            if (
                record.state is JobState.RUNNING
                and steps
                and steps[record.current_step % max(len(steps), 1)].renders_project
                and (record.cursor.get("render_job_id") or record.cursor.get("journal"))
                and record.target.project_id is not None
            ):
                self._render_inflight[record.target.project_id] = record.job_id

    def _note_quarantine(self, record: JobRecord, now: float) -> None:
        """Record a quarantine once (the row itself is never rewritten)."""
        events = self.store.events(record.job_id)
        if events and events[-1].kind == "quarantined":
            return
        assert record.error is not None
        _log.warning("job %s quarantined: %s", record.job_id, record.error.message)
        self.store.append_events(
            record.job_id, [("quarantined", {"error": record.error.to_dict()}, now)]
        )

    def _count_restart(self, record: JobRecord, now: float) -> None:
        """A RUNNING job on load was interrupted by a restart: count it, persistently (P5b)."""
        record.restarts += 1
        record.updated_at = now
        if record.restarts > self.max_restarts:
            self._stall(
                record,
                StructuredError(
                    "crash_loop",
                    f"the node restarted {record.restarts} times while this step ran (limit "
                    f"{self.max_restarts}); the step is not making progress",
                    "check the node log for what kills it during this step; resume(job_id) "
                    "once it is fixed",
                    detail={"restarts": record.restarts, "max_restarts": self.max_restarts},
                ),
                now,
            )
            return
        self.store.save(record, [("restarted", {"restarts": record.restarts}, now)])

    def _job_actor(self, record: JobRecord) -> Actor:
        """The actor a job consumes its approvals as: the job's agent, bound to this job
        (``agent:<agent id>#<job id>``), so the hub's audit names the agent and the job, and a
        token one job consumed never verifies for another (P2, P13b)."""
        return Actor("agent", f"{record.agent_id}#{record.job_id}")

    @staticmethod
    def _approval_target(record: JobRecord, key: str) -> str:
        """The exact ``project.write`` target an approval key needs: the project id for the
        F15 approval, ``<project id>#op:<op>`` for an override that names one op (P4)."""
        project = record.target.project_id or ""
        if key == _PROJECT_APPROVAL:
            return project
        return op_approval_target(project, key.rsplit("/", 1)[-1])

    def _consume_verified(self, record: JobRecord, token_id: str | None, key: str) -> bool:
        """Whether the hub shows ``token_id`` consumed by THIS job, for the key's target."""
        if token_id is None:
            return False
        approval = self.hublink.approval(token_id)
        return (
            approval is not None
            and approval.consumed_at is not None
            and approval.consumed_by == self._job_actor(record)
            and approval.scope is _APPROVAL_SCOPE
            and approval.target == self._approval_target(record, key)
        )

    def _verify_approval(self, record: JobRecord, now: float) -> None:
        """On load: trust a consumed approval only when the hub shows this job's own consume
        for its exact target (L1, P13b), and finish a consume a crash interrupted (L2):
        ``approval_pending`` is the token a consume was started with (for the approval key in
        ``approval_pending_key``); if the hub shows it consumed by this job, the consume
        happened and is recovered, otherwise the intent is dropped. Raises ``OSError`` when the
        hub is unreachable (the caller verifies again before the job runs)."""
        events: list[tuple[str, dict[str, Any], float]] = []
        verified = {
            key: token
            for key, token in record.approvals.items()
            if self._consume_verified(record, token, key)
        }
        for key, token in record.approvals.items():
            if key not in verified:
                events.append(("approval_unverified", {"token_id": token, "key": key}, now))
        if record.approval_consumed and not verified and not record.approvals:
            events.append(("approval_unverified", {"token_id": record.approval_token}, now))
        record.approvals = verified
        pending, pending_key = record.approval_pending, record.approval_pending_key
        if pending is not None:
            key = pending_key or _PROJECT_APPROVAL
            record.approval_pending = record.approval_pending_key = None
            if key not in record.approvals and self._consume_verified(record, pending, key):
                record.approvals[key] = pending
                if record.approval_token == pending:
                    record.approval_token = None
                events.append(("approval_recovered", {"token_id": pending, "key": key}, now))
            else:
                events.append(("approval_pending_dropped", {"token_id": pending, "key": key}, now))
        record.approval_consumed = bool(record.approvals)
        if events:
            self.store.save(record, events)

    def _parse_params(self, record: JobRecord) -> BaseModel:
        return get_kind(record.kind).params_model.model_validate(record.params)

    def _build_steps(self, record: JobRecord) -> list[Step]:
        return get_kind(record.kind).build(self._parse_params(record), record.target)

    # -- agent-facing API ------------------------------------------------------------------

    def submit(
        self,
        kind: str,
        params: dict[str, Any] | BaseModel,
        target: ProjectLocator,
        *,
        agent_id: str,
        priority: int = 0,
        approval_token: str | None = None,
    ) -> str:
        """Queue a new job for the agent ``agent_id`` and return its id. The caller drives it
        with :meth:`pump`.

        Every op of the job runs as ``Holder.agent(agent_id)`` (P2): branch ownership, leases
        and the hub audit all name the agent that asked for the work. A job without a valid
        agent id (letters, digits, ``.``, ``_``, ``-``; no ``:``) is refused here, before it
        exists."""
        if not isinstance(agent_id, str) or not valid_agent_id(agent_id):
            raise ValueError(
                f"a job needs the id of the agent it works for (letters, digits, '.', '_' or "
                f"'-', at most 128 characters), got {agent_id!r}"
            )
        if target.project_id is None:
            raise ValueError("a job's target locator needs a project_id (every op asserts by id)")
        spec = get_kind(kind)
        model = (
            params
            if isinstance(params, spec.params_model)
            else spec.params_model.model_validate(params)
        )
        steps = spec.build(model, target)  # validates the plan (ops exist, timeouts are finite)
        now = self.clock()
        with self._lock:
            job_id = "job_" + uuid.uuid4().hex
            record = JobRecord(
                job_id=job_id,
                kind=kind,
                node_id=self.node_id,
                target=target,
                params=model.model_dump(mode="json"),
                state=JobState.QUEUED,
                approval_token=approval_token,
                agent_id=agent_id,
                priority=priority,
                step_deadline=steps[0].timeout if steps else None,
                created_at=now,
                updated_at=now,
            )
            self.store.save(
                record,
                [
                    (
                        "submitted",
                        {
                            "kind": kind,
                            "target": target.to_dict(),
                            "steps": len(steps),
                            "agent_id": agent_id,
                        },
                        now,
                    )
                ],
            )
            self._jobs[job_id] = record
            self._runtime[job_id] = _Runtime(steps=steps, params=model)
        return job_id

    def get(self, job_id: str) -> JobView | None:
        with self._lock:
            record = self._jobs.get(job_id)
            return self._view(record) if record is not None else None

    def poll(self, job_id: str) -> JobView | None:
        """An alias for :meth:`get`: read a job's current state and measured values."""
        return self.get(job_id)

    def list(
        self, *, state: JobState | str | None = None, project_id: str | None = None
    ) -> list[JobView]:
        want = JobState(state) if state is not None else None
        with self._lock:
            records = sorted(self._jobs.values(), key=lambda r: (r.created_at, r.job_id))
            return [
                self._view(r)
                for r in records
                if (want is None or r.state is want)
                and (project_id is None or r.target.project_id == project_id)
            ]

    def alerts(self) -> list[dict[str, Any]]:
        """Every alert raised on any job, oldest first, each with its ``job_id``."""
        with self._lock:
            out = [
                {"job_id": job_id, **event.to_dict()}
                for job_id in sorted(self._jobs)
                for event in self.store.events(job_id)
                if event.kind == "alert"
            ]
        return sorted(out, key=lambda a: a["seq"])

    def cancel(self, job_id: str, reason: str = "cancelled by request") -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in TERMINAL_STATES or record.readonly:
                return False
            now = self.clock()
            in_flight = self._drop_pending(record, abandon=True)
            events = [] if in_flight else self._orphan_alerts(record, None, now)
            self._terminate(
                record, JobState.CANCELLED, "cancelled", {"reason": reason}, now, events
            )
            return True

    def approve(self, job_id: str, token_id: str, actor: Actor | None = None) -> JobView | None:
        """Give a waiting job a human approval token and let it try the write again next pump.

        The token is consumed at the first write, not here; a wrong token leaves the job
        waiting (and is not retried until a different token is offered).
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in TERMINAL_STATES or record.readonly:
                return self._view(record) if record is not None else None
            record.approval_token = token_id
            record.updated_at = self.clock()
            if (
                record.state is JobState.WAITING
                and record.waiting is not None
                and record.waiting.kind == "needs_approval"
            ):
                record.waiting = None
                record.state = JobState.QUEUED
            self._checkpoint(
                record,
                [
                    (
                        "approval_offered",
                        {"token_id": token_id, "by": _actor_dict(actor)},
                        record.updated_at,
                    )
                ],
                record.updated_at,
            )
            return self._view(record)

    def resume(self, job_id: str, actor: Actor | None = None) -> JobView | None:
        """Clear a stall (the explicit human/agent recovery) and let the job run again."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            if record.state is not JobState.STALLED:
                return self._view(record)
            now = self.clock()
            rt = self._runtime[job_id]
            record.stall = None
            record.restarts = 0
            record.updated_at = now
            rt.broker_error = None
            if rt.pending_op_id is not None:
                # The op that hung is still out; keep tracking it, with a fresh op window.
                record.state = JobState.RUNNING
                rt.deadline_anchor = now
                rt.op_submitted_at = now
            else:
                record.state = JobState.QUEUED
                rt.deadline_anchor = None
                rt.op_submitted_at = None
            self._checkpoint(
                record, [("resumed", {"from": "stalled", "by": _actor_dict(actor)}, now)], now
            )
            self._clear_project_stall(record)
            return self._view(record)

    def close(self, timeout: float = 5.0) -> bool:
        """Stop the runner: no op it submitted runs any more. Returns whether the drain ended."""
        with self._drain_lock:
            self._closed = True
            done = self._drain_done
        return done.wait(timeout) if done is not None else True

    def wait_drain(self, timeout: float | None = None) -> bool:
        """Block until the current drain (if any) has finished. For tests and shutdown."""
        with self._drain_lock:
            done = self._drain_done
        return done.wait(timeout) if done is not None else True

    # -- driving ---------------------------------------------------------------------------

    def pump(self) -> PumpOutcome:
        """Stop-check, run the broker off-thread, advance every job, then run the watchdog."""
        now = self.clock()
        with self._lock:
            for record in self._order():
                self._pre_check_stop(record, now)
        self._run_broker()  # off the pump thread, bounded; the lock is released here
        with self._lock:
            self._note_drain_errors(now)
            self._reconcile_late(now)
            for record in self._order():
                self._advance(record, now)
            self._run_watchdog(now)
            outcome = self._outcome()
        self._sync_hub(now)  # outside the job lock: the feed may talk to a remote hub
        return outcome

    def _sync_hub(self, now: float) -> None:
        """Report presence and flush the lease/presence feed (S11 wiring, P5).

        Both are called on every pump: ``PresenceReporter.sync`` turns the presence guard's
        verdict (refreshed by the broker's tick) into hub presence events, then
        ``NodeFeed.flush`` sends what is queued (it backs off by itself while the hub is
        unreachable). A concurrent pump skips it rather than flush the same feed twice; a
        feed error never escapes into the pump."""
        if not self._hub_sync_lock.acquire(blocking=False):
            return
        try:
            try:
                self.presence_reporter.sync(now, self.wall_clock())
            except Exception:  # a bad presence report must not stop the jobs
                _log.exception("job runner %s: presence sync failed", self.node_id)
            report = self.feed.flush()
            if report.stopped:
                _log.error("job runner %s: the hub replaced this node's feed", self.node_id)
            self.last_flush = report
        finally:
            self._hub_sync_lock.release()

    def run_until_settled(self, max_pumps: int = 10000) -> list[PumpOutcome]:
        """Pump until no job can make progress on its own. For tests and batch drains."""
        outcomes: list[PumpOutcome] = []
        for _ in range(max_pumps):
            outcome = self.pump()
            outcomes.append(outcome)
            if outcome.active == 0:
                return outcomes
        raise RuntimeError(f"jobs still active after {max_pumps} pumps")

    def _order(self) -> list[JobRecord]:
        return sorted(self._jobs.values(), key=lambda r: (-r.priority, r.created_at, r.job_id))

    def _run_broker(self) -> None:
        """Tick the broker on a worker thread -- one drain at a time -- and wait for it.

        It ticks even with nothing queued: that is how the broker restores the human's context
        after the agents are done. A hung gateway call keeps the worker busy; the pump returns
        after ``drain_budget`` and the watchdog can stall the job. The worker is never killed
        (DESIGN §6); a concurrent or later pump waits on the same drain instead of starting a
        second one, so Resolve never sees two calls at once.
        """
        with self._drain_lock:
            if self._closed:
                return
            done = self._drain_done
            if done is None or done.is_set():
                done = threading.Event()
                thread = threading.Thread(
                    target=self._drain, args=(done,), name=f"aar-drain-{self.node_id}", daemon=True
                )
                self._drain_thread, self._drain_done = thread, done
                thread.start()
        done.wait(self._drain_budget)

    def _drain(self, done: threading.Event) -> None:
        try:
            if _drain_delay:
                time.sleep(_drain_delay)
            self.broker.run_until_idle()
        except Exception as exc:  # op failures live in their OpResults; this is the broker
            _log.exception("job runner %s: the broker crashed during a drain", self.node_id)
            error = {
                "code": "broker_crashed",
                "exception": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc()[-4000:],
            }
            with self._drain_lock:
                self._drain_errors.append(error)
        finally:
            done.set()

    def _note_drain_errors(self, now: float) -> None:
        """Record a broker crash on every job whose op it left outstanding (once per error)."""
        with self._drain_lock:
            errors, self._drain_errors = self._drain_errors, []
        if not errors:
            return
        error = errors[-1]
        brief = {k: error[k] for k in ("code", "exception", "message")}
        for job in self._jobs.values():
            rt = self._runtime[job.job_id]
            if job.state in TERMINAL_STATES or rt.pending_op_id is None:
                continue
            if (
                rt.broker_error is not None
                and {k: rt.broker_error[k] for k in ("code", "exception", "message")} == brief
            ):
                continue  # the same crash again: already recorded (bounded events)
            rt.broker_error = error
            self._checkpoint(job, [("broker_error", brief | {"count": len(errors)}, now)], now)

    def _reconcile_late(self, now: float) -> None:
        """Handle the late result of an op a cancelled/failed job dropped while it ran (P15)."""
        for op_id, job_id in list(self._late.items()):
            result = self.broker.result(op_id)
            if result is None:
                continue
            del self._late[op_id]
            self._abandoned.discard(op_id)
            job = self._jobs.get(job_id)
            if job is None:
                continue
            data = {
                "op_id": op_id,
                "status": result.status.value,
                "code": result.error.code if result.error is not None else None,
            }
            events = [("late_result", data, now)]
            value = result.value if result.status is OpStatus.OK else None
            events += self._orphan_alerts(job, value, now)
            self._checkpoint(job, events, now)

    # -- broker hooks (called on the drain thread) -----------------------------------------

    def _wrap(self, job_id: str, hot: HotOp) -> HotOp:
        """Give an op this job's precheck and journal."""
        inner = hot.fn
        journal = functools.partial(self._journal, job_id, hot.id)

        def run(gateway: Any) -> Any:
            with op_journal(journal):
                return inner(gateway)

        return replace(hot, fn=run, precheck=functools.partial(self._precheck, job_id, hot.id))

    def _op_wanted(self, job_id: str, op_id: str) -> bool:
        record = self._jobs.get(job_id)
        return (
            not self._closed
            and record is not None
            and record.state not in TERMINAL_STATES
            and op_id not in self._abandoned
        )

    def _precheck(self, job_id: str, op_id: str) -> BrokerOpError | None:
        """Right before the op runs: refuse it if the job no longer wants it or is stopped."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or not self._op_wanted(job_id, op_id):
                return BrokerOpError("job_inactive", f"job {job_id} no longer wants op {op_id}")
            target = record.target
            if target.project_id in self._stalled_projects:
                return BrokerOpError(
                    "project_stalled", f"project {target.project_id} is stalled; nothing runs"
                )
        try:
            self.hublink.assert_running(project=target, node_id=self.node_id)
        except Stopped as stop:
            return BrokerOpError("emergency_stop", str(stop), {"scope": str(stop.scope)})
        except OSError as exc:  # fail closed: a stop we cannot read may be in force
            return BrokerOpError("hub_unreachable", f"the hub cannot be reached: {exc}")
        return None

    def _journal(self, job_id: str, op_id: str, event: str, data: dict[str, Any]) -> bool:
        """Persist an op's side effect on its job's checkpoint; say whether the job wants it."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.readonly:
                return False
            wanted = self._op_wanted(job_id, op_id)
            if event == "deleting":  # render.delete_job: only the render job this job recorded
                ours = record.cursor.get("render_job_id")
                return wanted and ours is not None and data.get("render_job_id") == ours
            now = self.clock()
            journal = record.cursor.setdefault("journal", {})
            if event == "snapshot_saving":
                journal["snapshot_deleted"] = False
            if event == "snapshot_deleted":
                journal["snapshot_deleted"] = True
            elif event == "deleted":
                if journal.get("render_job_id") == data.get("render_job_id"):
                    journal.pop("render_job_id", None)
            else:
                journal.update(data)
            if event not in _NOT_A_STATE:
                journal["state"] = event
            events: list[tuple[str, dict[str, Any], float]] = []
            if event in _JOURNALED_EVENTS:
                events.append(("journal", {"event": event, "late": not wanted, **data}, now))
            if event == "orphaned":
                events.append(
                    _alert(
                        "render_orphaned",
                        f"render job {data.get('render_job_id')} was added by this job but could "
                        "not be removed; it is not tracked",
                        "remove it from the render queue on the Deliver page (only that JobId)",
                        {
                            "render_job_id": data.get("render_job_id"),
                            "cleanup": data.get("cleanup"),
                        },
                        now,
                    )
                )
            if event == "restore_failed":
                events.append(
                    _alert(
                        "deliver_not_restored",
                        "the Deliver settings could not be put back after the render was set up",
                        f"load the preset {data.get('snapshot')!r} on the Deliver page if it is "
                        "still there (it holds the original settings), then delete it",
                        {
                            "restore_error": data.get("restore_error"),
                            "snapshot": data.get("snapshot"),
                        },
                        now,
                    )
                )
            self._checkpoint(record, events, now)
            return wanted

    # -- the step machine ------------------------------------------------------------------

    def _stop_of(self, job: JobRecord) -> Stopped | OSError | None:
        """Why the job may not run now: an emergency stop, or a hub it cannot reach (a stop it
        cannot read may be in force, so that pauses it too: fail closed). ``None``: it may."""
        try:
            self.hublink.assert_running(project=job.target, node_id=self.node_id)
        except Stopped as stop:
            return stop
        except OSError as exc:
            return exc
        return None

    def _enter_wait_blocked(self, job: JobRecord, why: Stopped | OSError, now: float) -> None:
        if isinstance(why, Stopped):
            self._enter_wait_stop(job, why, now)
        else:
            self._enter_wait_unreachable(job, why, now)

    def _pre_check_stop(self, job: JobRecord, now: float) -> None:
        """Pause a stopped job (cancelling its queued op) before the broker runs anything (H3)."""
        if job.state in TERMINAL_STATES or job.state is JobState.STALLED or job.readonly:
            return
        if job.target.project_id in self._stalled_projects:
            return
        stop = self._stop_of(job)
        if stop is not None:
            self._enter_wait_blocked(job, stop, now)

    def _advance(self, job: JobRecord, now: float) -> None:
        if job.readonly:
            return
        rt = self._runtime[job.job_id]
        if rt.branch_records:  # a finished job's branch records are sent too
            self._retry_branch_records(job, rt, now)
        if job.state in TERMINAL_STATES or job.state is JobState.STALLED:
            return
        pid = job.target.project_id
        if pid in self._stalled_projects:
            return  # a stalled project stops scheduling for every job in it
        if rt.verify_pending:
            try:
                self._verify_approval(job, now)
            except OSError as exc:
                self._enter_wait_unreachable(job, exc, now)
                return
            rt.verify_pending = False

        if rt.pending_op_id is not None:
            result = self.broker.result(rt.pending_op_id)
            if result is None:
                stop = self._stop_of(job)
                if stop is not None:
                    self._enter_wait_blocked(job, stop, now)  # a queued op is cancelled
                elif job.state is JobState.WAITING:
                    # The stop lifted while an op it could not cancel is still out: run again,
                    # so the watchdog covers that op.
                    events = self._begin_running(job, rt, now)
                    rt.op_submitted_at = now
                    job.updated_at = now
                    self._checkpoint(job, events, now)
                return  # still outstanding; the watchdog may stall it
            # The op came back: its result is ALWAYS collected, even if a stop landed while it
            # ran (P3c) -- dropping it would make the step run the op a second time.
            op_name, rt.pending_op_name = rt.pending_op_name, None
            rt.pending_op_id = None
            rt.op_submitted_at = None
            rt.broker_error = None
            rt.op_outcome = None if _not_run(result) else OpOutcome.from_result(result)
            if op_name == _BRANCH_OP and result.status is OpStatus.OK:
                self._record_branch(job, rt, result.value, now)

        stop = self._stop_of(job)
        if stop is not None:
            self._enter_wait_blocked(job, stop, now)  # a collected outcome waits in rt.op_outcome
            return

        if job.state is JobState.WAITING:
            assert job.waiting is not None
            if (
                job.waiting.kind == "needs_approval"
                and rt.op_outcome is None
                and not self._approval_retryable(job)
            ):
                return  # waiting for a human to offer a different token

        steps = rt.steps
        if job.current_step >= len(steps):
            self._succeed(job, now)
            return
        step = steps[job.current_step]
        if step.renders_project and not self._acquire_render_slot(job):
            return  # another render holds this project's render slot; wait our turn
        ctx = StepContext(
            job_id=job.job_id,
            params=rt.params,
            target=job.target,
            prior=job.results,
            cursor=job.cursor,
            outcome=rt.op_outcome,
        )
        rt.op_outcome = None
        action = step.advance(ctx)

        if isinstance(action, Submit):
            self._handle_submit(job, rt, step, action, now)
        elif isinstance(action, Done):
            self._complete_step(job, rt, step, action.value, now)
        elif isinstance(action, Fail):
            self._fail(job, action.error, now)
        elif isinstance(action, Pause):
            self._note_progress(job, action, now)
        else:  # pragma: no cover - the action union is closed
            raise TypeError(f"step {step.name} returned {type(action).__name__}")

    def _note_progress(self, job: JobRecord, action: Pause, now: float) -> None:
        """A polling step: persist its progress mark when it changes, and forget the node
        restarts counted against the step once it demonstrably moved on (LOW-4). No event per
        poll; one ``progress_after_restart`` event when a count is forgotten."""
        if action.progress is None or job.cursor.get("_progress") == action.progress:
            return  # no observable progress: nothing to persist (polls are not checkpointed)
        job.cursor["_progress"] = action.progress
        events: list[tuple[str, dict[str, Any], float]] = []
        if job.restarts:
            events.append(("progress_after_restart", {"restarts": job.restarts}, now))
            job.restarts = 0
        job.updated_at = now
        self._checkpoint(job, events, now)

    def _handle_submit(
        self, job: JobRecord, rt: _Runtime, step: Step, action: Submit, now: float
    ) -> None:
        try:
            spec = get_op(action.op)
        except KeyError as exc:
            self._fail(job, StructuredError("unknown_op", str(exc), "use a registered op"), now)
            return
        holder = Holder.agent(job.agent_id or "")
        override: str | None = None
        try:
            try:
                # The S11 branch guard is on, in every project (P1): the op must edit a branch
                # this job's agent owns, or be exempt (timeline.branch/duplicate, render ops).
                hot = make_hot_op(
                    action.op,
                    action.params,
                    target=job.target,
                    holder=holder,
                    priority=job.priority,
                    branch_registry=self.branch_registry,
                )
            except OpError as exc:
                if exc.code not in BRANCH_GUARD_CODES:
                    raise
                # P4: an op the guard refuses runs only after a consumed project.write approval
                # that names this op, then past the guard for this op alone.
                override = _override_key(job, action)
                if override not in job.approvals and not self._consume_approval(
                    job, override, now, guard=exc
                ):
                    return
                hot = self._override_op(job, action, holder)
            # F15: the first mutating op against a production project needs a consumed human
            # token, at the moment of the write. An override approval (above) is one.
            if (
                spec.mutates
                and self.policy.is_production(job.target.project_id)
                and not job.approvals
                and not self._consume_approval(job, _PROJECT_APPROVAL, now)
            ):
                return
            op_id = self.broker.submit(self._wrap(job.job_id, hot))
        except HotOpFailure as exc:
            self._fail(job, StructuredError.from_broker(exc), now)
            return
        except (ValidationError, ValueError) as exc:
            self._fail(
                job, StructuredError("op_submit_error", str(exc), "check the op params"), now
            )
            return
        rt.pending_op_id = op_id
        rt.pending_op_name = action.op
        rt.op_submitted_at = now
        events = self._begin_running(job, rt, now)
        # Polls are high-frequency; do not emit an event for each (bounds the log, L3).
        if action.key != "poll":
            events.append(
                ("op_submitted", {"op": action.op, "key": action.key, "op_id": op_id}, now)
            )
        if override is not None:
            events.append(
                (
                    "guard_override",
                    {
                        "op": action.op,
                        "key": override,
                        "token_id": job.approvals[override],
                        "op_id": op_id,
                    },
                    now,
                )
            )
        job.updated_at = now
        self._checkpoint(job, events, now)

    def _override_op(self, job: JobRecord, action: Submit, holder: Holder) -> HotOp:
        """The op past the branch guard. This is the ONLY production call with
        ``require_branch=False`` (P4; pinned by a test that greps the source): it is reached
        only once ``_consume_approval`` has consumed a ``project.write`` approval naming this
        op for this job, and it builds exactly that one op."""
        if _override_key(job, action) not in job.approvals:  # pragma: no cover - defence
            raise RuntimeError("an override needs a consumed approval for this op")
        return make_hot_op(
            action.op,
            action.params,
            target=job.target,
            holder=holder,
            priority=job.priority,
            require_branch=False,
            branch_registry=self.branch_registry,
        )

    # -- approval --------------------------------------------------------------------------

    def _approval_retryable(self, job: JobRecord) -> bool:
        """Whether the offered token is worth trying: present and not the one already refused."""
        token = job.approval_token
        if token is None:
            return False
        refused = job.waiting.detail.get("refused_token") if job.waiting is not None else None
        return token != refused

    def _consume_approval(
        self, job: JobRecord, key: str, now: float, *, guard: OpError | None = None
    ) -> bool:
        """Consume the offered token as approval ``key`` (``"project"`` or an op override).

        Returns whether the approval is now held. Otherwise the job waits: for a token, for a
        different one (refused), or for the hub (unreachable: nothing is marked refused, and a
        consume that may have committed is recovered from the pending intent)."""
        target = self._approval_target(job, key)
        token = job.approval_token
        if job.approval_pending is not None and job.approval_pending_key == key:
            # A consume of this token was started and its answer lost (hub unreachable).
            try:
                if self._consume_verified(job, job.approval_pending, key):
                    self._hold_approval(job, key, job.approval_pending, None, now)
                    return True
            except OSError as exc:
                self._enter_wait_unreachable(job, exc, now)
                return False
        if token is None or (
            job.waiting is not None and token == job.waiting.detail.get("refused_token")
        ):
            self._enter_wait_approval(job, key, target, now, None, guard)
            return False
        # L2: record the intent before consuming, so a crash in between is recovered on load.
        job.approval_pending, job.approval_pending_key = token, key
        job.updated_at = now
        self._checkpoint(
            job,
            [("approval_consuming", {"token_id": token, "target": target, "key": key}, now)],
            now,
        )
        try:
            approval = self.hublink.consume_approval(
                token, _APPROVAL_SCOPE, target, self._job_actor(job)
            )
        except ApprovalError as exc:
            job.approval_pending = job.approval_pending_key = None
            self._enter_wait_approval(job, key, target, now, exc, guard)
            return False
        except OSError as exc:  # unknown outcome: keep the intent, ask again when reachable
            self._enter_wait_unreachable(job, exc, now)
            return False
        self._hold_approval(job, key, token, approval.consumed_at, now)
        return True

    def _hold_approval(
        self, job: JobRecord, key: str, token: str, consumed_at: str | None, now: float
    ) -> None:
        job.approvals[key] = token
        job.approval_consumed = True
        job.approval_pending = job.approval_pending_key = None
        if job.approval_token == token:
            job.approval_token = None  # the offered slot is spent; a later approval needs a new one
        job.updated_at = now
        self._checkpoint(
            job,
            [
                (
                    "approval_consumed",
                    {
                        "token_id": token,
                        "key": key,
                        "scope": _APPROVAL_SCOPE.value,
                        "target": self._approval_target(job, key),
                        "consumed_at": consumed_at,
                        "by": {"kind": "agent", "id": self._job_actor(job).id},
                    },
                    now,
                )
            ],
            now,
        )

    def _enter_wait_approval(
        self,
        job: JobRecord,
        key: str,
        target: str,
        now: float,
        refusal: ApprovalError | None,
        guard: OpError | None = None,
    ) -> None:
        token = job.approval_token
        detail: dict[str, Any] = {"scope": _APPROVAL_SCOPE.value, "target": target, "key": key}
        if guard is not None:
            op = key.rsplit("/", 1)[-1]
            detail["op"] = op
            detail["guard"] = {"code": guard.code, "message": guard.message}
            need = (
                f"op {op} is refused by the branch guard ({guard.code}: {guard.message}); it "
                f"may run only with a human approval that names it"
            )
        else:
            need = f"writing to production project {target} needs a human approval token"
        if refusal is not None:
            detail["refused_code"] = refusal.code
            detail["refused_message"] = str(refusal)
            detail["refused_token"] = token  # do not retry this token (M4)
            message = (
                f"the approval token was refused ({refusal.code}); {need}: a valid, unconsumed "
                f"'{_APPROVAL_SCOPE.value}' token on target {target!r}"
            )
        else:
            message = need
        reason = WaitReason(
            "needs_approval",
            "needs_approval",
            message,
            fix_hint=(
                f"ask a human to grant an approval with scope '{_APPROVAL_SCOPE.value}' on "
                f"target '{target}', then call approve(job_id, token_id)"
            ),
            detail=detail,
        )
        self._enter_wait(job, reason, now)

    def _enter_wait_unreachable(self, job: JobRecord, exc: OSError, now: float) -> None:
        reason = WaitReason(
            "hub_unreachable",
            "hub_unreachable",
            f"the hub cannot be reached ({type(exc).__name__}); the job waits, fail-closed",
            fix_hint="nothing to do by hand: the job carries on once the hub answers again",
            detail={"exception": type(exc).__name__},
        )
        self._enter_wait(job, reason, now)

    # -- hub records -----------------------------------------------------------------------

    def _record_branch(self, job: JobRecord, rt: _Runtime, value: object, now: float) -> None:
        """Record a ``timeline.branch`` result on the hub (P5): under the hub's ``prj_`` id
        (looked up from the Resolve project), written as this node (the feed's node id). A
        reused branch is recorded again: the hub keeps the first node's record and accepts
        the same version from it (idempotent), so a record a crash lost is caught up."""
        if not isinstance(value, BranchResult):
            return
        record = {
            "source_timeline_id": value.source.unique_id,
            "source_timeline_name": value.source.name,
            "agent_id": value.agent_id,
            "branch_timeline_id": value.timeline.unique_id,
            "branch_timeline_name": value.timeline.name,
            "version": value.version,
            "handoff": f"job {job.job_id}",
        }
        if self._send_branch_record(job, record, now) == "unreachable":
            rt.branch_records = [*(rt.branch_records or []), record]
            self._checkpoint(job, [("branch_record_pending", record, now)], now)

    def _send_branch_record(self, job: JobRecord, record: dict[str, Any], now: float) -> str:
        """Send one record: ``"recorded"``, ``"refused"`` (alerted) or ``"unreachable"``."""
        try:
            hub_id = self.hublink.record_branch(job.target, self.actor, **record)
        except OSError:  # unreachable: the caller keeps it and tries again on a later pump
            return "unreachable"
        except ValueError as exc:  # refused: no single hub project, or the hub said no
            self._checkpoint(
                job,
                [
                    ("branch_not_recorded", {**record, "error": str(exc)}, now),
                    _alert(
                        "branch_not_recorded",
                        f"agent branch {record['branch_timeline_name']!r} could not be recorded "
                        f"on the hub: {exc}",
                        "register the project in the hub registry with a source naming this "
                        "Resolve project, then record the branch",
                        {"branch_timeline_id": record["branch_timeline_id"]},
                        now,
                    ),
                ],
                now,
            )
            return "refused"
        self._checkpoint(job, [("branch_recorded", {**record, "hub_project_id": hub_id}, now)], now)
        return "recorded"

    def _retry_branch_records(self, job: JobRecord, rt: _Runtime, now: float) -> None:
        """Send the records an unreachable hub held back, in order; stop at the first that
        is still unreachable (no event per retry: the pending event was recorded once)."""
        pending = list(rt.branch_records or [])
        while pending and self._send_branch_record(job, pending[0], now) != "unreachable":
            pending.pop(0)
        rt.branch_records = pending or None

    def _requeue_branch_records(self, record: JobRecord, rt: _Runtime) -> None:
        """On load: branch records a previous process could not send are sent again."""
        open_: dict[tuple[str, int], dict[str, Any]] = {}
        for event in self.store.events(record.job_id):
            data = event.data
            key = (str(data.get("branch_timeline_id")), int(data.get("version") or 0))
            if event.kind == "branch_record_pending":
                open_[key] = {k: v for k, v in data.items() if k != "error"}
            elif event.kind in ("branch_recorded", "branch_not_recorded"):
                open_.pop(key, None)
        if open_:
            rt.branch_records = list(open_.values())

    # -- stop ------------------------------------------------------------------------------

    def _enter_wait_stop(self, job: JobRecord, stop: Stopped, now: float) -> None:
        reason = WaitReason(
            "stop",
            "emergency_stop",
            str(stop),
            fix_hint="a human must lift the stop from the console (aar.hub.control.resume)",
            detail={
                "scope": str(stop.scope),
                "stop_reason": stop.reason,
                "by": {"kind": stop.by.kind, "id": stop.by.id},
            },
        )
        self._enter_wait(job, reason, now)

    # -- transitions -----------------------------------------------------------------------

    def _begin_running(
        self, job: JobRecord, rt: _Runtime, now: float
    ) -> list[tuple[str, dict[str, Any], float]]:
        """Move a queued or waiting job to running (once) and resume its step deadline."""
        if job.state is JobState.RUNNING:
            if rt.deadline_anchor is None:
                rt.deadline_anchor = now
            return []
        previous = job.state
        job.state = JobState.RUNNING
        job.waiting = None
        rt.deadline_anchor = now  # the step's countdown resumes from its persisted remaining
        return [
            (
                "resumed" if previous is JobState.WAITING else "running",
                {"from": previous.value},
                now,
            )
        ]

    def _enter_wait(self, job: JobRecord, reason: WaitReason, now: float) -> None:
        rt = self._runtime[job.job_id]
        self._drop_pending(job, abandon=False)  # a queued op is cancelled; a running one kept
        if job.state is JobState.WAITING and job.waiting == reason:
            return  # already paused for exactly this; no churn
        self._pause_deadline(job, rt, now)
        job.state = JobState.WAITING
        job.waiting = reason
        job.updated_at = now
        self._checkpoint(job, [("waiting", reason.to_dict(), now)], now)

    def _complete_step(
        self, job: JobRecord, rt: _Runtime, step: Step, value: dict[str, Any], now: float
    ) -> None:
        events = self._begin_running(job, rt, now)
        index = job.current_step
        job.results.append(value)
        job.current_step += 1
        job.cursor = {}
        job.restarts = 0  # the step made progress
        if job.current_step < len(rt.steps):
            job.step_deadline = rt.steps[job.current_step].timeout
            rt.deadline_anchor = now  # the next step gets its own deadline
        job.updated_at = now
        events.append(("step_completed", {"index": index, "step": step.name, "value": value}, now))
        self._checkpoint(job, events, now)
        if job.current_step >= len(rt.steps):
            self._succeed(job, now)

    def _succeed(self, job: JobRecord, now: float) -> None:
        measured = job.results[-1] if job.results else {}
        self._terminate(job, JobState.SUCCEEDED, "succeeded", {"measured": measured}, now)

    def _fail(self, job: JobRecord, error: StructuredError, now: float) -> None:
        in_flight = self._drop_pending(job, abandon=True)
        job.error = error
        events: list[tuple[str, dict[str, Any], float]] = []
        if error.code == "render_build_ambiguous":
            events.append(
                _alert(
                    "render_build_ambiguous",
                    error.message,
                    error.fix_hint,
                    {"candidates": error.detail.get("candidates")},
                    now,
                )
            )
        elif not in_flight and error.code not in _RENDER_ENDED:
            events += self._orphan_alerts(job, None, now)
        self._terminate(job, JobState.FAILED, "failed", {"error": error.to_dict()}, now, events)

    def _stall(self, job: JobRecord, error: StructuredError, now: float) -> None:
        rt = self._runtime[job.job_id]
        if rt.broker_error is not None:  # say WHY the op never returned (finding 10)
            error = replace(error, detail=error.detail | {"broker_error": rt.broker_error})
        self._drop_pending(job, abandon=False)  # a running op is kept: it may still come back
        job.stall = error
        job.state = JobState.STALLED
        job.updated_at = now
        if job.target.project_id is not None:
            self._stalled_projects.add(job.target.project_id)
        self._release_render_slot(job)
        self._checkpoint(job, [("stalled", {"reason": error.to_dict()}, now)], now)

    def _terminate(
        self,
        job: JobRecord,
        state: JobState,
        event: str,
        data: dict[str, Any],
        now: float,
        extra: list[tuple[str, dict[str, Any], float]] | None = None,
    ) -> None:
        job.state = state
        job.updated_at = now
        self._release_render_slot(job)
        self._checkpoint(job, [*(extra or []), (event, data, now)], now)

    def _drop_pending(self, job: JobRecord, *, abandon: bool) -> bool:
        """Let go of the job's op. A queued op is cancelled. An op that already ran or is
        running is never silently dropped: for a pause (``abandon=False``) it stays tracked
        and its result is collected later; a cancelled/failed job (``abandon=True``) abandons
        it (its precheck and journal now refuse it) and reconciles its late result. Returns
        whether an op was abandoned (its late result is reconciled by the pump)."""
        rt = self._runtime[job.job_id]
        op_id = rt.pending_op_id
        if abandon:
            rt.op_outcome = None
        if op_id is None:
            return False
        if self.broker.cancel(op_id, reason="job paused"):
            rt.pending_op_id = None
            rt.op_submitted_at = None
            return False
        if not abandon:
            return False
        rt.pending_op_id = None
        rt.op_submitted_at = None
        self._abandoned.add(op_id)
        self._late[op_id] = job.job_id
        return True

    def _orphan_alerts(
        self, job: JobRecord, value: object, now: float
    ) -> list[tuple[str, dict[str, Any], float]]:
        """An alert if this ended job leaves a render it started running untracked."""
        rid = live_render_id(job.cursor, value)
        if rid is None or any(
            e.kind == "alert" and e.data.get("render_job_id") == rid
            for e in self.store.events(job.job_id)
        ):
            return []
        return [
            _alert(
                "render_orphaned",
                f"the job ended ({job.state.value}) while its render job {rid} may still be "
                "running in Resolve; nothing tracks it now",
                "check the render queue on the Deliver page; stop or remove that JobId by hand "
                "if it is not wanted",
                {"render_job_id": rid},
                now,
            )
        ]

    def _clear_project_stall(self, job: JobRecord) -> None:
        pid = job.target.project_id
        if pid is None or pid not in self._stalled_projects:
            return
        if not any(
            r.state is JobState.STALLED and r.target.project_id == pid for r in self._jobs.values()
        ):
            self._stalled_projects.discard(pid)

    # -- deadlines and checkpoints ---------------------------------------------------------

    def _pause_deadline(self, job: JobRecord, rt: _Runtime, now: float) -> None:
        """Stop the step deadline counting down (it does not run while a job waits, H6)."""
        if rt.deadline_anchor is not None and job.step_deadline is not None:
            job.step_deadline = max(0.0, job.step_deadline - (now - rt.deadline_anchor))
            rt.deadline_anchor = None

    def _checkpoint(
        self, job: JobRecord, events: list[tuple[str, dict[str, Any], float]], now: float
    ) -> None:
        """Save the job. A running step's REMAINING deadline is what gets persisted, so a
        restart resumes the countdown instead of granting a fresh deadline (P5b). The in-memory
        countdown (anchor + deadline at the anchor) is left as it is."""
        rt = self._runtime.get(job.job_id)
        if (
            rt is not None
            and job.state is JobState.RUNNING
            and rt.deadline_anchor is not None
            and job.step_deadline is not None
        ):
            remaining = job.step_deadline - (now - rt.deadline_anchor)
            self.store.save(replace(job, step_deadline=remaining), events)
            rt.persisted_at = now
            return
        self.store.save(job, events)

    # -- per-project render lock -----------------------------------------------------------

    def _acquire_render_slot(self, job: JobRecord) -> bool:
        pid = job.target.project_id
        if pid is None:
            return True
        holder = self._render_inflight.get(pid)
        if holder is None or holder == job.job_id:
            self._render_inflight[pid] = job.job_id
            return True
        return False  # another render holds this project's slot

    def _release_render_slot(self, job: JobRecord) -> None:
        pid = job.target.project_id
        if pid is not None and self._render_inflight.get(pid) == job.job_id:
            del self._render_inflight[pid]

    # -- watchdog --------------------------------------------------------------------------

    def _run_watchdog(self, now: float) -> None:
        rendering_elsewhere = set(self._render_inflight)
        for job in list(self._jobs.values()):
            if job.state is not JobState.RUNNING:
                continue
            pid = job.target.project_id
            if pid in self._stalled_projects:
                continue
            rt = self._runtime[job.job_id]
            # M2: a job whose op is deferred behind a render in ANOTHER project is not hung.
            op_submitted = rt.op_submitted_at
            if op_submitted is not None and (rendering_elsewhere - {pid}):
                op_submitted = None
            overdue = self.watchdog.overdue(
                now=now,
                step_started_at=rt.deadline_anchor,
                step_timeout=job.step_deadline,
                op_submitted_at=op_submitted,
            )
            if overdue is not None:
                self._stall(job, overdue, now)
            elif (
                rt.deadline_anchor is not None
                and job.step_deadline is not None
                and (rt.persisted_at is None or now - rt.persisted_at >= DEADLINE_PERSIST_EVERY)
            ):
                self._checkpoint(job, [], now)  # keep the persisted countdown current

    # -- views -----------------------------------------------------------------------------

    def _outcome(self) -> PumpOutcome:
        running = waiting = stalled = active = 0
        for job in self._jobs.values():
            if job.state is JobState.RUNNING:
                running += 1
            elif job.state is JobState.WAITING:
                waiting += 1
            elif job.state is JobState.STALLED:
                stalled += 1
            if (
                job.state in (JobState.QUEUED, JobState.RUNNING)
                and job.target.project_id not in self._stalled_projects
            ):
                active += 1
        return PumpOutcome(active=active, running=running, waiting=waiting, stalled=stalled)

    def _view(self, record: JobRecord) -> JobView:
        rt = self._runtime.get(record.job_id)
        steps_total = len(rt.steps) if rt is not None else record.current_step
        step_name = (
            rt.steps[record.current_step].name
            if rt is not None and record.current_step < len(rt.steps)
            else None
        )
        events = [event.to_dict() for event in self.store.events(record.job_id)]
        return JobView(
            job_id=record.job_id,
            kind=record.kind,
            node_id=record.node_id,
            state=record.state.value,
            target=record.target.to_dict(),
            current_step=record.current_step,
            steps_total=steps_total,
            step_name=step_name,
            priority=record.priority,
            results=list(record.results),
            measured=record.results[-1] if record.results else {},
            error=record.error.to_dict() if record.error is not None else None,
            waiting=record.waiting.to_dict() if record.waiting is not None else None,
            stall=record.stall.to_dict() if record.stall is not None else None,
            approval_token=record.approval_token,
            approval_consumed=record.approval_consumed,
            approvals=dict(record.approvals),
            agent_id=record.agent_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            events=events,
            alerts=[e["data"] for e in events if e["kind"] == "alert"],
        )


class _Empty(BaseModel):
    """A placeholder params model for a quarantined job (it never runs)."""


def op_approval_target(project_id: str, op: str) -> str:
    """The ``project.write`` approval target that lets op ``op`` past the branch guard in
    project ``project_id`` (P4): a human grants exactly this to approve exactly this op."""
    return f"{project_id}#op:{op}"


def _override_key(job: JobRecord, action: Submit) -> str:
    """The approval key of one guarded op occurrence: its step, its key in the step, its op.
    Stable across a restart, so a resubmitted op reuses the approval it already consumed."""
    return f"op:{job.current_step}/{action.key}/{action.op}"


def _not_run(result: OpResult) -> bool:
    """A precheck refused the op: it never ran, so it is re-issued, not reported as failed."""
    return (
        result.status is OpStatus.CANCELLED
        and result.error is not None
        and result.error.code in _NOT_RUN
    )


def _alert(
    code: str, message: str, fix_hint: str, detail: dict[str, Any], now: float
) -> tuple[str, dict[str, Any], float]:
    _log.warning("job alert %s: %s", code, message)
    return ("alert", {"code": code, "message": message, "fix_hint": fix_hint, **detail}, now)


def _actor_dict(actor: Actor | None) -> dict[str, str] | None:
    return {"kind": actor.kind, "id": actor.id} if actor is not None else None
