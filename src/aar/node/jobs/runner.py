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
from dataclasses import asdict, dataclass, replace
from typing import Any

from pydantic import BaseModel, ValidationError

from aar.hub.approvals import ApprovalError, ApprovalScope
from aar.hub.control import Stopped
from aar.hub.events import Actor
from aar.node.broker import Broker, HotOp, HotOpFailure, OpResult, OpStatus
from aar.node.broker import OpError as BrokerOpError
from aar.node.hublink import HubLink
from aar.node.leases import Clock, Holder
from aar.node.locator import ProjectLocator
from aar.node.ops import get_op, make_hot_op
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

_NOT_RUN = frozenset({"emergency_stop", "job_inactive", "project_stalled"})
"""Precheck refusals: the op never ran, so the step re-issues it later (not a failure)."""

_JOURNALED_EVENTS = frozenset({"added", "started", "deleted", "orphaned", "restore_failed"})
"""Journal entries that also become job events (a handful per build: bounded)."""

_NOT_A_STATE = frozenset(
    {"build_started", "snapshot_recovered", "snapshot_deleted", "restore_failed"}
)
"""Journal entries that record a fact without moving the build's ``state``."""

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


class JobRunner:
    """Submits, drives, checkpoints and reports node jobs over one broker."""

    def __init__(
        self,
        broker: Broker,
        store: JobStore,
        hublink: HubLink,
        *,
        node_id: str,
        clock: Clock = time.monotonic,
        holder: Holder | None = None,
        actor: Actor | None = None,
        approval: ApprovalPolicy | None = None,
        watchdog: Watchdog | None = None,
        drain_budget: float | None = _DRAIN_BUDGET,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
    ) -> None:
        if not node_id:
            raise ValueError("a job runner needs a non-empty node_id")
        if drain_budget is not None and not (math.isfinite(drain_budget) and drain_budget > 0):
            raise ValueError("drain_budget must be a positive number of seconds, or None")
        if max_restarts < 0:
            raise ValueError("max_restarts must not be negative")
        self.broker = broker
        self.store = store
        self.hublink = hublink
        self.node_id = node_id
        self.clock = clock
        self.holder = holder or Holder.agent(f"job-runner:{node_id}")
        self.actor = actor or Actor("node", node_id)
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
            self._verify_approval(record, now)
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

    def _job_actor(self, job_id: str) -> Actor:
        """The actor a job consumes its approval as: binds the hub's consume to this job."""
        return Actor(self.actor.kind, f"{self.actor.id}#{job_id}")

    def _consume_verified(self, record: JobRecord, token_id: str | None) -> bool:
        """Whether the hub shows ``token_id`` consumed by THIS job, for its scope and target."""
        if token_id is None:
            return False
        approval = self.hublink.approval(token_id)
        return (
            approval is not None
            and approval.consumed_at is not None
            and approval.consumed_by == self._job_actor(record.job_id)
            and approval.scope is _APPROVAL_SCOPE
            and approval.target == (record.target.project_id or "")
        )

    def _verify_approval(self, record: JobRecord, now: float) -> None:
        """On load: trust ``approval_consumed`` only when the hub shows this job's own consume
        (L1, P13b), and finish a consume a crash interrupted (L2): ``approval_pending`` is the
        token a consume was started with; if the hub shows it consumed by this job, the
        consume happened and is recovered, otherwise the intent is dropped."""
        if record.approval_consumed and not self._consume_verified(record, record.approval_token):
            record.approval_consumed = False
            self.store.save(
                record, [("approval_unverified", {"token_id": record.approval_token}, now)]
            )
        pending = record.approval_pending
        if pending is None:
            return
        record.approval_pending = None
        if not record.approval_consumed and self._consume_verified(record, pending):
            record.approval_consumed = True
            record.approval_token = pending
            event = ("approval_recovered", {"token_id": pending}, now)
        else:
            event = ("approval_pending_dropped", {"token_id": pending}, now)
        self.store.save(record, [event])

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
        priority: int = 0,
        approval_token: str | None = None,
    ) -> str:
        """Queue a new job and return its id. The caller drives it with :meth:`pump`."""
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
                        {"kind": kind, "target": target.to_dict(), "steps": len(steps)},
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
            return self._outcome()

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
        return None

    def _journal(self, job_id: str, op_id: str, event: str, data: dict[str, Any]) -> bool:
        """Persist an op's side effect on its job's checkpoint; say whether the job wants it."""
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.readonly:
                return False
            wanted = self._op_wanted(job_id, op_id)
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

    def _stop_of(self, job: JobRecord) -> Stopped | None:
        try:
            self.hublink.assert_running(project=job.target, node_id=self.node_id)
        except Stopped as stop:
            return stop
        return None

    def _pre_check_stop(self, job: JobRecord, now: float) -> None:
        """Pause a stopped job (cancelling its queued op) before the broker runs anything (H3)."""
        if job.state in TERMINAL_STATES or job.state is JobState.STALLED or job.readonly:
            return
        if job.target.project_id in self._stalled_projects:
            return
        stop = self._stop_of(job)
        if stop is not None:
            self._enter_wait_stop(job, stop, now)

    def _advance(self, job: JobRecord, now: float) -> None:
        if job.state in TERMINAL_STATES or job.state is JobState.STALLED or job.readonly:
            return
        pid = job.target.project_id
        if pid in self._stalled_projects:
            return  # a stalled project stops scheduling for every job in it
        rt = self._runtime[job.job_id]

        if rt.pending_op_id is not None:
            result = self.broker.result(rt.pending_op_id)
            if result is None:
                stop = self._stop_of(job)
                if stop is not None:
                    self._enter_wait_stop(job, stop, now)  # a queued op is cancelled
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
            rt.pending_op_id = None
            rt.op_submitted_at = None
            rt.broker_error = None
            rt.op_outcome = None if _not_run(result) else OpOutcome.from_result(result)

        stop = self._stop_of(job)
        if stop is not None:
            self._enter_wait_stop(job, stop, now)  # a collected outcome waits in rt.op_outcome
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
            return  # the step is polling; nothing to persist
        else:  # pragma: no cover - the action union is closed
            raise TypeError(f"step {step.name} returned {type(action).__name__}")

    def _handle_submit(
        self, job: JobRecord, rt: _Runtime, step: Step, action: Submit, now: float
    ) -> None:
        try:
            spec = get_op(action.op)
        except KeyError as exc:
            self._fail(job, StructuredError("unknown_op", str(exc), "use a registered op"), now)
            return
        # The first mutating op against a production project consumes a human token here, at the
        # moment of the write. A missing or refused token leaves the job waiting.
        if (
            spec.mutates
            and self.policy.is_production(job.target.project_id)
            and not job.approval_consumed
            and not self._ensure_approval(job, now)
        ):
            return
        try:
            hot = make_hot_op(
                action.op,
                action.params,
                target=job.target,
                holder=self.holder,
                priority=job.priority,
            )
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
        rt.op_submitted_at = now
        events = self._begin_running(job, rt, now)
        # Polls are high-frequency; do not emit an event for each (bounds the log, L3).
        if action.key != "poll":
            events.append(
                ("op_submitted", {"op": action.op, "key": action.key, "op_id": op_id}, now)
            )
        job.updated_at = now
        self._checkpoint(job, events, now)

    # -- approval --------------------------------------------------------------------------

    def _approval_retryable(self, job: JobRecord) -> bool:
        """Whether the offered token is worth trying: present and not the one already refused."""
        token = job.approval_token
        if token is None:
            return False
        refused = job.waiting.detail.get("refused_token") if job.waiting is not None else None
        return token != refused

    def _ensure_approval(self, job: JobRecord, now: float) -> bool:
        target = job.target.project_id
        token = job.approval_token
        if token is None or (
            job.waiting is not None and token == job.waiting.detail.get("refused_token")
        ):
            self._enter_wait_approval(job, target, now, None)
            return False
        # L2: record the intent before consuming, so a crash in between is recovered on load.
        job.approval_pending = token
        job.updated_at = now
        self._checkpoint(
            job, [("approval_consuming", {"token_id": token, "target": target}, now)], now
        )
        try:
            approval = self.hublink.consume_approval(
                token, _APPROVAL_SCOPE, target or "", self._job_actor(job.job_id)
            )
        except ApprovalError as exc:
            job.approval_pending = None
            self._enter_wait_approval(job, target, now, exc)
            return False
        job.approval_consumed = True
        job.approval_pending = None
        job.updated_at = now
        self._checkpoint(
            job,
            [
                (
                    "approval_consumed",
                    {
                        "token_id": token,
                        "scope": _APPROVAL_SCOPE.value,
                        "target": target,
                        "consumed_at": approval.consumed_at,
                    },
                    now,
                )
            ],
            now,
        )
        return True

    def _enter_wait_approval(
        self, job: JobRecord, target: str | None, now: float, refusal: ApprovalError | None
    ) -> None:
        token = job.approval_token
        detail: dict[str, Any] = {"scope": _APPROVAL_SCOPE.value, "target": target}
        if refusal is not None:
            detail["refused_code"] = refusal.code
            detail["refused_message"] = str(refusal)
            detail["refused_token"] = token  # do not retry this token (M4)
            message = (
                f"the approval token was refused ({refusal.code}); writing to production project "
                f"{target} needs a valid, unconsumed human approval"
            )
        else:
            message = f"writing to production project {target} needs a human approval token"
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
            elif rt.persisted_at is None or now - rt.persisted_at >= DEADLINE_PERSIST_EVERY:
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
            created_at=record.created_at,
            updated_at=record.updated_at,
            events=events,
            alerts=[e["data"] for e in events if e["kind"] == "alert"],
        )


class _Empty(BaseModel):
    """A placeholder params model for a quarantined job (it never runs)."""


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
