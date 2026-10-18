"""The node job runner: drives checkpointed, resumable jobs over the session broker.

The runner owns no durable job state; the :class:`~aar.node.jobs.store.JobStore` does. A fresh
runner opened on the same store resumes every job from its last checkpoint, which is how a
node recovers after a restart. Only timers (step and op deadlines) live in memory and reset on
restart, which is correct: a restart is itself recovery, and the step re-runs idempotently.

Each pump makes one broker scheduling decision and then advances every live job by at most one
op, so several projects share the one Resolve instance without any job blocking another.

Gates, in order, around every step:

- **Emergency stop** (DESIGN R5, §5): before each step the runner asks the hub whether work in
  this project or node is stopped; if so the job pauses at its checkpoint as ``waiting`` and
  resumes automatically when a human lifts the stop.
- **Approval** (F15, §5.2): the first mutating op against a production project consumes a human
  ``project.write`` token at the moment of the write. Without a valid unconsumed token the job
  waits with a structured result naming exactly what to ask a human for.
- **Watchdog**: a step or op that runs past its deadline stalls the job and stops the project.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from aar.hub.approvals import ApprovalError, ApprovalScope
from aar.hub.control import Stopped
from aar.hub.events import Actor
from aar.node.broker import Broker, HotOpFailure
from aar.node.hublink import HubLink
from aar.node.leases import Clock, Holder
from aar.node.locator import ProjectLocator
from aar.node.ops import get_op, submit_op

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
from .store import JobRecord, JobStore
from .watchdog import Watchdog, WatchdogConfig

_APPROVAL_SCOPE = ApprovalScope.PROJECT_WRITE


@dataclass(frozen=True, slots=True)
class ApprovalPolicy:
    """When a mutating op needs a human approval token.

    ``live`` is whether this node runs against a real, production Resolve: it defaults to
    ``True`` so every real project counts as production until the D3 sandbox approval is given.
    The test fake gateway is not production, so tests set ``live=False``. ``sandbox_exempt``
    names project ids a human has marked as a sandbox; it defaults to none.
    """

    live: bool = True
    sandbox_exempt: frozenset[str] = frozenset()

    def is_production(self, project_id: str | None) -> bool:
        if not self.live:
            return False
        return project_id is not None and project_id not in self.sandbox_exempt


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Runtime:
    """Per-job state that lives only in this process (timers, the pending op, built steps)."""

    steps: list[Step]
    params: BaseModel
    pending_op_id: str | None = None
    op_submitted_at: float | None = None
    step_started_at: float | None = None
    op_outcome: OpOutcome | None = None


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
    ) -> None:
        if not node_id:
            raise ValueError("a job runner needs a non-empty node_id")
        self.broker = broker
        self.store = store
        self.hublink = hublink
        self.node_id = node_id
        self.clock = clock
        self.holder = holder or Holder.agent(f"job-runner:{node_id}")
        self.actor = actor or Actor("node", node_id)
        self.policy = approval or ApprovalPolicy()
        self.watchdog = watchdog or Watchdog(WatchdogConfig())
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._runtime: dict[str, _Runtime] = {}
        self._stalled_projects: set[str] = set()
        self._load()

    # -- startup ---------------------------------------------------------------------------

    def _load(self) -> None:
        now = self.clock()
        for record in self.store.load_all():
            self._jobs[record.job_id] = record
            self._runtime[record.job_id] = _Runtime(
                steps=self._build_steps(record),
                params=self._parse_params(record),
                step_started_at=now
                if record.state in (JobState.RUNNING, JobState.QUEUED)
                else None,
            )
            if record.state is JobState.STALLED and record.target.project_id is not None:
                self._stalled_projects.add(record.target.project_id)

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
        steps = spec.build(model, target)  # validates the plan (e.g. the ops exist)
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
                created_at=now,
                updated_at=now,
            )
            self._jobs[job_id] = record
            self._runtime[job_id] = _Runtime(steps=steps, params=model, step_started_at=now)
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

    def cancel(self, job_id: str, reason: str = "cancelled by request") -> bool:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in TERMINAL_STATES:
                return False
            self._drop_pending(record)
            self._terminate(
                record, JobState.CANCELLED, "cancelled", {"reason": reason}, self.clock()
            )
            self._clear_project_stall(record)
            return True

    def approve(self, job_id: str, token_id: str, actor: Actor | None = None) -> JobView | None:
        """Give a waiting job a human approval token and let it try again next pump.

        The token is consumed at the first write, not here, so a wrong token leaves the job
        waiting rather than consuming anything.
        """
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None or record.state in TERMINAL_STATES:
                return self._view(record) if record is not None else None
            record.approval_token = token_id
            record.updated_at = self.clock()
            # Re-activate a job that was waiting for exactly this, so the next pump retries it.
            if (
                record.state is JobState.WAITING
                and record.waiting is not None
                and (record.waiting.kind == "needs_approval")
            ):
                record.waiting = None
                record.state = JobState.QUEUED
            self.store.save(
                record,
                [
                    (
                        "approval_offered",
                        {"token_id": token_id, "by": _actor_dict(actor)},
                        record.updated_at,
                    )
                ],
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
            record.stall = None
            record.state = JobState.QUEUED
            record.updated_at = now
            rt = self._runtime[job_id]
            rt.step_started_at = now
            rt.op_submitted_at = None
            rt.pending_op_id = None
            self.store.save(
                record, [("resumed", {"from": "stalled", "by": _actor_dict(actor)}, now)]
            )
            self._clear_project_stall(record)
            return self._view(record)

    # -- driving ---------------------------------------------------------------------------

    def pump(self) -> PumpOutcome:
        """Make one broker decision, advance every live job once, then run the watchdog."""
        with self._lock:
            now = self.clock()
            self.broker.tick()
            for record in sorted(self._jobs.values(), key=lambda r: (-r.priority, r.created_at)):
                self._advance(record, now)
            self._run_watchdog(now)
            return self._outcome()

    def run_until_settled(self, max_pumps: int = 10000) -> list[PumpOutcome]:
        """Pump until no job can make progress on its own. For tests and batch drains.

        A job that is waiting (on a stop or approval) or stalled is settled; a human or agent
        action, or a lifted stop, is needed to move it. Raises if a job keeps the runner busy
        past ``max_pumps`` (for example a render that never finishes and whose deadline the
        caller has not advanced the clock past).
        """
        outcomes: list[PumpOutcome] = []
        for _ in range(max_pumps):
            outcome = self.pump()
            outcomes.append(outcome)
            if outcome.active == 0:
                return outcomes
        raise RuntimeError(f"jobs still active after {max_pumps} pumps")

    # -- the step machine ------------------------------------------------------------------

    def _advance(self, job: JobRecord, now: float) -> None:
        if job.state in TERMINAL_STATES or job.state is JobState.STALLED:
            return
        pid = job.target.project_id
        if pid in self._stalled_projects:
            return  # a stalled project stops scheduling for every job in it
        rt = self._runtime[job.job_id]

        # Emergency stop: checked before every step.
        try:
            self.hublink.assert_running(project_id=pid, node_id=self.node_id)
        except Stopped as stop:
            self._enter_wait_stop(job, stop, now)
            return

        if job.state is JobState.WAITING:
            assert job.waiting is not None
            if job.waiting.kind == "needs_approval" and job.approval_token is None:
                return  # still waiting for a human to grant and offer a token

        # Resolve the op this job last submitted (it may have run this pump's broker.tick).
        if rt.pending_op_id is not None:
            result = self.broker.result(rt.pending_op_id)
            if result is None:
                return  # still outstanding; the watchdog may stall it
            rt.pending_op_id = None
            rt.op_submitted_at = None
            rt.op_outcome = OpOutcome.from_result(result)

        steps = rt.steps
        if job.current_step >= len(steps):
            self._succeed(job, now)
            return
        step = steps[job.current_step]
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
            # No state change (the step is polling); the cursor is unchanged, so nothing to save.
            return
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
        # The first mutating op against a production project consumes a human token here, at
        # the moment of the write. A missing or refused token leaves the job waiting.
        if (
            spec.mutates
            and self.policy.is_production(job.target.project_id)
            and not job.approval_consumed
            and not self._ensure_approval(job, now)
        ):
            return
        try:
            op_id = submit_op(
                self.broker,
                action.op,
                action.params,
                target=job.target,
                holder=self.holder,
                priority=job.priority,
            )
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
        events = self._begin_running(job, now)
        events.append(("op_submitted", {"op": action.op, "key": action.key, "op_id": op_id}, now))
        job.updated_at = now
        self.store.save(job, events)

    # -- approval --------------------------------------------------------------------------

    def _ensure_approval(self, job: JobRecord, now: float) -> bool:
        target = job.target.project_id
        assert target is not None  # is_production() is False when it is None
        token = job.approval_token
        if token is None:
            self._enter_wait_approval(job, target, now, None)
            return False
        try:
            approval = self.hublink.consume_approval(token, _APPROVAL_SCOPE, target, self.actor)
        except ApprovalError as exc:
            self._enter_wait_approval(job, target, now, exc)
            return False
        job.approval_consumed = True
        job.updated_at = now
        self.store.save(
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
        )
        return True

    def _enter_wait_approval(
        self, job: JobRecord, target: str, now: float, refusal: ApprovalError | None
    ) -> None:
        detail: dict[str, Any] = {"scope": _APPROVAL_SCOPE.value, "target": target}
        if refusal is not None:
            detail["refused_code"] = refusal.code
            detail["refused_message"] = str(refusal)
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

    def _begin_running(self, job: JobRecord, now: float) -> list[tuple[str, dict[str, Any], float]]:
        """Move a queued or waiting job to running (once), returning the transition event."""
        if job.state is JobState.RUNNING:
            return []
        previous = job.state
        job.state = JobState.RUNNING
        job.waiting = None
        return [
            (
                "resumed" if previous is JobState.WAITING else "running",
                {"from": previous.value},
                now,
            )
        ]

    def _enter_wait(self, job: JobRecord, reason: WaitReason, now: float) -> None:
        if job.state is JobState.WAITING and job.waiting == reason:
            return  # already paused for exactly this; no churn
        self._drop_pending(job)
        job.state = JobState.WAITING
        job.waiting = reason
        job.updated_at = now
        self.store.save(job, [("waiting", reason.to_dict(), now)])

    def _complete_step(
        self, job: JobRecord, rt: _Runtime, step: Step, value: dict[str, Any], now: float
    ) -> None:
        events = self._begin_running(job, now)
        index = job.current_step
        job.results.append(value)
        job.current_step += 1
        job.cursor = {}
        rt.step_started_at = now  # the next step starts its own deadline
        job.updated_at = now
        events.append(("step_completed", {"index": index, "step": step.name, "value": value}, now))
        self.store.save(job, events)
        if job.current_step >= len(rt.steps):
            self._succeed(job, now)

    def _succeed(self, job: JobRecord, now: float) -> None:
        measured = job.results[-1] if job.results else {}
        self._terminate(job, JobState.SUCCEEDED, "succeeded", {"measured": measured}, now)

    def _fail(self, job: JobRecord, error: StructuredError, now: float) -> None:
        self._drop_pending(job)
        job.error = error
        self._terminate(job, JobState.FAILED, "failed", {"error": error.to_dict()}, now)

    def _stall(self, job: JobRecord, error: StructuredError, now: float) -> None:
        self._drop_pending(job)
        job.stall = error
        job.state = JobState.STALLED
        job.updated_at = now
        if job.target.project_id is not None:
            self._stalled_projects.add(job.target.project_id)
        self.store.save(job, [("stalled", {"reason": error.to_dict()}, now)])

    def _terminate(
        self, job: JobRecord, state: JobState, event: str, data: dict[str, Any], now: float
    ) -> None:
        job.state = state
        job.updated_at = now
        self.store.save(job, [(event, data, now)])

    def _drop_pending(self, job: JobRecord) -> None:
        """Take the job's op off the broker queue so a pause/stall/cancel truly stops work.

        The op is only cancelled if it has not run yet; if it already ran, the step re-runs it
        idempotently on resume. This only cancels work the runner itself queued; it never
        touches Resolve or any process.
        """
        rt = self._runtime[job.job_id]
        if rt.pending_op_id is not None:
            self.broker.cancel(rt.pending_op_id, reason="job paused")
            rt.pending_op_id = None
            rt.op_submitted_at = None
            rt.op_outcome = None

    def _clear_project_stall(self, job: JobRecord) -> None:
        pid = job.target.project_id
        if pid is None or pid not in self._stalled_projects:
            return
        if not any(
            r.state is JobState.STALLED and r.target.project_id == pid for r in self._jobs.values()
        ):
            self._stalled_projects.discard(pid)

    # -- watchdog --------------------------------------------------------------------------

    def _run_watchdog(self, now: float) -> None:
        for job in list(self._jobs.values()):
            if job.state is not JobState.RUNNING:
                continue
            if job.target.project_id in self._stalled_projects:
                continue
            rt = self._runtime[job.job_id]
            step = rt.steps[job.current_step] if job.current_step < len(rt.steps) else None
            overdue = self.watchdog.overdue(
                now=now,
                step_started_at=rt.step_started_at,
                step_timeout=step.timeout if step is not None else 0.0,
                op_submitted_at=rt.op_submitted_at,
            )
            if overdue is not None:
                self._stall(job, overdue, now)

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
            events=[event.to_dict() for event in self.store.events(record.job_id)],
        )


def _actor_dict(actor: Actor | None) -> dict[str, str] | None:
    return {"kind": actor.kind, "id": actor.id} if actor is not None else None
