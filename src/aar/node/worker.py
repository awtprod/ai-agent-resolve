"""The bounded outbound-only loop used by ``aar node run``.

The hub queue is coordination state; the node's :class:`JobStore` remains the execution
journal.  A hub id is committed as the local id before the first pump, making claim retries
and reconnects incapable of executing a second local copy.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from aar.node.broker import OpError as BrokerOpError
from aar.node.hublink import HubUnreachable
from aar.node.jobs import JobStore
from aar.node.jobs.model import TERMINAL_STATES
from aar.node.locator import ProjectLocator
from aar.node.remote import HubRefused, NodeNotEnrolled, RemoteJobClient
from aar.node.runtime import NodeRuntime

DEFAULT_HEARTBEAT_SECONDS = 15.0
DEFAULT_TICK_SECONDS = 0.25
MAX_BACKOFF_SECONDS = 30.0
_REMOTE_STATES = frozenset(
    {
        "waiting_for_node",
        "claimed",
        "node_offline",
        "cancel_requested",
        "lost_on_node",
        "queued",
        "running",
        "waiting",
        "stalled",
        *(state.value for state in TERMINAL_STATES),
    }
)


@dataclass(frozen=True, slots=True)
class TickResult:
    connected: bool
    claimed: str | None = None
    pumped: bool = False
    stopped: bool = False
    retry_in: float = 0.0


class RemoteWorker:
    """Reconcile, claim, pump, heartbeat and report one authenticated node's jobs."""

    def __init__(
        self,
        runtime: NodeRuntime,
        store: JobStore,
        jobs: RemoteJobClient,
        projects: tuple[ProjectLocator, ...],
        *,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        tick_seconds: float = DEFAULT_TICK_SECONDS,
        max_backoff: float = MAX_BACKOFF_SECONDS,
        clock: Any = time.monotonic,
    ) -> None:
        if not projects or any(project.project_id is None for project in projects):
            raise ValueError("remote worker projects need at least one locator with project_id")
        ids = tuple(project.project_id for project in projects)
        if ids != jobs.projects or len(set(ids)) != len(ids):
            raise ValueError("runtime project locators must exactly match RemoteJobClient projects")
        for name, value in (
            ("heartbeat_seconds", heartbeat_seconds),
            ("tick_seconds", tick_seconds),
            ("max_backoff", max_backoff),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if heartbeat_seconds >= jobs.lease_seconds:
            raise ValueError("heartbeat_seconds must be shorter than the hub lease")
        self.runtime = runtime
        self.store = store
        self.jobs = jobs
        self.projects = projects
        self._by_id = {project.project_id: project for project in projects}
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.tick_seconds = float(tick_seconds)
        self.max_backoff = float(max_backoff)
        self.clock = clock
        self._connected = False
        self._permanent_stop = False
        self._stop_reason: str | None = None
        self._stop_claiming = False
        self._closed = False
        self._failures = 0
        self._retry_at = 0.0
        self._next_heartbeat: dict[str, float] = {}
        self._generation: dict[str, int] = {}
        self._reported: dict[str, str] = {}
        self._terminal_acked: set[str] = set()
        # Attaching a worker closes dispatch immediately, including already queued local ops.
        self.runtime.runner.remote_claim_check = self._check_claim

    @property
    def permanently_stopped(self) -> bool:
        return self._permanent_stop

    @property
    def stop_reason(self) -> str | None:
        return self._stop_reason

    def stop_claiming(self) -> None:
        self._stop_claiming = True

    def _summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for view in self.runtime.runner.list():
            generation = self.runtime.runner.external_generation(view.job_id)
            if generation is None:
                self.runtime.runner.quarantine_external(
                    view.job_id, "missing or malformed assignment"
                )
                continue
            summaries.append(
                {
                    "job_id": view.job_id,
                    "generation": generation,
                    "state": view.state,
                }
            )
        return summaries

    def reconcile(self) -> None:
        """Prove every local job against its original durable assignment before pumping."""
        self._connected = False
        self._generation.clear()
        summaries = self._summaries()
        self.runtime.rebuild_branches(self.projects)
        response = self.jobs.reconcile(summaries)
        if not isinstance(response, dict) or not isinstance(response.get("jobs"), list):
            raise HubUnreachable("the hub returned malformed reconciliation")
        known = response.get("known_job_ids")
        if not isinstance(known, list) or not all(isinstance(item, str) for item in known):
            raise HubUnreachable("the hub returned malformed known_job_ids")
        proven: dict[str, tuple[int, dict[str, Any]]] = {}
        for raw in response["jobs"]:
            job_id, generation = self._identity(raw)
            if job_id in proven:
                raise HubUnreachable("the hub returned duplicate reconciliation jobs")
            local_generation = self.runtime.runner.external_generation(job_id)
            if (
                local_generation == generation
                and raw.get("assigned_node") == self.runtime.node_id
                and raw.get("claimed_node") == self.runtime.node_id
                and raw.get("project") in [project.to_dict() for project in self.projects]
                and raw.get("state") != "lost_on_node"
            ):
                proven[job_id] = (generation, raw)
        for view in self.runtime.runner.list():
            if view.job_id not in proven:
                self.runtime.runner.quarantine_external(
                    view.job_id, "no current matching hub claim"
                )
        for job_id, (generation, raw) in proven.items():
            self._generation[job_id] = generation
            self._next_heartbeat[job_id] = self.clock()
            view = self.runtime.runner.get(job_id)
            if view is None:
                continue
            if raw.get("cancel_requested") and view.state not in {
                state.value for state in TERMINAL_STATES
            }:
                self.runtime.runner.cancel(job_id, raw.get("cancel_reason") or "hub cancellation")
                view = self.runtime.runner.get(job_id)
            if view is not None and view.state in {state.value for state in TERMINAL_STATES}:
                self._report(view.to_dict(), generation)
        self._connected = True
        self._failures = 0
        self._retry_at = 0.0

    def _check_claim(self, job_id: str) -> BrokerOpError | None:
        """Broker precheck: ask the hub immediately before each remote op dispatch."""
        if not self._connected or self._closed or self._permanent_stop:
            return BrokerOpError("hub_unreachable", "remote assignment is not reconciled")
        generation = self.runtime.runner.external_generation(job_id)
        if generation is None or self._generation.get(job_id) != generation:
            self.runtime.runner.quarantine_external(job_id, "unproven remote assignment")
            return BrokerOpError("job_inactive", "no current durable assignment")
        try:
            raw = self.jobs.heartbeat(job_id, generation)
            identity = self._identity(raw)
            if (
                identity != (job_id, generation)
                or raw.get("claimed_node") != self.runtime.node_id
                or raw.get("assigned_node") != self.runtime.node_id
                or raw.get("state") == "lost_on_node"
            ):
                raise HubRefused("stale_generation", "hub no longer confirms this assignment")
            if raw.get("cancel_requested"):
                self.runtime.runner.cancel(job_id, raw.get("cancel_reason") or "hub cancellation")
                return BrokerOpError("job_inactive", "hub cancellation requested")
        except OSError as exc:
            self._connected = False
            return BrokerOpError("hub_unreachable", str(exc))
        except ValueError as exc:
            self._connected = False
            self.runtime.runner.quarantine_external(job_id, str(exc))
            self._stop_claiming = True
            self._permanent_stop = True
            self._stop_reason = str(exc)
            return BrokerOpError("job_inactive", "remote assignment refused")
        return None

    def _identity(self, raw: Any) -> tuple[str, int]:
        if not isinstance(raw, dict):
            raise HubUnreachable("the hub returned a non-object job")
        job_id, generation = raw.get("job_id"), raw.get("generation")
        if (
            not isinstance(job_id, str)
            or not job_id.startswith("job_")
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
            or not isinstance(raw.get("state"), str)
            or not 0 < len(raw["state"]) <= 128
            or (raw["state"] not in _REMOTE_STATES and not raw["state"].startswith("waiting:"))
            or not isinstance(raw.get("cancel_requested"), bool)
            or (
                raw.get("cancel_reason") is not None
                and (not isinstance(raw["cancel_reason"], str) or len(raw["cancel_reason"]) > 1000)
            )
        ):
            raise HubUnreachable("the hub returned a malformed job identity")
        return job_id, generation

    def _accept_claim(self, raw: dict[str, Any]) -> str:
        job_id, generation = self._identity(raw)
        project_raw = raw.get("project")
        try:
            target = ProjectLocator.from_dict(project_raw)
        except (TypeError, ValueError, KeyError) as exc:
            raise HubRefused(
                "invalid_job", f"hub job {job_id} has an invalid project: {exc}"
            ) from exc
        configured = self._by_id.get(target.project_id)
        if configured is None or configured != target:
            raise HubRefused(
                "project_not_configured",
                f"hub job {job_id} targets {target}, which is not this node's exact configuration",
            )
        if (
            raw.get("assigned_node") != self.runtime.node_id
            or raw.get("claimed_node") != self.runtime.node_id
        ):
            raise HubRefused("node_mismatch", f"hub job {job_id} is not assigned to this node")
        if self.runtime.runner.get(job_id) is not None:
            self.runtime.runner.quarantine_external(job_id, "job id was reassigned")
            raise HubRefused("duplicate_local_job", f"hub job {job_id} already exists locally")
        params = raw.get("params")
        if not isinstance(params, dict):
            raise HubRefused("invalid_job", f"hub job {job_id} params are not an object")
        # JobRunner revalidates the registered kind, typed params, finite timeouts, target,
        # agent id and approval gates. Its store commit happens before this method returns.
        try:
            self.runtime.runner.submit(
                raw.get("kind"),
                params,
                target,
                agent_id=raw.get("agent_id"),
                priority=raw.get("priority", 0),
                approval_token=raw.get("approval_token"),
                job_id=job_id,
                generation=generation,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HubRefused("invalid_job", f"hub job {job_id} is invalid: {exc}") from exc
        self._generation[job_id] = generation
        self._next_heartbeat[job_id] = self.clock()
        return job_id

    @staticmethod
    def _report_body(view: dict[str, Any]) -> dict[str, Any]:
        return {
            key: view.get(key)
            for key in (
                "current_step",
                "steps_total",
                "step_name",
                "results",
                "measured",
                "error",
                "waiting",
                "stall",
                "alerts",
                "updated_at",
            )
        }

    def _report(self, view: dict[str, Any], generation: int) -> None:
        job_id, state = view["job_id"], view["state"]
        body = self._report_body(view)
        fingerprint = json.dumps({"state": state, "report": body}, sort_keys=True, default=str)
        terminal = state in {member.value for member in TERMINAL_STATES}
        if terminal and (
            job_id in self._terminal_acked
            or any(
                event.kind == "external_result_acked"
                and event.data == {"generation": generation, "fingerprint": fingerprint}
                for event in self.store.events(job_id)
            )
        ):
            self._terminal_acked.add(job_id)
            return
        if not terminal and self._reported.get(job_id) == fingerprint:
            return
        try:
            result = self.jobs.report(job_id, generation, state, body)
        except ValueError as exc:
            if getattr(exc, "code", None) == "stale_generation":
                self.runtime.runner.quarantine_external(job_id, "assignment was released")
            raise
        if self._identity(result) != (job_id, generation):
            raise HubUnreachable("the hub returned an inconsistent report acknowledgement")
        self._reported[job_id] = fingerprint
        if terminal:
            self.store.append_events(
                job_id,
                [
                    (
                        "external_result_acked",
                        {
                            "generation": generation,
                            "fingerprint": fingerprint,
                        },
                        self.clock(),
                    )
                ],
            )
            self._terminal_acked.add(job_id)

    def _heartbeats(self, now: float) -> None:
        terminal = {member.value for member in TERMINAL_STATES}
        for job_id, generation in list(self._generation.items()):
            view = self.runtime.runner.get(job_id)
            if view is None or view.state in terminal:
                continue
            if now >= self._next_heartbeat.get(job_id, 0.0):
                try:
                    remote = self.jobs.heartbeat(job_id, generation)
                except ValueError as exc:
                    if getattr(exc, "code", None) == "stale_generation":
                        self.runtime.runner.quarantine_external(job_id, "assignment was released")
                    raise
                if self._identity(remote) != (job_id, generation):
                    raise HubUnreachable("the hub returned an inconsistent heartbeat")
                if remote.get("cancel_requested"):
                    self.runtime.runner.cancel(
                        job_id, remote.get("cancel_reason") or "hub cancellation"
                    )
                self._next_heartbeat[job_id] = now + self.heartbeat_seconds

    def _reports(self) -> None:
        for job_id, generation in list(self._generation.items()):
            view = self.runtime.runner.get(job_id)
            if view is not None:
                self._report(view.to_dict(), generation)

    def _backoff(self, now: float) -> TickResult:
        self._connected = False
        self._failures += 1
        delay = min(self.max_backoff, self.tick_seconds * (2 ** min(self._failures - 1, 10)))
        self._retry_at = now + delay
        return TickResult(False, retry_in=delay)

    def tick(self) -> TickResult:
        """Run one bounded loop iteration; safe for deterministic tests and service loops."""
        if self._closed or self._permanent_stop:
            return TickResult(False, stopped=True)
        now = self.clock()
        if now < self._retry_at:
            return TickResult(False, retry_in=self._retry_at - now)
        try:
            if not self._connected:
                self.reconcile()
            self._heartbeats(now)
            outcome = self.runtime.pump()
            if not self._connected or self._permanent_stop:
                return TickResult(False, pumped=True, stopped=self._permanent_stop)
            self._reports()
            claimed = None
            if not self._stop_claiming and outcome.active == 0:
                raw = self.jobs.claim()
                if raw is not None:
                    claimed = self._accept_claim(raw)
            self._failures = 0
            return TickResult(True, claimed=claimed, pumped=True)
        except NodeNotEnrolled:
            self._connected = False
            self._permanent_stop = True
            self._stop_claiming = True
            self._stop_reason = "node enrollment is missing, revoked, or unavailable"
            return TickResult(False, stopped=True)
        except HubUnreachable:
            return self._backoff(now)
        except ValueError as exc:
            self._connected = False
            # A typed refusal is permanent for this process. Stopping the whole worker keeps
            # a stale fenced owner from pumping any more work; reconnect reconciliation can
            # then identify and quarantine only the assignments it no longer owns.
            self._permanent_stop = True
            self._stop_claiming = True
            code = exc.code if isinstance(exc, HubRefused) else type(exc).__name__
            self._stop_reason = f"{code}: {exc}"
            return TickResult(False, stopped=True)

    def run(self, stop: threading.Event, *, max_ticks: int | None = None) -> int:
        ticks = 0
        while not stop.is_set() and not self._permanent_stop:
            result = self.tick()
            ticks += 1
            if max_ticks is not None and ticks >= max_ticks:
                break
            delay = max(self.tick_seconds, result.retry_in)
            stop.wait(delay)
        self.stop_claiming()
        return 2 if self._permanent_stop else 0

    def close(self, timeout: float = 5.0) -> bool:
        if self._closed:
            return True
        self._closed = True
        self.stop_claiming()
        try:
            return self.runtime.close(timeout)
        finally:
            self.store.close()
