"""The hot-step port: how a recipe hands work to a Resolve node, and a fake for tests.

A hot step (one that must touch Resolve) never runs in the recipe engine. It goes through a
:class:`HotDispatcher`: the engine ``submit``\\ s a request with a deterministic ``request_id``
and later ``poll``\\ s the resulting job. The real dispatcher -- hub jobs plus a remote node --
is S15b. This slice ships only the protocol and :class:`FakeHotDispatcher`, an in-memory,
deterministic implementation the tests (and subprocess workers) drive.

The contract the engine relies on:

- ``submit`` is idempotent on ``request_id``: the same id submitted twice returns the same
  ``job_ref`` and runs the work once. That is what lets the engine write the submit intent ahead
  of the submit and re-submit safely after a crash in between.
- ``poll`` reports the job's state. ``waiting_for_node`` and ``node_offline`` show through to the
  status view unchanged (the PC is off, or busy). A finished job reports ``done`` with a result,
  or ``failed`` with an error.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Job states the engine understands. waiting_for_node / node_offline are not failures; they are
# shown to agents and humans as a wait reason so "the Linux hub works with the PC off" is visible.
HOT_STATES = frozenset({"queued", "waiting_for_node", "node_offline", "running", "done", "failed"})
_TERMINAL = frozenset({"done", "failed"})


@dataclass(frozen=True, slots=True)
class HotResult:
    """A finished hot job's outputs: bounded JSON plus any artifact ids it produced."""

    outputs: dict[str, Any] = field(default_factory=dict)
    artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HotJobView:
    """What :meth:`HotDispatcher.poll` reports about one job.

    ``state`` is one of :data:`HOT_STATES`. ``wait_reason`` is a short token shown while the job
    is not finished (e.g. ``waiting_for_node``). ``result`` is set only when ``state == "done"``;
    ``error`` (``{code, message, fix_hint}``) only when ``state == "failed"``.
    """

    state: str
    wait_reason: str | None = None
    result: HotResult | None = None
    error: dict[str, Any] | None = None

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL


@runtime_checkable
class HotDispatcher(Protocol):
    """A port that submits hot work to a Resolve node and polls the resulting job."""

    def submit(
        self,
        *,
        request_id: str,
        project_id: str,
        kind: str,
        params: Mapping[str, Any],
        agent_id: str,
    ) -> str:
        """Submit (idempotently on ``request_id``) and return an opaque ``job_ref``."""
        ...

    def poll(self, job_ref: str) -> HotJobView:
        """Report the current state of the job named by ``job_ref``."""
        ...


# Per-kind behaviour for the fake: given the submitted params, return the HotJobView sequence.
FakeBehaviour = Callable[[Mapping[str, Any]], HotJobView]


def _done(**outputs: Any) -> FakeBehaviour:
    return lambda _params: HotJobView("done", result=HotResult(outputs=dict(outputs)))


class FakeHotDispatcher:
    """An in-memory, deterministic :class:`HotDispatcher` for tests and subprocess workers.

    ``behaviours`` maps a hot ``kind`` to a callable that turns the submitted params into a
    :class:`HotJobView`; the default completes immediately with empty outputs. ``submit`` is
    idempotent on ``request_id`` (a second submit of the same id returns the first ``job_ref``
    and does not re-run the behaviour), so the engine's write-ahead-then-submit is safe across a
    crash. ``submits`` records every ``request_id`` actually dispatched (once each); ``calls``
    records every ``submit`` invocation (including idempotent repeats), so a test can prove the
    same ``request_id`` was re-sent after a crash and that it ran only once.
    """

    def __init__(self, behaviours: Mapping[str, FakeBehaviour] | None = None) -> None:
        self._behaviours = dict(behaviours or {})
        self._by_request: dict[str, str] = {}
        self._views: dict[str, HotJobView] = {}
        self.submits: list[str] = []
        self.calls: list[str] = []
        self._next = 0

    def set_behaviour(self, kind: str, behaviour: FakeBehaviour) -> None:
        self._behaviours[kind] = behaviour

    def submit(
        self,
        *,
        request_id: str,
        project_id: str,
        kind: str,
        params: Mapping[str, Any],
        agent_id: str,
    ) -> str:
        self.calls.append(request_id)
        existing = self._by_request.get(request_id)
        if existing is not None:
            # Idempotent: already dispatched this request_id; do not re-run the behaviour.
            return existing
        self._next += 1
        job_ref = f"job-{self._next}"
        self._by_request[request_id] = job_ref
        self.submits.append(request_id)
        behaviour = self._behaviours.get(kind, _done())
        self._views[job_ref] = behaviour(dict(params))
        return job_ref

    def poll(self, job_ref: str) -> HotJobView:
        view = self._views.get(job_ref)
        if view is None:
            return HotJobView("failed", error={"code": "unknown_job", "message": job_ref})
        return view

    # -- test controls ------------------------------------------------------

    def complete(self, job_ref: str, **outputs: Any) -> None:
        """Force a job to ``done`` with ``outputs`` (for a two-phase submit/poll test)."""
        self._views[job_ref] = HotJobView("done", result=HotResult(outputs=dict(outputs)))

    def set_state(self, job_ref: str, state: str, *, wait_reason: str | None = None) -> None:
        """Force a non-terminal job state (e.g. ``waiting_for_node``) for the status-view tests."""
        self._views[job_ref] = HotJobView(state, wait_reason=wait_reason)
