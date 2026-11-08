"""The cold worker: the D6 "cold worker on the Linux hub host".

``aar recipe worker`` claims pending cold ops with a lease, runs the registered callable with a
:class:`~aar.recipes.context.StepContext`, records the result and calls :func:`advance`. It
heartbeats the lease while a step runs. A lapsed-lease op is re-claimed only when its step is
``idempotent``; a non-idempotent one is failed ``lease_lost`` by ``advance`` and needs an explicit
retry. A step cannot be killed in-process: timeouts are cooperative (the context deadline), and a
callable that returns late is recorded as ``timeout``. SIGINT/SIGTERM stop claiming new work and
let the current step finish.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
from datetime import timedelta
from typing import Any

from aar.hub.artifacts import ArtifactStore
from aar.hub.common import Clock, utcnow
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.hub.writes import typed_brief
from aar.plan.profile import ProfileError, load_profile
from aar.recipes import registry
from aar.recipes.context import StepContext, StepFailure, StepOutput
from aar.recipes.engine import (
    DEFAULT_HEARTBEAT_S,
    DEFAULT_LEASE_S,
    advance,
    claim,
    heartbeat,
    op_lease_live,
    record_result,
    record_worker_error,
)
from aar.recipes.events import OUTCOME_DONE, OUTCOME_FAILED
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.state import OpAttempt, RunState, load_state

_LOG = logging.getLogger("aar.recipes.worker")


def active_recipes(log: EventLog) -> list[tuple[str, str]]:
    """Every ``(project_id, recipe_node)`` whose run is active or blocked by evidence.

    Liveness is read from the folded evidence status, never the recipe node's graph status (which
    is only a mirror), so an agent cannot park or revive a run by moving the recipe node (F1).
    """
    found: list[tuple[str, str]] = []
    registry_ = Registry(log)
    for project in registry_.list():
        graph = WorkGraph(log, project.id)
        for node in graph.nodes():
            if node.kind != "recipe":
                continue
            state = load_state(log, node.id, project_id=project.id)
            if state.recipe_status in ("active", "blocked"):
                found.append((project.id, node.id))
    return found


def _claimable(
    state: RunState, recipe_def: RecipeDef, now: Any
) -> tuple[OpAttempt, StepDef] | None:
    current = state.current_round()
    for step in recipe_def.steps:
        if step.kind != "cold":
            continue
        op = state.latest_in_round(step.id, current)
        if op is None or op.is_cancelled:
            continue
        if op.status == "pending":
            return op, step
        if op.status == "active" and step.idempotent and not op_lease_live(state, op, now):
            return op, step  # lapsed, idempotent: re-claimable
    return None


def _run_step(
    log: EventLog,
    store: ArtifactStore,
    state: RunState,
    recipe_def: RecipeDef,
    op: OpAttempt,
    step: StepDef,
    worker: Actor,
    *,
    clock: Clock,
    stop: threading.Event,
    lease_s: int,
    heartbeat_s: int,
) -> None:
    """Run one claimed cold step: build the context, run the callable, record the result."""
    fn = registry.callable_for(step.callable_id or "")
    brief = _brief_of(log, state)
    profile = _profile_of(brief)
    try:
        params = _params_of(state, recipe_def)
    except Exception as exc:  # F12: never hand a step the raw dict; fail with a typed error
        record_result(
            log,
            project_id=state.project_id,
            op_id=op.op_id,
            actor=worker,
            outcome=OUTCOME_FAILED,
            error={
                "code": "invalid_params",
                "message": f"the stored recipe params do not validate: {exc}"[:500],
                "fix_hint": "Recreate the recipe with params the registered model accepts.",
            },
        )
        return
    needs = {need: dict(state.done_outputs(need)) for need in step.needs}
    deadline = clock() + timedelta(seconds=step.timeout_s)
    ctx = StepContext(
        log=log,
        store=store,
        project_id=state.project_id,
        recipe_node=state.recipe_node,
        op_node=op.op_id,
        actor=worker,
        params=params,
        brief=brief,
        profile=profile,
        needs=needs,
        deadline=deadline,
        _cancelled=stop.is_set,
        _clock=clock,
    )
    beat = _Heartbeat(log, state.project_id, op, worker, clock, lease_s, heartbeat_s, stop)
    beat.start()
    try:
        result = fn(ctx)
        outcome, outputs, artifacts, error = _interpret(result, ctx, clock)
    except StepFailure as failure:
        outcome, outputs, artifacts, error = OUTCOME_FAILED, {}, (), failure.to_error()
    except Exception as exc:  # a callable that raised: fail the step, never crash the worker
        outcome = OUTCOME_FAILED
        outputs, artifacts = {}, ()
        error = {"code": "error", "message": f"{type(exc).__name__}: {exc}"[:500], "fix_hint": ""}
    finally:
        beat.stop()
    record_result(
        log,
        project_id=state.project_id,
        op_id=op.op_id,
        actor=worker,
        outcome=outcome,
        outputs=outputs,
        artifact_ids=artifacts,
        error=error,
    )


def _interpret(
    result: Any, ctx: StepContext, clock: Clock
) -> tuple[str, dict[str, Any], tuple[str, ...], dict[str, Any] | None]:
    if not isinstance(result, StepOutput):
        raise StepFailure("bad_output", "a cold callable must return a StepOutput")
    if (clock() - ctx.deadline).total_seconds() > 0:
        return (
            OUTCOME_FAILED,
            {},
            (),
            {
                "code": "timeout",
                "message": "the step returned after its deadline (it cannot be killed in-process)",
                "fix_hint": "Raise timeout_s, or make the step cooperative (check time_left()).",
            },
        )
    return OUTCOME_DONE, dict(result.outputs), tuple(result.artifact_ids), None


def _brief_of(log: EventLog, state: RunState) -> Any:
    if state.recipe_data is None:
        return None
    graph = WorkGraph(log, state.project_id)
    return typed_brief(graph.node(state.recipe_data.brief_id))


def _profile_of(brief: Any) -> Any:
    if brief is None:
        return None
    try:
        return load_profile(brief.rule_profile)
    except (ProfileError, ValueError):
        return None


def _params_of(state: RunState, recipe_def: RecipeDef) -> Any:
    """The parsed params model. Raises if the stored params do not validate (F12): the worker
    never hands a step the raw dict."""
    raw = state.recipe_data.params if state.recipe_data is not None else {}
    return recipe_def.params_model.model_validate(dict(raw))


class _Heartbeat(threading.Thread):
    """Beats the lease every ``heartbeat_s`` while a step runs; stops when told."""

    def __init__(
        self,
        log: EventLog,
        project_id: str,
        op: OpAttempt,
        worker: Actor,
        clock: Clock,
        lease_s: int,
        heartbeat_s: int,
        stop: threading.Event,
    ) -> None:
        super().__init__(daemon=True)
        self._log = log
        self._project_id = project_id
        self._op = op
        self._worker = worker
        self._clock = clock
        self._lease_s = lease_s
        self._heartbeat_s = heartbeat_s
        self._stop = stop
        self._done = threading.Event()
        self._n = 1

    def run(self) -> None:
        while not self._done.wait(self._heartbeat_s):
            try:
                heartbeat(
                    self._log,
                    project_id=self._project_id,
                    op_id=self._op.op_id,
                    worker=self._worker,
                    clock=self._clock,
                    lease_s=self._lease_s,
                    n=self._n,
                )
                self._n += 1
            except Exception:  # a heartbeat failure must never crash the step
                return

    def stop(self) -> None:
        self._done.set()


def run_worker(
    log: EventLog,
    store: ArtifactStore,
    actor: Actor,
    *,
    dispatcher: Any = None,
    clock: Clock = utcnow,
    once: bool = False,
    poll_s: float = 2.0,
    lease_s: int = DEFAULT_LEASE_S,
    heartbeat_s: int = DEFAULT_HEARTBEAT_S,
    stop: threading.Event | None = None,
    install_signals: bool = False,
) -> int:
    """Run the worker loop until stopped; return the number of steps it ran.

    With ``once`` it claims and runs at most one step and returns. ``stop`` (a threading.Event)
    lets a caller or a signal handler ask it to drain. ``install_signals`` wires SIGINT/SIGTERM.
    """
    stop = stop or threading.Event()
    if install_signals:
        _install_signals(stop)
    ran = 0
    while not stop.is_set():
        progressed = _tick(
            log,
            store,
            actor,
            dispatcher=dispatcher,
            clock=clock,
            lease_s=lease_s,
            heartbeat_s=heartbeat_s,
            stop=stop,
        )
        if progressed:
            ran += 1
        if once:
            break
        if not progressed and not stop.wait(poll_s):
            continue
    return ran


def _tick(
    log: EventLog,
    store: ArtifactStore,
    actor: Actor,
    *,
    dispatcher: Any,
    clock: Clock,
    lease_s: int,
    heartbeat_s: int,
    stop: threading.Event,
) -> bool:
    now = clock()
    recipes = active_recipes(log)
    for _project_id, recipe_node in recipes:
        # N3: one recipe's failure (a raising dispatcher, a bad advance) must never stop the loop.
        try:
            advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
        except Exception as exc:
            _note_worker_error(log, _project_id, recipe_node, exc)
            continue
    for project_id, recipe_node in recipes:
        if stop.is_set():  # drain: load no further recipe once a stop is requested
            return False
        try:
            state = load_state(log, recipe_node, project_id=project_id)
            if state.recipe_def is None:
                continue
            found = _claimable(state, state.recipe_def, now)
            if found is None:
                continue
            op, step = found
            if stop.is_set():  # drain: never start a new step (callables cannot be interrupted)
                return False
            if not claim(
                log,
                project_id=project_id,
                op=op,
                worker=actor,
                idempotent=step.idempotent,
                clock=clock,
                lease_s=lease_s,
            ):
                return True  # a race; another worker took it -- retry the loop
            _run_step(
                log,
                store,
                state,
                state.recipe_def,
                op,
                step,
                actor,
                clock=clock,
                stop=stop,
                lease_s=lease_s,
                heartbeat_s=heartbeat_s,
            )
            advance(log, recipe_node, actor=actor, dispatcher=dispatcher, clock=clock)
            return True
        except Exception as exc:
            # The step result is recorded in _run_step; anything else (a per-recipe fault) is
            # logged, recorded for the status view, and swallowed so the worker goes on (N3/R3-3).
            _note_worker_error(log, project_id, recipe_node, exc)
            continue
    return False


# Per (recipe, exception type) WARNING rate-limit state (R4-1): the monotonic time of the last
# line logged and the occurrences suppressed since, so a recipe that fails on every tick writes at
# most one log line per window. Bounded: stale keys are pruned once the table grows past the cap.
_LOG_WINDOW_S = 60.0
_LOG_STATE_MAX = 1024
_LOG_TEXT_CHARS = 4096
_log_state: dict[tuple[str, str], tuple[float, int]] = {}
_log_lock = threading.Lock()
_monotonic = time.monotonic


def _describe(exc: BaseException) -> tuple[str, str]:
    """``(type name, "Type: message")`` for a swallowed exception, never raising (a broken
    ``__str__`` must not take the worker loop down) and cut to 4096 characters."""
    exc_type = type(exc).__name__
    try:
        text = str(exc)
    except Exception:
        text = "<unprintable exception>"
    return exc_type, f"{exc_type}: {text[:_LOG_TEXT_CHARS]}"[:_LOG_TEXT_CHARS]


def _should_log(key: tuple[str, str]) -> tuple[bool, int]:
    """Rate limit: ``(log it now?, occurrences suppressed since the last line)``."""
    with _log_lock:
        now = _monotonic()
        last, suppressed = _log_state.get(key, (None, 0))
        if last is not None and now - last < _LOG_WINDOW_S:
            _log_state[key] = (last, suppressed + 1)
            return False, 0
        if key not in _log_state and len(_log_state) >= _LOG_STATE_MAX:
            for stale in [k for k, (t, _) in _log_state.items() if now - t >= _LOG_WINDOW_S]:
                del _log_state[stale]
            if len(_log_state) >= _LOG_STATE_MAX:
                _log_state.clear()
        _log_state[key] = (now, 0)
        return True, suppressed


def _note_worker_error(log: EventLog, project_id: str, recipe_node: str, exc: Exception) -> None:
    """Account for a swallowed per-recipe worker exception (R3-3/R4-1), never raising.

    - ``logging`` WARNING for every occurrence, rate-limited to one line per (recipe, exception
      type) per 60 s; the next line carries the "suppressed N" count of the ones in between.
    - At most ONE ``recipe.worker_error`` hub.write record per (recipe, exception type), whatever
      the message (see :func:`~aar.recipes.engine.record_worker_error`); a repeat writes nothing.
    """
    exc_type, message = _describe(exc)
    with contextlib.suppress(Exception):
        emit, suppressed = _should_log((recipe_node, exc_type))
        if emit:
            extra = f" (suppressed {suppressed} since the last report)" if suppressed else ""
            _LOG.warning("recipe %s: worker swallowed %s%s", recipe_node, message, extra)
    with contextlib.suppress(Exception):
        record_worker_error(
            log,
            project_id=project_id,
            recipe_node=recipe_node,
            exc_type=exc_type,
            message=message,
        )


def _install_signals(stop: threading.Event) -> None:
    def handler(signum: int, frame: Any) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        # not in the main thread (tests): the caller drives stop itself
        with contextlib.suppress(ValueError):
            signal.signal(sig, handler)
