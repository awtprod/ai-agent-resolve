"""S19f: ``recipe_reopen`` -- a new attempt of an agent step in the same round, after a later
cold step that declares ``on_fail`` failed because of its output.

Engine level, on the ``reopentest`` recipe of ``recipe_support`` (``c`` cold, ``p`` agent plan,
``b`` cold build with ``on_fail="p"`` and two attempts, ``g`` a gate on ``p`` that rejects back to
``p``, ``report``). The build fails on the plans listed in ``BUILD_FAILS``, marked upstream or
not. Everything runs on the real engine, worker and log; only crash points are injected.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from typing import Any

import pytest
from mcp import Client
from recipe_support import (
    BUILD_FAILS,
    RecTestParams,
    make_reopen_run,
    register_reopen_recipe,
    submit_variant,
)
from write_support import AGENT, HUMAN, OTHER, Hub

from aar.cli import main
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import EventLog
from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.hub.reviews import decide, withdraw
from aar.hub.writes import WriteError
from aar.recipes import engine, registry
from aar.recipes import run as run_mod
from aar.recipes.engine import advance, record_result
from aar.recipes.models import PlanExpect, RecipeDef, RecipeDefError, StepDef, validate_recipe
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_cancel, recipe_reopen, recipe_retry, recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view
from aar.recipes.worker import run_worker


@pytest.fixture
def reopen_recipe() -> Iterator[RecipeDef]:
    definition = register_reopen_recipe()
    yield definition
    registry.clear()
    BUILD_FAILS.clear()


def work(hub: Hub, log: EventLog | None = None, store: ArtifactStore | None = None) -> int:
    ran = 0
    for _ in range(30):
        if run_worker(log or hub.log, store or hub.store, AGENT, once=True) == 0:
            break
        ran += 1
    return ran


def complete(hub: Hub, rn: str, node: str, key: str, actor: Any = AGENT) -> dict[str, Any]:
    return step_complete(
        hub.log, hub.store, rn, "p", None, {"plan_node": node}, actor, idempotency_key=key
    )


def failed_build(
    hub: Hub, *, upstream: bool = True, creator: Any = AGENT, rounds: int | None = None
) -> tuple[str, str]:
    """A started run whose plan (variant 1) the build failed on; ``(recipe node, plan sha)``."""
    rn = make_reopen_run(hub, actor=creator, max_review_rounds=rounds)
    recipe_start(hub.log, rn, creator, idempotency_key="start")
    work(hub)
    node, sha = submit_variant(hub, rn, AGENT, 1)
    BUILD_FAILS[sha] = upstream
    complete(hub, rn, node, "p1")
    work(hub)
    return rn, sha


def latest(hub: Hub, rn: str, step: str) -> Any:
    state = load_state(hub.log, rn)
    return state.latest_in_round(step, state.current_round())


def live(hub: Hub, rn: str, step: str) -> list[Any]:
    return [op for op in load_state(hub.log, rn).attempts_for(step) if op.is_live]


def refusal(call: Any) -> str:
    with pytest.raises(WriteError) as exc:
        call()
    return exc.value.code


def reopen(hub: Hub, rn: str, key: str = "re1", actor: Any = AGENT, step: str = "p") -> Any:
    return recipe_reopen(hub.log, rn, step, actor, idempotency_key=key)


def action(view: dict[str, Any], step: str) -> dict[str, Any]:
    return next(a for a in view["next_actions"] if a["step_id"] == step)


# -- the declaration -------------------------------------------------------------------------


def _def(*steps: StepDef) -> RecipeDef:
    return RecipeDef(name="ondef", version=1, params_model=RecTestParams, steps=steps)


AGENT_P = StepDef(id="p", kind="agent", instruction="x", expects=PlanExpect())


def test_on_fail_is_validated_when_the_recipe_is_defined() -> None:
    mid = StepDef(id="m", kind="cold", callable_id="x", needs=("p",))
    through = StepDef(id="b", kind="cold", callable_id="x", needs=("m",), on_fail="p")
    validate_recipe(_def(AGENT_P, mid, through))  # needed through another step: accepted
    bad = [
        # not a cold step
        (
            AGENT_P,
            StepDef(id="q", kind="agent", instruction="x", expects=PlanExpect(), on_fail="p"),
        ),
        (AGENT_P, StepDef(id="g", kind="gate", subject="p", on_reject="p", on_fail="p")),
        (AGENT_P, StepDef(id="h", kind="hot", hot_kind="k", needs=("p",), on_fail="p")),
        # not an earlier step / itself / unknown
        (AGENT_P, StepDef(id="b", kind="cold", callable_id="x", needs=("p",), on_fail="z")),
        (AGENT_P, StepDef(id="b", kind="cold", callable_id="x", needs=("p",), on_fail="b")),
        # an earlier step that is not an agent step
        (
            StepDef(id="c", kind="cold", callable_id="x"),
            StepDef(id="b", kind="cold", callable_id="x", needs=("c",), on_fail="c"),
        ),
        # an earlier agent step this step does not need
        (AGENT_P, StepDef(id="b", kind="cold", callable_id="x", on_fail="p")),
    ]
    for steps in bad:
        with pytest.raises(RecipeDefError, match="on_fail"):
            validate_recipe(_def(*steps))
    # the same check runs at registration
    with pytest.raises(RecipeDefError, match="on_fail"):
        registry.register(_def(*bad[-1]))


# -- the happy path ---------------------------------------------------------------------------


def test_reopen_opens_the_next_attempt_in_the_same_round(hub: Hub, reopen_recipe: Any) -> None:
    rn, sha1 = failed_build(hub)
    failed = latest(hub, rn, "b")
    assert failed.status == "failed" and failed.upstream is True
    assert failed.result_error == {
        "code": "bad_plan",
        "message": f"the build of plan {sha1[:12]} failed",
        "fix_hint": "change the plan",
        "upstream": True,
    }
    first = latest(hub, rn, "p")

    body = reopen(hub, rn)
    assert body["ok"] is True and body["op"] == "recipe.reopen" and body["round"] == 1
    state = load_state(hub.log, rn)
    assert state.current_round() == 1
    # every op of the round from p onward is superseded; c (before p) is untouched
    assert state.op_by_id(first.op_id).is_cancelled
    assert state.op_by_id(failed.op_id).is_cancelled
    assert state.done_op("c") is not None
    new = state.latest_in_round("p", 1)
    assert (new.attempt, new.status) == (2, "active")
    assert new.reopened_for == {
        "step": "b",
        "attempt": 1,
        "code": "bad_plan",
        "message": f"the build of plan {sha1[:12]} failed",
        "fix_hint": "change the plan",
        "op_node": failed.op_id,
    }
    assert [op.op_id for op in live(hub, rn, "p")] == [new.op_id]
    assert live(hub, rn, "b") == []

    # the status view tells the agent why it is back
    entry = next(s for s in body["steps"] if s["step_id"] == "p")
    assert entry["attempt"] == 2 and entry["reopened_for"]["code"] == "bad_plan"
    todo = action(body, "p")
    assert todo["tool"] == "step_complete" and todo["args"]["attempt"] == 2
    assert "attempt 2, reopened because step 'b' attempt 1 failed bad_plan" in todo["what"]

    # a changed plan goes on: the build's new attempt cannot collide with the superseded one
    node2, _ = submit_variant(hub, rn, AGENT, 2)
    complete(hub, rn, node2, "p2")
    assert work(hub) == 1
    built = latest(hub, rn, "b")
    assert (built.attempt, built.status) == (2, "done")
    gate = latest(hub, rn, "g")
    assert gate.round == 1 and gate.review_id is not None
    decide(hub.log, hub.project_id, gate.review_id, "approve", "ok", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    work(hub)
    state = load_state(hub.log, rn)
    assert state.recipe_status == "done" and state.current_round() == 1


# -- every refusal ----------------------------------------------------------------------------


def test_permission_and_unknown_step(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    assert refusal(lambda: reopen(hub, rn, actor=OTHER)) == "not_creator"
    assert refusal(lambda: reopen(hub, rn, step="nope")) == "unknown_step"
    assert refusal(lambda: reopen(hub, "nd_0000000000000000")) == "unknown_node"
    assert reopen(hub, rn, actor=HUMAN)["ok"] is True  # a human may


def test_not_open_runs_are_refused(hub: Hub, reopen_recipe: Any) -> None:
    rn = make_reopen_run(hub)
    assert refusal(lambda: reopen(hub, rn)) == "recipe_not_open"  # not started
    recipe_start(hub.log, rn, AGENT, idempotency_key="start")
    recipe_cancel(hub.log, rn, AGENT, idempotency_key="cancel")
    assert refusal(lambda: reopen(hub, rn)) == "recipe_not_open"


def test_a_done_run_is_refused(hub: Hub, reopen_recipe: Any) -> None:
    rn = make_reopen_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="start")
    work(hub)
    node, _ = submit_variant(hub, rn, AGENT, 1)
    complete(hub, rn, node, "p1")
    work(hub)
    decide(hub.log, hub.project_id, latest(hub, rn, "g").review_id, "approve", "ok", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    work(hub)
    assert load_state(hub.log, rn).recipe_status == "done"
    assert refusal(lambda: reopen(hub, rn)) == "recipe_not_open"


def test_step_kind_done_and_upstream_preconditions(hub: Hub, reopen_recipe: Any) -> None:
    rn = make_reopen_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="start")
    work(hub)
    assert refusal(lambda: reopen(hub, rn)) == "step_not_done"  # p is still open
    assert refusal(lambda: reopen(hub, rn, step="b")) == "not_agent_step"
    node, sha = submit_variant(hub, rn, AGENT, 1)
    BUILD_FAILS[sha] = False  # a failure that is not the plan's
    complete(hub, rn, node, "p1")
    work(hub)
    assert latest(hub, rn, "b").status == "failed" and not latest(hub, rn, "b").upstream
    assert "upstream" not in latest(hub, rn, "b").result_error
    assert refusal(lambda: reopen(hub, rn)) == "no_upstream_failure"
    # the status offers the retry, not a reopen
    assert action(status_view(hub.log, rn), "b")["tool"] == "recipe_retry"
    # a passing build: nothing failed, nothing to reopen
    del BUILD_FAILS[sha]
    recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="rb")
    work(hub)
    assert latest(hub, rn, "b").status == "done"
    assert refusal(lambda: reopen(hub, rn)) == "no_upstream_failure"


def test_a_withdrawn_review_blocks_the_reopen(hub: Hub, reopen_recipe: Any) -> None:
    rn = make_reopen_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="start")
    work(hub)
    node, _ = submit_variant(hub, rn, AGENT, 1)
    complete(hub, rn, node, "p1")
    work(hub)
    withdraw(hub.log, hub.project_id, latest(hub, rn, "g").review_id, "gone", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    assert load_state(hub.log, rn).blocked_reason() == "review_withdrawn"
    with pytest.raises(WriteError) as raised:
        reopen(hub, rn)
    err = raised.value
    assert err.code == "recipe_blocked" and err.details["reason"] == "review_withdrawn"


def test_exhausted_rounds_block_the_reopen(hub: Hub, reopen_recipe: Any) -> None:
    rn = make_reopen_run(hub, max_review_rounds=1)
    recipe_start(hub.log, rn, AGENT, idempotency_key="start")
    work(hub)
    node, _ = submit_variant(hub, rn, AGENT, 1)
    complete(hub, rn, node, "p1")
    work(hub)
    decide(hub.log, hub.project_id, latest(hub, rn, "g").review_id, "reject", "no", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    assert load_state(hub.log, rn).rounds_exhausted()
    with pytest.raises(WriteError) as raised:
        reopen(hub, rn)
    err = raised.value
    assert err.code == "recipe_blocked" and err.details["reason"] == "review_rounds_exhausted"


def test_a_run_blocked_only_by_the_failed_step_can_still_be_reopened(
    hub: Hub, reopen_recipe: Any
) -> None:
    rn, _ = failed_build(hub)
    recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="rb")  # b has two attempts
    work(hub)
    state = load_state(hub.log, rn)
    assert state.recipe_status == "blocked" and state.blocked_reason() == "bad_plan"
    assert action(status_view(hub.log, rn), "b")["tool"] == "recipe_reopen"
    reopen(hub, rn)
    state = load_state(hub.log, rn)
    assert state.recipe_status == "active" and state.latest_in_round("p", 1).attempt == 2


def test_the_agent_step_has_max_attempts_in_a_round(hub: Hub, reopen_recipe: Any) -> None:
    """p has 3 attempts: the first plus two reopens. The third reopen is refused, and the status
    says so and names what remains."""
    rn, _ = failed_build(hub)
    for n in (2, 3):
        reopen(hub, rn, key=f"re{n}")
        node, sha = submit_variant(hub, rn, AGENT, n)
        BUILD_FAILS[sha] = True
        complete(hub, rn, node, f"p{n}")
        work(hub)
    state = load_state(hub.log, rn)
    assert [op.attempt for op in state.all_in_round("p", 1)] == [1, 2, 3]
    with pytest.raises(WriteError) as raised:
        reopen(hub, rn, key="re4")
    err = raised.value
    assert err.code == "no_attempts_left" and err.details["used"] == 3
    offer = action(status_view(hub.log, rn), "b")
    assert offer["who"] == "human" and "tool" not in offer
    assert "no attempts left (3 of 3 in round 1)" in offer["what"]
    assert "recipe_cancel and a new run" in offer["what"]


def test_an_unchanged_plan_is_refused_after_a_reopen(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    first = load_state(hub.log, rn).latest_in_round("p", 1).result_outputs["plan_node"]
    reopen(hub, rn)
    assert refusal(lambda: complete(hub, rn, first, "again")) == "plan_unchanged"
    # the same bytes as a new plan node are refused too
    again, _ = submit_variant(hub, rn, OTHER, 1)
    assert refusal(lambda: complete(hub, rn, again, "again2")) == "plan_unchanged"
    node2, _ = submit_variant(hub, rn, AGENT, 2)
    assert complete(hub, rn, node2, "p2")["ok"] is True


# -- idempotency, crashes and races -----------------------------------------------------------


def test_replaying_the_key_changes_nothing(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    reopen(hub, rn, key="k1")
    before = hub.counts()
    body = reopen(hub, rn, key="k1")
    assert body["replayed"] is True
    assert hub.counts() == before
    assert len(load_state(hub.log, rn).all_in_round("p", 1)) == 2
    # the key names that one request: another step under it is a conflict
    assert refusal(lambda: reopen(hub, rn, key="k1", step="c")) == "idempotency_conflict"
    # a new key is a new request, refused because the step is open again
    assert refusal(lambda: reopen(hub, rn, key="k2")) == "step_not_done"


def _crash_after_first_write(monkeypatch: pytest.MonkeyPatch) -> None:
    def crash(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("killed after the first write")

    monkeypatch.setattr(run_mod, "finish_reopen", crash)


def _assert_finished(hub: Hub, rn: str, failed_op: str) -> None:
    state = load_state(hub.log, rn)
    new = state.latest_in_round("p", 1)
    assert (new.attempt, new.status) == (2, "active") and new.reopened_for["op_node"] == failed_op
    assert [op.op_id for op in live(hub, rn, "p")] == [new.op_id]
    assert live(hub, rn, "b") == []


def test_a_crash_after_the_first_write_is_finished_by_the_next_tick(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn, _ = failed_build(hub)
    failed = latest(hub, rn, "b")
    with monkeypatch.context() as patch:
        _crash_after_first_write(patch)
        with pytest.raises(RuntimeError, match="killed"):
            reopen(hub, rn, key="k1")
    state = load_state(hub.log, rn)
    assert state.latest_in_round("p", 1) is None  # the done op is superseded, carrying the plan
    assert state.op_by_id(failed.op_id).status == "failed"  # not yet superseded
    work(hub)  # a worker tick (advance) finishes it, with reopened_for
    _assert_finished(hub, rn, failed.op_id)
    # the crashed caller's re-run then records its request (nothing else changes) ...
    before = hub.counts()
    assert reopen(hub, rn, key="k1").get("replayed") is None
    assert hub.counts()["hub.write"] == before["hub.write"] + 1
    assert hub.counts()["recipe.cancelled"] == before["recipe.cancelled"]
    # ... and from then on replays
    assert reopen(hub, rn, key="k1")["replayed"] is True
    _assert_finished(hub, rn, failed.op_id)


def test_a_crash_after_the_first_write_is_finished_by_rerunning_the_command(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn, _ = failed_build(hub)
    failed = latest(hub, rn, "b")
    with monkeypatch.context() as patch:
        _crash_after_first_write(patch)
        with pytest.raises(RuntimeError):
            reopen(hub, rn, key="k1")
    assert reopen(hub, rn, key="k1")["ok"] is True
    _assert_finished(hub, rn, failed.op_id)
    # another caller cannot ride on that reopen
    assert refusal(lambda: reopen(hub, rn, key="k2")) == "step_not_done"


def test_a_crash_between_the_supersede_and_the_create_is_finished_by_the_next_tick(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn, _ = failed_build(hub)
    failed = latest(hub, rn, "b")
    real = engine.create_attempt

    def crash(*args: Any, **kwargs: Any) -> Any:
        if kwargs["step"].id == "p" and kwargs["attempt"] == 2:
            raise RuntimeError("killed before the create")
        return real(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine, "create_attempt", crash)
        with pytest.raises(RuntimeError, match="before the create"):
            reopen(hub, rn, key="k1")
    state = load_state(hub.log, rn)
    assert state.op_by_id(failed.op_id).is_cancelled and state.latest_in_round("p", 1) is None
    # any engine tick finishes it: here a recipe tool's advance
    advance(hub.log, rn, actor=HUMAN)
    _assert_finished(hub, rn, failed.op_id)


def test_a_worker_restart_finishes_and_runs_the_reopened_state(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn, _ = failed_build(hub)
    failed = latest(hub, rn, "b")
    with monkeypatch.context() as patch:
        _crash_after_first_write(patch)
        with pytest.raises(RuntimeError):
            reopen(hub, rn, key="k1")
    hub.log.close()
    # a new process: its own log and store on the same files
    log = EventLog(hub.root / "hub.db")
    store = ArtifactStore(hub.root / "artifacts", log)
    hub.log, hub.store = log, store
    assert work(hub) == 0  # finishes the reopen (no cold step is ready: p is the agent's)
    _assert_finished(hub, rn, failed.op_id)
    node2, _ = submit_variant(hub, rn, AGENT, 2)
    complete(hub, rn, node2, "p2")
    hub.log.close()
    log = EventLog(hub.root / "hub.db")
    hub.log, hub.store = log, ArtifactStore(hub.root / "artifacts", log)
    assert work(hub) == 1
    assert (latest(hub, rn, "b").attempt, latest(hub, rn, "b").status) == (2, "done")


def test_two_reopen_calls_interleaved_make_one_attempt(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second call passes every precondition on the same state as the first, then the first
    writes first: the second helps finish that reopen and is refused already_reopened."""
    rn, _ = failed_build(hub)
    real = run_mod.begin_reopen
    results: list[Any] = []
    fired: list[bool] = []

    def racing(*args: Any, **kwargs: Any) -> bool:
        if not fired:  # the outer call: let the other call run to the end first
            fired.append(True)
            results.append(reopen(hub, rn, key="first", actor=HUMAN))
        return real(*args, **kwargs)

    monkeypatch.setattr(run_mod, "begin_reopen", racing)
    assert refusal(lambda: reopen(hub, rn, key="second")) == "already_reopened"
    assert results[0]["ok"] is True
    state = load_state(hub.log, rn)
    assert [op.attempt for op in state.all_in_round("p", 1)] == [1, 2]
    assert len(live(hub, rn, "p")) == 1


def test_two_reopen_calls_racing_in_two_connections_make_one_attempt(
    hub: Hub, reopen_recipe: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rn, _ = failed_build(hub)
    real = run_mod.begin_reopen
    barrier = threading.Barrier(2, timeout=30)

    def together(*args: Any, **kwargs: Any) -> bool:
        barrier.wait()  # both calls have passed every precondition on the same state
        return real(*args, **kwargs)

    monkeypatch.setattr(run_mod, "begin_reopen", together)
    outcomes: dict[str, str] = {}

    def call(key: str, actor: Any) -> None:
        log = EventLog(hub.root / "hub.db")
        try:
            recipe_reopen(log, rn, "p", actor, idempotency_key=key)
            outcomes[key] = "ok"
        except WriteError as exc:
            outcomes[key] = exc.code
        finally:
            log.close()

    threads = [
        threading.Thread(target=call, args=("a", AGENT)),
        threading.Thread(target=call, args=("b", HUMAN)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert sorted(outcomes.values()) == ["already_reopened", "ok"]
    state = load_state(hub.log, rn)
    assert [op.attempt for op in state.all_in_round("p", 1)] == [1, 2]
    assert len(live(hub, rn, "p")) == 1 and live(hub, rn, "b") == []


# -- after a reopen: cancel, a gate rejection --------------------------------------------------


def test_cancel_after_a_reopen(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    reopen(hub, rn)
    recipe_cancel(hub.log, rn, AGENT, idempotency_key="cancel")
    state = load_state(hub.log, rn)
    assert state.recipe_status == "cancelled" and not any(op.is_live for op in state.ops)
    assert refusal(lambda: reopen(hub, rn, key="re2")) == "recipe_not_open"
    assert work(hub) == 0


def test_a_rejection_after_a_reopen_opens_the_next_round(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    reopen(hub, rn)
    node2, _ = submit_variant(hub, rn, AGENT, 2)
    complete(hub, rn, node2, "p2")
    work(hub)
    gate = latest(hub, rn, "g")
    assert gate.round == 1
    assert WorkGraph(hub.log, hub.project_id).node(gate.review_id).title == "gate g (round 1)"
    decide(hub.log, hub.project_id, gate.review_id, "reject", "tighter", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    state = load_state(hub.log, rn)
    assert state.current_round() == 2
    p2 = state.latest_in_round("p", 2)
    assert (p2.attempt, p2.status, p2.reopened_for) == (1, "active", None)  # attempts restart
    # the derivation rule of round 2 holds: from the rejecting review, not the recipe node
    node3, _ = submit_variant(hub, rn, AGENT, 3)
    assert refusal(lambda: complete(hub, rn, node3, "p3")) == "expects_unmet"
    # every plan the recipe took is refused: the reopened one (rejected) and the superseded one
    for variant in (1, 2):
        again, _ = submit_variant(hub, gate.review_id, AGENT, variant)
        assert refusal(lambda a=again, v=variant: complete(hub, rn, a, f"x{v}")) == (
            "plan_unchanged"
        )
    node4, sha4 = submit_variant(hub, gate.review_id, AGENT, 4)
    BUILD_FAILS[sha4] = True
    complete(hub, rn, node4, "p4")
    work(hub)
    # a reopen in round 2 counts round 2's attempts only and keeps the round
    body = reopen(hub, rn, key="re-r2")
    assert body["round"] == 2
    entry = next(s for s in body["steps"] if s["step_id"] == "p")
    assert entry["attempt"] == 2 and entry["reopened_for"]["code"] == "bad_plan"
    node5, _ = submit_variant(hub, gate.review_id, AGENT, 5)
    complete(hub, rn, node5, "p5")
    work(hub)
    gate2 = latest(hub, rn, "g")
    assert gate2.round == 2
    assert WorkGraph(hub.log, hub.project_id).node(gate2.review_id).title == "gate g (round 2)"
    report = build_report_text(hub.log, rn)
    assert "- review rounds: used 2 of 3" in report
    assert "- fix attempts: round 1: 2, round 2: 2" in report


# -- the old log, the status view, the report, the surfaces ------------------------------------


def test_a_result_without_upstream_folds_as_not_upstream(hub: Hub, reopen_recipe: Any) -> None:
    """Results written before S19f carry no ``upstream``; anything but exactly True is dropped
    when a result is recorded, so it folds as false."""
    rn, _ = failed_build(hub, upstream=False)
    op = latest(hub, rn, "b")
    assert set(op.result_error) == {"code", "message", "fix_hint"} and op.upstream is False
    assert op.reopen is None and op.reopened_for is None
    recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="rb")
    retried = latest(hub, rn, "b")
    record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=retried.op_id,
        actor=AGENT,
        outcome="failed",
        enforce_holder=False,
        error={"code": "bad_plan", "message": "m", "fix_hint": "h", "upstream": "true"},
    )
    op = latest(hub, rn, "b")
    assert "upstream" not in op.result_error and op.upstream is False
    assert refusal(lambda: reopen(hub, rn)) == "recipe_blocked"  # out of attempts, not upstream


def test_status_names_the_reopen_in_text_and_json(
    hub: Hub, reopen_recipe: Any, capsys: Any
) -> None:
    rn, _ = failed_build(hub)
    view = status_view(hub.log, rn)
    entry = next(s for s in view["steps"] if s["step_id"] == "b")
    assert entry["state"] == "failed" and entry["error"]["upstream"] is True
    offer = action(view, "b")
    assert offer == {
        "step_id": "b",
        "who": "agent",
        "what": (
            "recipe_reopen p: step 'b' failed (bad_plan) because of the 'p' step's output; "
            "reopen 'p' and complete it with a changed result (an unchanged plan is refused)"
        ),
        "tool": "recipe_reopen",
        "args": {"recipe_node": rn, "step_id": "p"},
        "fill": ["agent_id", "idempotency_key"],
    }
    hub.log.close()
    assert main(["recipe", "status", "--db", str(hub.root / "hub.db"), rn]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert action(printed, "b") == offer


def test_the_report_shows_fix_attempts_only_when_a_round_had_more_than_one(
    hub: Hub, reopen_recipe: Any
) -> None:
    rn, _ = failed_build(hub)
    assert "fix attempts" not in build_report_text(hub.log, rn)
    reopen(hub, rn)
    report = build_report_text(hub.log, rn)
    section = report.split("### p (agent)")[1].split("###")[0]
    assert "- fix attempts: round 1: 2" in section
    assert "- round 1 attempt 1: done" in section and "(superseded)" in section
    assert "fix attempts" not in report.split("### b (cold)")[1].split("###")[0]
    assert "- review rounds: used 1 of 3" in report


def test_the_cli_reopen(hub: Hub, reopen_recipe: Any, capsys: Any) -> None:
    rn, _ = failed_build(hub)
    hub.log.close()
    db = str(hub.root / "hub.db")
    argv = ["recipe", "reopen", "--db", db, "--actor", "agent:agent-2", "--idempotency-key", "k"]
    assert main([*argv, rn, "p"]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "not_creator"
    argv[5] = "agent:agent-1"
    assert main([*argv, rn, "p"]) == 0
    body = json.loads(capsys.readouterr().out)
    assert body["op"] == "recipe.reopen"
    assert next(s for s in body["steps"] if s["step_id"] == "p")["attempt"] == 2
    assert main([*argv, rn, "p"]) == 0
    assert json.loads(capsys.readouterr().out)["replayed"] is True


def test_the_mcp_tool(hub: Hub, reopen_recipe: Any) -> None:
    rn, _ = failed_build(hub)
    server = build_server(
        hub.log,
        Registry(hub.log),
        hub.store,
        params_validator=registry.registry_params_validator,
        agent_writes=True,
    )

    async def body() -> tuple[Any, Any, list[str]]:
        async with Client(server) as client:
            names = [t.name for t in (await client.list_tools()).tools]
            args = {"agent_id": "agent-1", "recipe_node": rn, "step_id": "p"}
            ok = await client.call_tool("recipe_reopen", {**args, "idempotency_key": "m1"})
            no = await client.call_tool("recipe_reopen", {**args, "idempotency_key": "m2"})
            return ok, no, names

    ok, no, names = asyncio.run(body())
    assert "recipe_reopen" in names
    assert ok.structured_content["op"] == "recipe.reopen"
    assert no.is_error and no.structured_content["error"]["code"] == "step_not_done"
