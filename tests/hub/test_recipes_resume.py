"""Kill and resume from the log alone: lapsed leases, and the hot write-ahead re-send."""

from __future__ import annotations

import signal
import time
from collections.abc import Iterator
from typing import Any

import pytest
import recipe_proc
from pydantic import BaseModel, ConfigDict
from recipe_support import (
    hot_fake,
    make_run,
    submit_test_plan,
)
from write_support import AGENT, HUMAN, Hub, make_hub

from aar.hub.common import parse_ts, utcnow
from aar.hub.events import Actor, EventLog
from aar.hub.reviews import decide
from aar.hub.writes import set_own_status
from aar.recipes import registry
from aar.recipes.engine import advance
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


# -- hot write-ahead ----------------------------------------------------------


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _CrashOnce:
    """A dispatcher that raises on the first submit (a crash before the submitted record)."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.crashed = False

    def submit(self, **kw: Any) -> str:
        if not self.crashed:
            self.crashed = True
            raise RuntimeError("crash between the hot intent and the submitted record")
        return self.inner.submit(**kw)

    def poll(self, job_ref: str) -> Any:
        return self.inner.poll(job_ref)


@pytest.fixture
def hot_recipe() -> Iterator[None]:
    registry.clear()
    registry.register(
        RecipeDef(
            name="hotwa",
            version=1,
            params_model=_Params,
            steps=(
                StepDef(id="h", kind="hot", hot_kind="x"),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("h",),
                ),
            ),
        )
    )
    yield
    registry.clear()


def test_hot_write_ahead_resends_same_request_id(hub: Hub, hot_recipe: None) -> None:
    from write_support import brief_data

    from aar.hub.writes import create_brief, create_recipe

    brief_id = create_brief(
        hub.log, brief_data(hub.project_id, recipe="hotwa"), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    recipe_node = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "hotwa", "brief_id": brief_id, "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]

    from aar.recipes.ports import HotJobView, HotResult

    fake = hot_fake()
    fake.set_behaviour("x", lambda params: HotJobView("done", result=HotResult(outputs={"x": 1})))
    crasher = _CrashOnce(fake)

    with pytest.raises(RuntimeError, match="crash between"):
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start", dispatcher=crasher)

    # The intent is on the log; the submitted record is not.
    state = load_state(hub.log, recipe_node)
    op = state.latest_in_round("h", 1)
    assert op is not None and op.hot_intent and not op.hot_submitted
    assert fake.calls == []  # nothing reached the real dispatcher yet

    # Resume: advance re-submits the SAME request_id (the op id) to the (now working) dispatcher.
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=crasher)
    state = load_state(hub.log, recipe_node)
    op = state.latest_in_round("h", 1)
    assert op.hot_submitted and op.hot_request_id == op.op_id
    assert fake.calls == [op.op_id], "exactly the same request_id, sent once to the dispatcher"
    assert state.is_step_done("h")


# -- review rounds exhausted --------------------------------------------------


def test_review_rounds_exhausted_blocks(hub: Hub, registered_recipe: Any) -> None:
    fake = hot_fake()
    recipe_node = make_run(hub, max_review_rounds=1)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start", dispatcher=fake)
    from aar.recipes.worker import run_worker

    run_worker(hub.log, hub.store, AGENT, dispatcher=fake, once=True)
    plan_node = submit_test_plan(hub, recipe_node, AGENT, key="plan")
    step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        None,
        {"plan_node": plan_node},
        AGENT,
        idempotency_key="done-p",
        dispatcher=fake,
    )
    review = load_state(hub.log, recipe_node).latest_in_round("g", 1).review_id
    decide(hub.log, hub.project_id, review, "reject", "no", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)

    state = load_state(hub.log, recipe_node)
    assert state.recipe_status == "blocked"
    gate_op = next(op for op in state.attempts_for("g") if op.round == 1)
    assert gate_op.status == "failed"
    assert gate_op.result_error["code"] == "review_rounds_exhausted"


# -- kill mid-step, lease lapses ----------------------------------------------


def _claimed_lease_until(log: EventLog, project_id: str) -> Any:
    for event in log.read(0, project_id=project_id, types=["recipe.claimed"], limit=-1):
        return parse_ts(event.payload["lease_until"])
    return None


def _kill_worker_mid_slow_step(hub: Hub, tmp_path: Any, recipe_name: str) -> tuple[Any, Any, str]:
    moddir, driver = recipe_proc.materialize(tmp_path)
    recipe_node = recipe_proc.make_cold_recipe(hub, recipe_name)
    set_own_status(hub.log, hub.project_id, recipe_node, "active", AGENT, idempotency_key="act")
    hub.log.close()
    db = tmp_path / "hub.db"
    proc = recipe_proc.spawn(
        driver,
        moddir,
        "worker",
        db,
        hub.root / "artifacts",
        "agent:w1",
        recipe_node,
        AAR_LEASE="1",
        AAR_HB="100",
        AAR_SLEEP="30",
        AAR_POLL="0.05",
    )

    def claimed() -> bool:
        probe = EventLog(db, read_only=True)
        try:
            return _claimed_lease_until(probe, hub.project_id) is not None
        finally:
            probe.close()

    assert recipe_proc.wait_for(claimed, timeout=30), "the worker should claim the slow step"
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)
    # Wait for the 1s lease to lapse.
    probe = EventLog(db, read_only=True)
    lease_until = _claimed_lease_until(probe, hub.project_id)
    probe.close()
    while utcnow() <= lease_until:
        time.sleep(0.05)
    return moddir, db, recipe_node


def test_kill_idempotent_step_is_reclaimed(hub: Hub, tmp_path: Any) -> None:
    moddir, db, recipe_node = _kill_worker_mid_slow_step(hub, tmp_path, "slow_idem")
    recipe_proc.load_in_process(moddir)  # register the recipes in this process too
    from aar.recipes.worker import run_worker

    hub.log = EventLog(db)
    assert not load_state(hub.log, recipe_node).is_step_done("slow")
    # A fresh worker re-claims the lapsed idempotent op and completes it (its callable sleeps 0s).
    for _ in range(10):
        if run_worker(hub.log, hub.store, Actor("agent", "w2"), once=True) == 0:
            break
    state = load_state(hub.log, recipe_node)
    assert state.is_step_done("slow")


def test_kill_nonidempotent_step_fails_lease_lost(hub: Hub, tmp_path: Any) -> None:
    moddir, db, recipe_node = _kill_worker_mid_slow_step(hub, tmp_path, "slow_nonidem")
    recipe_proc.load_in_process(moddir)
    hub.log = EventLog(db)
    # advance reaps the lapsed non-idempotent claim: the step fails lease_lost, recipe blocks.
    advance(hub.log, recipe_node, actor=Actor("agent", "reaper"))
    state = load_state(hub.log, recipe_node)
    slow = state.latest_in_round("slow", 1)
    assert slow.status == "failed"
    assert slow.result_error["code"] == "lease_lost"
    assert state.recipe_status == "blocked"
