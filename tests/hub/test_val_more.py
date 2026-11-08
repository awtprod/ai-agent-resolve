"""Validator probes aimed at the mutant survivors (pins the tip already satisfies)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel
from recipe_support import FixedClock
from val_support import start_run, step_states, to_gate, worker_idle
from write_support import AGENT, HUMAN, brief_data, plan_fixture

from aar.hub.events import Actor
from aar.hub.reviews import decide
from aar.hub.writes import WriteError, create_brief, create_recipe, submit_plan
from aar.recipes import registry
from aar.recipes.context import StepFailure, StepOutput
from aar.recipes.engine import advance, claim, heartbeat, record_result
from aar.recipes.models import PlanExpect, RecipeDef, StepDef
from aar.recipes.run import recipe_cancel, recipe_retry, recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view

A = Actor("agent", "worker-a")
B = Actor("agent", "worker-b")


class P(BaseModel):
    pass


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)}")


def _mk(hub: Any, name: str, steps: tuple, *, activate: bool = True) -> str:
    registry.register(RecipeDef(name=name, version=1, params_model=P, steps=steps))
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe=name), AGENT, idempotency_key=f"b-{name}"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": name, "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key=f"r-{name}",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    if activate:
        __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
            hub.log, rn, AGENT, idempotency_key=f"a-{name}"
        )  # A1: start via recipe_start
    return rn


@pytest.fixture
def clean_registry() -> Any:
    registry.clear()
    yield
    registry.clear()


def _subject(ctx: Any) -> StepOutput:
    aid = ctx.put(b"render", meta={"kind": "short"})
    return StepOutput(
        outputs={"subject_node": ctx.op_node, "artifact_ids": [aid]}, artifact_ids=(aid,)
    )


def test_recipe_ending_in_a_gate_is_not_done_before_the_human(
    hub: Any, clean_registry: Any
) -> None:
    registry.register_callable("v.subject", _subject)
    rn = _mk(
        hub,
        "vgatelast",
        (
            StepDef(id="a", kind="cold", callable_id="v.subject"),
            StepDef(id="g", kind="gate", subject="a", on_reject="a", needs=("a",)),
        ),
    )
    advance(hub.log, rn, actor=AGENT)
    worker_idle(hub, None)
    st = load_state(hub.log, rn)
    _obs(
        "gate-last recipe before decision",
        [st.recipe_status, step_states(status_view(hub.log, rn))],
    )
    assert st.recipe_status == "active"
    g = st.latest_in_round("g", 1)
    decide(hub.log, hub.project_id, g.review_id, "approve", "ok", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    assert load_state(hub.log, rn).recipe_status == "done"


def test_retry_refused_when_out_of_attempts(hub: Any, clean_registry: Any) -> None:
    def boom(ctx: Any) -> Any:
        raise StepFailure("boom", "always")

    registry.register_callable("v.boom", boom)
    rn = _mk(hub, "vboom2", (StepDef(id="b", kind="cold", callable_id="v.boom", max_attempts=2),))
    advance(hub.log, rn, actor=AGENT)
    worker_idle(hub, None)
    recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="r1")
    worker_idle(hub, None)
    view = status_view(hub.log, rn)
    with pytest.raises(WriteError) as excinfo:
        recipe_retry(hub.log, rn, "b", AGENT, idempotency_key="r2")
    acts = [a for a in view["next_actions"] if a.get("step_id") == "b"]
    _obs("out of attempts", [excinfo.value.code, acts, load_state(hub.log, rn).recipe_status])
    assert excinfo.value.code == "no_attempts_left"
    assert all(a.get("tool") != "recipe_retry" for a in acts)
    assert load_state(hub.log, rn).recipe_status == "blocked"


def test_heartbeating_worker_keeps_its_lease(hub: Any, clean_registry: Any) -> None:
    registry.register_callable("v.ok", lambda ctx: StepOutput())
    rn = _mk(hub, "vhb", (StepDef(id="s", kind="cold", callable_id="v.ok", idempotent=True),))
    clock = FixedClock()
    advance(hub.log, rn, actor=AGENT, clock=clock)
    op = load_state(hub.log, rn).latest_in_round("s", 1)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=A,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )
    for n in (1, 2, 3):
        clock.tick(15)
        heartbeat(
            hub.log,
            project_id=hub.project_id,
            op_id=op.op_id,
            worker=A,
            clock=clock,
            lease_s=60,
            n=n,
        )
    clock.tick(25)  # t0+70: past the claim's lease, inside the last heartbeat's
    stolen = claim(
        hub.log,
        project_id=hub.project_id,
        op=load_state(hub.log, rn).latest_in_round("s", 1),
        worker=B,
        idempotent=True,
        clock=clock,
    )
    advance(hub.log, rn, actor=AGENT, clock=clock)
    _obs("steal at t0+70 with heartbeats", stolen)
    assert stolen is False


def test_result_after_cancel_is_refused_quietly(hub: Any, registered_recipe: Any) -> None:
    rn, _fake = start_run(hub)
    c = load_state(hub.log, rn).latest_in_round("c", 1)
    claim(
        hub.log,
        project_id=hub.project_id,
        op=c,
        worker=A,
        idempotent=True,
        clock=__import__("aar.hub.common", fromlist=["utcnow"]).utcnow,
    )
    recipe_cancel(hub.log, rn, AGENT, idempotency_key="cx")
    try:
        wrote = record_result(
            hub.log, project_id=hub.project_id, op_id=c.op_id, actor=A, outcome="done", outputs={}
        )
    except Exception as exc:
        wrote = type(exc).__name__
    _obs("late result after cancel", wrote)
    assert wrote is False


def test_cancel_right_after_start_cancels_cold_ops(hub: Any, registered_recipe: Any) -> None:
    from recipe_support import make_run

    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")  # c pending, h pending, p active
    recipe_cancel(hub.log, rn, AGENT, idempotency_key="cx")
    live = [(o.step_id, o.kind) for o in load_state(hub.log, rn).ops if o.is_live]
    _obs("live after cancel", live)
    assert live == []


def test_gate_status_is_needs_human(hub: Any, registered_recipe: Any) -> None:
    rn, fake = to_gate(hub)
    view = status_view(hub.log, rn, dispatcher=fake)
    g = next(s for s in view["steps"] if s["step_id"] == "g")
    act = [a for a in view["next_actions"] if a.get("step_id") == "g"]
    _obs("gate state", [g["state"], act])
    assert g["state"] == "needs_human" and act and act[0]["who"] == "human"
    assert "tool" not in act[0]


def test_plan_required_rule_that_did_not_run(hub: Any, clean_registry: Any) -> None:
    rn = _mk(
        hub,
        "vreq",
        (
            StepDef(
                id="p",
                kind="agent",
                instruction="plan",
                expects=PlanExpect(required_rules=("end-on-laugh",)),
            ),
        ),
    )
    advance(hub.log, rn, actor=AGENT)
    plan = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=rn,
        idempotency_key="p",
    ).body["node"]["id"]
    with pytest.raises(WriteError) as excinfo:
        step_complete(
            hub.log, hub.store, rn, "p", None, {"plan_node": plan}, AGENT, idempotency_key="c"
        )
    _obs("required rule skipped", excinfo.value.code)
    assert excinfo.value.code == "expects_unmet"


def test_media_kind_and_json_each_alone(hub: Any, clean_registry: Any) -> None:
    from aar.recipes.models import ArtifactExpect, JsonExpect

    class J(BaseModel):
        n: int

    rn = _mk(
        hub,
        "vmk",
        (
            StepDef(id="art", kind="agent", instruction="r", expects=ArtifactExpect("short")),
            StepDef(id="js", kind="agent", instruction="j", expects=JsonExpect(J)),
        ),
    )
    advance(hub.log, rn, actor=AGENT)
    thumb = hub.store.put(b"t", actor=AGENT, project_id=hub.project_id, meta={"kind": "thumbnail"})
    codes = []
    for step, out in (("art", {"artifact_id": thumb}), ("js", {"json": {"n": "many"}})):
        try:
            step_complete(hub.log, hub.store, rn, step, None, out, AGENT, idempotency_key=step)
            codes.append("accepted")
        except WriteError as exc:
            codes.append(exc.code)
    _obs("media kind / json alone", codes)
    assert codes == ["expects_unmet", "expects_unmet"]
