"""Validator probes: hot steps and the expects contracts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from recipe_support import hot_fake, make_run, submit_test_plan
from val_support import complete_plan, gate_op, reject, start_run, step_states, worker_idle
from write_support import AGENT, OTHER, brief_data, plan_fixture

from aar.hub.writes import (
    WriteError,
    create_brief,
    create_recipe,
    set_own_status,
    submit_plan,
)
from aar.recipes import engine, registry
from aar.recipes.engine import advance
from aar.recipes.models import ArtifactExpect, JsonExpect, RecipeDef, StepDef
from aar.recipes.ports import FakeHotDispatcher, HotJobView
from aar.recipes.run import recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:2000]}")


# -- hot -------------------------------------------------------------------------------------


def test_hot_request_id_is_the_op_id_and_crash_resends_it(
    hub: Any, registered_recipe: Any, monkeypatch: Any
) -> None:
    fake = FakeHotDispatcher()
    fake.set_behaviour("rectest.hot", lambda p: HotJobView("running"))
    real = engine._record_hot_submitted
    calls = {"n": 0}

    def crash_once(*a: Any, **k: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("crash between intent and submitted")
        real(*a, **k)

    monkeypatch.setattr(engine, "_record_hot_submitted", crash_once)
    recipe_node = make_run(hub)
    # A2: F10 now swallows the crash inside advance (by design); the call no longer raises.
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s", dispatcher=fake)
    h = load_state(hub.log, recipe_node).latest_in_round("h", 1)
    _obs("after crash: intent/submitted", [h.hot_intent, h.hot_submitted, h.hot_request_id])
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    _obs("status after crash", step_states(view)["h"])
    for _ in range(5):
        advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    h = load_state(hub.log, recipe_node).latest_in_round("h", 1)
    _obs("calls / submits / op", [fake.calls, fake.submits, h.op_id, h.hot_job_ref])
    assert fake.calls == [h.op_id, h.op_id], "re-sent once with the same request_id, then polled"
    assert fake.submits == [h.op_id]
    assert h.hot_request_id == h.op_id


def test_hot_states_pass_through_and_no_dispatcher(hub: Any, registered_recipe: Any) -> None:
    fake = FakeHotDispatcher()
    fake.set_behaviour("rectest.hot", lambda p: HotJobView("queued"))
    recipe_node = make_run(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s")  # no dispatcher
    worker_idle(hub, None)
    v0 = status_view(hub.log, recipe_node)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    h = load_state(hub.log, recipe_node).latest_in_round("h", 1)
    got = {"no_dispatcher": step_states(v0)["h"], "c_done_pc_off": step_states(v0)["c"]}
    for state in ("waiting_for_node", "node_offline", "running", "queued"):
        fake.set_state(h.hot_job_ref, state, wait_reason=state)
        got[state] = step_states(status_view(hub.log, recipe_node, dispatcher=fake))["h"]
    got["submitted_but_viewed_without_dispatcher"] = step_states(status_view(hub.log, recipe_node))[
        "h"
    ]
    _obs("hot states", got)
    assert got["no_dispatcher"] == "waiting:no_node_dispatcher"
    assert got["c_done_pc_off"] == "done"
    assert got["waiting_for_node"] == "waiting:waiting_for_node"
    assert got["node_offline"] == "waiting:node_offline"


def test_dispatcher_error_does_not_crash_the_worker(hub: Any, registered_recipe: Any) -> None:
    class Down:
        def submit(self, **kw: Any) -> str:
            raise ConnectionError("node unreachable")

        def poll(self, job_ref: str) -> HotJobView:
            raise ConnectionError("node unreachable")

    recipe_node = make_run(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s")
    outcome: Any = "ok"
    try:
        worker_idle(hub, Down())
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    st = load_state(hub.log, recipe_node)
    _obs("worker with a failing dispatcher", [outcome, st.is_step_done("c")])
    assert outcome == "ok", "a dispatcher exception escapes advance and kills the worker loop"


# -- expects: plan ------------------------------------------------------------------------------


def _second_recipe(hub: Any) -> str:
    brief = create_brief(
        hub.log,
        brief_data(hub.project_id, recipe="rectest", show="Other"),
        AGENT,
        idempotency_key="brief-2",
    )
    qa = load_state  # noqa: F841
    return create_recipe(
        hub.log,
        hub.project_id,
        {
            "recipe": "rectest",
            "brief_id": brief.body["node"]["id"],
            "params": {"qa_report": "0" * 64},
        },
        AGENT,
        idempotency_key="recipe-2",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]


def _try_complete(hub: Any, recipe_node: str, outputs: Any, key: str, actor: Any = AGENT) -> Any:
    try:
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "p",
            None,
            outputs,
            actor,
            idempotency_key=key,
            dispatcher=hot_fake(),
        )
        return "accepted"
    except WriteError as exc:
        return exc.code


def test_plan_contract_refusals(hub: Any, registered_recipe: Any) -> None:
    recipe_node, _fake = start_run(hub)
    res: dict[str, Any] = {}
    # another project's plan
    ob = create_brief(hub.log, brief_data(hub.other_project_id), AGENT, idempotency_key="ob")
    other_plan = submit_plan(
        hub.log,
        hub.store,
        hub.other_project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=ob.body["node"]["id"],
        idempotency_key="op",
    ).body["node"]["id"]
    res["other_project"] = _try_complete(hub, recipe_node, {"plan_node": other_plan}, "k1")
    # derived from a different recipe in the same project
    r2 = _second_recipe(hub)
    p_r2 = submit_test_plan(hub, r2, AGENT, key="p-r2")
    res["other_recipe"] = _try_complete(hub, recipe_node, {"plan_node": p_r2}, "k2")
    # F2: lint-blocked plan moved to active by its agent
    bad = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_broken"),
        AGENT,
        derives_from=recipe_node,
        idempotency_key="bad",
    ).body["node"]
    set_own_status(hub.log, hub.project_id, bad["id"], "active", AGENT, idempotency_key="unblk")
    res["f2_blocked_moved_active"] = _try_complete(hub, recipe_node, {"plan_node": bad["id"]}, "k3")
    # a plan deriving from a plan (PLAN_PARENT_KINDS allows plan parents)
    good = submit_test_plan(hub, recipe_node, AGENT, key="good")
    child = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=good,
        idempotency_key="child",
    ).body["node"]["id"]
    res["plan_of_plan"] = _try_complete(hub, recipe_node, {"plan_node": child}, "k4")
    # not a plan / missing
    res["not_a_plan_node"] = _try_complete(hub, recipe_node, {"plan_node": recipe_node}, "k5")
    res["missing_key"] = _try_complete(hub, recipe_node, {}, "k6")
    still_open = load_state(hub.log, recipe_node).latest_in_round("p", 1).status
    _obs("plan contract (before any review exists)", [res, still_open])
    assert all(v == "expects_unmet" for v in res.values()), res
    assert still_open == "active"


def test_plan_from_another_recipes_review_or_an_earlier_round(
    hub: Any, registered_recipe: Any
) -> None:
    # A second run of the same recipe in the same project, driven to an open (or decided) review.
    r_a, fake = start_run(hub, max_review_rounds=5)
    worker_idle(hub, fake)
    complete_plan(hub, r_a, fake, "a1")
    review_a1 = gate_op(hub, r_a).review_id
    reject(hub, r_a, fake)  # round 2
    complete_plan(hub, r_a, fake, "a2")
    review_a2 = gate_op(hub, r_a).review_id
    reject(hub, r_a, fake)  # round 3 now; review_a1 is an EARLIER round's review
    res: dict[str, Any] = {}
    p_early = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=review_a1,
        idempotency_key="early",
    ).body["node"]["id"]
    res["round3_plan_from_round1_review"] = _try_complete(hub, r_a, {"plan_node": p_early}, "e1")
    # undo for the next case is impossible (it may have been accepted); use a fresh recipe
    r_b = _second_recipe(hub)
    recipe_start(hub.log, r_b, AGENT, idempotency_key="sb", dispatcher=fake)
    p_cross = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=review_a2,
        idempotency_key="cross",
    ).body["node"]["id"]
    res["plan_from_other_recipes_review"] = _try_complete(hub, r_b, {"plan_node": p_cross}, "x1")
    _obs("plan parents that are reviews", res)
    assert res["plan_from_other_recipes_review"] == "expects_unmet", res
    assert res["round3_plan_from_round1_review"] == "expects_unmet", res


# -- expects: artifact and json ---------------------------------------------------------------


class J(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str
    n: int


class Pm(BaseModel):
    pass


@pytest.fixture
def contract_recipe(hub: Any) -> Any:
    registry.clear()
    registry.register(
        RecipeDef(
            name="valexp",
            version=1,
            params_model=Pm,
            steps=(
                StepDef(
                    id="art", kind="agent", instruction="render", expects=ArtifactExpect("short")
                ),
                StepDef(id="js", kind="agent", instruction="json", expects=JsonExpect(J)),
            ),
        )
    )
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valexp"), AGENT, idempotency_key="b"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "valexp", "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, rn, AGENT, idempotency_key="s")
    yield rn
    registry.clear()


def _complete(hub: Any, rn: str, step: str, outputs: Any, key: str) -> Any:
    try:
        step_complete(hub.log, hub.store, rn, step, None, outputs, OTHER, idempotency_key=key)
        return "accepted"
    except WriteError as exc:
        return exc.code


def test_artifact_and_json_contracts(hub: Any, contract_recipe: str) -> None:
    rn = contract_recipe
    res: dict[str, Any] = {}
    thumb = hub.store.put(
        b"thumb", actor=AGENT, project_id=hub.project_id, meta={"kind": "thumbnail"}
    )
    res["wrong_media_kind"] = _complete(hub, rn, "art", {"artifact_id": thumb}, "a1")
    res["unknown_artifact"] = _complete(hub, rn, "art", {"artifact_id": "f" * 64}, "a2")
    res["json_invalid"] = _complete(hub, rn, "js", {"json": {"title": "x", "n": "many"}}, "j1")
    res["json_extra_field"] = _complete(
        hub, rn, "js", {"json": {"title": "x", "n": 1, "z": 1}}, "j2"
    )
    res["json_missing"] = _complete(hub, rn, "js", {}, "j3")
    st = load_state(hub.log, rn)
    open_after = [st.latest_in_round("art", 1).status, st.latest_in_round("js", 1).status]
    _obs("artifact/json refusals", [res, open_after])
    assert all(v == "expects_unmet" for v in res.values()), res
    assert open_after == ["active", "active"]
    # another project's artifact, right kind
    foreign = hub.store.put(
        b"foreign short", actor=AGENT, project_id=hub.other_project_id, meta={"kind": "short"}
    )
    meta = hub.store.get_meta(foreign)
    foreign_res = _complete(hub, rn, "art", {"artifact_id": foreign}, "a3")
    _obs("other project's artifact", [meta.project_id, foreign_res])
    # outputs over 64 KiB
    big = {"json": {"title": "x" * (200 * 1024), "n": 1}}
    big_res = _complete(hub, rn, "js", big, "j4")
    ev = [e for e in hub.log.iter_events(types=["recipe.result"])]
    sizes = [len(json.dumps(e.payload)) for e in ev]
    view = status_view(hub.log, rn)
    _obs("200 KiB json outputs", [big_res, sizes, len(json.dumps(view))])
    assert foreign_res == "expects_unmet", "another project's artifact satisfied the contract"
    assert (
        big_res in ("expects_unmet", "outputs_too_large") and max(sizes or [0]) <= 64 * 1024
    ), (  # A3: brief F6 code
        f"outputs are not bounded to 64 KiB: recipe.result payload sizes {sizes}, status view "
        f"{len(json.dumps(view))} bytes"
    )
