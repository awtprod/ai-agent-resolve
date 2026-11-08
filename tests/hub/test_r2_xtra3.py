"""Round-3 extra probes (run explicitly) aimed at the slice-test survivors: N11, N12, N14, R05,
R09, R16, R17. Each asserts the brief-required behaviour on the tip."""

from __future__ import annotations

import threading
from typing import Any

import pytest
from pydantic import BaseModel
from recipe_support import RECIPE_NAME, make_run, passing_qa_report, store_qa_report
from val_support import approve, complete_plan, start_run, worker_idle
from write_support import AGENT, brief_data

from aar.hub.graph import WorkGraph
from aar.hub.writes import WriteError, create_brief
from aar.recipes import engine, registry, worker
from aar.recipes.context import StepOutput
from aar.recipes.models import RecipeDef, RecipeDefError, StepDef, validate_recipe
from aar.recipes.run import recipe_start
from aar.recipes.state import load_state
from aar.recipes.status import status_view


class P(BaseModel):
    pass


def test_n11_start_revalidates_params_of_a_generic_graph_recipe(
    hub: Any, registered_recipe: Any
) -> None:
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe=RECIPE_NAME), AGENT, idempotency_key="b"
    )
    node = WorkGraph(hub.log, hub.project_id).add_node(
        "recipe",
        AGENT,
        title="raw",
        data={
            "schema": "aar.recipe/1",
            "recipe": RECIPE_NAME,
            "brief_id": brief.body["node"]["id"],
            "params": {"bogus": 1},
            "max_review_rounds": 3,
        },
        idempotency_key="raw-recipe",
    )
    with pytest.raises(WriteError) as exc:
        recipe_start(hub.log, node.id, AGENT, idempotency_key="s")
    assert exc.value.code == "invalid_params"


@pytest.mark.parametrize("on_reject", ["g", "later", "nope"])
def test_n12_on_reject_self_later_unknown_typed_refusal(on_reject: str) -> None:
    steps = [
        StepDef(id="p", kind="cold", callable_id="recipe.report"),
        StepDef(id="g", kind="gate", subject="p", on_reject=on_reject, needs=("p",)),
    ]
    if on_reject == "later":
        steps.append(StepDef(id="later", kind="cold", callable_id="recipe.report"))
    with pytest.raises(RecipeDefError):
        validate_recipe(RecipeDef(name="x", version=1, params_model=P, steps=tuple(steps)))


def test_n14_stored_report_is_not_active(hub: Any, registered_recipe: Any) -> None:
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    complete_plan(hub, rn, fake, "r1")
    approve(hub, rn, fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, rn)
    stored = hub.store.read_bytes(st.done_op("report").result_artifacts[0]).decode()
    status = [ln for ln in stored.splitlines() if ln.startswith("- status:")]
    print("\nOBS stored status line:", status)
    assert status and "active" not in status[0]


class _NonePoll:
    def submit(self, **kw: Any) -> str:
        return "job-1"

    def poll(self, job_ref: Any) -> Any:
        return None


def test_r16_r17_none_poll_shows_unknown_then_fails_after_the_limit(
    hub: Any, registered_recipe: Any
) -> None:
    disp = _NonePoll()
    rn = make_run(hub)
    recipe_start(hub.log, rn, AGENT, idempotency_key="s", dispatcher=disp)
    first = {s["step_id"]: s["state"] for s in status_view(hub.log, rn, dispatcher=disp)["steps"]}
    for _ in range(8):
        engine.advance(hub.log, rn, actor=AGENT, dispatcher=disp)
    h = load_state(hub.log, rn).latest_in_round("h", 1)
    print("\nOBS none poll:", first["h"], h.status, (h.result_error or {}).get("code"))
    assert first["h"] == "dispatcher_unknown"
    assert h.status == "failed" and h.result_error["code"] == "dispatcher_bad_response"


def test_r09_worker_goes_on_when_one_recipe_advance_raises(
    hub: Any, registered_recipe: Any, monkeypatch: Any
) -> None:
    registry.register_callable("val.ok", lambda ctx: StepOutput(outputs={"ok": True}))
    registry.register(
        RecipeDef(
            name="valok",
            version=1,
            params_model=P,
            steps=(StepDef(id="c", kind="cold", callable_id="val.ok"),),
        )
    )
    bad = make_run(hub)
    recipe_start(hub.log, bad, AGENT, idempotency_key="s-bad")
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valok"), AGENT, idempotency_key="b-ok"
    )
    from aar.hub.writes import create_recipe

    good = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "valok", "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="r-ok",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, good, AGENT, idempotency_key="s-ok")
    real = worker.advance

    def flaky(log: Any, recipe_node: str, **kw: Any) -> Any:
        if recipe_node == bad:
            raise RuntimeError("advance blew up for one recipe")
        return real(log, recipe_node, **kw)

    monkeypatch.setattr(worker, "advance", flaky)
    stop = threading.Event()
    err: list[str] = []

    def run() -> None:
        try:
            worker.run_worker(hub.log, hub.store, AGENT, stop=stop, poll_s=0.05)
        except BaseException as exc:
            err.append(type(exc).__name__)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(3)
    stop.set()
    th.join(10)
    c = load_state(hub.log, good).latest_in_round("c", 1)
    print("\nOBS worker with one raising recipe:", err, c.status if c else None)
    assert not err and c is not None and c.status == "done"
    _ = (passing_qa_report, store_qa_report)
