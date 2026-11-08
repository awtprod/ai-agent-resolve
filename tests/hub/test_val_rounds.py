"""Validator probes: review rounds, rollback order, limits 1 and 10."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel
from val_support import approve, complete_plan, gate_op, reject, start_run, worker_idle
from write_support import AGENT, HUMAN, brief_data

from aar.hub.reviews import decide
from aar.hub.writes import WriteError, create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.engine import advance
from aar.recipes.models import JsonExpect, RecipeDef, StepDef
from aar.recipes.run import step_complete
from aar.recipes.state import load_state


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)}")


class J(BaseModel):
    v: int


class Pm(BaseModel):
    pass


def _subject(ctx: Any) -> StepOutput:
    aid = ctx.put(f"render {ctx.op_node}".encode(), meta={"kind": "short"})
    return StepOutput(
        outputs={"subject_node": ctx.op_node, "artifact_ids": [aid]}, artifact_ids=(aid,)
    )


def _plain(ctx: Any) -> StepOutput:
    return StepOutput(outputs={"ok": True})


@pytest.fixture
def round_recipe(hub: Any) -> Any:
    registry.clear()
    registry.register_callable("val.subject", _subject)
    registry.register_callable("val.plain", _plain)
    registry.register(
        RecipeDef(
            name="valround",
            version=1,
            params_model=Pm,
            steps=(
                StepDef(id="a", kind="cold", callable_id="val.plain"),
                StepDef(id="p", kind="agent", instruction="j", expects=JsonExpect(J)),
                StepDef(id="x", kind="cold", callable_id="val.plain"),  # independent, after p
                StepDef(id="b", kind="cold", callable_id="val.subject", needs=("p",)),
                StepDef(id="g", kind="gate", subject="b", on_reject="p", needs=("b",)),
                StepDef(id="z", kind="cold", callable_id="val.plain", needs=("g",)),
            ),
        )
    )
    yield
    registry.clear()


def _mk(hub: Any, rounds: int | None, key: str = "r") -> str:
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valround"), AGENT, idempotency_key=f"b{key}"
    )
    body: dict[str, Any] = {
        "recipe": "valround",
        "brief_id": brief.body["node"]["id"],
        "params": {},
    }
    if rounds is not None:
        body["max_review_rounds"] = rounds
    rn = create_recipe(
        hub.log,
        hub.project_id,
        body,
        AGENT,
        idempotency_key=key,
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
        hub.log, rn, AGENT, idempotency_key=f"a{key}"
    )  # A1: start via recipe_start
    advance(hub.log, rn, actor=AGENT)
    return rn


def _round_to_gate(hub: Any, rn: str, rnd: int) -> str:
    worker_idle(hub, None)
    step_complete(
        hub.log, hub.store, rn, "p", None, {"json": {"v": rnd}}, AGENT, idempotency_key=f"c{rnd}"
    )
    worker_idle(hub, None)
    st = load_state(hub.log, rn)
    op = st.latest_in_round("g", st.current_round())
    assert op is not None and op.review_id, f"gate open in round {rnd}"
    return op.review_id


def test_reject_rolls_back_from_on_reject_in_recipe_order(hub: Any, round_recipe: Any) -> None:
    rn = _mk(hub, 3)
    review = _round_to_gate(hub, rn, 1)
    seq_before = max(e.seq for e in hub.log.iter_events())
    decide(hub.log, hub.project_id, review, "reject", "again", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    st = load_state(hub.log, rn)
    by_op = {op.op_id: op.step_id for op in st.ops}
    cancelled = [
        by_op.get(e.payload["op_node"])
        for e in hub.log.iter_events(types=["recipe.cancelled"])
        if e.seq > seq_before
    ]
    r1 = {op.step_id: (op.status, op.superseded) for op in st.ops if op.round == 1}
    _obs("superseded in order", cancelled)
    _obs("round-1 ops after reject", r1)
    _obs(
        "round now / ops in round 2",
        [st.current_round(), sorted({op.step_id for op in st.ops if op.round == 2})],
    )
    assert cancelled == ["p", "x", "b", "g"], cancelled
    assert r1["a"] == ("done", False)


@pytest.mark.parametrize("limit", [1, 10])
def test_limits(hub: Any, round_recipe: Any, limit: int) -> None:
    rn = _mk(hub, limit)
    reviews = []
    for rnd in range(1, limit + 1):
        review = _round_to_gate(hub, rn, rnd)
        reviews.append(review)
        st = load_state(hub.log, rn)
        assert st.current_round() == rnd
        decide(hub.log, hub.project_id, review, "reject", f"no {rnd}", HUMAN)
        advance(hub.log, rn, actor=AGENT)
    st = load_state(hub.log, rn)
    g = st.latest_in_round("g", limit)
    _obs(
        f"limit {limit}",
        [
            len(set(reviews)),
            st.current_round(),
            g.status,
            (g.result_error or {}).get("code"),
            st.recipe_status,
        ],
    )
    assert len(set(reviews)) == limit
    assert g.status == "failed" and g.result_error["code"] == "review_rounds_exhausted"
    assert st.recipe_status == "blocked"
    # nothing opens a round past the limit
    for _ in range(3):
        advance(hub.log, rn, actor=AGENT)
        worker_idle(hub, None)
    st = load_state(hub.log, rn)
    assert max(op.round for op in st.ops) == limit
    assert not any(op.is_live for op in st.ops if op.step_id in ("p", "g"))


@pytest.mark.parametrize("bad", [0, 11, True, "3"])
def test_max_review_rounds_bounds(hub: Any, round_recipe: Any, bad: Any) -> None:
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valround"), AGENT, idempotency_key="bb"
    )
    with pytest.raises(WriteError) as excinfo:
        create_recipe(
            hub.log,
            hub.project_id,
            {
                "recipe": "valround",
                "brief_id": brief.body["node"]["id"],
                "params": {},
                "max_review_rounds": bad,
            },
            AGENT,
            idempotency_key="rb",
            params_validator=registry.registry_params_validator,
        )
    _obs(f"max_review_rounds={bad!r}", excinfo.value.code)


def test_full_run_three_rounds_on_test_recipe(hub: Any, registered_recipe: Any) -> None:
    recipe_node, fake = start_run(hub, max_review_rounds=3)
    worker_idle(hub, fake)
    for k in ("r1", "r2"):
        complete_plan(hub, recipe_node, fake, k)
        reject(hub, recipe_node, fake)
    complete_plan(hub, recipe_node, fake, "r3")
    approve(hub, recipe_node, fake)
    worker_idle(hub, fake)
    st = load_state(hub.log, recipe_node)
    _obs(
        "3 rounds",
        [
            st.recipe_status,
            st.current_round(),
            sorted((o.step_id, o.round, o.attempt, o.status, o.superseded) for o in st.ops),
        ],
    )
    assert st.recipe_status == "done" and st.current_round() == 3
    g = gate_op(hub, recipe_node)
    assert g.round == 3
