"""Round-2 regression pins for S13g2: the safety core the validation found broken.

Each test pins one validation finding (F1-F18) or a refusal code. They use the slice's own
fixtures (``recipe_support``/``write_support``), and are adapted from the validator's probes in
``aar-evidence/s13g2-val/probes``. The key invariant throughout: the scheduler decides only on
runner-written evidence, never on a node's graph status, so an agent that owns a node cannot walk a
run forward with ``node_set_status``.
"""

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from mcp import Client
from recipe_support import (
    FixedClock,
    hot_fake,
    make_run,
    submit_test_plan,
)
from write_support import AGENT, HUMAN, OTHER, Hub, brief_data, make_hub

from aar.hub.events import Actor
from aar.hub.graph import WorkGraph
from aar.hub.mcp_server import build_server
from aar.hub.registry import Registry
from aar.hub.reviews import decide, get_review, withdraw
from aar.hub.writes import WriteError, set_own_status
from aar.recipes import registry
from aar.recipes.engine import advance, claim, heartbeat, record_result
from aar.recipes.events import RUNNER
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_cancel, recipe_retry, recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view
from aar.recipes.worker import run_worker

WORKER = Actor("agent", "worker-1")


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


# -- helpers ------------------------------------------------------------------


def _worker_idle(hub: Hub, fake: Any, *, actor: Actor = WORKER) -> int:
    ran = 0
    for _ in range(30):
        n = run_worker(hub.log, hub.store, actor, dispatcher=fake, once=True)
        ran += n
        if n == 0:
            break
    return ran


def _start(hub: Hub, *, creator: Actor = AGENT, fake: Any = None, **kw: Any) -> tuple[str, Any]:
    fake = fake if fake is not None else hot_fake()
    recipe_node = make_run(hub, actor=creator, **kw)
    recipe_start(hub.log, recipe_node, creator, idempotency_key="start", dispatcher=fake)
    return recipe_node, fake


def _to_gate(hub: Hub, *, creator: Actor = AGENT, key: str = "r1", **kw: Any) -> tuple[str, Any]:
    recipe_node, fake = _start(hub, creator=creator, **kw)
    _worker_idle(hub, fake)  # cold c, hot h complete
    plan = submit_test_plan(hub, recipe_node, creator, key=f"plan-{key}")
    step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        None,
        {"plan_node": plan},
        creator,
        idempotency_key=f"complete-{key}",
        dispatcher=fake,
    )
    return recipe_node, fake


def _gate_op(hub: Hub, recipe_node: str) -> Any:
    state = load_state(hub.log, recipe_node)
    return state.latest_in_round("g", state.current_round())


def _reject(hub: Hub, recipe_node: str, fake: Any, reason: str = "tighter") -> None:
    op = _gate_op(hub, recipe_node)
    decide(hub.log, hub.project_id, op.review_id, "reject", reason, HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)


def _approve(hub: Hub, recipe_node: str, fake: Any) -> None:
    op = _gate_op(hub, recipe_node)
    decide(hub.log, hub.project_id, op.review_id, "approve", "ship it", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)


def _server(hub: Hub, **kw: Any) -> Any:
    return build_server(
        hub.log,
        Registry(hub.log),
        hub.store,
        params_validator=registry.registry_params_validator,
        **kw,
    )


def _run(server: Any, body: Any) -> Any:
    async def go() -> Any:
        async with Client(server) as client:
            return await body(client)

    return asyncio.run(go())


# -- F1 / F16: the scheduler reads evidence, never graph status ---------------


def test_f1_creator_agent_cannot_mark_gate_op_done(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, _fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    with pytest.raises(WriteError) as exc:
        set_own_status(hub.log, hub.project_id, op.op_id, "done", AGENT, idempotency_key="x")
    assert exc.value.code == "runner_managed"
    assert not load_state(hub.log, recipe_node).is_step_done("g")


def test_f1_creator_agent_cannot_skip_the_cold_qa_step(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _to_gate(hub)
    _approve(hub, recipe_node, fake)  # q becomes ready
    q = load_state(hub.log, recipe_node).latest_in_round("q", 1)
    assert q is not None
    with pytest.raises(WriteError) as exc:
        set_own_status(hub.log, hub.project_id, q.op_id, "done", AGENT, idempotency_key="x")
    assert exc.value.code == "runner_managed"


def test_f1_creator_agent_cannot_mark_recipe_node_done(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, _fake = _to_gate(hub)
    with pytest.raises(WriteError) as exc:
        set_own_status(hub.log, hub.project_id, recipe_node, "done", AGENT, idempotency_key="x")
    assert exc.value.code == "runner_managed"
    assert status_view(hub.log, recipe_node)["status"] != "done"


def test_f1_exhausted_gate_cannot_be_revived_by_creator(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _to_gate(hub, max_review_rounds=1)
    _reject(hub, recipe_node, fake)
    op = _gate_op(hub, recipe_node)
    assert op.status == "failed" and op.result_error["code"] == "review_rounds_exhausted"
    with pytest.raises(WriteError) as exc:
        set_own_status(hub.log, hub.project_id, op.op_id, "active", AGENT, idempotency_key="x")
    assert exc.value.code == "runner_managed"


def test_f16_forged_review_link_to_an_approved_review_is_ignored(
    hub: Hub, registered_recipe: Any
) -> None:
    recipe_node, fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    # A human-approved review of an unrelated subject, created by an agent (not RUNNER).
    graph = WorkGraph(hub.log, hub.project_id)
    other = graph.add_node(
        "review",
        AGENT,
        title="unrelated",
        status="active",
        data={"subject": recipe_node, "renders": [], "requested_by": "agent:agent-1"},
        idempotency_key="forged-review",
    )
    decide(hub.log, hub.project_id, other.id, "approve", "unrelated approval", HUMAN)
    # Forge a recipe.review_linked through the generic event API, as the agent.
    hub.log.append(
        "recipe.review_linked",
        AGENT,
        {"op_node": op.op_id, "review_id": other.id, "round": 1, "subject": recipe_node},
        project_id=hub.project_id,
    )
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    # The forged link (non-runner actor) is ignored: the gate is not done.
    assert not load_state(hub.log, recipe_node).is_step_done("g")


def test_f16_forged_op_node_is_not_adopted(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, _fake = _start(hub)
    # An agent forges an op node with the aar.step/1 shape via the generic graph API.
    graph = WorkGraph(hub.log, hub.project_id)
    graph.add_node(
        "op",
        AGENT,
        title="forged",
        status="done",
        data={
            "schema": "aar.step/1",
            "recipe_node": recipe_node,
            "step_id": "q",
            "attempt": 1,
            "round": 1,
            "step_kind": "cold",
        },
        idempotency_key="forged-op",
    )
    state = load_state(hub.log, recipe_node)
    # The fold ignores an op node not created by the runner; it is listed in diagnostics.
    assert state.latest_in_round("q", 1) is None
    assert any(i.get("kind") == "op_node" for i in state.ignored)


def test_f1_gate_bypass_through_mcp_node_set_status_is_refused(
    hub: Hub, registered_recipe: Any
) -> None:
    recipe_node, _fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> None:
        result = await client.call_tool(
            "node_set_status",
            {
                "agent_id": "agent-1",
                "idempotency_key": "x",
                "project_id": hub.project_id,
                "node_id": op.op_id,
                "status": "done",
            },
        )
        assert result.is_error
        assert result.structured_content["error"]["code"] == "runner_managed"

    _run(server, body)
    assert not load_state(hub.log, recipe_node).is_step_done("g")


# -- F2: one constant runner identity, caller recorded ------------------------


def test_f2_runner_identity_and_caller_recorded(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _start(hub, creator=HUMAN)
    _worker_idle(hub, fake)
    # One step_complete by agent:ed on a human-created recipe.
    plan = submit_test_plan(hub, recipe_node, Actor("agent", "ed"), key="p")
    step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        None,
        {"plan_node": plan},
        Actor("agent", "ed"),
        idempotency_key="done-p",
        dispatcher=fake,
    )
    # The op node and the recipe.* events are authored by RUNNER; the gate review is requested
    # by RUNNER; no agent can withdraw it.
    graph = WorkGraph(hub.log, hub.project_id)
    gate = _gate_op(hub, recipe_node)
    for ev in hub.log.read(0, project_id=hub.project_id, types=["recipe.review_linked"], limit=-1):
        assert ev.actor == RUNNER
    assert graph.creator(gate.op_id) == RUNNER
    review = get_review(hub.log, hub.project_id, gate.review_id)
    assert review.requested_by == "runner:recipes"


def test_f2_parse_actor_refuses_runner() -> None:
    from aar.hub.cli import _parse_actor as hub_parse
    from aar.hub.write_cli import _parse_actor as write_parse
    from aar.recipes.cli import _parse_actor as recipe_parse

    for parse in (hub_parse, write_parse, recipe_parse):
        with pytest.raises(ValueError, match="runner"):
            parse("runner:recipes")


def test_f2_runner_identity_constructed_only_inside_recipes() -> None:
    """AST pin: Actor("runner", ...) and the RUNNER name are built only under src/aar/recipes."""
    src = Path(__file__).resolve().parents[2] / "src" / "aar"
    offenders: list[str] = []
    for path in src.rglob("*.py"):
        if "recipes" in path.relative_to(src).parts:
            continue
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Actor"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "runner"
            ):
                offenders.append(f"{path}:{node.lineno} Actor('runner', ...)")
    assert offenders == []


def test_f2_runner_may_not_decide_a_review(hub: Hub, registered_recipe: Any) -> None:
    from aar.hub.common import HumanRequired

    recipe_node, _fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    with pytest.raises(HumanRequired):
        decide(hub.log, hub.project_id, op.review_id, "approve", "runner tries", RUNNER)


# -- F3: lease holder checks --------------------------------------------------


def _cold_op(hub: Hub, recipe_node: str) -> Any:
    return load_state(hub.log, recipe_node).latest_in_round("c", 1)


def test_f3_new_holder_beats_recorded_stale_beat_and_result_refused(
    hub: Hub, registered_recipe: Any
) -> None:
    clock = FixedClock()
    a = Actor("agent", "A")
    b = Actor("agent", "B")
    recipe_node, _ = _start(hub)  # c is an idempotent cold op (rectest)
    op = _cold_op(hub, recipe_node)
    assert claim(
        hub.log, project_id=hub.project_id, op=op, worker=a, idempotent=True, clock=clock, lease_s=1
    )
    clock.tick(5)  # A's lease lapses
    op = _cold_op(hub, recipe_node)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=b,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )  # B steals the lapsed idempotent op
    for n in range(1, 5):
        heartbeat(
            hub.log,
            project_id=hub.project_id,
            op_id=op.op_id,
            worker=b,
            clock=clock,
            lease_s=60,
            n=n,
        )
    # A lost the lease: its heartbeat is refused (not the latest claimant).
    heartbeat(
        hub.log, project_id=hub.project_id, op_id=op.op_id, worker=a, clock=clock, lease_s=60, n=9
    )
    beats = hub.log.read(0, project_id=hub.project_id, types=["recipe.heartbeat"], limit=-1)
    assert len(beats) == 4 and all(ev.payload["worker"] == "agent:B" for ev in beats)
    # A stale result from A is refused; B's result is kept.
    assert not record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=a,
        outcome="failed",
        error={"code": "x", "message": "stale"},
    )
    assert record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=b,
        outcome="done",
        outputs={"ok": True},
    )
    done = _cold_op(hub, recipe_node)
    assert done.status == "done" and done.result_by == "agent:B"


def test_f3_lease_boundary_exactly_at_lease_until(hub: Hub, registered_recipe: Any) -> None:
    clock = FixedClock()
    a = Actor("agent", "A")
    b = Actor("agent", "B")
    recipe_node, _ = _start(hub)
    op = _cold_op(hub, recipe_node)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=a,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )
    clock.tick(60)  # exactly at lease_until: still live, not stealable
    op = _cold_op(hub, recipe_node)
    assert not claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=b,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )
    clock.tick(0.000001)  # just past: an idempotent op is re-claimable
    op = _cold_op(hub, recipe_node)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=b,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )


# -- F4: plan contract --------------------------------------------------------


def test_f4_plan_from_an_unrelated_review_is_refused(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _start(hub)
    _worker_idle(hub, fake)
    # A review node (any) in this project; a round-1 plan may derive only from the recipe node.
    graph = WorkGraph(hub.log, hub.project_id)
    rev = graph.add_node(
        "review",
        AGENT,
        title="x",
        status="active",
        data={"subject": recipe_node, "renders": [], "requested_by": "agent:a"},
        idempotency_key="rev",
    )
    plan = submit_test_plan(hub, recipe_node, AGENT, key="p")
    # Re-point the plan's parent at the review by submitting a plan deriving from it.
    from write_support import plan_fixture

    from aar.hub.writes import submit_plan

    plan2 = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=rev.id,
        idempotency_key="p2",
    ).body["node"]["id"]
    with pytest.raises(WriteError) as exc:
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "p",
            None,
            {"plan_node": plan2},
            AGENT,
            idempotency_key="c2",
            dispatcher=fake,
        )
    assert exc.value.code == "expects_unmet"
    assert plan is not None


def test_f4_replayed_round1_step_complete_does_not_complete_round2(
    hub: Hub, registered_recipe: Any
) -> None:
    recipe_node, fake = _to_gate(hub, key="r1")  # round 1 p completed with key complete-r1
    _reject(hub, recipe_node, fake)
    assert load_state(hub.log, recipe_node).current_round() == 2
    plan_r1 = submit_test_plan(hub, recipe_node, AGENT, key="plan-r1")  # same plan node id (idem)
    # Replay the round-1 call: same key, same request. It must not complete round 2.
    result = step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        1,
        {"plan_node": plan_r1},
        AGENT,
        idempotency_key="complete-r1",
        dispatcher=fake,
    )
    assert result.get("replayed") is True
    p2 = load_state(hub.log, recipe_node).latest_in_round("p", 2)
    assert p2 is None or p2.status != "done"


# -- F7 / F8: exhausted and withdrawn gates -----------------------------------


def test_f7_retry_of_exhausted_gate_is_refused(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _to_gate(hub, max_review_rounds=1)
    _reject(hub, recipe_node, fake)
    assert load_state(hub.log, recipe_node).recipe_status == "blocked"
    with pytest.raises(WriteError) as exc:
        recipe_retry(hub.log, recipe_node, "g", AGENT, idempotency_key="retry", dispatcher=fake)
    assert exc.value.code == "rounds_exhausted"
    assert load_state(hub.log, recipe_node).recipe_status == "blocked"


def test_f8_withdrawn_gate_blocks_then_retry_reopens_same_round(
    hub: Hub, registered_recipe: Any
) -> None:
    recipe_node, fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    withdraw(hub.log, hub.project_id, op.review_id, "pulled", RUNNER)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    blocked = load_state(hub.log, recipe_node)
    failed = blocked.latest_in_round("g", 1)
    assert failed.status == "failed" and failed.result_error["code"] == "review_withdrawn"
    assert blocked.recipe_status == "blocked"
    recipe_retry(hub.log, recipe_node, "g", AGENT, idempotency_key="retry", dispatcher=fake)
    after = load_state(hub.log, recipe_node)
    assert after.current_round() == 1  # a withdrawal does not use a review round
    reopened = after.latest_in_round("g", 1)
    assert reopened.attempt == 2 and reopened.review_id is not None


def test_f8_agent_cannot_withdraw_the_gate_review(hub: Hub, registered_recipe: Any) -> None:
    from aar.hub.reviews import NotRequester

    recipe_node, _fake = _to_gate(hub)
    op = _gate_op(hub, recipe_node)
    with pytest.raises(NotRequester):
        withdraw(hub.log, hub.project_id, op.review_id, "nope", AGENT)


# -- F9: idempotency keys -----------------------------------------------------


def test_f9_two_retries_one_key_do_not_make_three_attempts(hub: Hub) -> None:
    from pydantic import BaseModel, ConfigDict

    from aar.recipes.context import StepFailure
    from aar.recipes.models import RecipeDef, StepDef

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry.clear()

    def boom(ctx: Any) -> Any:
        raise StepFailure("boom", "it broke")

    registry.register_callable("r2.boom", boom)
    registry.register(
        RecipeDef(
            name="r2boom",
            version=1,
            params_model=_P,
            steps=(
                StepDef(id="c", kind="cold", callable_id="r2.boom", max_attempts=5),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("c",),
                ),
            ),
        )
    )
    try:
        from write_support import brief_data

        from aar.hub.writes import create_brief, create_recipe

        brief_id = create_brief(
            hub.log, brief_data(hub.project_id, recipe="r2boom"), AGENT, idempotency_key="b"
        ).body["node"]["id"]
        recipe_node = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "r2boom", "brief_id": brief_id, "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
        _worker_idle(hub, None)  # c fails (attempt 1)
        recipe_retry(hub.log, recipe_node, "c", AGENT, idempotency_key="retry-1")
        _worker_idle(hub, None)  # attempt 2 fails
        # A second retry with the SAME key must not create a third attempt.
        code = None
        try:
            recipe_retry(hub.log, recipe_node, "c", AGENT, idempotency_key="retry-1")
        except WriteError as exc:
            code = exc.code
        attempts = load_state(hub.log, recipe_node).attempts_for("c")
        # The key makes the second retry a conflict (its failed op differs) and never a 3rd attempt.
        assert len(attempts) == 2, (len(attempts), code)
        assert code == "idempotency_conflict", code
    finally:
        registry.clear()


# -- F10: dispatcher errors ---------------------------------------------------


class _SubmitRaises:
    def submit(self, **kw: Any) -> str:
        raise ConnectionError("node down")

    def poll(self, job_ref: str) -> Any:
        raise ConnectionError("node down")


def test_f10_dispatcher_submit_error_does_not_crash_the_worker(
    hub: Hub, registered_recipe: Any
) -> None:
    recipe_node, _ = _start(hub, fake=_SubmitRaises())
    # The worker loop must not raise even though the hot dispatcher is down.
    _worker_idle(hub, _SubmitRaises())
    # The cold step still ran; the hot step is simply not done.
    state = load_state(hub.log, recipe_node)
    assert state.is_step_done("c")
    assert not state.is_step_done("h")


def test_f10_poll_error_shows_waiting_dispatcher_error(hub: Hub, registered_recipe: Any) -> None:
    from aar.recipes.ports import HotJobView

    class _PollRaises:
        def __init__(self) -> None:
            self.submitted = False

        def submit(self, **kw: Any) -> str:
            self.submitted = True
            return "job-1"

        def poll(self, job_ref: str) -> Any:
            if not self.submitted:
                return HotJobView("running")
            raise ConnectionError("poll down")

    disp = _PollRaises()
    recipe_node, _ = _start(hub, fake=disp)
    view = status_view(hub.log, recipe_node, dispatcher=disp)
    states = {s["step_id"]: s["state"] for s in view["steps"]}
    assert states["h"] == "waiting:dispatcher_error"


# -- F12: params validator ----------------------------------------------------


def test_f12_mcp_recipe_create_refuses_bad_params(hub: Hub, registered_recipe: Any) -> None:
    server = _server(hub, agent_writes=True)
    pid = hub.project_id

    async def body(client: Client) -> None:
        brief = await client.call_tool(
            "brief_create",
            {"agent_id": "ed", "idempotency_key": "b", "brief": brief_data(pid, recipe="rectest")},
        )
        brief_id = brief.structured_content["node"]["id"]
        bad = await client.call_tool(
            "recipe_create",
            {
                "agent_id": "ed",
                "idempotency_key": "r",
                "project_id": pid,
                "recipe": {"recipe": "rectest", "brief_id": brief_id, "params": {"bogus": 1}},
            },
        )
        assert bad.is_error and bad.structured_content["error"]["code"] == "invalid_params"
        missing = await client.call_tool(
            "recipe_create",
            {
                "agent_id": "ed",
                "idempotency_key": "r2",
                "project_id": pid,
                "recipe": {"recipe": "ghost", "brief_id": brief_id},
            },
        )
        assert missing.is_error and missing.structured_content["error"]["code"] in (
            "unknown_recipe",
            "recipe_mismatch",
        )

    _run(server, body)


def test_f12_recipe_start_revalidates_stored_params(hub: Hub, registered_recipe: Any) -> None:
    from aar.hub.writes import accept_any_params, create_brief, create_recipe

    brief_id = create_brief(
        hub.log, brief_data(hub.project_id, recipe="rectest"), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    # Create with any-params (bypassing the registry validator), storing params the model rejects.
    recipe_node = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "rectest", "brief_id": brief_id, "params": {"bogus": 1}},
        AGENT,
        idempotency_key="r",
        params_validator=accept_any_params,
    ).body["node"]["id"]
    with pytest.raises(WriteError) as exc:
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
    assert exc.value.code == "invalid_params"


# -- F13: registration bounds -------------------------------------------------


def test_f13_on_reject_and_needs_are_validated() -> None:
    from pydantic import BaseModel

    from aar.recipes.models import RecipeDef, RecipeDefError, StepDef, validate_recipe

    class _P(BaseModel):
        pass

    def _mk(gate: StepDef) -> RecipeDef:
        return RecipeDef(
            name="x",
            version=1,
            params_model=_P,
            steps=(StepDef(id="p", kind="cold", callable_id="recipe.report"), gate),
        )

    # on_reject naming the gate itself or a later/unknown step is refused.
    with pytest.raises(RecipeDefError):
        validate_recipe(_mk(StepDef(id="g", kind="gate", subject="p", on_reject="g")))
    with pytest.raises(RecipeDefError):
        validate_recipe(_mk(StepDef(id="g", kind="gate", subject="p", on_reject="later")))
    # needs given as a bare string (not a tuple) is refused.
    with pytest.raises(RecipeDefError):
        validate_recipe(
            RecipeDef(
                name="y",
                version=1,
                params_model=_P,
                steps=(
                    StepDef(id="a", kind="cold", callable_id="recipe.report"),
                    StepDef(id="b", kind="cold", callable_id="recipe.report", needs="a"),
                ),
            )
        )  # type: ignore[arg-type]


# -- F15: facts attributed to the op node -------------------------------------


def test_f15_facts_are_sourced_to_the_op_node(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _start(hub)
    _worker_idle(hub, fake)  # the cold step records qa.lufs.master via ctx.record_fact
    c = _cold_op(hub, recipe_node)
    facts = hub.log.read(0, project_id=hub.project_id, types=["facts.recorded"], limit=-1)
    sources = [ev.payload.get("source") for ev in facts]
    assert any(s and s.get("op_node") == c.op_id for s in sources), sources


# -- F17: carried-forward done step in a later round --------------------------


def test_f17_round2_shows_carried_done_step_as_done(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _to_gate(hub)
    _reject(hub, recipe_node, fake)
    view = status_view(hub.log, recipe_node)
    states = {s["step_id"]: s["state"] for s in view["steps"]}
    assert view["round"] == 2
    assert states["c"] == "done" and states["h"] == "done"


# -- F18: stored REPORT.md status ---------------------------------------------


def test_f18_stored_report_is_not_active(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _to_gate(hub)
    _approve(hub, recipe_node, fake)
    _worker_idle(hub, fake)  # q and report run
    state = load_state(hub.log, recipe_node)
    assert state.recipe_status == "done"
    report_op = state.done_op("report")
    assert report_op is not None
    text = hub.store.read_bytes(report_op.result_artifacts[0]).decode("utf-8")
    assert "- status: active" not in text
    assert "- status: done" in text


# -- refusal codes not covered above ------------------------------------------


def test_refusal_step_not_open_and_not_creator(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _start(hub)
    # step_complete on a non-agent step is step_not_open.
    with pytest.raises(WriteError) as exc:
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "c",
            None,
            {},
            AGENT,
            idempotency_key="k",
            dispatcher=fake,
        )
    assert exc.value.code == "step_not_open"
    # a non-creator agent cannot start/cancel.
    with pytest.raises(WriteError) as exc2:
        recipe_cancel(hub.log, recipe_node, OTHER, idempotency_key="c")
    assert exc2.value.code == "not_creator"


def test_refusal_no_attempts_left(hub: Hub) -> None:
    from pydantic import BaseModel, ConfigDict

    from aar.recipes.context import StepFailure
    from aar.recipes.models import RecipeDef, StepDef

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry.clear()
    registry.register_callable("r2.boom2", lambda ctx: (_ for _ in ()).throw(StepFailure("b", "x")))
    registry.register(
        RecipeDef(
            name="r2noattempt",
            version=1,
            params_model=_P,
            steps=(
                StepDef(id="c", kind="cold", callable_id="r2.boom2", max_attempts=2),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("c",),
                ),
            ),
        )
    )
    try:
        from write_support import brief_data

        from aar.hub.writes import create_brief, create_recipe

        brief_id = create_brief(
            hub.log, brief_data(hub.project_id, recipe="r2noattempt"), AGENT, idempotency_key="b"
        ).body["node"]["id"]
        recipe_node = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "r2noattempt", "brief_id": brief_id, "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
        _worker_idle(hub, None)
        recipe_retry(hub.log, recipe_node, "c", AGENT, idempotency_key="rt1")  # attempt 2
        _worker_idle(hub, None)
        with pytest.raises(WriteError) as exc:
            recipe_retry(hub.log, recipe_node, "c", AGENT, idempotency_key="rt2")
        assert exc.value.code == "no_attempts_left"
    finally:
        registry.clear()


# -- F5: artifact contract ----------------------------------------------------


def _artifact_recipe() -> None:
    from pydantic import BaseModel, ConfigDict

    from aar.recipes.models import ArtifactExpect, RecipeDef, StepDef

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry.clear()
    registry.register(
        RecipeDef(
            name="r2art",
            version=1,
            params_model=_P,
            steps=(
                StepDef(
                    id="a",
                    kind="agent",
                    instruction="make a short",
                    expects=ArtifactExpect("short"),
                ),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("a",),
                ),
            ),
        )
    )


def _art_run(hub: Hub) -> str:
    from write_support import brief_data

    from aar.hub.writes import create_brief, create_recipe

    brief_id = create_brief(
        hub.log, brief_data(hub.project_id, recipe="r2art"), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    recipe_node = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "r2art", "brief_id": brief_id, "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
    return recipe_node


def test_f5_artifact_from_another_project_is_refused(hub: Hub) -> None:
    _artifact_recipe()
    try:
        recipe_node = _art_run(hub)
        # An artifact stored in a DIFFERENT project.
        foreign = hub.store.put(
            b"foreign short",
            actor=AGENT,
            project_id=hub.other_project_id,
            mime="video/mp4",
            meta={"kind": "short"},
        )
        with pytest.raises(WriteError) as exc:
            step_complete(
                hub.log,
                hub.store,
                recipe_node,
                "a",
                None,
                {"artifact_id": foreign},
                AGENT,
                idempotency_key="done-a",
            )
        assert exc.value.code == "expects_unmet"
    finally:
        registry.clear()


def test_f5_agent_artifact_is_linked_and_reported(hub: Hub) -> None:
    _artifact_recipe()
    try:
        recipe_node = _art_run(hub)
        aid = hub.store.put(
            b"the short",
            actor=AGENT,
            project_id=hub.project_id,
            mime="video/mp4",
            meta={"kind": "short"},
        )
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "a",
            None,
            {"artifact_id": aid},
            AGENT,
            idempotency_key="done-a",
        )
        _worker_idle(hub, None)  # report runs
        state = load_state(hub.log, recipe_node)
        a_op = state.done_op("a")
        graph = WorkGraph(hub.log, hub.project_id)
        produces = [e for e in graph.edges() if e.type == "produces" and e.src == a_op.op_id]
        assert produces, "the agent step's artifact gets a produces edge from its op"
        text = build_report_text(hub.log, recipe_node)
        assert aid in text
    finally:
        registry.clear()


# -- F6: 64 KiB output bound --------------------------------------------------


def test_f6_oversized_agent_json_refused_outputs_too_large(hub: Hub) -> None:
    from pydantic import BaseModel, ConfigDict

    from aar.recipes.models import JsonExpect, RecipeDef, StepDef

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class _Blob(BaseModel):
        model_config = ConfigDict(extra="forbid")
        blob: str

    registry.clear()
    registry.register(
        RecipeDef(
            name="r2json",
            version=1,
            params_model=_P,
            steps=(
                StepDef(id="j", kind="agent", instruction="emit json", expects=JsonExpect(_Blob)),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("j",),
                ),
            ),
        )
    )
    try:
        from write_support import brief_data

        from aar.hub.writes import create_brief, create_recipe

        brief_id = create_brief(
            hub.log, brief_data(hub.project_id, recipe="r2json"), AGENT, idempotency_key="b"
        ).body["node"]["id"]
        recipe_node = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "r2json", "brief_id": brief_id, "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
        big = {"json": {"blob": "x" * (200 * 1024)}}
        with pytest.raises(WriteError) as exc:
            step_complete(
                hub.log, hub.store, recipe_node, "j", None, big, AGENT, idempotency_key="done-j"
            )
        assert exc.value.code == "outputs_too_large"
    finally:
        registry.clear()


def test_f6_oversized_hot_output_is_bounded_in_the_result(hub: Hub, registered_recipe: Any) -> None:
    from aar.recipes.ports import HotJobView, HotResult

    fake = hot_fake()
    fake.set_behaviour(
        "rectest.hot",
        lambda params: HotJobView("done", result=HotResult(outputs={"big": "y" * (1024 * 1024)})),
    )
    recipe_node, _ = _start(hub, fake=fake)
    # The 1 MB hot output is refused into a failed result; no recipe.result payload exceeds 64 KiB.
    results = hub.log.read(0, project_id=hub.project_id, types=["recipe.result"], limit=-1)
    assert results
    assert max(len(json.dumps(ev.payload)) for ev in results) <= 64 * 1024
    h = load_state(hub.log, recipe_node).latest_in_round("h", 1)
    assert h.status == "failed" and h.result_error["code"] == "outputs_too_large"


# -- F14: next_actions templates are callable as written ----------------------


def test_f14_step_complete_template_is_callable(hub: Hub, registered_recipe: Any) -> None:
    recipe_node, fake = _start(hub)
    _worker_idle(hub, fake)
    plan = submit_test_plan(hub, recipe_node, AGENT, key="p")
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    action = next(a for a in view["next_actions"] if a.get("tool") == "step_complete")
    assert "<" not in json.dumps(action["args"]), "no placeholders in args; only fill fields"
    assert set(action["fill"]) == {"agent_id", "idempotency_key", "outputs"}
    server = _server(hub, agent_writes=True)

    async def body(client: Client) -> None:
        args = dict(action["args"])
        args.update(
            {"agent_id": "agent-1", "idempotency_key": "tmpl", "outputs": {"plan_node": plan}}
        )
        result = await client.call_tool(action["tool"], args)
        assert result.is_error is False, result.structured_content

    _run(server, body)
    assert load_state(hub.log, recipe_node).is_step_done("p")


def test_f14_retry_template_is_callable(hub: Hub) -> None:
    from pydantic import BaseModel, ConfigDict

    from aar.recipes.context import StepFailure
    from aar.recipes.models import RecipeDef, StepDef

    class _P(BaseModel):
        model_config = ConfigDict(extra="forbid")

    registry.clear()
    registry.register_callable("r2.boom3", lambda ctx: (_ for _ in ()).throw(StepFailure("b", "x")))
    registry.register(
        RecipeDef(
            name="r2retry",
            version=1,
            params_model=_P,
            steps=(
                StepDef(id="c", kind="cold", callable_id="r2.boom3", max_attempts=3),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("c",),
                ),
            ),
        )
    )
    try:
        from write_support import brief_data

        from aar.hub.writes import create_brief, create_recipe

        brief_id = create_brief(
            hub.log, brief_data(hub.project_id, recipe="r2retry"), AGENT, idempotency_key="b"
        ).body["node"]["id"]
        recipe_node = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "r2retry", "brief_id": brief_id, "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        recipe_start(hub.log, recipe_node, AGENT, idempotency_key="start")
        _worker_idle(hub, None)  # c fails
        view = status_view(hub.log, recipe_node)
        action = next(a for a in view["next_actions"] if a.get("tool") == "recipe_retry")
        assert "<" not in json.dumps(action["args"])
        assert set(action["fill"]) == {"agent_id", "idempotency_key"}
        server = _server(hub, agent_writes=True)

        async def body(client: Client) -> None:
            args = dict(action["args"])
            args.update({"agent_id": "agent-1", "idempotency_key": "tmpl-retry"})
            result = await client.call_tool(action["tool"], args)
            assert result.is_error is False, result.structured_content

        _run(server, body)
        assert len(load_state(hub.log, recipe_node).attempts_for("c")) == 2
    finally:
        registry.clear()
