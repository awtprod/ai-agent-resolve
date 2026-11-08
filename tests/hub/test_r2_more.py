"""Round-2 validator probes: F9 replay semantics and key scoping, F12 CLI registry, F13 on_reject,
F4 for non-plan subjects, F15 fact source, F18 claimed/report, cancel after done.

Each test asserts the SAFE (or brief-required) property; a failure is a finding.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel
from recipe_support import make_run
from val_support import approve, complete_plan, gate_op, reject, start_run, to_gate, worker_idle
from write_support import AGENT, HUMAN, OTHER, brief_data

from aar.hub.common import utcnow
from aar.hub.graph import WorkGraph
from aar.hub.reviews import get_review
from aar.hub.writes import WriteError, create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.context import StepFailure, StepOutput
from aar.recipes.engine import advance, claim
from aar.recipes.models import ArtifactExpect, PlanExpect, RecipeDef, StepDef
from aar.recipes.report import build_report_text
from aar.recipes.run import recipe_cancel, recipe_retry, recipe_start, step_complete
from aar.recipes.state import load_state
from aar.recipes.status import status_view

AAR = str(Path(sys.executable).parent / "aar")
HERE = Path(__file__).parent


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)[:3000]}")


def _code(fn: Any) -> str:
    try:
        res = fn()
        return "replayed" if isinstance(res, dict) and res.get("replayed") else "accepted"
    except WriteError as exc:
        return exc.code


class P(BaseModel):
    pass


# -- F9: replay semantics and key scoping --------------------------------------------------


def test_retry_same_key_twice_replays(hub: Any, registered_recipe: Any) -> None:
    registry.register_callable(
        "rectest.cold", lambda ctx: (_ for _ in ()).throw(StepFailure("boom", "fails"))
    )
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    first = _code(lambda: recipe_retry(hub.log, rn, "c", AGENT, idempotency_key="rk"))
    second = _code(lambda: recipe_retry(hub.log, rn, "c", AGENT, idempotency_key="rk"))
    attempts = [o.attempt for o in load_state(hub.log, rn).attempts_for("c")]
    _obs(
        "retry, same key twice (attempt 2 still pending)",
        {"first": first, "second": second, "attempts": attempts},
    )
    assert attempts == [1, 2]
    assert second == "replayed", "brief F9: the same caller/key/request replays the first result"


def test_client_keys_are_actor_scoped(hub: Any, registered_recipe: Any) -> None:
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    plan = __import__("recipe_support").submit_test_plan(hub, rn, AGENT, key="pk")
    a1 = _code(
        lambda: step_complete(
            hub.log,
            hub.store,
            rn,
            "p",
            None,
            {"plan_node": plan},
            AGENT,
            idempotency_key="shared",
            dispatcher=fake,
        )
    )
    a1_again = _code(
        lambda: step_complete(
            hub.log,
            hub.store,
            rn,
            "p",
            None,
            {"plan_node": plan},
            AGENT,
            idempotency_key="shared",
            dispatcher=fake,
        )
    )
    a2_same_key = _code(
        lambda: step_complete(
            hub.log,
            hub.store,
            rn,
            "p",
            None,
            {"plan_node": plan},
            OTHER,
            idempotency_key="shared",
            dispatcher=fake,
        )
    )
    a1_reuse_for_retry = _code(
        lambda: recipe_retry(hub.log, rn, "c", AGENT, idempotency_key="shared")
    )
    keys = sorted(
        e.idempotency_key
        for e in hub.log.iter_events(types=["hub.write"])
        if e.payload.get("op") == "recipe.step_complete"
    )
    _obs(
        "key scoping",
        {
            "agent1": a1,
            "agent1_replay": a1_again,
            "agent2_same_key": a2_same_key,
            "agent1_key_reused_for_retry": a1_reuse_for_retry,
            "step_complete record keys": keys,
        },
    )
    assert a1 == "accepted" and a1_again == "replayed"
    assert a2_same_key != "replayed", "agent-2 replayed agent-1's key"
    # c is not failed here, so the retry is refused before its key is looked up (probe design).
    assert a1_reuse_for_retry in ("idempotency_conflict", "step_not_open")


# -- F12: the CLI default validator with a registered recipe module ------------------------


def test_cli_write_recipe_accepts_a_recipe_registered_through_aar_recipes(
    hub: Any, tmp_path: Path
) -> None:
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valcli"), AGENT, idempotency_key="bcli"
    )
    req = tmp_path / "req.json"
    env = {k: v for k, v in os.environ.items() if k != "AAR_LIVE"}
    env["AAR_RECIPES"] = "valclimod"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(HERE / "recipemod"), *filter(None, [os.environ.get("PYTHONPATH")])]
    )
    out = {}
    for label, params in (("good", {"n": 2}), ("bad", {"n": "x"})):
        req.write_text(
            json.dumps(
                {
                    "project_id": hub.project_id,
                    "recipe": {
                        "recipe": "valcli",
                        "brief_id": brief.body["node"]["id"],
                        "params": params,
                    },
                }
            )
        )
        cp = subprocess.run(
            [
                AAR,
                "hub",
                "write",
                "recipe",
                "--db",
                str(hub.root / "hub.db"),
                "--actor",
                "agent:agent-1",
                "--idempotency-key",
                f"k-{label}",
                str(req),
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        err = cp.stderr.strip()
        out[label] = [
            cp.returncode,
            json.loads(err)["error"]["code"] if err.startswith("{") else err[-200:],
        ]
    _obs("aar hub write recipe with AAR_RECIPES=valclimod", out)
    assert out["good"][0] == 0, "the CLI refuses a recipe that AAR_RECIPES registers"
    assert out["bad"][1] == "invalid_params"


# -- F13: on_reject between the subject and the gate ---------------------------------------


@pytest.fixture
def onreject_recipe(hub: Any) -> Iterator[None]:
    registry.clear()
    registry.register_callable("val.x", lambda ctx: StepOutput(outputs={"x": 1}))
    yield
    registry.clear()


def test_on_reject_after_the_subject_is_refused_at_registration(onreject_recipe: None) -> None:
    try:
        registry.register(
            RecipeDef(
                name="valonrej",
                version=1,
                params_model=P,
                steps=(
                    StepDef(id="p", kind="agent", instruction="plan", expects=PlanExpect()),
                    StepDef(id="x", kind="cold", callable_id="val.x", needs=("p",)),
                    StepDef(id="g", kind="gate", subject="p", on_reject="x", needs=("x",)),
                ),
            )
        )
        got = "registered"
    except Exception as exc:
        got = f"refused: {exc}"
    _obs("gate subject=p on_reject=x (x after p)", got)
    assert got.startswith("refused"), (
        "on_reject after the subject: a reject rolls back x and g only, so round 2 re-reviews the "
        "unchanged rejected plan (models.py comment says 'at or before the subject')"
    )


# -- F4 for a non-plan (artifact) gate subject -------------------------------------------


@pytest.fixture
def art_gate_recipe(hub: Any) -> Iterator[str]:
    registry.clear()
    registry.register(
        RecipeDef(
            name="valartg",
            version=1,
            params_model=P,
            steps=(
                StepDef(
                    id="a", kind="agent", instruction="render", expects=ArtifactExpect("short")
                ),
                StepDef(id="g", kind="gate", subject="a", on_reject="a", needs=("a",)),
            ),
        )
    )
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="valartg"), AGENT, idempotency_key="ba"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "valartg", "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="ra",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    recipe_start(hub.log, rn, AGENT, idempotency_key="sa")
    yield rn
    registry.clear()


def test_rejected_artifact_cannot_be_resubmitted_in_round_2(hub: Any, art_gate_recipe: str) -> None:
    rn = art_gate_recipe
    art = hub.store.put(
        b"acme render v1", actor=AGENT, project_id=hub.project_id, meta={"kind": "short"}
    )
    step_complete(
        hub.log, hub.store, rn, "a", None, {"artifact_id": art}, AGENT, idempotency_key="a1"
    )
    g = load_state(hub.log, rn).latest_in_round("g", 1)
    from aar.hub.reviews import decide

    decide(hub.log, hub.project_id, g.review_id, "reject", "redo it", HUMAN)
    advance(hub.log, rn, actor=AGENT)
    again = _code(
        lambda: step_complete(
            hub.log, hub.store, rn, "a", None, {"artifact_id": art}, AGENT, idempotency_key="a2"
        )
    )
    st = load_state(hub.log, rn)
    g2 = st.latest_in_round("g", 2)
    r2 = get_review(hub.log, hub.project_id, g2.review_id) if g2 and g2.review_id else None
    _obs(
        "round 2: the rejected artifact resubmitted with a new key",
        {
            "result": again,
            "round2_review_renders": None if r2 is None else list(r2.renders),
            "rejected_artifact": art,
        },
    )
    assert again != "accepted", "the rejected artifact is re-reviewed in round 2"


# -- F15, F18, cancel after done -------------------------------------------------------


def test_fact_source_names_the_op(hub: Any, registered_recipe: Any) -> None:
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    c = load_state(hub.log, rn).latest_in_round("c", 1)
    sources = [e.payload.get("source") for e in hub.log.iter_events(types=["fact.recorded"])]
    if not sources:
        sources = [
            e.payload.get("source") for e in hub.log.iter_events() if e.type.startswith("fact")
        ]
    _obs("fact sources", {"sources": sources, "c_op": c.op_id})
    assert any(isinstance(s, dict) and s.get("op_node") == c.op_id for s in sources)


def test_claimed_state_and_stored_report(hub: Any, registered_recipe: Any) -> None:
    rn, _fake = start_run(hub)
    c = load_state(hub.log, rn).latest_in_round("c", 1)
    claim(hub.log, project_id=hub.project_id, op=c, worker=AGENT, idempotent=True, clock=utcnow)
    claimed_state = {s["step_id"]: s["state"] for s in status_view(hub.log, rn)["steps"]}["c"]
    _obs("claimed cold op state", claimed_state)


def test_cancel_after_done(hub: Any, registered_recipe: Any) -> None:
    rn, fake = to_gate(hub)
    approve(hub, rn, fake)
    worker_idle(hub, fake)
    assert load_state(hub.log, rn).recipe_status == "done"
    res = _code(lambda: recipe_cancel(hub.log, rn, AGENT, idempotency_key="late"))
    st = load_state(hub.log, rn)
    _obs(
        "cancel after done",
        {
            "cancel": res,
            "evidence": st.recipe_status,
            "mirror": WorkGraph(hub.log, hub.project_id).node(rn).status,
            "report_status_line": [
                ln
                for ln in build_report_text(hub.log, rn).splitlines()
                if ln.startswith("- status")
            ],
        },
    )
    assert st.recipe_status == "done", "a finished run was turned into a cancelled one"
    _ = (complete_plan, gate_op, reject, make_run)


def test_replays_in_round_3(hub: Any, registered_recipe: Any) -> None:
    """Brief F4: pin the replayed round-1 step_complete in round 2 AND round 3; also a fresh-key
    resubmission of each rejected plan."""
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    plan1 = complete_plan(hub, rn, fake, "r1")  # key complete-r1
    review1 = gate_op(hub, rn).review_id
    reject(hub, rn, fake)
    from write_support import plan_fixture

    from aar.hub.writes import submit_plan

    plan2 = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        plan_fixture("short_clean"),
        AGENT,
        derives_from=review1,
        idempotency_key="plan-r2",
    ).body["node"]["id"]
    step_complete(
        hub.log,
        hub.store,
        rn,
        "p",
        None,
        {"plan_node": plan2},
        AGENT,
        idempotency_key="complete-r2",
        dispatcher=fake,
    )
    reject(hub, rn, fake)
    assert load_state(hub.log, rn).current_round() == 3
    res = {
        "replay_r1_key": _code(
            lambda: step_complete(
                hub.log,
                hub.store,
                rn,
                "p",
                1,
                {"plan_node": plan1},
                AGENT,
                idempotency_key="complete-r1",
                dispatcher=fake,
            )
        ),
        "replay_r2_key": _code(
            lambda: step_complete(
                hub.log,
                hub.store,
                rn,
                "p",
                1,
                {"plan_node": plan2},
                AGENT,
                idempotency_key="complete-r2",
                dispatcher=fake,
            )
        ),
        "fresh_key_plan1": _code(
            lambda: step_complete(
                hub.log,
                hub.store,
                rn,
                "p",
                1,
                {"plan_node": plan1},
                AGENT,
                idempotency_key="f1",
                dispatcher=fake,
            )
        ),
        "fresh_key_plan2": _code(
            lambda: step_complete(
                hub.log,
                hub.store,
                rn,
                "p",
                1,
                {"plan_node": plan2},
                AGENT,
                idempotency_key="f2",
                dispatcher=fake,
            )
        ),
    }
    p3 = load_state(hub.log, rn).latest_in_round("p", 3)
    _obs("round 3 replays", {**res, "p3": None if p3 is None else p3.status})
    assert p3 is not None and p3.status != "done"
    assert res["fresh_key_plan1"] == res["fresh_key_plan2"] == "expects_unmet"


def test_round2_fresh_key_resubmission_of_the_rejected_plan(
    hub: Any, registered_recipe: Any
) -> None:
    """Brief F4: in round r > 1 the plan must derive from this recipe's round r-1 gate review;
    anything else is expects_unmet. The rejected round-1 plan derives from the recipe node."""
    rn, fake = start_run(hub)
    worker_idle(hub, fake)
    plan1 = complete_plan(hub, rn, fake, "r1")
    reject(hub, rn, fake)
    from val_support import mcp_run

    from aar.hub.mcp_server import build_server as _bs
    from aar.hub.registry import Registry as _Reg

    server = _bs(hub.log, _Reg(hub.log), hub.store, agent_writes=True)

    async def body(client: Any) -> Any:
        r = await client.call_tool(
            "step_complete",
            {
                "agent_id": "agent-9",
                "idempotency_key": "fresh",
                "recipe_node": rn,
                "step_id": "p",
                "outputs": {"plan_node": plan1},
            },
        )
        return [r.is_error, (r.structured_content or {}).get("error", {}).get("code")]

    got = mcp_run(server, body)
    st = load_state(hub.log, rn)
    g2 = st.latest_in_round("g", 2)
    r2 = get_review(hub.log, hub.project_id, g2.review_id) if g2 and g2.review_id else None
    _obs(
        "round 2: MCP step_complete by another agent, fresh key, the rejected round-1 plan",
        {
            "mcp": got,
            "p2": st.latest_in_round("p", 2).status,
            "round2_review_subject": None if r2 is None else r2.subject,
            "rejected_plan": plan1,
        },
    )
    assert got[0] and got[1] == "expects_unmet", "the rejected plan completed round 2"
