"""Validator probes: leases, heartbeats, late results, timeouts, facts attribution, needs."""

from __future__ import annotations

import json
import threading
from datetime import timedelta
from typing import Any

import pytest
from pydantic import BaseModel
from recipe_support import FixedClock
from write_support import AGENT, brief_data

from aar.hub.events import Actor
from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.context import StepOutput
from aar.recipes.engine import advance, claim, heartbeat, record_result
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.state import load_state
from aar.recipes.worker import run_worker

A = Actor("agent", "worker-a")
B = Actor("agent", "worker-b")


def _obs(label: str, value: Any) -> None:
    print(f"\nOBS {label}: {json.dumps(value, default=str, sort_keys=True)}")


class P(BaseModel):
    pass


SEEN: dict[str, Any] = {}


def _ok(step: str) -> Any:
    def fn(ctx: Any) -> StepOutput:
        SEEN[step] = {"needs": sorted(ctx.needs), "op": ctx.op_node}
        return StepOutput(outputs={"step": step})

    return fn


@pytest.fixture
def lease_recipe(hub: Any) -> Any:
    registry.clear()
    registry.register_callable("val.a", _ok("a"))
    registry.register_callable("val.b", _ok("b"))
    registry.register_callable("val.c", _ok("c"))
    registry.register(
        RecipeDef(
            name="vallease",
            version=1,
            params_model=P,
            steps=(
                StepDef(id="idem", kind="cold", callable_id="val.a", idempotent=True),
                StepDef(id="nonidem", kind="cold", callable_id="val.b", idempotent=False),
                StepDef(id="c", kind="cold", callable_id="val.c", needs=("idem",)),
            ),
        )
    )
    brief = create_brief(
        hub.log, brief_data(hub.project_id, recipe="vallease"), AGENT, idempotency_key="b"
    )
    rn = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "vallease", "brief_id": brief.body["node"]["id"], "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
        hub.log, rn, AGENT, idempotency_key="act"
    )  # A1: start via recipe_start
    yield rn
    registry.clear()


def _op(hub: Any, rn: str, step: str) -> Any:
    return load_state(hub.log, rn).latest_in_round(step, 1)


def test_lease_boundary_exactly_at_lease_until(hub: Any, lease_recipe: str) -> None:
    rn = lease_recipe
    clock = FixedClock()
    advance(hub.log, rn, actor=AGENT, clock=clock)
    for step, idem in (("idem", True), ("nonidem", False)):
        op = _op(hub, rn, step)
        assert claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=A,
            idempotent=idem,
            clock=clock,
            lease_s=60,
        )
    t0 = clock.now
    clock.now = t0 + timedelta(seconds=60)  # exactly lease_until
    res = {}
    for step, idem in (("idem", True), ("nonidem", False)):
        op = _op(hub, rn, step)
        res[f"{step}@lease_until"] = claim(
            hub.log, project_id=hub.project_id, op=op, worker=B, idempotent=idem, clock=clock
        )
    advance(hub.log, rn, actor=AGENT, clock=clock)
    res["nonidem status @lease_until after advance"] = _op(hub, rn, "nonidem").status
    clock.now = t0 + timedelta(seconds=60, microseconds=1)
    for step, idem in (("idem", True), ("nonidem", False)):
        op = _op(hub, rn, step)
        res[f"{step}@+1us"] = claim(
            hub.log, project_id=hub.project_id, op=op, worker=B, idempotent=idem, clock=clock
        )
    advance(hub.log, rn, actor=AGENT, clock=clock)
    non = _op(hub, rn, "nonidem")
    res["nonidem after advance @+1us"] = [non.status, (non.result_error or {}).get("code")]
    _obs("lease boundary", res)
    assert res["idem@lease_until"] is False and res["nonidem@lease_until"] is False
    assert res["nonidem status @lease_until after advance"] == "active"
    assert res["idem@+1us"] is True and res["nonidem@+1us"] is False
    assert res["nonidem after advance @+1us"] == ["failed", "lease_lost"]


def test_heartbeat_from_worker_that_lost_its_lease_is_refused(hub: Any, lease_recipe: str) -> None:
    rn = lease_recipe
    clock = FixedClock()
    advance(hub.log, rn, actor=AGENT, clock=clock)
    op = _op(hub, rn, "idem")
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=A,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )
    for n in range(1, 5):  # A runs 60 s, beating every 15 s
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
    clock.tick(61)  # A stalls; its lease lapses
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=_op(hub, rn, "idem"),
        worker=B,
        idempotent=True,
        clock=clock,
        lease_s=60,
    )
    t_claim_b = clock.now
    # B (the holder) heartbeats with its own counter starting at 1, like _Heartbeat does
    for n in range(1, 5):
        clock.tick(15)
        heartbeat(
            hub.log,
            project_id=hub.project_id,
            op_id=op.op_id,
            worker=B,
            clock=clock,
            lease_s=60,
            n=n,
        )
    lease_b = _op(hub, rn, "idem").lease_until
    hb = [
        (e.payload.get("worker"), e.payload["lease_until"])  # A5
        for e in hub.log.iter_events(types=["recipe.heartbeat"])
    ]
    _obs("heartbeat events (actor, lease_until)", hb)
    _obs(
        "lease_until after B's 4 beats vs B claim+60", [lease_b, t_claim_b + timedelta(seconds=60)]
    )
    clock.tick(1)
    stolen = claim(
        hub.log,
        project_id=hub.project_id,
        op=_op(hub, rn, "idem"),
        worker=Actor("agent", "worker-c"),
        idempotent=True,
        clock=clock,
    )
    _obs("worker C re-claims while B is alive and beating", stolen)
    # Now the stale A heartbeats again
    clock.tick(1)
    accepted_a = True
    try:
        heartbeat(
            hub.log,
            project_id=hub.project_id,
            op_id=op.op_id,
            worker=A,
            clock=clock,
            lease_s=60,
            n=5,
        )
    except Exception as exc:
        accepted_a = type(exc).__name__
    a_beats = [
        e
        for e in hub.log.iter_events(types=["recipe.heartbeat"])
        if e.payload.get("worker") == f"{A.kind}:{A.id}"
    ]  # A5: worker is in the payload; actor is RUNNER
    _obs("stale A heartbeat after losing the lease", [accepted_a, len(a_beats)])
    b_beats = [
        e
        for e in hub.log.iter_events(types=["recipe.heartbeat"])
        if e.payload.get("worker") == f"{B.kind}:{B.id}"
    ]  # A5
    assert len(b_beats) == 4, (
        f"the lease holder B heartbeated 4 times but only {len(b_beats)} were recorded: "
        "heartbeat keys are aar.recipe.heartbeat/<op>/<n> with n restarting at 1 per worker, so "
        "B's beats collide with the old holder's and are silently dropped"
    )
    assert not stolen, "B's lease lapsed although B was alive and heartbeating"
    assert len(a_beats) == 4, "a heartbeat from the worker that lost its lease was accepted"


def test_result_from_worker_that_lost_its_lease(hub: Any, lease_recipe: str) -> None:
    rn = lease_recipe
    clock = FixedClock()
    advance(hub.log, rn, actor=AGENT, clock=clock)
    op = _op(hub, rn, "idem")
    claim(hub.log, project_id=hub.project_id, op=op, worker=A, idempotent=True, clock=clock)
    clock.tick(61)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=_op(hub, rn, "idem"),
        worker=B,
        idempotent=True,
        clock=clock,
    )
    # Order 1: B (holder) finishes first, then stale A: A must not overwrite.
    wrote_b = record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=B,
        outcome="done",
        outputs={"by": "B"},
    )
    wrote_a = record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=A,
        outcome="failed",
        outputs={"by": "A"},
    )
    st = _op(hub, rn, "idem")
    _obs("B then A", [wrote_b, wrote_a, st.status, st.result_outputs])
    assert st.result_outputs == {"by": "B"} and st.status == "done"


def test_stale_result_first_is_refused(hub: Any, lease_recipe: str) -> None:
    rn = lease_recipe
    clock = FixedClock()
    advance(hub.log, rn, actor=AGENT, clock=clock)
    op = _op(hub, rn, "idem")
    claim(hub.log, project_id=hub.project_id, op=op, worker=A, idempotent=True, clock=clock)
    clock.tick(61)
    assert claim(
        hub.log,
        project_id=hub.project_id,
        op=_op(hub, rn, "idem"),
        worker=B,
        idempotent=True,
        clock=clock,
    )
    wrote_a = record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=A,
        outcome="failed",
        outputs={"by": "A"},
        error={"code": "x", "message": "stale"},
    )
    wrote_b = record_result(
        hub.log,
        project_id=hub.project_id,
        op_id=op.op_id,
        actor=B,
        outcome="done",
        outputs={"by": "B"},
    )
    st = _op(hub, rn, "idem")
    _obs("stale A first, then holder B", [wrote_a, wrote_b, st.status, st.result_outputs])
    assert wrote_a is False, (
        "a result from the worker that lost its lease (A) was recorded and the current holder's "
        f"(B) result was then dropped: op is {st.status} with outputs {st.result_outputs}"
    )


def test_timeout_and_needs_and_facts_attribution(hub: Any) -> None:
    clock = FixedClock()

    def late(ctx: Any) -> StepOutput:
        ctx.record_fact("qa.lufs.master", {"value": -14.0, "unit": "LUFS"})
        clock.tick(11)  # returns after its 10 s deadline
        return StepOutput(outputs={"late": True})

    registry.clear()
    registry.register_callable("val.late", late)
    registry.register_callable("val.a", _ok("a"))
    registry.register_callable("val.b", _ok("b"))
    registry.register_callable("val.c", _ok("c"))
    registry.register(
        RecipeDef(
            name="valtime",
            version=1,
            params_model=P,
            steps=(
                StepDef(id="a", kind="cold", callable_id="val.a"),
                StepDef(id="b", kind="cold", callable_id="val.b"),
                StepDef(id="c", kind="cold", callable_id="val.c", needs=("a",)),
                StepDef(id="late", kind="cold", callable_id="val.late", timeout_s=10),
            ),
        )
    )
    try:
        brief = create_brief(
            hub.log, brief_data(hub.project_id, recipe="valtime"), AGENT, idempotency_key="b"
        )
        rn = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "valtime", "brief_id": brief.body["node"]["id"], "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
            hub.log, rn, AGENT, idempotency_key="act"
        )  # A1: start via recipe_start
        for _ in range(8):
            if run_worker(hub.log, hub.store, A, once=True, clock=clock) == 0:
                break
        st = load_state(hub.log, rn)
        late_op = st.latest_in_round("late", 1)
        _obs("late step", [late_op.status, (late_op.result_error or {}).get("code")])
        _obs("needs seen by c", SEEN.get("c"))
        facts = [e for e in hub.log.iter_events(types=["facts.recorded"])]
        _obs(
            "fact events", [(e.actor.id, e.payload.get("source"), e.idempotency_key) for e in facts]
        )
        assert late_op.status == "failed" and late_op.result_error["code"] == "timeout"
        assert SEEN["c"]["needs"] == ["a"]
        src = facts[0].payload.get("source") or {}
        assert late_op.op_id in json.dumps(src), (
            "the facts writer attributes the fact to the recipe node only "
            f"(source={src}); the op node appears only inside the idempotency key"
        )
    finally:
        registry.clear()


def test_sigterm_drain_in_process(hub: Any) -> None:
    """stop set mid-step: the current step finishes and is recorded; nothing new is claimed."""
    stop = threading.Event()
    started = threading.Event()
    release = threading.Event()

    def slow(ctx: Any) -> StepOutput:
        started.set()
        release.wait(10)
        return StepOutput(outputs={"cancelled_seen": ctx.cancelled()})

    registry.clear()
    registry.register_callable("val.slow", slow)
    registry.register_callable("val.b", _ok("b"))
    registry.register(
        RecipeDef(
            name="valdrain",
            version=1,
            params_model=P,
            steps=(
                StepDef(id="s1", kind="cold", callable_id="val.slow"),
                StepDef(id="s2", kind="cold", callable_id="val.b"),
            ),
        )
    )
    try:
        brief = create_brief(
            hub.log, brief_data(hub.project_id, recipe="valdrain"), AGENT, idempotency_key="b"
        )
        rn = create_recipe(
            hub.log,
            hub.project_id,
            {"recipe": "valdrain", "brief_id": brief.body["node"]["id"], "params": {}},
            AGENT,
            idempotency_key="r",
            params_validator=registry.registry_params_validator,
        ).body["node"]["id"]
        __import__("aar.recipes.run", fromlist=["recipe_start"]).recipe_start(
            hub.log, rn, AGENT, idempotency_key="act"
        )  # A1: start via recipe_start
        result: dict[str, Any] = {}
        t = threading.Thread(
            target=lambda: result.setdefault(
                "ran", run_worker(hub.log, hub.store, A, stop=stop, poll_s=0.05)
            )
        )
        t.start()
        assert started.wait(10)
        stop.set()
        release.set()
        t.join(10)
        st = load_state(hub.log, rn)
        s1, s2 = st.latest_in_round("s1", 1), st.latest_in_round("s2", 1)
        _obs("drain", [result, s1.status, s1.result_outputs, s2.status])
        assert s1.status == "done" and s2.status == "pending" and result["ran"] == 1
    finally:
        registry.clear()
