"""R3-1: the log keeps the released serialization (ensure_ascii=True), so idempotent retries of
non-ASCII events still replay; and R05: a byte-identical rejected plan is refused plan_unchanged.
Named test_r2_* so mutate_r3val's slice glob includes it."""

from __future__ import annotations

from typing import Any

import pytest
from recipe_support import make_run
from val_support import gate_op, start_run, worker_idle
from write_support import AGENT, HUMAN, brief_data, plan_fixture

from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.hub.reviews import decide
from aar.hub.writes import WriteError, create_brief, submit_plan
from aar.recipes.engine import advance
from aar.recipes.run import step_complete
from aar.recipes.state import load_state

ACCENT = "Café Comédie — Montée été ☃"


def test_nonascii_events_replay_under_the_tip(hub: Any) -> None:
    """Every keyed write path with accented text, retried with the same key, replays (no conflict),
    and a project with a non-ASCII name folds back identically."""
    log, pid = hub.log, hub.project_id

    # 1. a raw keyed append with non-ASCII text.
    key = "raw-accent-1"
    log.append("note.test", AGENT, {"text": ACCENT}, project_id=pid, idempotency_key=key)
    before = len(list(log.iter_events()))
    log.append("note.test", AGENT, {"text": ACCENT}, project_id=pid, idempotency_key=key)
    assert len(list(log.iter_events())) == before, "raw keyed append must replay, not re-add"

    # 2. a graph node with a non-ASCII title, retried.
    graph = WorkGraph(log, pid)
    n1 = graph.add_node(
        "brief", AGENT, title=ACCENT, status="pending", data={}, idempotency_key="node-accent"
    )
    n2 = graph.add_node(
        "brief", AGENT, title=ACCENT, status="pending", data={}, idempotency_key="node-accent"
    )
    assert n1.id == n2.id, "a node retry with the same key returns the same node"

    # 3. create_brief with a non-ASCII show, retried.
    brief = create_brief(log, brief_data(pid, show=ACCENT), AGENT, idempotency_key="brief-accent")
    again = create_brief(log, brief_data(pid, show=ACCENT), AGENT, idempotency_key="brief-accent")
    assert again.replayed and again.body["node"] == brief.body["node"]

    # 4. submit_plan with a non-ASCII timeline, retried: same plan node and the same content hash.
    plan = dict(plan_fixture("short_clean"))
    plan["timeline"] = f"{plan['timeline']} — {ACCENT}"
    brief_id = brief.body["node"]["id"]
    r1 = submit_plan(log, hub.store, pid, plan, AGENT, derives_from=brief_id, idempotency_key="pa")
    r2 = submit_plan(log, hub.store, pid, plan, AGENT, derives_from=brief_id, idempotency_key="pa")
    assert r2.replayed and r1.body["node"]["id"] == r2.body["node"]["id"]

    # 5. a project with a non-ASCII name folds back.
    reg = Registry(log)
    made = reg.create_project(ACCENT, HUMAN, project_id="prj_accent")
    assert reg.get(made.id) is not None


def test_plan_unchanged_refuses_a_byte_identical_rejected_plan(
    hub: Any, registered_recipe: Any
) -> None:
    """R05: resubmitting the exact rejected plan bytes in a later round is plan_unchanged."""
    recipe_node, fake = start_run(hub)
    worker_idle(hub, fake)
    bytes_v1 = dict(plan_fixture("short_clean"))
    bytes_v1["timeline"] = "Short — version un"
    p1 = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        bytes_v1,
        AGENT,
        derives_from=recipe_node,
        idempotency_key="pv1",
    ).body["node"]["id"]
    step_complete(
        hub.log,
        hub.store,
        recipe_node,
        "p",
        None,
        {"plan_node": p1},
        AGENT,
        idempotency_key="cv1",
        dispatcher=fake,
    )
    review1 = gate_op(hub, recipe_node).review_id
    decide(hub.log, hub.project_id, review1, "reject", "tighter", HUMAN)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    assert load_state(hub.log, recipe_node).current_round() == 2
    # Resubmit the IDENTICAL bytes, derived from the round-1 rejected review.
    same = submit_plan(
        hub.log,
        hub.store,
        hub.project_id,
        bytes_v1,
        AGENT,
        derives_from=review1,
        idempotency_key="pv2",
    ).body["node"]["id"]
    with pytest.raises(WriteError) as exc:
        step_complete(
            hub.log,
            hub.store,
            recipe_node,
            "p",
            None,
            {"plan_node": same},
            AGENT,
            idempotency_key="cv2",
            dispatcher=fake,
        )
    assert exc.value.code == "plan_unchanged"
    assert make_run  # keep the import used
