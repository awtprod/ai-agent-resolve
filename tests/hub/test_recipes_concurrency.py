"""Concurrency on one SQLite file: one op per attempt, each step run exactly once."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
import recipe_proc
from write_support import AGENT, Hub, make_hub

from aar.hub.graph import WorkGraph
from aar.hub.writes import set_own_status
from aar.recipes.state import load_state


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _op_nodes(hub: Hub, recipe_node: str) -> list[Any]:
    graph = WorkGraph(hub.log, hub.project_id)
    return [
        n
        for n in graph.nodes()
        if n.kind == "op"
        and isinstance(n.payload, dict)
        and n.payload.get("recipe_node") == recipe_node
    ]


def test_two_concurrent_advance_callers_one_op_per_attempt(hub: Hub, tmp_path: Any) -> None:
    moddir, driver = recipe_proc.materialize(tmp_path)
    recipe_node = recipe_proc.make_cold_recipe(hub, "parallel")
    # Activate WITHOUT advancing, so the two concurrent advance passes race to create the ops.
    set_own_status(hub.log, hub.project_id, recipe_node, "active", AGENT, idempotency_key="act")
    hub.log.close()  # release this process's handle; the subprocesses open their own

    procs = [
        recipe_proc.spawn(
            driver,
            moddir,
            "advance",
            tmp_path / "hub.db",
            hub.root / "artifacts",
            f"agent:racer-{i}",
            recipe_node,
        )
        for i in range(2)
    ]
    for proc in procs:
        assert proc.wait(timeout=60) == 0

    from aar.hub.events import EventLog

    hub.log = EventLog(tmp_path / "hub.db")
    ops = _op_nodes(hub, recipe_node)
    # Five independent cold steps become ready at once; each must have exactly one op.
    per_step = Counter(n.payload["step_id"] for n in ops)
    assert set(per_step) == {f"s{i}" for i in range(5)}
    assert all(count == 1 for count in per_step.values()), per_step


def test_two_worker_processes_each_step_runs_once(hub: Hub, tmp_path: Any) -> None:
    moddir, driver = recipe_proc.materialize(tmp_path)
    recipe_node = recipe_proc.make_cold_recipe(hub, "parallel")
    set_own_status(hub.log, hub.project_id, recipe_node, "active", AGENT, idempotency_key="act")
    hub.log.close()

    sidefile = tmp_path / "ran.log"
    procs = [
        recipe_proc.spawn(
            driver,
            moddir,
            "worker",
            tmp_path / "hub.db",
            hub.root / "artifacts",
            f"agent:worker-{i}",
            recipe_node,
            AAR_SIDEFILE=str(sidefile),
            AAR_POLL="0.05",
        )
        for i in range(2)
    ]
    from aar.hub.events import EventLog

    def done() -> bool:
        probe = EventLog(tmp_path / "hub.db", read_only=True)
        try:
            return load_state(probe, recipe_node).recipe_status == "done"
        finally:
            probe.close()

    assert recipe_proc.wait_for(done, timeout=60), "the recipe should finish"
    for proc in procs:
        proc.terminate()
        proc.wait(timeout=30)

    # Each cold step ran exactly once (the sidefile has no duplicates).
    ran = Counter(sidefile.read_text().split())
    for i in range(5):
        assert ran[f"s{i}"] == 1, ran

    hub.log = EventLog(tmp_path / "hub.db")
    state = load_state(hub.log, recipe_node)
    assert state.recipe_status == "done"
    for i in range(5):
        assert state.is_step_done(f"s{i}")
