"""Lease claims and reaping: never steal a live lease; re-claim only idempotent lapsed ops."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from recipe_support import FixedClock
from write_support import AGENT, Hub, brief_data, make_hub

from aar.hub.events import Actor
from aar.hub.writes import create_brief, create_recipe, set_own_status
from aar.recipes import registry
from aar.recipes.engine import DEFAULT_LEASE_S, advance, claim
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.state import load_state

W1 = Actor("agent", "w1")
W2 = Actor("agent", "w2")


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


@pytest.fixture
def lease_recipes() -> Iterator[None]:
    registry.clear()
    registry.register_callable("lease.noop", lambda ctx: None)
    for name, idem in (("leasetest", True), ("leasetest_ni", False)):
        registry.register(
            RecipeDef(
                name=name,
                version=1,
                params_model=_P,
                steps=(
                    StepDef(
                        id="c",
                        kind="cold",
                        callable_id="lease.noop",
                        idempotent=idem,
                        max_attempts=1,
                    ),
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
    yield
    registry.clear()


@pytest.fixture
def hub(tmp_path: Any) -> Iterator[Hub]:
    made = make_hub(tmp_path)
    yield made
    made.log.close()


def _started(hub: Hub, recipe_name: str, clock: FixedClock) -> str:
    brief_id = create_brief(
        hub.log, brief_data(hub.project_id, recipe=recipe_name), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    recipe_node = create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": recipe_name, "brief_id": brief_id, "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]
    set_own_status(hub.log, hub.project_id, recipe_node, "active", AGENT, idempotency_key="act")
    advance(hub.log, recipe_node, actor=AGENT, clock=clock)  # creates the pending cold op "c"
    return recipe_node


def _cold_op(hub: Hub, recipe_node: str) -> Any:
    return load_state(hub.log, recipe_node).latest_in_round("c", 1)


def test_pending_is_claimed_then_a_live_lease_is_not_stolen(hub: Hub, lease_recipes: None) -> None:
    clock = FixedClock()
    recipe_node = _started(hub, "leasetest", clock)
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W1,
            idempotent=True,
            clock=clock,
            lease_s=DEFAULT_LEASE_S,
        )
        is True
    )
    # Still well within the 60s lease: a second worker must NOT steal it, idempotent or not.
    clock.tick(10)
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W2,
            idempotent=True,
            clock=clock,
            lease_s=DEFAULT_LEASE_S,
        )
        is False
    )


def test_lapsed_idempotent_is_reclaimed(hub: Hub, lease_recipes: None) -> None:
    clock = FixedClock()
    recipe_node = _started(hub, "leasetest", clock)
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W1,
            idempotent=True,
            clock=clock,
            lease_s=1,
        )
        is True
    )
    clock.tick(5)  # lease lapsed
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W2,
            idempotent=True,
            clock=clock,
            lease_s=1,
        )
        is True
    )  # stolen because lapsed + idempotent


def test_lapsed_nonidempotent_is_not_reclaimed(hub: Hub, lease_recipes: None) -> None:
    clock = FixedClock()
    recipe_node = _started(hub, "leasetest_ni", clock)
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W1,
            idempotent=False,
            clock=clock,
            lease_s=1,
        )
        is True
    )
    clock.tick(5)
    op = _cold_op(hub, recipe_node)
    assert (
        claim(
            hub.log,
            project_id=hub.project_id,
            op=op,
            worker=W2,
            idempotent=False,
            clock=clock,
            lease_s=1,
        )
        is False
    )


def test_reap_fails_nonidempotent_lapsed_lease_lost(hub: Hub, lease_recipes: None) -> None:
    clock = FixedClock()
    recipe_node = _started(hub, "leasetest_ni", clock)
    op = _cold_op(hub, recipe_node)
    claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=W1,
        idempotent=False,
        clock=clock,
        lease_s=1,
    )
    clock.tick(5)
    advance(hub.log, recipe_node, actor=AGENT, clock=clock)  # reaps the lapsed non-idempotent claim
    op = _cold_op(hub, recipe_node)
    assert op.status == "failed"
    assert op.result_error["code"] == "lease_lost"
    assert load_state(hub.log, recipe_node).recipe_status == "blocked"


def test_reap_leaves_idempotent_lapsed_claimable(hub: Hub, lease_recipes: None) -> None:
    clock = FixedClock()
    recipe_node = _started(hub, "leasetest", clock)
    op = _cold_op(hub, recipe_node)
    claim(
        hub.log,
        project_id=hub.project_id,
        op=op,
        worker=W1,
        idempotent=True,
        clock=clock,
        lease_s=1,
    )
    clock.tick(5)
    advance(hub.log, recipe_node, actor=AGENT, clock=clock)
    op = _cold_op(hub, recipe_node)
    assert op.status == "active" and not op.is_cancelled  # left active, re-claimable
