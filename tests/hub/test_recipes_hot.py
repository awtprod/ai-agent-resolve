"""Hot steps through the port: submit once, node states show through, one op per attempt."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from recipe_support import hot_fake
from write_support import AGENT, Hub, brief_data, make_hub

from aar.hub.events import Actor
from aar.hub.writes import create_brief, create_recipe
from aar.recipes import registry
from aar.recipes.engine import advance
from aar.recipes.models import RecipeDef, StepDef
from aar.recipes.ports import HotJobView
from aar.recipes.run import recipe_start
from aar.recipes.state import load_state
from aar.recipes.status import status_view


class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


@pytest.fixture
def hottest() -> Iterator[None]:
    registry.clear()
    registry.register_callable("hot.cold", lambda ctx: None)
    registry.register(
        RecipeDef(
            name="hottest",
            version=1,
            params_model=_P,
            steps=(
                StepDef(id="h", kind="hot", hot_kind="x"),
                StepDef(id="c", kind="cold", callable_id="hot.cold", idempotent=True),
                StepDef(
                    id="report",
                    kind="cold",
                    callable_id="recipe.report",
                    idempotent=True,
                    needs=("h", "c"),
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


def _make(hub: Hub) -> str:
    brief_id = create_brief(
        hub.log, brief_data(hub.project_id, recipe="hottest"), AGENT, idempotency_key="b"
    ).body["node"]["id"]
    return create_recipe(
        hub.log,
        hub.project_id,
        {"recipe": "hottest", "brief_id": brief_id, "params": {}},
        AGENT,
        idempotency_key="r",
        params_validator=registry.registry_params_validator,
    ).body["node"]["id"]


def test_hot_submitted_once_and_not_resubmitted(hub: Hub, hottest: None) -> None:
    fake = hot_fake()
    fake.set_behaviour("x", lambda params: HotJobView("waiting_for_node", wait_reason="n"))
    recipe_node = _make(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s", dispatcher=fake)
    op = load_state(hub.log, recipe_node).latest_in_round("h", 1)
    assert op.hot_submitted and fake.calls == [op.op_id]
    # A job ref already recorded is never submitted again without an explicit retry.
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    advance(hub.log, recipe_node, actor=AGENT, dispatcher=fake)
    assert fake.calls == [op.op_id], "the hot op is submitted exactly once"


def test_hot_node_states_show_through(hub: Hub, hottest: None) -> None:
    fake = hot_fake()
    fake.set_behaviour("x", lambda params: HotJobView("node_offline", wait_reason="node_offline"))
    recipe_node = _make(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s", dispatcher=fake)
    view = status_view(hub.log, recipe_node, dispatcher=fake)
    h = next(s for s in view["steps"] if s["step_id"] == "h")
    assert h["state"] == "waiting:node_offline"


def test_advance_twice_creates_one_op_per_attempt(hub: Hub, hottest: None) -> None:
    fake = hot_fake()
    fake.set_behaviour("x", lambda params: HotJobView("waiting_for_node", wait_reason="n"))
    recipe_node = _make(hub)
    recipe_start(hub.log, recipe_node, AGENT, idempotency_key="s", dispatcher=fake)
    # A second advance (even by a different caller) must not create duplicate op nodes.
    advance(hub.log, recipe_node, actor=Actor("agent", "other"), dispatcher=fake)
    state = load_state(hub.log, recipe_node)
    per_step = Counter(op.step_id for op in state.ops)
    assert per_step["h"] == 1 and per_step["c"] == 1
