"""RecipeDef bounds (validator's own pins)."""

from typing import Any

import pytest
from pydantic import BaseModel

from aar.recipes import registry
from aar.recipes.models import RecipeDef, RecipeDefError, StepDef


class P(BaseModel):
    pass


def _cold(i: int, **kw: Any) -> StepDef:
    return StepDef(id=kw.pop("id", f"s{i}"), kind="cold", callable_id="recipe.report", **kw)


def _reg(steps: tuple) -> Any:
    registry.clear()
    try:
        return registry.register(RecipeDef(name="vb", version=1, params_model=P, steps=steps))
    finally:
        registry.clear()


def test_step_count_bound() -> None:
    _reg(tuple(_cold(i) for i in range(64)))
    with pytest.raises(RecipeDefError):
        _reg(tuple(_cold(i) for i in range(65)))


@pytest.mark.parametrize(
    "bad",
    [
        {"max_attempts": 6},
        {"max_attempts": 0},
        {"timeout_s": 86400.001},
        {"timeout_s": 0},
        {"timeout_s": float("inf")},
        {"id": "a" * 33},
        {"id": "A"},
    ],
)
def test_step_field_bounds(bad: dict) -> None:
    with pytest.raises(RecipeDefError):
        _reg((_cold(0, **bad),))


@pytest.mark.parametrize(
    "ok", [{"max_attempts": 5}, {"max_attempts": 1}, {"timeout_s": 86400}, {"id": "a" * 32}]
)
def test_step_field_bounds_ok(ok: dict) -> None:
    _reg((_cold(0, **ok),))


def test_dag_and_unique_ids() -> None:
    with pytest.raises(RecipeDefError):
        _reg((_cold(0, needs=("s1",)), _cold(1)))
    with pytest.raises(RecipeDefError):
        _reg((_cold(0, id="a"), _cold(1, id="a")))
