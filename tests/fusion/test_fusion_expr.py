"""Tests for ``aar.fusion.expr``: Fusion Lua expression evaluator."""

from __future__ import annotations

import math

import pytest

from aar.fusion.errors import FusionError
from aar.fusion.expr import ExprContext, Point2D, evaluate_expression, expression_references


def _ctx(
    time: float = 0.0,
    render_start: float = 0.0,
    render_end: float = 149.0,
    current_tool: str = "Tool1",
    resolve_property=None,
    resolve_output_attr=None,
) -> ExprContext:
    return ExprContext(
        time=time,
        render_start=render_start,
        render_end=render_end,
        current_tool=current_tool,
        resolve_property=resolve_property or (lambda _t, _p: 0.0),
        resolve_output_attr=resolve_output_attr or (lambda _t, _a: None),
    )


def test_arithmetic_precedence_and_right_assoc_power() -> None:
    ctx = _ctx(time=10.0)
    assert evaluate_expression("2 + 3 * 4", ctx) == pytest.approx(14.0)
    assert evaluate_expression("(2 + 3) * 4", ctx) == pytest.approx(20.0)
    assert evaluate_expression("2 ^ 3 ^ 2", ctx) == pytest.approx(512.0)
    assert evaluate_expression("-2 ^ 2", ctx) == pytest.approx(-4.0)
    assert evaluate_expression("10 % 3", ctx) == pytest.approx(1.0)


def test_time_comp_and_builtins() -> None:
    ctx = _ctx(time=15.0, render_start=0.0, render_end=149.0)
    assert evaluate_expression("time / comp.RenderEnd", ctx) == pytest.approx(15.0 / 149.0)
    assert evaluate_expression("comp.CurrentTime", ctx) == pytest.approx(15.0)
    assert evaluate_expression("min(1, max(0, (time - 5) / 10))", ctx) == pytest.approx(1.0)
    assert evaluate_expression("clamp(time, 0, 10)", ctx) == pytest.approx(10.0)
    assert evaluate_expression("iif(time > 10, 100, 200)", ctx) == pytest.approx(100.0)
    assert evaluate_expression("iif(time < 10 or time == 0, 1, 2)", ctx) == pytest.approx(2.0)
    assert evaluate_expression("math.sin(math.pi / 2)", ctx) == pytest.approx(1.0)
    assert evaluate_expression(
        "math.sqrt(9) + math.floor(2.9) + math.abs(-4)", ctx
    ) == pytest.approx(3.0 + 2.0 + 4.0)


def test_point2d_and_datawindow_one_based_indexing() -> None:
    def prop_lookup(tool: str, prop: str) -> object:
        if tool == "SelfTool" and prop == "Width":
            return 0.4
        raise KeyError((tool, prop))

    def out_lookup(tool: str, attr: str) -> object:
        if tool == "NameText" and attr == "DataWindow":
            return [100.0, 200.0, 580.0, 260.0]
        raise KeyError((tool, attr))

    ctx = _ctx(
        time=30.0,
        current_tool="SelfTool",
        resolve_property=prop_lookup,
        resolve_output_attr=out_lookup,
    )
    width_expr = "(NameText.Output.DataWindow[3] - NameText.Output.DataWindow[1]) / 1920"
    assert evaluate_expression(width_expr, ctx) == pytest.approx(480.0 / 1920.0)
    assert evaluate_expression("Width * 2", ctx) == pytest.approx(0.8)

    pt = evaluate_expression("Point(0.25, 0.75)", ctx)
    assert isinstance(pt, Point2D)
    assert pt.x == pytest.approx(0.25)
    assert pt.y == pytest.approx(0.75)
    assert evaluate_expression("Point(0.25, 0.75).X", ctx) == pytest.approx(0.25)
    assert evaluate_expression("Point(0.25, 0.75)[2]", ctx) == pytest.approx(0.75)


def test_expression_references_detects_time_and_nodes() -> None:
    uses_time, refs = expression_references(
        "Point(0.1 + NameText.Output.DataWindow[3] / 1920, iif(time > 10, 0.5, 0.2))"
    )
    assert uses_time is True
    assert "time" in refs
    assert "NameText" in refs


def test_invalid_expression_raises_fusion_error() -> None:
    ctx = _ctx()
    with pytest.raises(FusionError) as exc:
        evaluate_expression("1 + * 2", ctx)
    assert exc.value.code == "expression_error"

    with pytest.raises(FusionError) as exc2:
        evaluate_expression("unknown_fn(1)", ctx)
    assert exc2.value.code == "expression_error"

    # Guard against NaN / Infinity
    assert math.isfinite(float(evaluate_expression("1 / 4", ctx)))


def test_short_circuit_and_lua_truthiness() -> None:
    # Short-circuit prevents division by zero on the untaken branch
    assert evaluate_expression("time ~= 0 and 1 / time", _ctx(time=0.0)) is False
    assert evaluate_expression("time ~= 0 and 1 / time", _ctx(time=2.0)) == pytest.approx(0.5)
    assert evaluate_expression("true or 1 / 0", _ctx()) is True
    assert evaluate_expression("false or 42", _ctx()) == pytest.approx(42.0)

    # Lua truthiness: 0 is truthy in `and`/`or`, while iif() keeps Fusion checkbox semantics
    assert evaluate_expression("0 and 5", _ctx()) == pytest.approx(5.0)
    assert evaluate_expression("0 or 5", _ctx()) == pytest.approx(0.0)
    assert evaluate_expression("iif(0, 10, 20)", _ctx()) == pytest.approx(20.0)
    assert evaluate_expression("iif(1, 10, 20)", _ctx()) == pytest.approx(10.0)


def test_expression_math_errors_and_depth_limit() -> None:
    ctx = _ctx()
    for bad in ("2 ^ 10000", "math.sqrt(-1)", "math.log(0)", "math.exp(10000)", "1 / 0"):
        with pytest.raises(FusionError) as exc:
            evaluate_expression(bad, ctx)
        assert exc.value.code == "expression_error", f"expected expression_error for {bad!r}"

    deep = "(" * 200 + "1" + ")" * 200
    with pytest.raises(FusionError) as exc_depth:
        evaluate_expression(deep, ctx)
    assert exc_depth.value.code == "expression_error"
