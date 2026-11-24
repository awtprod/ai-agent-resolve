"""Customize a Fusion ``.setting`` template by overriding published inputs or asset paths."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aar.fusion.errors import FusionError
from aar.fusion.parse import (
    FusionTemplate,
    LuaEntry,
    LuaTable,
    dump_setting,
    map_lines_to_inputs,
)

__all__ = ["apply_inputs_to_ast", "customize_setting"]


def _to_lua_value(value: Any, existing: Any) -> Any:
    """Convert a Python override into the appropriate Lua AST node matching ``existing``."""
    if isinstance(value, bool):
        return (
            int(value)
            if isinstance(existing, (int, float)) and not isinstance(existing, bool)
            else value
        )
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        if isinstance(existing, LuaTable) and existing.constructor == "FuID":
            return LuaTable(constructor="FuID", entries=[LuaEntry(key=None, value=value)])
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        x, y = value
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
        ):
            raise FusionError(
                "invalid_input_value",
                f"point input must contain two numbers, got {value!r}",
            )
        return LuaTable(entries=[LuaEntry(key=None, value=x), LuaEntry(key=None, value=y)])
    raise FusionError(
        "invalid_input_value",
        f"unsupported Fusion input value {value!r} ({type(value).__name__})",
    )


def _set_tool_input(tool: LuaTable, source: str, value: Any) -> None:
    inputs_tbl = tool.get("Inputs")
    if not isinstance(inputs_tbl, LuaTable):
        inputs_tbl = LuaTable()
        tool.set("Inputs", inputs_tbl)

    inp = inputs_tbl.get(source)
    if isinstance(inp, LuaTable):
        existing = inp.get("Value")
        # Constant override replaces any competing expression or spline connection on this input.
        inp.remove("Expression")
        inp.remove("SourceOp")
        inp.remove("Source")
        inp.set("Value", _to_lua_value(value, existing))
    else:
        inputs_tbl.set(
            source,
            LuaTable(
                constructor="Input",
                entries=[LuaEntry(key="Value", value=_to_lua_value(value, None))],
            ),
        )


def _find_tools_in_ast(ast: LuaTable, template: FusionTemplate) -> dict[str, LuaTable]:
    top_tools = ast.get("Tools")
    if not isinstance(top_tools, LuaTable):
        return {}
    if template.operator_type in ("MacroOperator", "GroupOperator"):
        macro = top_tools.get(template.name)
        if isinstance(macro, LuaTable):
            inner = macro.get("Tools")
            if isinstance(inner, LuaTable):
                return {str(k): v for k, v in inner.items_dict().items() if isinstance(v, LuaTable)}
    return {str(k): v for k, v in top_tools.items_dict().items() if isinstance(v, LuaTable)}


def apply_inputs_to_ast(
    template: FusionTemplate,
    *,
    inputs: Mapping[str, Any] | None = None,
    lines: Sequence[str] = (),
    asset_map: Mapping[str, str] | None = None,
) -> LuaTable:
    """Return a deep copy of ``template.ast`` with ``inputs``/``lines`` and ``asset_map``
    applied."""
    resolved = map_lines_to_inputs(template, lines=lines, inputs=inputs)
    ast_copy = template.ast.copy()
    tools = _find_tools_in_ast(ast_copy, template)

    spec_by_name = {spec.name: spec for spec in template.inputs}

    for key, val in resolved.items():
        spec = spec_by_name.get(key)
        if spec is not None:
            tool = tools.get(spec.source_op)
            if tool is None:
                raise FusionError(
                    "invalid_setting",
                    f"input {spec.name!r} targets missing tool {spec.source_op!r}",
                )
            _set_tool_input(tool, spec.source, val)
        elif "." in key:
            tool_name, param = key.split(".", 1)
            tool = tools.get(tool_name)
            if tool is None:
                raise FusionError(
                    "unknown_input",
                    f"unknown tool {tool_name!r} in input override {key!r}",
                )
            _set_tool_input(tool, param, val)
        else:
            raise FusionError(
                "unknown_input",
                f"unknown input {key!r} for Fusion template {template.name!r}",
            )

    if asset_map:
        for tool_name, tool in tools.items():
            if tool.constructor != "Loader":
                continue
            clips = tool.get("Clips")
            if not isinstance(clips, LuaTable):
                continue
            for entry in clips.entries:
                clip = entry.value
                if not isinstance(clip, LuaTable):
                    continue
                fn = clip.get("Filename")
                if not isinstance(fn, str):
                    continue
                replacement = asset_map.get(tool_name) or asset_map.get(fn)
                if replacement is not None:
                    clip.set("Filename", replacement)

    return ast_copy


def customize_setting(
    template: FusionTemplate,
    *,
    inputs: Mapping[str, Any] | None = None,
    lines: Sequence[str] = (),
    asset_map: Mapping[str, str] | None = None,
) -> str:
    """Return customized ``.setting`` Lua text with inputs and asset overrides applied."""
    ast_copy = apply_inputs_to_ast(
        template,
        inputs=inputs,
        lines=lines,
        asset_map=asset_map,
    )
    return dump_setting(ast_copy)
