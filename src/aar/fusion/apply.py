"""Fusion manifest schema and DaVinci Resolve application helpers.

Resolve's FCPXML importer cannot instantiate Fusion ``.setting`` macros directly via
``<effect uid>``. Instead, ``aar plan export`` emits a companion ``.fusion.json`` manifest
(plus optional customized ``.setting`` files) describing every Fusion placement on the timeline.
This module defines that manifest schema and provides shared helpers to apply Fusion macro
inputs or ``.setting`` files to a Resolve ``TimelineItem`` (used both by ``aar.node.ops.build``
and ``aar fusion apply``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aar.fusion.errors import FusionError

__all__ = [
    "FusionManifest",
    "FusionPlacement",
    "apply_fusion_inputs_to_item",
    "load_fusion_manifest",
    "write_fusion_manifest",
]


class FusionPlacement(BaseModel):
    """One Fusion title or graphic placement on a timeline."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    kind: str
    template: str | None = None
    setting: str | None = None
    customized_setting: str | None = None
    track: int = Field(ge=1)
    record_in_s: float = Field(ge=0.0)
    record_out_s: float = Field(gt=0.0)
    duration_s: float = Field(gt=0.0)
    lines: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)


class FusionManifest(BaseModel):
    """Companion manifest written alongside FCPXML exports for Fusion graphics."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    timeline_name: str
    fps: float = Field(gt=0.0)
    resolution: tuple[int, int]
    fcpxml: str | None = None
    placements: list[FusionPlacement] = Field(default_factory=list)


def load_fusion_manifest(path: Path) -> FusionManifest:
    """Read and validate a ``.fusion.json`` manifest."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FusionError("invalid_setting", f"cannot read Fusion manifest {path}: {exc}") from exc
    return FusionManifest.model_validate(raw)


def write_fusion_manifest(manifest: FusionManifest, path: Path) -> Path:
    """Write a ``.fusion.json`` manifest with stable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _iter_comp_tools(comp: Any) -> list[Any]:
    get_tools = getattr(comp, "GetToolList", None)
    if get_tools is None:
        return []
    raw = get_tools(False)
    if isinstance(raw, dict):
        return list(raw.values())
    if isinstance(raw, list):
        return raw
    return []


def _tool_id(tool: Any) -> str:
    attrs = getattr(tool, "GetAttrs", None)
    if callable(attrs):
        a = attrs()
        if isinstance(a, dict):
            return str(a.get("TOOLS_RegID") or a.get("TOOLB_RegID") or "")
    return str(getattr(tool, "ID", "") or "")


def _values_match(actual: Any, expected: Any) -> bool:
    if actual == expected:
        return True
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-4
    if (
        isinstance(actual, (list, tuple))
        and isinstance(expected, (list, tuple))
        and len(actual) == len(expected)
    ):
        return all(_values_match(a, e) for a, e in zip(actual, expected, strict=True))
    return False


def _set_and_verify(tool: Any, key: str, value: Any) -> bool:
    """Call ``tool.SetInput(key, value)`` and verify read-back via ``GetInput(key)``."""
    set_input = getattr(tool, "SetInput", None)
    if not callable(set_input):
        return False
    set_input(key, value)
    get_input = getattr(tool, "GetInput", None)
    if callable(get_input):
        after = get_input(key)
        if after is not None and not _values_match(after, value):
            return False
    return True


def _find_tool_by_name(tools: list[Any], name: str) -> Any | None:
    for tool in tools:
        tname = getattr(tool, "Name", None)
        if tname == name:
            return tool
        attrs = getattr(tool, "GetAttrs", None)
        if callable(attrs):
            a = attrs()
            if isinstance(a, dict) and a.get("TOOLS_Name") == name:
                return tool
    return None


def apply_fusion_inputs_to_item(
    item: Any,
    *,
    inputs: Mapping[str, Any] | None = None,
    lines: Sequence[str] = (),
    setting_path: Path | None = None,
) -> dict[str, Any]:
    """Apply Fusion macro inputs (and/or a ``.setting`` file) to a Resolve ``TimelineItem``.

    Supports:
    1. Importing a ``.setting`` file via ``item.ImportFusionComp`` / ``comp.LoadSettings`` when
       ``setting_path`` is supplied and the item does not already contain the requested macro.
    2. Setting named inputs on a ``MacroOperator`` / ``GroupOperator`` (or ``Tool.Param`` direct
       targets), with ``GetInput`` read-back verification.
    3. Falling back to ``TextPlus`` ``StyledText`` when setting plain ``lines`` on a standard
       Text+ template.

    Returns a summary dict ``{"applied": {...}, "tool_id": ..., "loaded_setting": ...}``.
    Raises :class:`FusionError` (``apply_failed``) if a Fusion comp is missing or read-back fails.
    """
    loaded_setting: str | None = None
    comp_count_fn = getattr(item, "GetFusionCompCount", None)
    comp_count = int(comp_count_fn()) if callable(comp_count_fn) else 0

    if setting_path is not None:
        if not setting_path.is_file():
            raise FusionError("template_not_found", f"Fusion .setting not found: {setting_path}")
        import_comp = getattr(item, "ImportFusionComp", None)
        if callable(import_comp):
            comp = import_comp(str(setting_path))
            if comp is not None:
                loaded_setting = str(setting_path)
                comp_count = max(comp_count, 1)

    if comp_count < 1:
        add_comp = getattr(item, "AddFusionComp", None)
        if setting_path is not None and callable(add_comp):
            comp = add_comp()
            if comp is not None:
                load_settings = getattr(comp, "LoadSettings", None)
                if callable(load_settings):
                    load_settings(str(setting_path))
                    loaded_setting = str(setting_path)
                    comp_count = 1

    if comp_count < 1:
        raise FusionError("apply_failed", "timeline item has no Fusion composition")

    comp = item.GetFusionCompByIndex(1)
    if comp is None:
        raise FusionError("apply_failed", "GetFusionCompByIndex(1) returned None")

    tools = _iter_comp_tools(comp)
    macros = [t for t in tools if _tool_id(t) in ("MacroOperator", "GroupOperator")]
    text_tools = [t for t in tools if _tool_id(t) == "TextPlus"]

    applied: dict[str, Any] = {}
    target_tool_id: str | None = None

    if inputs:
        primary = macros[0] if macros else (text_tools[0] if text_tools else None)
        if primary is None:
            raise FusionError(
                "apply_failed",
                "Fusion comp has neither a MacroOperator/GroupOperator nor a TextPlus tool",
            )
        target_tool_id = _tool_id(primary)
        for key, value in inputs.items():
            if "." in key:
                tool_name, param = key.split(".", 1)
                sub_tool = _find_tool_by_name(tools, tool_name)
                if sub_tool is None:
                    raise FusionError(
                        "apply_failed",
                        f"tool {tool_name!r} not found in Fusion comp for input {key!r}",
                    )
                if not _set_and_verify(sub_tool, param, value):
                    raise FusionError(
                        "apply_failed",
                        f"SetInput({param!r}) on {tool_name!r} did not take effect",
                    )
                applied[key] = value
            else:
                if not _set_and_verify(primary, key, value):
                    raise FusionError(
                        "apply_failed",
                        f"SetInput({key!r}) on {target_tool_id or 'tool'} did not take effect",
                    )
                applied[key] = value

    elif lines:
        if text_tools:
            target = text_tools[0]
            target_tool_id = "TextPlus"
            joined = "\n".join(lines)
            if not _set_and_verify(target, "StyledText", joined):
                raise FusionError("apply_failed", "TextPlus StyledText did not take effect")
            applied["StyledText"] = joined
        elif macros:
            target = macros[0]
            target_tool_id = _tool_id(target)
            for idx, line in enumerate(lines, start=1):
                key = f"Input{idx}"
                if not _set_and_verify(target, key, line):
                    raise FusionError(
                        "apply_failed",
                        f"SetInput({key!r}) on {target_tool_id} did not take effect",
                    )
                applied[key] = line
        else:
            raise FusionError(
                "apply_failed",
                "Fusion comp has no TextPlus or MacroOperator tool to receive lines",
            )

    return {
        "applied": applied,
        "tool_id": target_tool_id,
        "loaded_setting": loaded_setting,
    }
