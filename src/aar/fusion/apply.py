"""Fusion manifest schema and DaVinci Resolve application helpers.

Resolve's FCPXML importer cannot instantiate Fusion ``.setting`` macros directly via
``<effect uid>``. Instead, ``aar plan export`` emits a companion ``.fusion.json`` manifest
(plus optional customized ``.setting`` files) describing every Fusion placement on the timeline.
This module defines that manifest schema and provides shared helpers to apply Fusion macro
inputs or ``.setting`` files to a Resolve ``TimelineItem`` (used by ``aar.node.ops.build`` and
programmatic callers).
"""

from __future__ import annotations

import contextlib
import json
import posixpath
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aar.fusion.errors import FusionError

__all__ = [
    "FusionManifest",
    "FusionPlacement",
    "activate_fusion_comp",
    "apply_fusion_inputs_to_item",
    "find_loaded_setting_comp",
    "find_tool_by_name",
    "iter_comp_tools",
    "load_fusion_manifest",
    "record_loaded_setting",
    "resolve_tool_input_id",
    "values_match",
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
    try:
        return FusionManifest.model_validate(raw)
    except ValidationError as exc:
        raise FusionError("invalid_setting", f"invalid Fusion manifest {path}: {exc}") from exc


def write_fusion_manifest(manifest: FusionManifest, path: Path) -> Path:
    """Write a ``.fusion.json`` manifest with stable JSON formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def iter_comp_tools(comp: Any) -> list[Any]:
    get_tools = getattr(comp, "GetToolList", None)
    if callable(get_tools):
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


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "Name", None)
    if isinstance(name, str) and name:
        return name
    attrs = getattr(tool, "GetAttrs", None)
    if callable(attrs):
        a = attrs()
        if isinstance(a, dict) and isinstance(a.get("TOOLS_Name"), str):
            return a["TOOLS_Name"]
    return ""


def _point_dict_to_seq(actual: Mapping[Any, Any], expected_len: int) -> list[Any] | None:
    """Extract a 1-based Resolve Point table ``{1: x, 2: y[, 3: z]}`` as a sequence."""
    # Accept integer or float 1-based keys (Lua tables in Resolve Python API).
    indexed: dict[int, Any] = {}
    for k, v in actual.items():
        if isinstance(k, bool):
            return None
        if isinstance(k, int):
            indexed[k] = v
        elif isinstance(k, float) and k.is_integer():
            indexed[int(k)] = v
        else:
            return None
    keys = set(indexed)
    wanted = set(range(1, expected_len + 1))
    if keys == wanted or (expected_len == 2 and keys == {1, 2, 3}):
        return [indexed[i] for i in range(1, expected_len + 1)]
    return None


def values_match(actual: Any, expected: Any) -> bool:
    """Compare a Fusion ``GetInput`` read-back value against an expected Python value.

    Handles bool/checkbox semantics (Resolve may return ``0``/``1`` for bools, while a bool
    read-back never matches a numeric expectation), float tolerance, and Resolve's 1-based
    Point table dicts ``{1: x, 2: y[, 3: z]}``.
    """
    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual is expected
        if isinstance(actual, (int, float)):
            return float(actual) == (1.0 if expected else 0.0)
        return False
    if isinstance(actual, bool):
        return False
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-4
    if isinstance(expected, (list, tuple)):
        if isinstance(actual, Mapping):
            seq = _point_dict_to_seq(actual, len(expected))
            if seq is None:
                return False
            actual = seq
        if isinstance(actual, (list, tuple)) and len(actual) == len(expected):
            return all(values_match(a, e) for a, e in zip(actual, expected, strict=True))
        return False
    return bool(actual == expected)


def find_tool_by_name(tools_or_comp: Any, name: str) -> Any | None:
    tools = tools_or_comp if isinstance(tools_or_comp, list) else iter_comp_tools(tools_or_comp)
    for tool in tools:
        if _tool_name(tool) == name:
            return tool
    return None


def resolve_tool_input_id(tool: Any, key: str) -> str:
    """Resolve a published Fusion input display name (``INPS_Name``) to its ``INPS_ID``.

    Returns ``key`` unchanged when it already matches an ``INPS_ID`` or when the tool does not
    expose ``GetInputList()``.
    """
    get_inputs = getattr(tool, "GetInputList", None)
    if not callable(get_inputs):
        return key
    raw = get_inputs()
    items = (
        list(raw.values())
        if isinstance(raw, Mapping)
        else (list(raw) if isinstance(raw, list) else [])
    )
    by_name: str | None = None
    for inp in items:
        inp_id: str | None = None
        inp_name: str | None = None
        attrs_fn = getattr(inp, "GetAttrs", None)
        if callable(attrs_fn):
            attrs = attrs_fn()
            if isinstance(attrs, Mapping):
                raw_id = attrs.get("INPS_ID")
                raw_name = attrs.get("INPS_Name")
                if isinstance(raw_id, str) and raw_id:
                    inp_id = raw_id
                if isinstance(raw_name, str) and raw_name:
                    inp_name = raw_name
        if inp_id is None:
            raw_id = getattr(inp, "ID", None)
            if isinstance(raw_id, str) and raw_id:
                inp_id = raw_id
        if inp_name is None:
            raw_name = getattr(inp, "Name", None)
            if isinstance(raw_name, str) and raw_name:
                inp_name = raw_name
        if inp_id == key:
            return key
        if inp_name == key and inp_id is not None and by_name is None:
            by_name = inp_id
    return by_name if by_name is not None else key


_SETTING_DATA_KEY = "aar.setting"


def _norm_setting_path(path_str: str) -> str:
    """Canonicalize a Windows or POSIX setting path for exact identity comparison."""
    raw = path_str.replace("\\", "/").strip()
    if not raw:
        return ""
    unc = raw.startswith("//")
    body = raw[2:] if unc else raw
    norm = posixpath.normpath(body)
    if unc:
        norm = "//" + norm.lstrip("/")
    return norm.casefold()


def record_loaded_setting(comp: Any, setting_path: str) -> None:
    """Persist the canonical ``setting_path`` identity on a Fusion ``comp``."""
    norm = _norm_setting_path(setting_path)
    if not norm:
        return
    set_data = getattr(comp, "SetData", None)
    if callable(set_data):
        with contextlib.suppress(Exception):
            set_data(_SETTING_DATA_KEY, norm)
    with contextlib.suppress(AttributeError, TypeError):
        comp._aar_setting_path = norm


def _comp_active_setting(comp: Any) -> str | None:
    """Return the active canonical setting path recorded on ``comp``, if any."""
    get_data = getattr(comp, "GetData", None)
    if callable(get_data):
        with contextlib.suppress(Exception):
            val = get_data(_SETTING_DATA_KEY)
            if isinstance(val, str) and val.strip():
                return _norm_setting_path(val)
    recorded_attr = getattr(comp, "_aar_setting_path", None)
    if isinstance(recorded_attr, str) and recorded_attr.strip():
        return _norm_setting_path(recorded_attr)
    single = getattr(comp, "setting_path", None)
    if isinstance(single, str) and single.strip():
        return _norm_setting_path(single)
    many = getattr(comp, "loaded_settings", None)
    if isinstance(many, (list, tuple)) and many:
        for entry in reversed(many):
            if isinstance(entry, str) and entry.strip():
                return _norm_setting_path(entry)
    return None


def _comp_name(comp: Any, listed_name: str | None = None) -> str:
    if isinstance(listed_name, str) and listed_name:
        return listed_name
    name = getattr(comp, "Name", None)
    if isinstance(name, str) and name:
        return name
    attrs_fn = getattr(comp, "GetAttrs", None)
    if callable(attrs_fn):
        attrs = attrs_fn()
        if isinstance(attrs, Mapping) and isinstance(attrs.get("COMPS_Name"), str):
            return attrs["COMPS_Name"]
    return ""


def _listed_comp_names(item: Any) -> dict[int, str]:
    name_list_fn = getattr(item, "GetFusionCompNameList", None)
    listed_names: dict[int, str] = {}
    if callable(name_list_fn):
        raw_names = name_list_fn()
        if isinstance(raw_names, list):
            for idx, nm in enumerate(raw_names, start=1):
                if isinstance(nm, str):
                    listed_names[idx] = nm
        elif isinstance(raw_names, Mapping):
            for k, nm in raw_names.items():
                if isinstance(nm, str) and isinstance(k, int):
                    listed_names[k] = nm
    return listed_names


def activate_fusion_comp(item: Any, comp: Any) -> None:
    """Activate ``comp`` on ``item`` via ``LoadFusionCompByName`` when available and unambiguous."""
    load_by_name = getattr(item, "LoadFusionCompByName", None)
    if not callable(load_by_name):
        return
    listed = _listed_comp_names(item)
    comp_count_fn = getattr(item, "GetFusionCompCount", None)
    comp_count = int(comp_count_fn()) if callable(comp_count_fn) else 0
    cname = ""
    for idx in range(1, comp_count + 1):
        if item.GetFusionCompByIndex(idx) is comp:
            cname = _comp_name(comp, listed.get(idx))
            break
    if not cname:
        cname = _comp_name(comp)
    if not cname:
        return
    if listed and list(listed.values()).count(cname) > 1:
        return
    with contextlib.suppress(Exception):
        load_by_name(cname)


def find_loaded_setting_comp(item: Any, setting_path: str) -> Any | None:
    """Return an existing Fusion comp on ``item`` whose recorded canonical setting path equals
    ``setting_path``, or ``None`` if no comp carries that exact identity.
    """
    comp_count_fn = getattr(item, "GetFusionCompCount", None)
    comp_count = int(comp_count_fn()) if callable(comp_count_fn) else 0
    if comp_count < 1:
        return None
    target_norm = _norm_setting_path(setting_path)
    if not target_norm:
        return None

    for idx in range(comp_count, 0, -1):
        comp = item.GetFusionCompByIndex(idx)
        if comp is None:
            continue
        active = _comp_active_setting(comp)
        if active is not None and active == target_norm:
            return comp
    return None


def _set_and_verify(tool: Any, key: str, value: Any) -> Any:
    """Call ``tool.SetInput(key, value)`` and verify read-back via ``GetInput(key)``.

    Returns the read-back value on success; raises :class:`FusionError` (``apply_failed``) if
    ``SetInput``/``GetInput`` is missing or the read-back is ``None`` or does not match.
    """
    set_input = getattr(tool, "SetInput", None)
    get_input = getattr(tool, "GetInput", None)
    if not callable(set_input) or not callable(get_input):
        raise FusionError("apply_failed", f"tool does not support SetInput/GetInput for {key!r}")
    set_input(key, value)
    after = get_input(key)
    if after is None or not values_match(after, value):
        raise FusionError(
            "apply_failed",
            f"SetInput({key!r}) read back {after!r}, expected {value!r}",
        )
    return after


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
       ``setting_path`` is supplied, targeting the imported composition (and failing if loading
       produces no positive evidence).
    2. Setting named inputs on a ``MacroOperator`` / ``GroupOperator`` (resolving display names via
       ``GetInputList``) or ``Tool.Param`` direct targets, with ``GetInput`` read-back verification.
    3. Applying positional ``lines`` alongside ``inputs`` with explicit ``inputs`` taking
       precedence.

    Returns a summary dict ``{"applied": {...}, "tool_id": ..., "loaded_setting": ...}``.
    Raises :class:`FusionError` (``apply_failed``) if a Fusion comp is missing or read-back fails.
    """
    loaded_setting: str | None = None
    comp: Any = None
    comp_count_fn = getattr(item, "GetFusionCompCount", None)
    comp_count = int(comp_count_fn()) if callable(comp_count_fn) else 0

    if setting_path is not None:
        if not setting_path.is_file():
            raise FusionError("template_not_found", f"Fusion .setting not found: {setting_path}")
        import_comp = getattr(item, "ImportFusionComp", None)
        if callable(import_comp):
            imported = import_comp(str(setting_path))
            if imported is not None and not isinstance(imported, bool):
                comp = imported
                loaded_setting = str(setting_path)
            elif imported is True:
                after_count = int(comp_count_fn()) if callable(comp_count_fn) else comp_count
                if after_count > comp_count:
                    comp = item.GetFusionCompByIndex(after_count)
                    if comp is not None:
                        loaded_setting = str(setting_path)
                comp_count = after_count

        if loaded_setting is None:
            if comp is None and comp_count == 1:
                comp = item.GetFusionCompByIndex(1)
            elif comp is None and comp_count == 0:
                add_comp = getattr(item, "AddFusionComp", None)
                if callable(add_comp):
                    comp = add_comp()
            if comp is not None:
                load_settings = getattr(comp, "LoadSettings", None)
                if callable(load_settings) and load_settings(str(setting_path)):
                    loaded_setting = str(setting_path)

        if loaded_setting is None or comp is None:
            raise FusionError(
                "apply_failed",
                f"could not load Fusion .setting {setting_path} onto timeline item",
            )
        record_loaded_setting(comp, loaded_setting)
        activate_fusion_comp(item, comp)
    else:
        if comp_count < 1:
            raise FusionError("apply_failed", "timeline item has no Fusion composition")
        comp = item.GetFusionCompByIndex(1)

    if comp is None:
        raise FusionError("apply_failed", "timeline item Fusion composition is None")

    tools = iter_comp_tools(comp)
    macros = [t for t in tools if _tool_id(t) in ("MacroOperator", "GroupOperator")]
    text_tools = [t for t in tools if _tool_id(t) == "TextPlus"]

    applied: dict[str, Any] = {}
    target_tool_id: str | None = None
    primary = macros[0] if macros else (text_tools[0] if text_tools else None)
    if primary is not None:
        target_tool_id = _tool_id(primary)

    # Track explicit input targets so explicit `inputs` win over positional `lines`.
    explicit_targets: set[tuple[int, str]] = set()
    resolved_inputs: list[tuple[str, Any, str, Any]] = []
    if inputs:
        for key, value in inputs.items():
            if "." in key:
                tool_name, param = key.split(".", 1)
                sub_tool = find_tool_by_name(tools, tool_name)
                if sub_tool is None:
                    raise FusionError(
                        "apply_failed",
                        f"tool {tool_name!r} not found in Fusion comp for input {key!r}",
                    )
                param_id = resolve_tool_input_id(sub_tool, param)
                resolved_inputs.append((key, sub_tool, param_id, value))
                explicit_targets.add((id(sub_tool), param_id))
            else:
                if primary is None:
                    raise FusionError(
                        "apply_failed",
                        "Fusion comp has neither a MacroOperator/GroupOperator nor a TextPlus tool",
                    )
                param_id = resolve_tool_input_id(primary, key)
                resolved_inputs.append((key, primary, param_id, value))
                explicit_targets.add((id(primary), param_id))

    if lines:
        if text_tools:
            target = text_tools[0]
            if target_tool_id is None:
                target_tool_id = "TextPlus"
            if (id(target), "StyledText") not in explicit_targets:
                joined = "\n".join(lines)
                after = _set_and_verify(target, "StyledText", joined)
                applied["StyledText"] = after
        elif macros:
            target = macros[0]
            if target_tool_id is None:
                target_tool_id = _tool_id(target)
            for idx, line in enumerate(lines, start=1):
                key = f"Input{idx}"
                if (id(target), key) not in explicit_targets:
                    after = _set_and_verify(target, key, line)
                    applied[key] = after
        else:
            raise FusionError(
                "apply_failed",
                "Fusion comp has no TextPlus or MacroOperator tool to receive lines",
            )

    for orig_key, target, param_id, value in resolved_inputs:
        after = _set_and_verify(target, param_id, value)
        applied[orig_key] = after

    return {
        "applied": applied,
        "tool_id": target_tool_id,
        "loaded_setting": loaded_setting,
    }
