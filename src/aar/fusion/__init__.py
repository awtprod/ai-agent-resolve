"""Native DaVinci Resolve Fusion (``.setting`` / ``.drfx``) graphics support.

Parsing, inspection, customization, and manifest handling are stdlib + pydantic only;
RGBA frame and alpha ``.mov`` rendering lazy-imports Pillow (the ``fusion`` optional extra).
"""

from __future__ import annotations

from aar.fusion.apply import (
    FusionManifest,
    FusionPlacement,
    apply_fusion_inputs_to_item,
    load_fusion_manifest,
    write_fusion_manifest,
)
from aar.fusion.customize import apply_inputs_to_ast, customize_setting
from aar.fusion.errors import FusionError
from aar.fusion.expr import ExprContext, Point2D, evaluate_expression, expression_references
from aar.fusion.parse import (
    ENV_TEMPLATES_DIR,
    FusionAssetRef,
    FusionInputSpec,
    FusionTemplate,
    LuaEntry,
    LuaTable,
    dump_setting,
    load_template,
    map_lines_to_inputs,
    parse_setting,
    resolve_asset_path,
    resolve_template_path,
    unwrap_value,
)
from aar.fusion.render import (
    ENV_FONTS_DIR,
    FusionRenderResult,
    render_fusion_clip,
    render_fusion_frame,
)

__all__ = [
    "ENV_FONTS_DIR",
    "ENV_TEMPLATES_DIR",
    "ExprContext",
    "FusionAssetRef",
    "FusionError",
    "FusionInputSpec",
    "FusionManifest",
    "FusionPlacement",
    "FusionRenderResult",
    "FusionTemplate",
    "LuaEntry",
    "LuaTable",
    "Point2D",
    "apply_fusion_inputs_to_item",
    "apply_inputs_to_ast",
    "customize_setting",
    "dump_setting",
    "evaluate_expression",
    "expression_references",
    "load_fusion_manifest",
    "load_template",
    "map_lines_to_inputs",
    "parse_setting",
    "render_fusion_clip",
    "render_fusion_frame",
    "resolve_asset_path",
    "resolve_template_path",
    "unwrap_value",
    "write_fusion_manifest",
]
