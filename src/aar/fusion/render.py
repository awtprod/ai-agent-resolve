"""Native RGBA renderer for DaVinci Resolve Fusion ``.setting`` and ``.drfx`` templates.

Evaluates a Fusion node graph (``Background``, ``RectangleMask``, ``EllipseMask``, ``TextPlus``,
``Loader``, ``Merge``, ``Transform``, ``Dissolve``, ``BezierSpline``, and Lua expressions) into
RGBA frames using Pillow (lazy-imported via the ``fusion`` optional extra) and optionally encodes
them to an alpha-preserving ``.mov`` via ``ffmpeg``.
"""

from __future__ import annotations

import contextlib
import functools
import io
import math
import os
import secrets
import shutil
import subprocess
import threading
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.fusion.customize import apply_inputs_to_ast
from aar.fusion.errors import FusionError
from aar.fusion.expr import ExprContext, Point2D, evaluate_expression, expression_references
from aar.fusion.parse import (
    MAX_ASSET_BYTES,
    FusionTemplate,
    LuaTable,
    load_template,
    read_zip_member_bounded,
    resolve_asset_path,
    unwrap_value,
)

__all__ = [
    "ENV_FONTS_DIR",
    "MAX_CANVAS_DIM",
    "MAX_CANVAS_PIXELS",
    "FusionRenderResult",
    "render_fusion_clip",
    "render_fusion_frame",
]

ENV_FONTS_DIR = "AAR_FONTS_DIR"
MAX_CANVAS_DIM = 8192
MAX_CANVAS_PIXELS = 40_000_000  # ~8K UHD; prevents OOM from hostile template sizes/scales
_MAX_STDERR_BYTES = 65536

_CODEC_ARGS: dict[str, list[str]] = {
    "qtrle": ["-c:v", "qtrle", "-pix_fmt", "argb"],
    "prores4444": ["-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le"],
    "png": ["-c:v", "png", "-pix_fmt", "rgba"],
}


def _check_image_size(width: int, height: int, context: str) -> tuple[int, int]:
    """Validate an image allocation against dimension and total pixel budgets."""
    if width < 1 or height < 1:
        raise FusionError(
            "invalid_setting",
            f"{context}: invalid dimensions {width}x{height} (must be positive)",
        )
    if width > MAX_CANVAS_DIM or height > MAX_CANVAS_DIM:
        raise FusionError(
            "invalid_setting",
            f"{context}: dimensions {width}x{height} exceed maximum side length {MAX_CANVAS_DIM}",
        )
    if width * height > MAX_CANVAS_PIXELS:
        raise FusionError(
            "invalid_setting",
            f"{context}: pixel count {width * height} ({width}x{height}) exceeds budget "
            f"of {MAX_CANVAS_PIXELS} pixels",
        )
    return width, height


def _require_pil() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
    except ImportError as exc:
        raise FusionError(
            "missing_extra",
            "Pillow is required to render Fusion graphics; install with: "
            "pip install 'ai-agent-resolve[fusion]'",
        ) from exc
    return Image, ImageChops, ImageDraw, ImageFilter, ImageFont


# ---------------------------------------------------------------------------
# Font discovery
# ---------------------------------------------------------------------------


def _font_search_dirs(extra_dirs: Sequence[Path]) -> list[Path]:
    dirs: list[Path] = []
    env_raw = os.environ.get(ENV_FONTS_DIR, "").strip()
    if env_raw:
        dirs.extend(Path(p) for p in env_raw.split(os.pathsep) if p.strip())
    dirs.extend(extra_dirs)
    dirs.extend(
        [
            Path.home() / ".local" / "share" / "fonts",
            Path.home() / ".fonts",
            Path("/usr/local/share/fonts"),
            Path("/usr/share/fonts"),
            Path("/Library/Fonts"),
            Path("/System/Library/Fonts"),
            Path("C:/Windows/Fonts"),
        ]
    )
    out: list[Path] = []
    seen: set[Path] = set()
    for d in dirs:
        if d.is_dir():
            r = d.resolve()
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def _norm_font_token(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def _find_font_file(
    font_family: str,
    style: str,
    extra_dirs: Sequence[Path],
) -> Path | None:
    fam_norm = _norm_font_token(font_family)
    style_norm = _norm_font_token(style) or "regular"
    want_bold = any(w in style_norm for w in ("bold", "extrabold", "semibold", "black", "heavy"))
    want_italic = any(w in style_norm for w in ("italic", "oblique"))

    # 1. Scan extra_dirs + system font dirs
    best_path: Path | None = None
    best_score = -1

    for base in _font_search_dirs(extra_dirs):
        for ext in ("*.ttf", "*.otf", "*.ttc"):
            for fp in sorted(base.rglob(ext)):
                stem_norm = _norm_font_token(fp.stem)
                if fam_norm and fam_norm not in stem_norm:
                    continue
                score = 10
                if style_norm in stem_norm:
                    score += 20
                has_bold = any(w in stem_norm for w in ("bold", "heavy", "black"))
                has_italic = any(w in stem_norm for w in ("italic", "oblique"))
                if has_bold == want_bold:
                    score += 5
                if has_italic == want_italic:
                    score += 5
                if score > best_score:
                    best_score = score
                    best_path = fp

    if best_path is not None:
        return best_path

    # 2. Try fontconfig's fc-match if present (handles aliases & system installs cleanly)
    fc_match = shutil.which("fc-match")
    if fc_match:
        pattern = f"{font_family}:style={style}" if style else font_family
        try:
            proc = subprocess.run(
                [fc_match, "-f", "%{file}", pattern],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                cand = Path(proc.stdout.strip())
                if cand.is_file():
                    return cand
        except OSError:
            pass

    # 3. Fallback to DejaVu / Liberation sans matching weight
    fallback_names = (
        ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "NotoSans-Bold.ttf"]
        if want_bold
        else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "NotoSans-Regular.ttf"]
    )
    for base in _font_search_dirs(extra_dirs):
        for fname in fallback_names:
            for fp in base.rglob(fname):
                return fp

    return None


@functools.lru_cache(maxsize=64)
def _cached_font_path(font_family: str, style: str, extra_dirs_key: tuple[str, ...]) -> str | None:
    found = _find_font_file(font_family, style, [Path(p) for p in extra_dirs_key])
    return str(found) if found else None


def _load_pil_font(
    font_family: str,
    style: str,
    size_px: int,
    extra_dirs: Sequence[Path],
) -> Any:
    _, _, _, _, ImageFont = _require_pil()
    key = tuple(str(p.resolve()) for p in extra_dirs if p.is_dir())
    path = _cached_font_path(font_family, style, key)
    if path is not None:
        try:
            return ImageFont.truetype(path, max(1, size_px))
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=max(1, size_px))
    except TypeError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Keyframe / BezierSpline evaluation
# ---------------------------------------------------------------------------


def _eval_bezier_spline(tool: LuaTable, time: float) -> float:
    kf_tbl = tool.get("KeyFrames")
    if not isinstance(kf_tbl, LuaTable):
        return 0.0
    points: list[tuple[float, float, tuple[float, float] | None, tuple[float, float] | None]] = []
    for entry in kf_tbl.entries:
        if not isinstance(entry.key, (int, float)) or not isinstance(entry.value, LuaTable):
            continue
        t = float(entry.key)
        pos = entry.value.positional()
        val = float(pos[0]) if pos and isinstance(pos[0], (int, float)) else 0.0
        lh_raw = unwrap_value(entry.value.get("LH"))
        rh_raw = unwrap_value(entry.value.get("RH"))
        lh = (
            (float(lh_raw[0]), float(lh_raw[1]))
            if isinstance(lh_raw, list) and len(lh_raw) == 2
            else None
        )
        rh = (
            (float(rh_raw[0]), float(rh_raw[1]))
            if isinstance(rh_raw, list) and len(rh_raw) == 2
            else None
        )
        points.append((t, val, lh, rh))

    if not points:
        return 0.0
    points.sort(key=lambda p: p[0])
    if time <= points[0][0]:
        return points[0][1]
    if time >= points[-1][0]:
        return points[-1][1]

    for i in range(len(points) - 1):
        t0, v0, _, rh0 = points[i]
        t1, v1, lh1, _ = points[i + 1]
        if t0 <= time <= t1:
            if t1 == t0:
                return v1
            u = (time - t0) / (t1 - t0)
            if rh0 is not None and lh1 is not None:
                # Cubic Bezier on value axis using normalized parameter u
                p0 = v0
                p1 = rh0[1]
                p2 = lh1[1]
                p3 = v1
                one_u = 1.0 - u
                return (
                    (one_u**3) * p0
                    + 3.0 * (one_u**2) * u * p1
                    + 3.0 * one_u * (u**2) * p2
                    + (u**3) * p3
                )
            return v0 + (v1 - v0) * u
    return points[-1][1]


# ---------------------------------------------------------------------------
# Node graph evaluator
# ---------------------------------------------------------------------------


def _detect_native_size(tools: dict[str, LuaTable], fallback: tuple[int, int]) -> tuple[int, int]:
    """Infer the template's native design resolution from Background tools, if specified."""
    for tool_name, tool in tools.items():
        if tool.constructor != "Background":
            continue
        inputs = tool.get("Inputs")
        if not isinstance(inputs, LuaTable):
            continue
        use_fmt = inputs.get("UseFrameFormatSettings")
        if isinstance(use_fmt, LuaTable) and unwrap_value(use_fmt.get("Value")) == 1:
            continue
        w_inp = inputs.get("Width")
        h_inp = inputs.get("Height")
        w_val = unwrap_value(w_inp.get("Value")) if isinstance(w_inp, LuaTable) else None
        h_val = unwrap_value(h_inp.get("Value")) if isinstance(h_inp, LuaTable) else None
        if isinstance(w_val, (int, float)) and isinstance(h_val, (int, float)):
            if not (math.isfinite(float(w_val)) and math.isfinite(float(h_val))):
                raise FusionError(
                    "invalid_setting",
                    f"Background {tool_name!r} has non-finite Width/Height ({w_val!r}, {h_val!r})",
                )
            w_i, h_i = round(w_val), round(h_val)
            if w_i > 0 and h_i > 0:
                return _check_image_size(w_i, h_i, f"Background {tool_name!r}")
    return fallback


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _to_u8(v: float) -> int:
    return round(_clamp01(v) * 255.0)


def _to_point(val: Any, default: tuple[float, float] = (0.5, 0.5)) -> tuple[float, float]:
    if isinstance(val, Point2D):
        return (val.x, val.y)
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (float(val[0]), float(val[1]))
    return default


class _FusionGraphEvaluator:
    """Evaluates a customized Fusion tool graph across frames with time-invariance caching."""

    def __init__(
        self,
        *,
        template: FusionTemplate,
        tools: dict[str, LuaTable],
        output_tool: str,
        comp_size: tuple[int, int],
        render_start: float,
        render_end: float,
        asset_dirs: Sequence[Path],
        confined_roots: Sequence[Path] | None = None,
    ) -> None:
        _check_image_size(comp_size[0], comp_size[1], f"template {template.name!r} composition")
        self._template = template
        self._tools = tools
        self._output_tool = output_tool
        self._comp_w, self._comp_h = comp_size
        self._render_start = render_start
        self._render_end = render_end
        self._asset_dirs = list(asset_dirs)
        self._confined_roots = list(confined_roots) if confined_roots is not None else None
        self._archive_members: list[str] = []
        if template.path.suffix.lower() == ".drfx":
            with zipfile.ZipFile(template.path, "r") as zf:
                self._archive_members = zf.namelist()

        self._loader_cache: dict[str, Any] = {}
        self._static_cache: dict[str, Any] = {}
        self._time_varying: dict[str, bool] = {}
        self._compute_time_varying()

    def _compute_time_varying(self) -> None:
        visiting: set[str] = set()

        def _check(tool_name: str) -> bool:
            if tool_name in self._time_varying:
                return self._time_varying[tool_name]
            if tool_name in visiting:
                return False
            tool = self._tools.get(tool_name)
            if tool is None:
                return False
            visiting.add(tool_name)
            varying = False
            if tool.constructor == "BezierSpline":
                kf = tool.get("KeyFrames")
                if isinstance(kf, LuaTable) and len(kf.entries) > 1:
                    varying = True
            inputs_tbl = tool.get("Inputs")
            if isinstance(inputs_tbl, LuaTable) and not varying:
                for val in inputs_tbl.items_dict().values():
                    if not isinstance(val, LuaTable):
                        continue
                    expr = val.get("Expression")
                    if isinstance(expr, str) and expr.strip():
                        uses_time, refs = expression_references(expr)
                        if uses_time:
                            varying = True
                            break
                        for ref in refs:
                            if ref != tool_name and ref in self._tools and _check(ref):
                                varying = True
                                break
                        if varying:
                            break
                    src_op = val.get("SourceOp")
                    if isinstance(src_op, str) and src_op in self._tools and _check(src_op):
                        varying = True
                        break
            visiting.discard(tool_name)
            self._time_varying[tool_name] = varying
            return varying

        for name in self._tools:
            _check(name)

    @property
    def is_animated(self) -> bool:
        return self._time_varying.get(self._output_tool, False)

    def render_frame(self, time: float) -> Any:
        frame_cache: dict[str, Any] = {}
        eval_stack: set[tuple[str, str]] = set()
        return self._eval_tool(self._output_tool, time, frame_cache, eval_stack)

    def _eval_input(
        self,
        tool_name: str,
        param: str,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
        default: Any = None,
    ) -> Any:
        key = (tool_name, param)
        if key in eval_stack:
            raise FusionError(
                "expression_error",
                f"cyclic expression/input reference at {tool_name}.{param}",
            )
        tool = self._tools.get(tool_name)
        if tool is None:
            return default
        inputs_tbl = tool.get("Inputs")
        if not isinstance(inputs_tbl, LuaTable):
            return default
        inp = inputs_tbl.get(param)
        if not isinstance(inp, LuaTable):
            return default

        eval_stack.add(key)
        try:
            expr = inp.get("Expression")
            if isinstance(expr, str) and expr.strip():
                ctx = ExprContext(
                    time=time,
                    render_start=self._render_start,
                    render_end=self._render_end,
                    current_tool=tool_name,
                    resolve_property=lambda t, p: self._resolve_expr_prop(
                        t, p, time, frame_cache, eval_stack
                    ),
                    resolve_output_attr=lambda t, a: self._resolve_output_attr(
                        t, a, time, frame_cache, eval_stack
                    ),
                )
                return evaluate_expression(expr, ctx)

            src_op = inp.get("SourceOp")
            if isinstance(src_op, str) and src_op in self._tools:
                src_tool = self._tools[src_op]
                if src_tool.constructor == "BezierSpline":
                    return _eval_bezier_spline(src_tool, time)

            if "Value" in inp:
                return unwrap_value(inp.get("Value"))
            return default
        finally:
            eval_stack.discard(key)

    def _resolve_expr_prop(
        self,
        tool_name: str,
        param: str,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        defaults: dict[str, Any] = {
            "Width": 0.5,
            "Height": 0.5,
            "Center": Point2D(0.5, 0.5),
            "Size": 1.0,
            "Angle": 0.0,
            "Blend": 1.0,
        }
        tool = self._tools.get(tool_name)
        if tool is not None and tool.constructor == "Background":
            defaults["Width"] = float(self._comp_w)
            defaults["Height"] = float(self._comp_h)
        val = self._eval_input(
            tool_name, param, time, frame_cache, eval_stack, default=defaults.get(param)
        )
        if (
            isinstance(val, list)
            and len(val) == 2
            and all(isinstance(x, (int, float)) for x in val)
        ):
            return Point2D(float(val[0]), float(val[1]))
        if val is None:
            raise FusionError(
                "expression_error",
                f"unresolved property {tool_name}.{param}",
            )
        return val

    def _resolve_output_attr(
        self,
        tool_name: str,
        attr: str,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        img = self._eval_tool(tool_name, time, frame_cache, eval_stack)
        w, h = img.size
        if attr == "DataWindow":
            bbox = img.getbbox()
            if bbox is None:
                return [0.0, 0.0, 0.0, 0.0]
            left, top, right, bottom = bbox
            return [float(left), float(h - bottom), float(right), float(h - top)]
        if attr in ("Width", "OriginalWidth"):
            return float(w)
        if attr in ("Height", "OriginalHeight"):
            return float(h)
        raise FusionError(
            "expression_error", f"unsupported Output attribute {tool_name}.Output.{attr}"
        )

    def _connected_tool(self, tool: LuaTable, input_name: str) -> str | None:
        inputs_tbl = tool.get("Inputs")
        if not isinstance(inputs_tbl, LuaTable):
            return None
        inp = inputs_tbl.get(input_name)
        if not isinstance(inp, LuaTable):
            return None
        src = inp.get("SourceOp")
        return src if isinstance(src, str) and src in self._tools else None

    def _eval_tool(
        self,
        tool_name: str,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        if not self._time_varying.get(tool_name, False) and tool_name in self._static_cache:
            return self._static_cache[tool_name]
        if tool_name in frame_cache:
            return frame_cache[tool_name]

        tool = self._tools.get(tool_name)
        if tool is None:
            raise FusionError("invalid_setting", f"referenced tool {tool_name!r} does not exist")

        kind = tool.constructor or ""
        if kind == "Background":
            img = self._render_background(tool_name, tool, time, frame_cache, eval_stack)
        elif kind == "RectangleMask":
            img = self._render_rectangle_mask(tool_name, tool, time, frame_cache, eval_stack)
        elif kind == "EllipseMask":
            img = self._render_ellipse_mask(tool_name, tool, time, frame_cache, eval_stack)
        elif kind == "TextPlus":
            img = self._render_text_plus(tool_name, tool, time, frame_cache, eval_stack)
        elif kind == "Loader":
            img = self._render_loader(tool_name, tool)
        elif kind == "Merge":
            img = self._render_merge(tool_name, tool, time, frame_cache, eval_stack)
        elif kind in ("Transform", "Xf"):
            img = self._render_transform(tool_name, tool, time, frame_cache, eval_stack)
        elif kind == "Dissolve":
            img = self._render_dissolve(tool_name, tool, time, frame_cache, eval_stack)
        else:
            raise FusionError(
                "unsupported_node",
                f"Fusion node {tool_name!r} of type {kind!r} is not supported natively",
            )

        frame_cache[tool_name] = img
        if not self._time_varying.get(tool_name, False):
            self._static_cache[tool_name] = img
        return img

    def _apply_effect_mask(
        self,
        img: Any,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, ImageChops, _, _, _ = _require_pil()
        mask_op = self._connected_tool(tool, "EffectMask")
        if mask_op is None:
            return img
        mask_img = self._eval_tool(mask_op, time, frame_cache, eval_stack)
        mask_alpha = mask_img.getchannel("A")
        if mask_alpha.size != img.size:
            mask_alpha = mask_alpha.resize(img.size, Image.Resampling.BILINEAR)
        out = img.copy()
        new_alpha = ImageChops.multiply(out.getchannel("A"), mask_alpha)
        out.putalpha(new_alpha)
        return out

    def _render_background(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, _, _, _, _ = _require_pil()
        r = float(self._eval_input(tool_name, "TopLeftRed", time, frame_cache, eval_stack, 0.0))
        g = float(self._eval_input(tool_name, "TopLeftGreen", time, frame_cache, eval_stack, 0.0))
        b = float(self._eval_input(tool_name, "TopLeftBlue", time, frame_cache, eval_stack, 0.0))
        a = float(self._eval_input(tool_name, "TopLeftAlpha", time, frame_cache, eval_stack, 1.0))

        img = Image.new(
            "RGBA", (self._comp_w, self._comp_h), (_to_u8(r), _to_u8(g), _to_u8(b), _to_u8(a))
        )
        return self._apply_effect_mask(img, tool, time, frame_cache, eval_stack)

    def _render_rectangle_mask(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, ImageChops, ImageDraw, ImageFilter, _ = _require_pil()
        center = _to_point(
            self._eval_input(tool_name, "Center", time, frame_cache, eval_stack, (0.5, 0.5))
        )
        w_norm = float(self._eval_input(tool_name, "Width", time, frame_cache, eval_stack, 0.5))
        h_norm = float(self._eval_input(tool_name, "Height", time, frame_cache, eval_stack, 0.5))
        corner = float(
            self._eval_input(tool_name, "CornerRadius", time, frame_cache, eval_stack, 0.0)
        )
        angle = float(self._eval_input(tool_name, "Angle", time, frame_cache, eval_stack, 0.0))
        soft = float(self._eval_input(tool_name, "SoftEdge", time, frame_cache, eval_stack, 0.0))
        invert = bool(self._eval_input(tool_name, "Invert", time, frame_cache, eval_stack, 0))
        level = float(self._eval_input(tool_name, "Level", time, frame_cache, eval_stack, 1.0))

        ss = 2  # 2x supersampling for crisp anti-aliased edges
        sw, sh = self._comp_w * ss, self._comp_h * ss
        mask = Image.new("L", (sw, sh), 0)
        draw = ImageDraw.Draw(mask)

        cx = center[0] * sw
        cy = (1.0 - center[1]) * sh
        rw = max(0.0, w_norm * sw)
        rh = max(0.0, h_norm * sh)

        if rw > 0.0 and rh > 0.0:
            x0, y0 = cx - rw / 2.0, cy - rh / 2.0
            x1, y1 = cx + rw / 2.0, cy + rh / 2.0
            rad = max(0.0, min(1.0, corner)) * min(rw, rh) * 0.5
            if rad > 0.5:
                draw.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=255)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=255)

        if abs(angle) > 1e-4:
            mask = mask.rotate(
                angle, resample=Image.Resampling.BICUBIC, center=(cx, cy), fillcolor=0
            )

        mask = mask.resize((self._comp_w, self._comp_h), Image.Resampling.LANCZOS)
        if soft > 1e-5:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=soft * self._comp_w * 0.5))
        if invert:
            mask = ImageChops.invert(mask)
        if level < 0.9999:
            scale = _clamp01(level)
            mask = mask.point(lambda p: round(p * scale))

        out = Image.new("RGBA", (self._comp_w, self._comp_h), (255, 255, 255, 0))
        out.putalpha(mask)
        return self._combine_mask_input(out, tool, time, frame_cache, eval_stack)

    def _render_ellipse_mask(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, ImageChops, ImageDraw, ImageFilter, _ = _require_pil()
        center = _to_point(
            self._eval_input(tool_name, "Center", time, frame_cache, eval_stack, (0.5, 0.5))
        )
        w_norm = float(self._eval_input(tool_name, "Width", time, frame_cache, eval_stack, 0.5))
        h_norm = float(self._eval_input(tool_name, "Height", time, frame_cache, eval_stack, 0.5))
        angle = float(self._eval_input(tool_name, "Angle", time, frame_cache, eval_stack, 0.0))
        soft = float(self._eval_input(tool_name, "SoftEdge", time, frame_cache, eval_stack, 0.0))
        invert = bool(self._eval_input(tool_name, "Invert", time, frame_cache, eval_stack, 0))
        level = float(self._eval_input(tool_name, "Level", time, frame_cache, eval_stack, 1.0))

        ss = 2
        sw, sh = self._comp_w * ss, self._comp_h * ss
        mask = Image.new("L", (sw, sh), 0)
        draw = ImageDraw.Draw(mask)
        cx = center[0] * sw
        cy = (1.0 - center[1]) * sh
        rw = max(0.0, w_norm * sw)
        rh = max(0.0, h_norm * sh)
        if rw > 0.0 and rh > 0.0:
            draw.ellipse([cx - rw / 2.0, cy - rh / 2.0, cx + rw / 2.0, cy + rh / 2.0], fill=255)
        if abs(angle) > 1e-4:
            mask = mask.rotate(
                angle, resample=Image.Resampling.BICUBIC, center=(cx, cy), fillcolor=0
            )
        mask = mask.resize((self._comp_w, self._comp_h), Image.Resampling.LANCZOS)
        if soft > 1e-5:
            mask = mask.filter(ImageFilter.GaussianBlur(radius=soft * self._comp_w * 0.5))
        if invert:
            mask = ImageChops.invert(mask)
        if level < 0.9999:
            scale = _clamp01(level)
            mask = mask.point(lambda p: round(p * scale))

        out = Image.new("RGBA", (self._comp_w, self._comp_h), (255, 255, 255, 0))
        out.putalpha(mask)
        return self._combine_mask_input(out, tool, time, frame_cache, eval_stack)

    def _combine_mask_input(
        self,
        mask_rgba: Any,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, ImageChops, _, _, _ = _require_pil()
        upstream = self._connected_tool(tool, "EffectMask")
        if upstream is None:
            return mask_rgba
        up_img = self._eval_tool(upstream, time, frame_cache, eval_stack)
        up_a = up_img.getchannel("A")
        if up_a.size != mask_rgba.size:
            up_a = up_a.resize(mask_rgba.size, Image.Resampling.BILINEAR)
        cur_a = mask_rgba.getchannel("A")
        paint_mode = str(
            unwrap_value(
                (tool.get("Inputs") or LuaTable()).get("PaintMode", LuaTable()).get("Value")
                if isinstance(tool.get("Inputs"), LuaTable)
                and isinstance(tool.get("Inputs").get("PaintMode"), LuaTable)
                else "Merge"
            )
            or "Merge"
        )
        if paint_mode == "Multiply":
            combined = ImageChops.multiply(up_a, cur_a)
        elif paint_mode == "Subtract":
            combined = ImageChops.subtract(up_a, cur_a)
        else:
            combined = ImageChops.lighter(up_a, cur_a)
        out = mask_rgba.copy()
        out.putalpha(combined)
        return out

    def _render_text_plus(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, _, ImageDraw, _, _ = _require_pil()
        text = str(
            self._eval_input(tool_name, "StyledText", time, frame_cache, eval_stack, "") or ""
        )
        font_family = str(
            self._eval_input(tool_name, "Font", time, frame_cache, eval_stack, "Open Sans")
            or "Open Sans"
        )
        style = str(
            self._eval_input(tool_name, "Style", time, frame_cache, eval_stack, "Regular")
            or "Regular"
        )
        size_norm = float(self._eval_input(tool_name, "Size", time, frame_cache, eval_stack, 0.08))
        center = _to_point(
            self._eval_input(tool_name, "Center", time, frame_cache, eval_stack, (0.5, 0.5))
        )
        h_align = float(
            self._eval_input(
                tool_name, "HorizontalLeftCenterRight", time, frame_cache, eval_stack, 0.0
            )
        )
        v_align = float(
            self._eval_input(
                tool_name, "VerticalTopCenterBottom", time, frame_cache, eval_stack, 0.0
            )
        )
        r = float(self._eval_input(tool_name, "Red1", time, frame_cache, eval_stack, 1.0))
        g = float(self._eval_input(tool_name, "Green1", time, frame_cache, eval_stack, 1.0))
        b = float(self._eval_input(tool_name, "Blue1", time, frame_cache, eval_stack, 1.0))
        a = float(self._eval_input(tool_name, "Alpha1", time, frame_cache, eval_stack, 1.0))
        opacity = float(self._eval_input(tool_name, "Opacity1", time, frame_cache, eval_stack, 1.0))
        line_spacing = float(
            self._eval_input(tool_name, "LineSpacing", time, frame_cache, eval_stack, 1.0)
        )

        img = Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))
        if not text:
            return img

        font_px = max(1, round(size_norm * self._comp_h))
        font = _load_pil_font(
            font_family,
            style,
            font_px,
            [self._template.path.parent, *self._asset_dirs],
        )

        draw = ImageDraw.Draw(img)
        fill = (_to_u8(r), _to_u8(g), _to_u8(b), _to_u8(a * opacity))

        anchor_h = "l" if h_align < -0.33 else ("r" if h_align > 0.33 else "m")
        anchor_v = "a" if v_align < -0.33 else ("d" if v_align > 0.33 else "m")
        pil_align = "left" if anchor_h == "l" else ("right" if anchor_h == "r" else "center")

        px = center[0] * self._comp_w
        py = (1.0 - center[1]) * self._comp_h

        if "\n" not in text:
            # Measure tight bbox to align cleanly on the requested horizontal & vertical anchor
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if anchor_h == "l":
                draw_x = px - bbox[0]
            elif anchor_h == "r":
                draw_x = px - tw - bbox[0]
            else:
                draw_x = px - tw / 2.0 - bbox[0]

            if anchor_v == "a":
                draw_y = py - bbox[1]
            elif anchor_v == "d":
                draw_y = py - th - bbox[1]
            else:
                draw_y = py - th / 2.0 - bbox[1]

            draw.text((draw_x, draw_y), text, font=font, fill=fill)
        else:
            spacing = max(0, round(font_px * 0.2 * line_spacing))
            bbox = draw.multiline_textbbox(
                (0, 0), text, font=font, spacing=spacing, align=pil_align
            )
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if anchor_h == "l":
                draw_x = px - bbox[0]
            elif anchor_h == "r":
                draw_x = px - tw - bbox[0]
            else:
                draw_x = px - tw / 2.0 - bbox[0]
            if anchor_v == "a":
                draw_y = py - bbox[1]
            elif anchor_v == "d":
                draw_y = py - th - bbox[1]
            else:
                draw_y = py - th / 2.0 - bbox[1]
            draw.multiline_text(
                (draw_x, draw_y),
                text,
                font=font,
                fill=fill,
                spacing=spacing,
                align=pil_align,
            )

        return self._apply_effect_mask(img, tool, time, frame_cache, eval_stack)

    def _render_loader(self, tool_name: str, tool: LuaTable) -> Any:
        if tool_name in self._loader_cache:
            return self._loader_cache[tool_name]
        Image, _, _, _, _ = _require_pil()
        from PIL import UnidentifiedImageError

        clips = tool.get("Clips")
        filename: str | None = None
        if isinstance(clips, LuaTable):
            for entry in clips.entries:
                if isinstance(entry.value, LuaTable) and isinstance(
                    entry.value.get("Filename"), str
                ):
                    filename = entry.value.get("Filename")
                    break
        if not filename:
            raise FusionError("missing_asset", f"Loader {tool_name!r} has no Filename")

        resolved_path, archive_member = resolve_asset_path(
            filename,
            template_path=self._template.path,
            asset_dirs=self._asset_dirs,
            archive_members=self._archive_members,
            confined_roots=self._confined_roots,
        )
        try:
            if resolved_path is not None:
                size = resolved_path.stat().st_size
                if size > MAX_ASSET_BYTES:
                    raise FusionError(
                        "invalid_setting",
                        f"Loader {tool_name!r} asset {resolved_path.name!r} ({size} bytes) "
                        f"exceeds limit of {MAX_ASSET_BYTES} bytes",
                    )
                with Image.open(resolved_path) as opened:
                    _check_image_size(opened.width, opened.height, f"Loader {tool_name!r}")
                    img = opened.convert("RGBA")
            elif archive_member is not None:
                with zipfile.ZipFile(self._template.path, "r") as zf:
                    raw = read_zip_member_bounded(zf, archive_member, MAX_ASSET_BYTES)
                with Image.open(io.BytesIO(raw)) as opened:
                    _check_image_size(opened.width, opened.height, f"Loader {tool_name!r}")
                    img = opened.convert("RGBA")
            else:
                raise FusionError(
                    "missing_asset",
                    f"Loader {tool_name!r} asset not found: {filename!r}",
                    {"tool": tool_name, "filename": filename},
                )
        except FusionError:
            raise
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,
            ValueError,
            zipfile.BadZipFile,
        ) as exc:
            raise FusionError(
                "render_failed",
                f"Loader {tool_name!r} could not decode asset {filename!r}: {exc}",
                {"tool": tool_name, "filename": filename},
            ) from exc
        self._loader_cache[tool_name] = img
        return img

    def _render_merge(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, ImageChops, _, _, _ = _require_pil()
        bg_op = self._connected_tool(tool, "Background")
        fg_op = self._connected_tool(tool, "Foreground")

        if bg_op is not None:
            bg = self._eval_tool(bg_op, time, frame_cache, eval_stack).copy()
        else:
            bg = Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))

        if fg_op is None:
            return bg

        fg = self._eval_tool(fg_op, time, frame_cache, eval_stack)
        center = _to_point(
            self._eval_input(tool_name, "Center", time, frame_cache, eval_stack, (0.5, 0.5))
        )
        size = float(self._eval_input(tool_name, "Size", time, frame_cache, eval_stack, 1.0))
        angle = float(self._eval_input(tool_name, "Angle", time, frame_cache, eval_stack, 0.0))
        blend = float(self._eval_input(tool_name, "Blend", time, frame_cache, eval_stack, 1.0))
        flip_h = bool(self._eval_input(tool_name, "FlipHoriz", time, frame_cache, eval_stack, 0))
        flip_v = bool(self._eval_input(tool_name, "FlipVert", time, frame_cache, eval_stack, 0))

        if not (math.isfinite(size) and math.isfinite(angle) and math.isfinite(blend)):
            raise FusionError(
                "invalid_setting", f"Merge {tool_name!r} has non-finite transform parameter"
            )
        if blend <= 0.0 or size <= 0.0:
            return bg

        fg_work = fg
        if flip_h:
            fg_work = fg_work.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if flip_v:
            fg_work = fg_work.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if abs(size - 1.0) > 1e-5:
            new_w = max(1, round(fg_work.width * size))
            new_h = max(1, round(fg_work.height * size))
            _check_image_size(new_w, new_h, f"Merge {tool_name!r} scaled foreground")
            fg_work = fg_work.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if abs(angle) > 1e-4:
            rad = math.radians(angle)
            rot_w = math.ceil(
                abs(fg_work.width * math.cos(rad)) + abs(fg_work.height * math.sin(rad))
            )
            rot_h = math.ceil(
                abs(fg_work.width * math.sin(rad)) + abs(fg_work.height * math.cos(rad))
            )
            _check_image_size(
                max(1, rot_w), max(1, rot_h), f"Merge {tool_name!r} rotated foreground"
            )
            fg_work = fg_work.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

        # Place fg_work onto a bg-sized transparent layer centered at (center.x, 1-center.y)
        bg_w, bg_h = bg.size
        cx = center[0] * bg_w
        cy = (1.0 - center[1]) * bg_h
        paste_x = round(cx - fg_work.width / 2.0)
        paste_y = round(cy - fg_work.height / 2.0)

        fg_layer = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))
        fg_layer.paste(fg_work, (paste_x, paste_y))

        # Apply Merge's EffectMask to the foreground layer before compositing onto Background
        fg_layer = self._apply_effect_mask(fg_layer, tool, time, frame_cache, eval_stack)

        if blend < 0.9999:
            scale = _clamp01(blend)
            alpha = fg_layer.getchannel("A").point(lambda p: round(p * scale))
            fg_layer.putalpha(alpha)

        apply_mode = str(
            self._eval_input(tool_name, "ApplyMode", time, frame_cache, eval_stack, "Normal")
            or "Normal"
        )
        if apply_mode == "Screen":
            blended_rgb = ImageChops.screen(bg.convert("RGB"), fg_layer.convert("RGB"))
            fg_a = fg_layer.getchannel("A")
            blended = blended_rgb.convert("RGBA")
            blended.putalpha(fg_a)
            return Image.alpha_composite(bg, blended)
        if apply_mode == "Multiply":
            blended_rgb = ImageChops.multiply(bg.convert("RGB"), fg_layer.convert("RGB"))
            fg_a = fg_layer.getchannel("A")
            blended = blended_rgb.convert("RGBA")
            blended.putalpha(fg_a)
            return Image.alpha_composite(bg, blended)
        if apply_mode == "Add":
            blended_rgb = ImageChops.add(bg.convert("RGB"), fg_layer.convert("RGB"))
            fg_a = fg_layer.getchannel("A")
            blended = blended_rgb.convert("RGBA")
            blended.putalpha(fg_a)
            return Image.alpha_composite(bg, blended)

        return Image.alpha_composite(bg, fg_layer)

    def _render_transform(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, _, _, _, _ = _require_pil()
        src_op = self._connected_tool(tool, "Input")
        if src_op is None:
            return Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))
        src = self._eval_tool(src_op, time, frame_cache, eval_stack)
        center = _to_point(
            self._eval_input(tool_name, "Center", time, frame_cache, eval_stack, (0.5, 0.5))
        )
        size = float(self._eval_input(tool_name, "Size", time, frame_cache, eval_stack, 1.0))
        aspect = float(self._eval_input(tool_name, "Aspect", time, frame_cache, eval_stack, 1.0))
        angle = float(self._eval_input(tool_name, "Angle", time, frame_cache, eval_stack, 0.0))
        blend = float(self._eval_input(tool_name, "Blend", time, frame_cache, eval_stack, 1.0))

        if not (
            math.isfinite(size)
            and math.isfinite(aspect)
            and math.isfinite(angle)
            and math.isfinite(blend)
        ):
            raise FusionError(
                "invalid_setting", f"Transform {tool_name!r} has non-finite transform parameter"
            )

        work = src
        sx = max(1e-4, size)
        sy = max(1e-4, size / max(1e-4, aspect))
        if abs(sx - 1.0) > 1e-5 or abs(sy - 1.0) > 1e-5:
            new_w = max(1, round(work.width * sx))
            new_h = max(1, round(work.height * sy))
            _check_image_size(new_w, new_h, f"Transform {tool_name!r} scaled image")
            work = work.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if abs(angle) > 1e-4:
            rad = math.radians(angle)
            rot_w = math.ceil(abs(work.width * math.cos(rad)) + abs(work.height * math.sin(rad)))
            rot_h = math.ceil(abs(work.width * math.sin(rad)) + abs(work.height * math.cos(rad)))
            _check_image_size(
                max(1, rot_w), max(1, rot_h), f"Transform {tool_name!r} rotated image"
            )
            work = work.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)

        out = Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))
        cx = center[0] * self._comp_w
        cy = (1.0 - center[1]) * self._comp_h
        out.paste(work, (round(cx - work.width / 2.0), round(cy - work.height / 2.0)))
        if blend < 0.9999:
            scale = _clamp01(blend)
            out.putalpha(out.getchannel("A").point(lambda p: round(p * scale)))
        return self._apply_effect_mask(out, tool, time, frame_cache, eval_stack)

    def _render_dissolve(
        self,
        tool_name: str,
        tool: LuaTable,
        time: float,
        frame_cache: dict[str, Any],
        eval_stack: set[tuple[str, str]],
    ) -> Any:
        Image, _, _, _, _ = _require_pil()
        bg_op = self._connected_tool(tool, "Background")
        fg_op = self._connected_tool(tool, "Foreground")
        mix = _clamp01(
            float(self._eval_input(tool_name, "Mix", time, frame_cache, eval_stack, 0.5))
        )
        bg = (
            self._eval_tool(bg_op, time, frame_cache, eval_stack)
            if bg_op
            else Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))
        )
        fg = (
            self._eval_tool(fg_op, time, frame_cache, eval_stack)
            if fg_op
            else Image.new("RGBA", (self._comp_w, self._comp_h), (0, 0, 0, 0))
        )
        if fg.size != bg.size:
            fg = fg.resize(bg.size, Image.Resampling.LANCZOS)
        return Image.blend(bg, fg, mix)


def _extract_tools_from_ast(ast: LuaTable, template: FusionTemplate) -> dict[str, LuaTable]:
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


def _build_evaluator(
    template: FusionTemplate,
    *,
    inputs: Mapping[str, Any] | None,
    lines: Sequence[str],
    resolution: tuple[int, int],
    render_start: float,
    render_end: float,
    asset_dirs: Sequence[Path],
    confined_roots: Sequence[Path] | None = None,
) -> tuple[_FusionGraphEvaluator, tuple[int, int]]:
    _check_image_size(resolution[0], resolution[1], "target resolution")
    ast = apply_inputs_to_ast(template, inputs=inputs, lines=lines)
    tools = _extract_tools_from_ast(ast, template)
    if not template.output_tool or template.output_tool not in tools:
        raise FusionError(
            "invalid_setting", f"template {template.name!r} has no resolvable output tool"
        )

    native = _detect_native_size(tools, resolution)
    # If target resolution matches native aspect ratio within 1%, evaluate at native resolution so
    # hardcoded pixel expressions (like `/ 1920`) stay exact, then resize cleanly to target.
    target_w, target_h = resolution
    nat_w, nat_h = native
    same_aspect = abs((nat_w / nat_h) - (target_w / target_h)) < 0.02
    comp_size = native if same_aspect else resolution

    evaluator = _FusionGraphEvaluator(
        template=template,
        tools=tools,
        output_tool=template.output_tool,
        comp_size=comp_size,
        render_start=render_start,
        render_end=render_end,
        asset_dirs=asset_dirs,
        confined_roots=confined_roots,
    )
    return evaluator, comp_size


def _drain_stream_bounded(stream: Any, out: bytearray, limit: int) -> None:
    """Drain ``stream`` to EOF on a helper thread, retaining at most the last ``limit`` bytes."""
    try:
        while True:
            chunk = stream.read(8192)
            if not chunk:
                break
            out.extend(chunk)
            if len(out) > limit:
                del out[: len(out) - limit]
    except OSError:
        pass
    finally:
        with contextlib.suppress(OSError):
            stream.close()


def _terminate_and_reap(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.terminate()
        try:
            proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                proc.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Public rendering API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionRenderResult:
    """Metadata produced by :func:`render_fusion_clip`."""

    output: Path
    template: str
    resolution: tuple[int, int]
    fps: float
    frames: int
    duration_s: float
    codec: str
    animated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "output": str(self.output),
            "template": self.template,
            "resolution": list(self.resolution),
            "fps": self.fps,
            "frames": self.frames,
            "duration_s": self.duration_s,
            "codec": self.codec,
            "animated": self.animated,
        }


def render_fusion_frame(
    template: FusionTemplate | str | Path,
    *,
    inputs: Mapping[str, Any] | None = None,
    lines: Sequence[str] = (),
    resolution: tuple[int, int] = (1920, 1080),
    time: float = 0.0,
    render_end: float = 149.0,
    asset_dirs: Sequence[Path] = (),
    confined_roots: Sequence[Path] | None = None,
    output: Path | None = None,
) -> Any:
    """Render a single RGBA frame of a Fusion template and return the ``PIL.Image.Image``.

    If ``output`` is provided, saves the RGBA PNG to that path as well.
    """
    Image, _, _, _, _ = _require_pil()
    _check_image_size(resolution[0], resolution[1], "target resolution")
    tpl = (
        template
        if isinstance(template, FusionTemplate)
        else load_template(template, asset_dirs=asset_dirs, confined_roots=confined_roots)
    )
    evaluator, comp_size = _build_evaluator(
        tpl,
        inputs=inputs,
        lines=lines,
        resolution=resolution,
        render_start=0.0,
        render_end=render_end,
        asset_dirs=asset_dirs,
        confined_roots=confined_roots,
    )
    img = evaluator.render_frame(time)
    if comp_size != resolution:
        img = img.resize(resolution, Image.Resampling.LANCZOS)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        img.save(output, format="PNG")
    return img


def render_fusion_clip(
    template: FusionTemplate | str | Path,
    output: Path,
    *,
    duration_s: float,
    fps: float = 30.0,
    resolution: tuple[int, int] = (1920, 1080),
    inputs: Mapping[str, Any] | None = None,
    lines: Sequence[str] = (),
    asset_dirs: Sequence[Path] = (),
    confined_roots: Sequence[Path] | None = None,
    codec: str = "qtrle",
    ffmpeg_bin: str = "ffmpeg",
    timeout_s: float | None = None,
) -> FusionRenderResult:
    """Render an animated or static Fusion template to an alpha-preserving ``.mov``."""
    Image, _, _, _, _ = _require_pil()
    if duration_s <= 0 or not math.isfinite(duration_s):
        raise FusionError("invalid_input_value", f"duration_s must be > 0, got {duration_s}")
    if fps <= 0 or not math.isfinite(fps):
        raise FusionError("invalid_input_value", f"fps must be > 0, got {fps}")
    if timeout_s is not None and (timeout_s <= 0 or not math.isfinite(timeout_s)):
        raise FusionError("invalid_input_value", f"timeout_s must be > 0, got {timeout_s}")
    _check_image_size(resolution[0], resolution[1], "target resolution")
    if codec not in _CODEC_ARGS:
        raise FusionError(
            "invalid_input_value",
            f"unsupported alpha codec {codec!r}; choose from {sorted(_CODEC_ARGS)}",
        )
    if shutil.which(ffmpeg_bin) is None and not Path(ffmpeg_bin).is_file():
        raise FusionError("render_failed", f"ffmpeg binary not found: {ffmpeg_bin!r}")

    deadline = (time.monotonic() + timeout_s) if timeout_s is not None else None

    def _check_deadline(stage: str) -> None:
        if deadline is not None and time.monotonic() > deadline:
            raise FusionError(
                "render_timeout",
                f"Fusion render timed out after {timeout_s:g}s while {stage}",
            )

    tpl = (
        template
        if isinstance(template, FusionTemplate)
        else load_template(template, asset_dirs=asset_dirs, confined_roots=confined_roots)
    )

    num_frames = max(1, math.ceil(duration_s * fps))
    render_end = float(max(0, num_frames - 1))
    evaluator, comp_size = _build_evaluator(
        tpl,
        inputs=inputs,
        lines=lines,
        resolution=resolution,
        render_start=0.0,
        render_end=render_end,
        asset_dirs=asset_dirs,
        confined_roots=confined_roots,
    )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix or ".mov"
    tmp_path = (
        out_path.parent / f".{out_path.stem}.{os.getpid()}.{secrets.token_hex(4)}.tmp{suffix}"
    )

    width, height = resolution
    tc_args: list[str] = []
    if any(abs(fps - rate) < 0.01 for rate in (23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0)):
        tc_args = ["-timecode", "00:00:00:00"]
    cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}".rstrip("0").rstrip("."),
        "-i",
        "pipe:0",
        *_CODEC_ARGS[codec],
        *tc_args,
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise FusionError("render_failed", f"could not launch ffmpeg: {exc}") from exc

    assert proc.stdin is not None
    assert proc.stderr is not None

    stderr_buf = bytearray()
    stderr_thread = threading.Thread(
        target=_drain_stream_bounded,
        args=(proc.stderr, stderr_buf, _MAX_STDERR_BYTES),
        daemon=True,
    )
    stderr_thread.start()

    animated = evaluator.is_animated
    timed_out = False
    try:
        if not animated:
            _check_deadline(f"rendering {tpl.name!r}")
            img = evaluator.render_frame(0.0)
            if comp_size != resolution:
                img = img.resize(resolution, Image.Resampling.LANCZOS)
            raw = img.tobytes()
            for _ in range(num_frames):
                _check_deadline(f"encoding {tpl.name!r}")
                proc.stdin.write(raw)
        else:
            for f in range(num_frames):
                _check_deadline(f"rendering frame {f} of {tpl.name!r}")
                img = evaluator.render_frame(float(f))
                if comp_size != resolution:
                    img = img.resize(resolution, Image.Resampling.LANCZOS)
                proc.stdin.write(img.tobytes())
    except BrokenPipeError:
        pass
    except FusionError as exc:
        if exc.code == "render_timeout":
            timed_out = True
        _terminate_and_reap(proc)
        with contextlib.suppress(OSError):
            proc.stdin.close()
        stderr_thread.join(timeout=2.0)
        tmp_path.unlink(missing_ok=True)
        raise
    except BaseException:
        _terminate_and_reap(proc)
        with contextlib.suppress(OSError):
            proc.stdin.close()
        stderr_thread.join(timeout=2.0)
        tmp_path.unlink(missing_ok=True)
        raise
    finally:
        with contextlib.suppress(OSError):
            proc.stdin.close()

    if not timed_out:
        try:
            remaining = None if deadline is None else max(0.05, deadline - time.monotonic())
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _terminate_and_reap(proc)
            stderr_thread.join(timeout=2.0)
            tmp_path.unlink(missing_ok=True)
            raise FusionError(
                "render_timeout",
                f"ffmpeg timed out after {timeout_s:g}s while finishing {tpl.name!r}",
            ) from None

    stderr_thread.join(timeout=2.0)

    if proc.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        msg = bytes(stderr_buf).decode("utf-8", errors="replace").strip()
        raise FusionError(
            "render_failed",
            f"ffmpeg exited {proc.returncode} while rendering {tpl.name!r}: {msg}",
        )

    os.replace(tmp_path, out_path)

    return FusionRenderResult(
        output=out_path,
        template=tpl.name,
        resolution=resolution,
        fps=fps,
        frames=num_frames,
        duration_s=num_frames / fps,
        codec=codec,
        animated=animated,
    )
