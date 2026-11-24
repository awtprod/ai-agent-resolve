"""``aar fusion`` CLI commands: inspect, customize, render, and manifest Fusion templates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aar import cli_io
from aar.fusion.apply import load_fusion_manifest
from aar.fusion.customize import customize_setting
from aar.fusion.errors import FusionError
from aar.fusion.parse import load_template, resolve_template_path
from aar.fusion.render import render_fusion_clip, render_fusion_frame

_PROG = "aar fusion"
EXIT_OK = 0
EXIT_ERROR = 2


def register(subparsers: Any) -> None:
    """Register the ``fusion`` command group on the top-level ``aar`` parser."""
    fusion = subparsers.add_parser(
        "fusion",
        help="Inspect, customize, and render DaVinci Resolve Fusion (.setting/.drfx) graphics.",
    )
    sub = fusion.add_subparsers(dest="fusion_command", metavar="COMMAND", required=True)

    insp = sub.add_parser(
        "inspect",
        help="Inspect published inputs, assets, fonts, and nodes of a .setting or .drfx template.",
    )
    insp.add_argument("template", help="Path or template name (.setting or .drfx).")
    insp.add_argument("--member", help="Archive member name when inspecting a .drfx package.")
    insp.add_argument(
        "--asset-dir",
        action="append",
        default=[],
        type=Path,
        help="Extra directory to search for Loader assets (repeatable).",
    )
    insp.add_argument("--json", action="store_true", help="Emit JSON output.")
    insp.set_defaults(func=_cmd_inspect)

    cust = sub.add_parser(
        "customize",
        help="Write a customized .setting with overridden macro inputs or asset paths.",
    )
    cust.add_argument("template", help="Path or template name (.setting or .drfx).")
    cust.add_argument("--out", required=True, type=Path, help="Output .setting path.")
    cust.add_argument("--member", help="Archive member name for .drfx packages.")
    cust.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Published input override (repeatable; JSON-parsed when valid, else string).",
    )
    cust.add_argument("--inputs-json", type=Path, help="JSON object of input overrides.")
    cust.add_argument(
        "--line",
        action="append",
        default=[],
        help="Positional text line mapped to string inputs (repeatable).",
    )
    cust.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Remap a Loader tool name or Filename to PATH (repeatable).",
    )
    cust.add_argument("--json", action="store_true", help="Emit JSON summary.")
    cust.set_defaults(func=_cmd_customize)

    rend = sub.add_parser(
        "render",
        help="Render a Fusion .setting/.drfx graphic to an alpha-preserving .mov or .png.",
    )
    rend.add_argument("template", help="Path or template name (.setting or .drfx).")
    rend.add_argument("--out", required=True, type=Path, help="Output .mov or .png path.")
    rend.add_argument("--member", help="Archive member name for .drfx packages.")
    rend.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds.")
    rend.add_argument("--fps", type=float, default=30.0, help="Frame rate (default: 30).")
    rend.add_argument("--width", type=int, default=1920, help="Output width (default: 1920).")
    rend.add_argument("--height", type=int, default=1080, help="Output height (default: 1080).")
    rend.add_argument(
        "--frame",
        type=float,
        help="Render a single RGBA PNG at this frame index instead of an animated .mov.",
    )
    rend.add_argument(
        "--codec",
        choices=["qtrle", "prores4444", "png"],
        default="qtrle",
        help="Alpha video codec for .mov output (default: qtrle).",
    )
    rend.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Published input override (repeatable).",
    )
    rend.add_argument("--inputs-json", type=Path, help="JSON object of input overrides.")
    rend.add_argument(
        "--line",
        action="append",
        default=[],
        help="Positional text line mapped to string inputs (repeatable).",
    )
    rend.add_argument(
        "--asset-dir",
        action="append",
        default=[],
        type=Path,
        help="Extra directory to search for Loader assets and fonts (repeatable).",
    )
    rend.add_argument("--json", action="store_true", help="Emit JSON summary.")
    rend.set_defaults(func=_cmd_render)

    man = sub.add_parser(
        "manifest",
        help="Inspect a .fusion.json export manifest or batch-render its placements.",
    )
    man.add_argument("manifest", type=Path, help="Path to a .fusion.json manifest.")
    man.add_argument(
        "--render-dir",
        type=Path,
        help="Render each placement with a resolvable .setting into this directory.",
    )
    man.add_argument(
        "--asset-dir",
        action="append",
        default=[],
        type=Path,
        help="Extra directory to search for templates, assets, and fonts (repeatable).",
    )
    man.add_argument(
        "--codec",
        choices=["qtrle", "prores4444", "png"],
        default="qtrle",
        help="Alpha codec when using --render-dir (default: qtrle).",
    )
    man.add_argument("--json", action="store_true", help="Emit JSON output.")
    man.set_defaults(func=_cmd_manifest)


def _parse_kv_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped in ("true", "false") or (
        stripped and (stripped[0] in "-0123456789[" or stripped.startswith('"'))
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    return raw


def _collect_inputs(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if getattr(args, "inputs_json", None) is not None:
        try:
            data = json.loads(args.inputs_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FusionError(
                "invalid_input_value", f"cannot read --inputs-json {args.inputs_json}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise FusionError("invalid_input_value", "--inputs-json must contain a JSON object")
        out.update(data)
    for item in getattr(args, "input", ()):
        if "=" not in item:
            raise FusionError("invalid_input_value", f"--input must be KEY=VALUE, got {item!r}")
        key, val = item.split("=", 1)
        out[key.strip()] = _parse_kv_value(val)
    return out


def _collect_kv_strings(items: list[str], flag: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise FusionError("invalid_input_value", f"{flag} must be KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _resolve_or_raise(ref: str, search_dirs: list[Path]) -> Path:
    resolved = resolve_template_path(ref, search_dirs=search_dirs)
    if resolved is None:
        raise FusionError("template_not_found", f"Fusion template not found: {ref!r}")
    return resolved


def _emit_error(exc: FusionError, as_json: bool) -> int:
    if as_json:
        code = cli_io.emit(
            _PROG, json.dumps({"ok": False, "error": exc.to_dict()}, indent=2) + "\n"
        )
        return code or EXIT_ERROR
    cli_io.warn(f"{_PROG}: {exc}")
    return EXIT_ERROR


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        path = _resolve_or_raise(args.template, list(args.asset_dir))
        tpl = load_template(path, member=args.member, asset_dirs=args.asset_dir)
    except FusionError as exc:
        return _emit_error(exc, args.json)

    if args.json:
        return cli_io.emit(_PROG, json.dumps(tpl.to_dict(), indent=2) + "\n") or EXIT_OK

    lines = [
        f"Template: {tpl.name} ({tpl.operator_type})",
        f"Path:     {tpl.path}" + (f" [{tpl.member}]" if tpl.member else ""),
        f"Output:   {tpl.output_tool or '(none)'} ({len(tpl.tools)} nodes)",
        "Inputs:",
    ]
    if not tpl.inputs:
        lines.append("  (none)")
    for inp in tpl.inputs:
        lines.append(
            f"  - {inp.name} ({inp.key} -> {inp.source_op}.{inp.source}, "
            f"{inp.value_type}, default={inp.default!r})"
        )
    if tpl.assets:
        lines.append("Assets:")
        for asset in tpl.assets:
            status = str(asset.resolved_path or asset.archive_member or "MISSING")
            lines.append(f"  - {asset.tool}: {asset.filename} -> {status}")
    if tpl.fonts:
        lines.append("Fonts:")
        for fam, style in tpl.fonts:
            lines.append(f"  - {fam} ({style})")
    return cli_io.emit(_PROG, "\n".join(lines) + "\n") or EXIT_OK


def _cmd_customize(args: argparse.Namespace) -> int:
    try:
        path = _resolve_or_raise(args.template, [])
        tpl = load_template(path, member=args.member)
        inputs = _collect_inputs(args)
        asset_map = _collect_kv_strings(args.asset, "--asset")
        text = customize_setting(
            tpl,
            inputs=inputs,
            lines=args.line,
            asset_map=asset_map or None,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    except (FusionError, OSError) as exc:
        err = exc if isinstance(exc, FusionError) else FusionError("render_failed", str(exc))
        return _emit_error(err, args.json)

    payload = {"ok": True, "template": tpl.name, "output": str(args.out.resolve())}
    if args.json:
        return cli_io.emit(_PROG, json.dumps(payload, indent=2) + "\n") or EXIT_OK
    return cli_io.emit(_PROG, f"Wrote customized Fusion setting: {args.out}\n") or EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    try:
        path = _resolve_or_raise(args.template, list(args.asset_dir))
        tpl = load_template(path, member=args.member, asset_dirs=args.asset_dir)
        inputs = _collect_inputs(args)
        res = (args.width, args.height)
        if args.frame is not None or args.out.suffix.lower() == ".png":
            frame_idx = float(args.frame or 0.0)
            render_end = max(frame_idx, args.duration * args.fps - 1.0)
            render_fusion_frame(
                tpl,
                inputs=inputs,
                lines=args.line,
                resolution=res,
                time=frame_idx,
                render_end=render_end,
                asset_dirs=args.asset_dir,
                output=args.out,
            )
            payload: dict[str, Any] = {
                "ok": True,
                "template": tpl.name,
                "output": str(args.out.resolve()),
                "resolution": list(res),
                "frame": frame_idx,
            }
        else:
            result = render_fusion_clip(
                tpl,
                args.out,
                duration_s=args.duration,
                fps=args.fps,
                resolution=res,
                inputs=inputs,
                lines=args.line,
                asset_dirs=args.asset_dir,
                codec=args.codec,
            )
            payload = {"ok": True, **result.to_dict()}
    except FusionError as exc:
        return _emit_error(exc, args.json)

    if args.json:
        return cli_io.emit(_PROG, json.dumps(payload, indent=2) + "\n") or EXIT_OK
    return cli_io.emit(_PROG, f"Rendered Fusion graphic: {args.out}\n") or EXIT_OK


def _cmd_manifest(args: argparse.Namespace) -> int:
    try:
        manifest = load_fusion_manifest(args.manifest)
        rendered: list[dict[str, Any]] = []
        if args.render_dir is not None:
            search = [args.manifest.parent, *args.asset_dir]
            args.render_dir.mkdir(parents=True, exist_ok=True)
            for p in manifest.placements:
                ref = p.customized_setting or p.setting or p.template
                if not ref:
                    continue
                tpl_path = resolve_template_path(ref, search_dirs=search)
                if tpl_path is None:
                    raise FusionError(
                        "template_not_found",
                        f"placement #{p.index} template not found: {ref!r}",
                    )
                out_mov = args.render_dir / f"fusion_{p.index:02d}_{tpl_path.stem}.mov"
                res = render_fusion_clip(
                    tpl_path,
                    out_mov,
                    duration_s=p.duration_s,
                    fps=manifest.fps,
                    resolution=manifest.resolution,
                    inputs=p.inputs,
                    lines=p.lines,
                    asset_dirs=search,
                    codec=args.codec,
                )
                rendered.append({"index": p.index, **res.to_dict()})
    except FusionError as exc:
        return _emit_error(exc, args.json)

    payload: dict[str, Any] = manifest.model_dump(mode="json")
    if args.render_dir is not None:
        payload["rendered"] = rendered
    if args.json:
        return cli_io.emit(_PROG, json.dumps(payload, indent=2) + "\n") or EXIT_OK

    lines = [
        f"Manifest: {manifest.timeline_name} "
        f"({manifest.resolution[0]}x{manifest.resolution[1]} @ {manifest.fps:g}fps)",
        f"Placements: {len(manifest.placements)}",
    ]
    for p in manifest.placements:
        ref = p.setting or p.template or "(none)"
        lines.append(
            f"  #{p.index} [{p.kind}] V{p.track} {p.record_in_s:.2f}s..{p.record_out_s:.2f}s "
            f"template={ref} inputs={p.inputs or p.lines}"
        )
    for r in rendered:
        lines.append(f"  rendered #{r['index']} -> {r['output']}")
    return cli_io.emit(_PROG, "\n".join(lines) + "\n") or EXIT_OK
