"""``aar node`` commands: list the node op library."""

from __future__ import annotations

import argparse
import json
from typing import Any

from aar import cli_io
from aar.node.ops import list_ops

_PROG = "aar node"


def register(subparsers: Any) -> None:
    """Add the ``node`` command group to the top-level parser."""
    node = subparsers.add_parser("node", help="Node op library (hot ops on a Resolve node).")
    sub = node.add_subparsers(dest="node_command", metavar="COMMAND", required=True)
    ops = sub.add_parser(
        "ops", help="List the ops: name, hot, mutates, the traps they handle, params schema."
    )
    ops.add_argument("--json", action="store_true", help="Print JSON, with each params schema.")
    ops.set_defaults(func=_cmd_ops)


def _params(schema: dict[str, Any]) -> str:
    required = set(schema.get("required", []))
    names = [f"{name}{'' if name in required else '?'}" for name in schema.get("properties", {})]
    return ", ".join(names) or "-"


def _cmd_ops(args: argparse.Namespace) -> int:
    described = [spec.describe() for spec in list_ops()]
    if args.json:
        return cli_io.emit(_PROG, json.dumps(described, indent=2) + "\n")
    width = max(len(d["name"]) for d in described)
    lines: list[str] = []
    for d in described:
        flags = f"{'hot' if d['hot'] else 'cold'} {'mutates' if d['mutates'] else 'read'}"
        lines.append(f"{d['name']:<{width}}  {flags:<12} {d['summary']}")
        lines.append(f"{'':<{width}}  params: {_params(d['params_schema'])}")
        if d["traps"]:
            lines.append(f"{'':<{width}}  traps: {', '.join(d['traps'])}")
    return cli_io.emit(_PROG, "\n".join(lines) + "\n")
