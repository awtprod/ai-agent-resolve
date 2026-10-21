"""``aar node`` commands: list the node op library."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
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

    run = sub.add_parser("run", help="Run the outbound-only durable job worker beside Resolve.")
    run.add_argument("--node-id", required=True)
    run.add_argument("--hub-url", required=True)
    run.add_argument("--token-file", required=True)
    run.add_argument("--store", required=True, help="Node JobStore SQLite path.")
    run.add_argument(
        "--project",
        action="append",
        required=True,
        help="Allowed ProjectLocator JSON object; repeat for each configured project.",
    )
    run.add_argument("--lease-seconds", type=float, default=60.0)
    run.add_argument("--heartbeat-seconds", type=float, default=15.0)
    run.add_argument("--tick-seconds", type=float, default=0.25)
    run.add_argument("--connect-timeout", type=float, default=10.0)
    run.add_argument("--read-timeout", type=float, default=10.0)
    run.add_argument("--response-cap", type=int, default=4 << 20)
    run.set_defaults(func=_cmd_run)


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


def _live_gateway() -> Any:
    """Construct the production gateway only on a Windows node at worker startup."""
    if os.name != "nt":
        raise RuntimeError("aar node run requires Windows and a running DaVinci Resolve Studio")
    import DaVinciResolveScript as dvr  # type: ignore[import-not-found]

    from aar.node.gateway import ScriptApiGateway

    return ScriptApiGateway(dvr.scriptapp("Resolve"))


def run_node(args: argparse.Namespace, *, gateway_factory: Any = _live_gateway) -> int:
    """Production loop with an injectable gateway factory for TCP acceptance tests."""
    from aar.node.jobs import JobStore
    from aar.node.locator import ProjectLocator
    from aar.node.remote import RemoteConfig, RemoteJobClient, _NodeClient
    from aar.node.runtime import NodeRuntime
    from aar.node.worker import RemoteWorker

    projects = tuple(ProjectLocator.from_dict(json.loads(raw)) for raw in args.project)
    ids = tuple(project.project_id for project in projects)
    if any(project_id is None for project_id in ids):
        raise ValueError("every --project locator needs project_id")
    config = RemoteConfig(
        hub_url=args.hub_url,
        token_file=args.token_file,
        connect_timeout=args.connect_timeout,
        read_timeout=args.read_timeout,
        response_cap=args.response_cap,
    )
    gateway = gateway_factory()
    store = JobStore(args.store)
    runtime = None
    worker = None
    stop = threading.Event()
    previous: dict[int, Any] = {}

    def stopping(signum: int, frame: Any) -> None:
        del signum, frame
        if worker is not None:
            worker.stop_claiming()
        stop.set()

    try:
        runtime = NodeRuntime.remote(
            gateway,
            store,
            node_id=args.node_id,
            hub_url=config.hub_url,
            token_file=config.token_file,
            connect_timeout=config.connect_timeout,
            read_timeout=config.read_timeout,
            response_cap=config.response_cap,
        )
        job_client = RemoteJobClient(
            _NodeClient(config),
            ids,
            lease_seconds=args.lease_seconds,  # type: ignore[arg-type]
        )
        worker = RemoteWorker(
            runtime,
            store,
            job_client,
            projects,
            heartbeat_seconds=args.heartbeat_seconds,
            tick_seconds=args.tick_seconds,
        )
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, stopping)
        code = worker.run(stop)
        if worker.permanently_stopped:
            cli_io.warn(f"aar node run stopped: {worker.stop_reason or 'permanent refusal'}")
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if worker is not None:
            closed = worker.close()
        else:
            closed = runtime is None or runtime.close()
            if closed:
                store.close()
        if not closed:
            cli_io.warn(
                "aar node run: shutdown incomplete; Resolve call still active. Store lock and "
                "node id retained until drain/checkpoint completes. Before restarting, inspect "
                "Resolve for uncertain side effects; human recovery may be required."
            )
    return code if closed else 2


def _cmd_run(args: argparse.Namespace) -> int:
    try:
        return run_node(args)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        cli_io.warn(f"aar node run: {exc}")
        return 2
