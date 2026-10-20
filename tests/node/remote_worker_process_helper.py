"""Subprocess-only fake gateway injection for the real node CLI loop; never live Resolve."""

from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from unittest.mock import patch

from aar.node.cli import run_node
from aar.node.gateway import ScriptApiGateway
from aar.node.runtime import NodeRuntime

from .fake_resolve import FakeClock, FakeResolve
from .jobs_support import IDLE, JOB_CONFIG, LAYOUT


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--observation", required=True)
    parser.add_argument("--block-third", action="store_true")
    args = parser.parse_args()
    journal = Path(args.observation)

    def events() -> list[dict]:
        return (
            [json.loads(line) for line in journal.read_text().splitlines()]
            if journal.exists()
            else []
        )

    def record(event: str) -> None:
        # fsync makes the observation survive SIGKILL independently of Python's buffers.
        with journal.open("a") as stream:
            stream.write(json.dumps({"event": event, "pid": os.getpid()}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    fake = FakeResolve(LAYOUT, current="Show", folder=("Shows",))
    project = fake.project("Show")
    timeline = project.timeline("Main")
    original_start = timeline.GetStartFrame

    def observed_start() -> int:
        completed = sum(event["event"] == "op_completed" for event in events())
        if args.block_third and completed == 2:
            record("op_blocked")
            if not threading.Event().wait(30):
                raise RuntimeError("test did not kill the blocked node within 30 seconds")
        result = original_start()
        record("op_completed")
        return result

    timeline.GetStartFrame = observed_start
    real_remote = NodeRuntime.remote

    def runtime_factory(*pos: object, **kw: object) -> NodeRuntime:
        clock = FakeClock()
        runtime = real_remote(
            *pos,
            **kw,
            clock=clock,
            broker_config=JOB_CONFIG,
            input_idle=None,
            idle_window=IDLE,
            drain_budget=None,
        )
        runtime.presence.observe(pos[0].get_context(), clock())
        clock.advance(IDLE)
        return runtime

    locator = {"folder_path": ["Shows"], "name": "Show", "project_id": project.GetUniqueId()}
    node_args = argparse.Namespace(
        node_id="edit-pc",
        hub_url=args.url,
        token_file=args.token,
        store=args.store,
        project=[json.dumps(locator)],
        lease_seconds=1,
        heartbeat_seconds=0.2,
        tick_seconds=0.02,
        connect_timeout=0.5,
        read_timeout=0.5,
        response_cap=4 << 20,
    )
    # Only the gateway and deterministic broker clock are test seams. run_node constructs
    # the production RemoteWorker/RemoteJobClient and handles signals/cleanup itself.
    with patch.object(NodeRuntime, "remote", side_effect=runtime_factory):
        return run_node(node_args, gateway_factory=lambda: ScriptApiGateway(fake))


if __name__ == "__main__":
    raise SystemExit(main())
