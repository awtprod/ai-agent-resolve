"""Seed a throwaway hub database with placeholder data, to look at the console.

Usage::

    python scripts/seed_console_demo.py DIR [--video demo.mp4]

It writes ``DIR/hub.db`` and ``DIR/artifacts``. Everything is placeholder data
(``acme``, ``demo-show``, ``agent-1``, ``operator-1``). One review title is an XSS
payload on purpose, to show it renders as text. Then run::

    aar hub console --db DIR/hub.db --artifacts DIR/artifacts --operator operator-1
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aar.hub import approvals, reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.leaseboard import record_lease_event, record_presence
from aar.hub.registry import Registry
from aar.node.leases import Holder, LeaseEvent, LeaseManager, Scope

AGENT_1 = Actor("agent", "agent-1")
AGENT_2 = Actor("agent", "agent-2")
OPERATOR = Actor("human", "operator-1")


def seed(root: Path, video: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root / "hub.db").exists():
        raise SystemExit(f"{root / 'hub.db'} already exists; use an empty directory")
    log = EventLog(root / "hub.db")
    store = ArtifactStore(root / "artifacts", log)
    registry = Registry(log)
    show = registry.create_project("demo-show", OPERATOR).id
    promo = registry.create_project("acme-promo", OPERATOR).id

    graph = WorkGraph(log, show)
    build = graph.add_node("op", AGENT_1, title="build long-form v3").id
    render = store.put(video, actor=AGENT_1, mime="video/mp4", project_id=show)

    xss = reviews.request_review(
        log,
        store,
        show,
        build,
        [render],
        AGENT_1,
        title='<script>alert("xss")</script> demo-show long-form v3',
        timeline="Main [agent:agent-1 v3]",
        fps=24,
    )
    reviews.add_note(log, show, xss.id, 24, "Jump cut on the same camera here.", AGENT_1)
    reviews.add_note(log, show, xss.id, 60, "Laugh tail is clipped; end 12 frames later?", AGENT_1)

    promo_graph = WorkGraph(log, promo)
    promo_op = promo_graph.add_node("op", AGENT_2, title="cut acme promo").id
    second = reviews.request_review(
        log,
        store,
        promo,
        promo_op,
        [render],
        AGENT_2,
        title="acme-promo 30s, captions pass",
        timeline="Promo [agent:agent-2 v1]",
        fps=24,
    )
    reviews.add_note(log, promo, second.id, 36, "Caption runs over 16 characters.", AGENT_2)

    done = reviews.request_review(
        log, store, show, build, [render], AGENT_1, title="demo-show cold open v1", fps=24
    )
    reviews.decide(log, show, done.id, "approve", "Clean. Ship the cold open.", OPERATOR)

    events: list[LeaseEvent] = []
    clock = {"now": 1000.0}
    manager = LeaseManager(clock=lambda: clock["now"], sink=events.append)
    manager.acquire(
        Holder.agent("agent-1"), Scope.TIMELINE, ("demo-show", "Main [agent:agent-1 v3]"), 900
    )
    manager.acquire(Holder.human("operator-1"), Scope.PROJECT, ("acme-promo",), None)
    clock["now"] = 1010.0
    manager.acquire(Holder.agent("agent-2"), Scope.INSTANCE, ("node-1",), 600)
    # The fake node clock read 1000 s a minute ago: map it onto wall time.
    offset = time.time() - 60 - 1000.0
    for event in events:
        record_lease_event(log, event, "node-1", wall_offset=offset)
    record_presence(
        log,
        "node-1",
        active=True,
        at=datetime.now(UTC) - timedelta(minutes=12),
        human_id="operator-1",
        ttl=timedelta(hours=2),
    )

    approvals.grant(
        log, "db.switch", "Local Database", timedelta(hours=1), "Sandbox round-trip", OPERATOR
    )
    used = approvals.grant(
        log, "project.load", show, timedelta(minutes=15), "Load for the colour pass", OPERATOR
    )
    approvals.consume(log, used.token_id, "project.load", show, Actor("node", "node-1"))
    stale = approvals.grant(
        log, "cloud.restore", "acme-archive", timedelta(hours=4), "Restore last week", OPERATOR
    )
    approvals.revoke(log, stale.token_id, "Restored by hand instead", OPERATOR)
    log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path, help="Empty directory for hub.db and artifacts/.")
    parser.add_argument("--video", type=Path, help="An MP4 to use as the render.")
    args = parser.parse_args(argv)
    video = args.video.read_bytes() if args.video else b"not a real video"
    seed(args.dir, video)
    print(f"seeded {args.dir / 'hub.db'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
