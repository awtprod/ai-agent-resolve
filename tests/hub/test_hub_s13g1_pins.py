"""S13g1 no-change pins: they pass on the base commit too, and must keep passing.

This module imports only APIs that exist on the base (925c5c5), so it can be run there.
"""

from __future__ import annotations

import json
from pathlib import Path

from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph

FIXTURE_LOG = Path(__file__).parent / "fixtures" / "graph_pre_s13g1.json"


def test_old_log_folds_to_the_same_state(tmp_path: Path) -> None:
    """No-change pin: a log written by the base code folds exactly as the base code folded it.

    The fixture (events plus the state the base fold produced) was generated on 925c5c5 by
    aar-evidence/s13g1/gen_graph_fixture.py before the graph changed. It passes on the base
    too; it guards the recipe kind (and add_node_in/add_edge_in) against changing old state.
    """
    fixture = json.loads(FIXTURE_LOG.read_text(encoding="utf-8"))
    log = EventLog(tmp_path / "replay.db")
    for event in fixture["events"]:
        log.append(
            event["type"],
            Actor(*event["actor"]),
            event["payload"],
            project_id=event["project_id"],
            idempotency_key=event["idempotency_key"],
            ts=event["ts"],
        )
    graph = WorkGraph(log, fixture["project_id"])
    snap = graph.snapshot()
    got = {
        **graph.to_dict(),
        "skipped": [s.to_dict() for s in graph.skipped()],
        "creators": {k: [v.kind, v.id] for k, v in sorted(snap.creators.items())},
        "last_change": {
            k: [seq, actor.kind, actor.id, to]
            for k, (seq, actor, to) in sorted(snap.last_change.items())
        },
    }
    expected = fixture["expected"]
    # Skipped event ids are minted by the log on append; compare everything else.
    for row in (*got["skipped"], *expected["skipped"]):
        row.pop("event_id")
    assert got == expected
    log.close()
