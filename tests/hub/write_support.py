"""Shared fixtures for the S13g1 hub write-surface tests."""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog, Transaction

PLAN_FIXTURES = Path(__file__).resolve().parents[1] / "plan" / "fixtures"
HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")
OTHER = Actor("agent", "agent-2")


def plan_fixture(name: str) -> dict[str, Any]:
    """A plan from tests/plan/fixtures (``short_clean`` lints clean, ``short_broken`` not)."""
    return json.loads((PLAN_FIXTURES / f"{name}.plan.json").read_text(encoding="utf-8"))


def brief_data(project_id: str, **changes: Any) -> dict[str, Any]:
    """A valid aar.brief/1 object for ``project_id``."""
    data: dict[str, Any] = {
        "project_id": project_id,
        "show": "Acme Comedy",
        "kind": "short",
        "recipe": "ccn-short",
        "rule_profile": "ccn",
        "sources": [
            {"kind": "cloud_project", "library": "Acme Library", "project": "Acme S01"},
            {"kind": "timeline", "unique_id": "5f0c-77aa-01", "name": "Main", "label": "show"},
            {"kind": "media", "path": "media/take1_camA.mov", "host": "edit-pc"},
        ],
        "deliverables": [
            {"name": "short-1", "kind": "short"},
            {"name": "thumb-1", "kind": "thumbnail"},
        ],
        "hook": "He never parks there again",
        "handle": "@acme",
        "title": "Parking Rant",
        "notes": "Keep the second laugh.\nNo whip-pans.",
    }
    data.update(changes)
    return copy.deepcopy(data)


@dataclass
class Hub:
    log: EventLog
    store: ArtifactStore
    root: Path
    project_id: str
    other_project_id: str

    def counts(self) -> Counter[str]:
        """How many events of each type the log holds."""
        return Counter(event.type for event in self.log.iter_events())

    def node_kinds(self) -> Counter[str]:
        """How many graph nodes of each kind were ever added."""
        return Counter(
            event.payload.get("kind")
            for event in self.log.iter_events(types=["graph.node_added"])
            if isinstance(event.payload, dict)
        )


def make_hub(tmp_path: Path) -> Hub:
    """A fresh hub with two registered projects, ``prj_acme`` and ``prj_other``."""
    from aar.hub.registry import Registry

    log = EventLog(tmp_path / "hub.db")
    store = ArtifactStore(tmp_path / "artifacts", log)
    registry = Registry(log)
    project = registry.create_project("acme-show", HUMAN, project_id="prj_acme")
    other = registry.create_project("acme-other", HUMAN, project_id="prj_other")
    return Hub(log, store, tmp_path, project.id, other.id)


class Crash(RuntimeError):
    """Stands in for a process dying between two events of one write."""


def crash_after_appends(monkeypatch: pytest.MonkeyPatch, n: int) -> list[str]:
    """Make the ``n``-th event appended (by a transaction or directly) raise ``Crash``.

    The event itself is written first, as a real crash after it would leave it; whether
    it survives is then up to the transaction around it. Returns the types appended so far.
    """
    seen: list[str] = []
    real_txn_append = Transaction.append
    real_log_append = EventLog.append

    def counting(real: Any) -> Any:
        def append(self: Any, event_type: str, *args: Any, **kwargs: Any) -> Any:
            event = real(self, event_type, *args, **kwargs)
            seen.append(event_type)
            if len(seen) == n:
                raise Crash(f"crash after event {n} ({event_type})")
            return event

        return append

    monkeypatch.setattr(Transaction, "append", counting(real_txn_append))
    monkeypatch.setattr(EventLog, "append", counting(real_log_append))
    return seen
