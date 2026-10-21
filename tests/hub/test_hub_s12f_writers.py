"""S12f item 3: the graph and registry writers refuse what their own fold would skip.

Before, such an event was committed and then either raised KeyError (the writer could
not find what it wrote) or was a silent no-op (rename, alias). Now it is refused with
ValueError and nothing is written.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import ProjectSource, Registry

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"title": 5},
        {"data": ["not", "an", "object"]},
        {"node_id": "n" * 200},
    ],
    ids=["title-not-text", "data-not-object", "node-id-too-long"],
)
def test_add_node_refuses_what_the_fold_would_skip(log: EventLog, kwargs: dict[str, Any]) -> None:
    graph = WorkGraph(log, "prj_demo")
    before = log.latest_seq()
    with pytest.raises(ValueError, match="refused before writing"):
        graph.add_node("op", AGENT, **kwargs)
    assert log.latest_seq() == before
    assert graph.nodes() == []


def test_add_edge_refuses_what_the_fold_would_skip(log: EventLog) -> None:
    graph = WorkGraph(log, "prj_demo")
    a = graph.add_node("op", AGENT).id
    b = graph.add_node("op", AGENT).id
    before = log.latest_seq()
    with pytest.raises(ValueError, match="refused before writing"):
        graph.add_edge("blocks", a, b, AGENT, data=["x"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refused before writing"):
        graph.add_edge("blocks", a, b, AGENT, edge_id="e" * 200)
    assert log.latest_seq() == before
    assert graph.edges() == []


def test_create_project_refuses_what_the_fold_would_skip(log: EventLog) -> None:
    registry = Registry(log)
    before = log.latest_seq()
    with pytest.raises(ValueError, match="refused before writing"):
        registry.create_project(5, HUMAN)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="refused before writing"):
        registry.create_project("demo-show", HUMAN, aliases=[5])  # type: ignore[list-item]
    assert log.latest_seq() == before
    assert registry.list() == []


@pytest.mark.parametrize("change", ["rename", "alias", "source"])
def test_rename_alias_and_source_are_refused_not_silent_no_ops(log: EventLog, change: str) -> None:
    registry = Registry(log)
    project = registry.create_project("demo-show", HUMAN)
    before = log.latest_seq()
    with pytest.raises(ValueError, match="refused before writing"):
        if change == "rename":
            registry.rename_project(project.id, 5, HUMAN)  # type: ignore[arg-type]
        elif change == "alias":
            registry.add_alias(project.id, ["acme"], HUMAN)  # type: ignore[arg-type]
        else:
            bad = ProjectSource("drp_file", ["not-a-locator"])  # type: ignore[arg-type]
            registry.add_source(project.id, bad, HUMAN)
    assert log.latest_seq() == before
    assert Registry(log).get(project.id) == project


def test_valid_writes_still_land(log: EventLog) -> None:
    registry = Registry(log)
    project = registry.create_project("demo-show", HUMAN, aliases=["acme"])
    registry.rename_project(project.id, "demo-show v2", HUMAN)
    registry.add_alias(project.id, "acme-2", HUMAN)
    renamed = registry.get(project.id)
    assert renamed is not None and renamed.name == "demo-show v2"
    assert renamed.aliases == ["acme", "acme-2"]
    graph = WorkGraph(log, project.id)
    node = graph.add_node("op", AGENT, title="build", data={"fps": 24})
    assert graph.set_status(node.id, "active", AGENT).status == "active"
