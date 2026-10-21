"""S12g: tests that pin the four behaviours the S12f validator's mutants removed without a red.

Each test names the mutant id it pins (F08, F10, F13, F20). The code was already right;
these just give each behaviour a test that fails when it is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aar.hub import reviews
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.hub.reviews import is_timecode

HUMAN = Actor("human", "operator-1")
AGENT = Actor("agent", "agent-1")

FILLER = 1100  # more than the 1000 events a default log.read() returns


@pytest.fixture
def log(tmp_path: Path) -> EventLog:
    return EventLog(tmp_path / "hub.db")


# -- F13: the review snapshot reads the whole project, not the first 1000 events ----------


def test_F13_a_review_requested_in_a_project_with_over_a_thousand_events_is_found(
    tmp_path: Path, log: EventLog
) -> None:
    store = ArtifactStore(tmp_path / "artifacts", log)
    pid = Registry(log).create_project("demo-show", HUMAN).id
    op = WorkGraph(log, pid).add_node("op", AGENT, title="build").id
    # One transaction, so the filler costs one commit. Each node is a valid graph event
    # that the review snapshot reads.
    with log.transaction() as txn:
        for i in range(FILLER):
            payload = {"node_id": f"nd_f{i}", "kind": "op", "status": "pending", "title": "t"}
            txn.append("graph.node_added", AGENT, {**payload, "data": {}}, project_id=pid)
    assert len(log.read(0, project_id=pid, limit=-1)) > 1000
    render = store.put(b"render", actor=AGENT, mime="video/mp4", project_id=pid)

    review = reviews.request_review(log, store, pid, op, [render], AGENT, title="cut v1", fps=24)

    # It is the newest event of the project, far past a 1000-event window.
    read_back, skipped = reviews.review_view(log, pid, review.id)
    assert read_back is not None and read_back.id == review.id and skipped == ()
    assert [r.id for r in reviews.scan_reviews(log, pid).reviews] == [review.id]


# -- F10: a refresh leaves _last_seq at the head, so a writer does not re-fold history ---


def test_F10_a_new_graph_starts_at_the_log_head_and_a_writer_does_not_refold(
    log: EventLog,
) -> None:
    pid = Registry(log).create_project("demo-show", HUMAN).id
    writer = WorkGraph(log, pid)
    node = writer.add_node("op", AGENT, title="build").id
    writer.set_status(node, "active", AGENT)  # the log head is now a graph event

    graph = WorkGraph(log, pid)
    assert graph._last_seq == log.latest_seq()
    graph.add_node("op", AGENT, title="second")
    graph.set_status(node, "done", AGENT)
    assert graph.skipped() == ()  # no "already exists", no "illegal transition"
    assert graph._last_seq == log.latest_seq()


def test_F10_refresh_catches_up_to_the_head_without_refolding(log: EventLog) -> None:
    pid = Registry(log).create_project("demo-show", HUMAN).id
    graph = WorkGraph(log, pid)
    node = graph.add_node("op", AGENT, title="build").id
    # Another writer on the same log moves the head past this graph's view.
    WorkGraph(log, pid).add_node("op", AGENT, title="elsewhere")
    assert graph._last_seq < log.latest_seq()

    graph.refresh()
    assert graph._last_seq == log.latest_seq()
    graph.set_status(node, "active", AGENT)
    assert graph.skipped() == ()
    assert len(graph.nodes()) == 2


# -- F08: from_events ignores the events of other projects and other types ---------------


def test_F08_from_events_ignores_other_projects_even_when_they_are_passed_in(
    log: EventLog,
) -> None:
    registry = Registry(log)
    mine = registry.create_project("mine", HUMAN).id
    theirs = registry.create_project("theirs", HUMAN).id
    # The two projects share node ids, so a graph that folded both would also list
    # "already exists" skips and overwrite the title.
    my_graph, their_graph = WorkGraph(log, mine), WorkGraph(log, theirs)
    shared = my_graph.add_node("op", AGENT, title="mine", node_id="nd_shared").id
    their_graph.add_node("op", AGENT, title="theirs", node_id="nd_shared")
    only_mine = my_graph.add_node("op", AGENT, title="only mine").id
    their_only = their_graph.add_node("op", AGENT, title="only theirs").id
    my_graph.add_edge("blocks", shared, only_mine, AGENT)
    their_graph.add_edge("blocks", shared, their_only, AGENT)

    everything = log.read(0, limit=-1)
    assert {e.project_id for e in everything} == {mine, theirs}

    graph = WorkGraph.from_events(log, mine, everything)
    assert sorted(n.id for n in graph.nodes()) == sorted([shared, only_mine])
    assert graph.node(shared) is not None and graph.node(shared).title == "mine"  # type: ignore[union-attr]
    assert [(e.src, e.dst) for e in graph.edges()] == [(shared, only_mine)]
    # Nothing was folded and rejected: the registry events of this very project (same
    # project id, other type) and the other project's graph events never reached the fold.
    assert graph.skipped() == ()
    mine_graph_events = [
        e for e in everything if e.project_id == mine and e.type.startswith("graph.")
    ]
    assert graph._last_seq == mine_graph_events[-1].seq


def test_F08_a_graph_built_from_events_still_writes_without_refolding(log: EventLog) -> None:
    pid = Registry(log).create_project("demo-show", HUMAN).id
    first = WorkGraph(log, pid).add_node("op", AGENT, title="build").id

    graph = WorkGraph.from_events(log, pid, log.read(0, limit=-1))
    graph.add_node("op", AGENT, title="after")
    assert graph.skipped() == ()
    assert sorted(n.title or "" for n in graph.nodes()) == ["after", "build"]
    assert graph.node(first) is not None


# -- F20: a timecode is a full match -----------------------------------------------------


def test_F20_a_timecode_with_a_trailing_newline_is_not_a_timecode() -> None:
    # "$" matches before a final newline, so only fullmatch rejects this.
    assert not is_timecode("01:00:12:05\n")
    assert not is_timecode("01:00:12;05\n")
    assert is_timecode("01:00:12:05") and is_timecode("01:00:12;05")


@pytest.mark.parametrize("value", [5, None, b"01:00:12:05", ["01:00:12:05"], 1.5])
def test_F20_a_value_that_is_not_text_is_not_a_timecode(value: object) -> None:
    # Not a TypeError from the regex: the answer is False.
    assert is_timecode(value) is False
