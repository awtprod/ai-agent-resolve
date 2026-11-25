"""Opt-in gate (S25c2): authored timeline items must reproduce Resolve's own, leave-one-out.

Set ``AAR_ITEM_PROJECTS`` to Resolve-saved ``Project.db`` files (separated by ``os.pathsep``).
Every picture and sound item of a supported shape is authored from its media clip's rows, its
own range, position, track and links, with another real item of its kind as the template (from
the same project, and in a second run from another project), and compared with Resolve's rows
field by field (:mod:`aar.drp.items.gate`); each sequence's extents are recomputed from its
items. Since S25e every sound transition of a supported shape is authored the same way from its
neighbours' cut, length, alignment and kind (:mod:`aar.drp.items.transition_gate`), with
templates of the same kind and of the other kind. Nothing from the private files is committed;
``scripts/items_scoreboard.py`` prints both scoreboards.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from aar.drp.items import gate
from aar.drp.library import LibraryOptions

PROJECTS = [Path(p) for p in os.environ.get("AAR_ITEM_PROJECTS", "").split(os.pathsep) if p]
needs_projects = pytest.mark.skipif(
    not PROJECTS, reason="set AAR_ITEM_PROJECTS to run the timeline-item gate"
)
_OPTS = LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)


@pytest.fixture(
    scope="module",
    params=[(False, False), (True, False), (False, True)],
    ids=["same-project", "other-project", "linked-templates"],
)
def board(request: pytest.FixtureRequest) -> gate.ItemScoreboard:
    other, linked = request.param
    result = gate.run(PROJECTS, options=_OPTS, other_project=other, linked_templates=linked)
    print("\n" + gate.format_scoreboard(result))
    return result


@needs_projects
def test_every_supported_item_reproduces_resolve(board: gate.ItemScoreboard) -> None:
    assert board.items_checked > 0
    for group in board.groups:
        assert group.status == "equal", (group.group, group.differing, group.refusals)
        assert group.fields == group.equal > 0


@needs_projects
def test_every_sequence_length_follows_its_items(board: gate.ItemScoreboard) -> None:
    assert board.extents_checked > 0
    assert board.extents_differing == []


@needs_projects
def test_every_exclusion_is_needed(board: gate.ItemScoreboard) -> None:
    from aar.drp.items.fields import ITEM_EXCLUDED

    hidden = {p for g in board.groups for p in g.hidden}
    unused = [e.pattern for e in ITEM_EXCLUDED if e.pattern not in hidden]
    assert unused == []


# -- S25e: sound transitions ---------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["same-kind", "other-kind"])
def transitions(request: pytest.FixtureRequest) -> Any:
    from aar.drp.items import transition_gate

    result = transition_gate.run(PROJECTS, options=_OPTS, other_kind=request.param)
    print("\n" + transition_gate.format_scoreboard(result))
    return result


@needs_projects
def test_every_supported_transition_reproduces_resolve(transitions: Any) -> None:
    """Each transition authored from its neighbours' cut, its length, alignment and kind
    alone, with another real transition as the template, equals Resolve's on every field."""
    assert transitions.transitions_checked > 0
    for group in transitions.groups:
        assert group.status == "equal", (group.group, group.differing, group.refusals)
        assert group.fields == group.equal > 0
    assert transitions.passed()


@needs_projects
def test_a_transition_leaves_its_neighbours_as_the_item_writer_makes_them(
    transitions: Any,
) -> None:
    """The neighbours of compared transitions, authored by the item writer (which knows
    nothing of transitions), equal Resolve's; and no transition reads media its neighbours do
    not have."""
    assert transitions.neighbours.status == "equal"
    assert transitions.violations == []


@needs_projects
def test_every_transition_exclusion_is_needed(transitions: Any) -> None:
    from aar.drp.items.transition_gate import TRANSITION_EXCLUDED

    hidden = {p for g in transitions.groups for p in g.hidden}
    assert [e.pattern for e in TRANSITION_EXCLUDED if e.pattern not in hidden] == []
