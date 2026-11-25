"""Opt-in gate (S25c1): authored media-pool rows must reproduce Resolve's own, leave-one-out.

Set ``AAR_MEDIA_FACTS`` to the facts file (JSON lines, one per file, as the edit PC's facts script
writes them) and ``AAR_MEDIA_PROJECTS`` to the Resolve-saved ``Project.db`` files that use those
files (separated by ``os.pathsep``). For every file with facts, its rows are authored from the
facts with another file of the same kind as the template and compared with Resolve's rows field
by field (:mod:`aar.drp.mediapool.gate`). Every kind :data:`aar.drp.mediapool.kinds.KINDS`
declares ``verified`` must be equal on every compared field for as many files as it claims;
every ``unverified`` kind must have one member whose own derived fields equal Resolve's. Nothing
from the private files is committed; ``scripts/media_scoreboard.py`` prints the scoreboard.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aar.drp.library import LibraryOptions
from aar.drp.mediapool import gate
from aar.drp.mediapool.kinds import KINDS

FACTS = os.environ.get("AAR_MEDIA_FACTS")
PROJECTS = [Path(p) for p in os.environ.get("AAR_MEDIA_PROJECTS", "").split(os.pathsep) if p]
needs_key = pytest.mark.skipif(
    not FACTS or not PROJECTS,
    reason="set AAR_MEDIA_FACTS and AAR_MEDIA_PROJECTS to run the media-pool gate",
)


@pytest.fixture(scope="module")
def board() -> gate.Scoreboard:
    assert FACTS is not None
    facts = gate.load_facts(Path(FACTS))
    result = gate.run(
        PROJECTS, facts, options=LibraryOptions(settle_seconds=0.0, deadline_seconds=0.0)
    )
    print("\n" + gate.format_scoreboard(result))
    return result


@needs_key
def test_every_verified_kind_reproduces_resolve(board: gate.Scoreboard) -> None:
    scores = {s.kind: s for s in board.kinds}
    for kind, status in KINDS.items():
        if status.status != "verified":
            continue
        score = scores.get(kind)
        assert score is not None, f"no file of the verified kind {kind.label()}"
        assert score.status == "equal", (kind.label(), score.differing, score.refusals)
        assert score.checked == status.files, (kind.label(), score.checked, status.files)
        assert score.fields == score.equal > 0


@needs_key
def test_every_unverified_kind_is_one_file_whose_derived_fields_match(
    board: gate.Scoreboard,
) -> None:
    scores = {s.kind: s for s in board.kinds}
    for kind, status in KINDS.items():
        if status.status == "verified":
            continue
        score = scores[kind]
        assert score.status == "single" and status.files == 1
        assert score.self_check is not None and score.self_check.startswith("equal"), (
            score.self_check
        )


@needs_key
def test_no_kind_that_differs_is_declared(board: gate.Scoreboard) -> None:
    for score in board.kinds:
        if score.status in ("differs", "refused"):
            assert score.kind not in KINDS, score.kind.label()
