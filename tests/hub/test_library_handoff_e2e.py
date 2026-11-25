"""S25d end to end, offline: the ``ccn-longform`` recipe hands the cut over as a library project
beside the FCPXML; the "owner" edits the placed ``Project.db`` (row edits) and saves; the
worker's sweep stores each save as a return exactly as a ``.drt`` return is stored."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from ccn_recipe_support import (
    PC_ROOT,
    WORKER,
    CcnEnv,
    agent_cut,
    make_run,
    outputs,
    start,
    work,
)
from library_handoff_support import (
    blade,
    delete,
    make_library_setup,
    pictures,
    save,
    swap_first_two,
    trim_tail,
)

from aar.recipes.desktop_returns import returned_edits
from aar.recipes.library_returns import EVENT_PICKUP, Limits, scan_library_returns
from aar.recipes.run import recipe_status
from ccn_support import needs_ffmpeg

pytestmark = needs_ffmpeg
pytest.importorskip("zstandard")

#: One continuous range: one segment, no join, so no audio crossfade (which a library project
#: cannot carry yet: the ccn profile puts one on every other join).
ONE = {"ranges": [{"start_s": 0.9, "end_s": 11.1}]}
THREE = {
    "ranges": [
        {"start_s": 0.9, "end_s": 4.1},
        {"start_s": 4.9, "end_s": 8.1},
        {"start_s": 8.9, "end_s": 11.1},
    ]
}
FAST = Limits(settle_seconds=0.0)


def _run(env: CcnEnv, keep: dict[str, Any] = ONE, **lib: Any) -> tuple[str, dict[str, Any]]:
    show = env.show("cam-a.mp4")
    library = make_library_setup(env.media / "share", [f"{PC_ROOT}\\cam-a.mp4"], **lib)
    run = make_run(env, env.params(show, library=library), key="lib")
    start(env, run, "lib")
    assert work(env) == 5
    agent_cut(env, run, keep=keep, key="r1")
    assert work(env) == 2  # render, export
    return run, outputs(env, run, "export")


def sweep_once(env: CcnEnv, limits: Limits = FAST) -> Any:
    return scan_library_returns(env.hub.log, env.hub.store, WORKER, limits=limits)


def _diff(env: CcnEnv, ret: dict[str, Any]) -> dict[str, Any]:
    return json.loads(env.hub.store.read_bytes(ret["diff"]))


def test_a_cut_is_handed_over_as_a_library_project_and_each_save_comes_back(
    ccn_env: CcnEnv,
) -> None:
    env = ccn_env
    run, export = _run(env)
    handoff = export["library_handoff"]
    assert handoff["status"] == "placed", handoff["error"]["message"]
    assert "warnings" not in export
    project = Path(handoff["library"]).joinpath(*handoff["parts"])
    db = project / "Project.db"
    assert handoff["folder"] == "Synthetic Show - Synthetic Long v1 r1" == project.name
    assert handoff["timeline"] == "Synthetic Long v1"
    assert handoff["plan_sha256"] == outputs(env, run, "cut")["plan_sha256"]
    assert hashlib.sha256(db.read_bytes()).hexdigest() == handoff["placed_sha256"]
    report = json.loads(env.hub.store.read_bytes(handoff["report"]))
    assert report["project"]["project_id"] == handoff["project_id"]
    assert len(pictures(db)) == 1

    # The FCPXML hand-off is published as before, and the notes name the library project.
    copy = export["copy"]
    assert Path(copy["fcpxml"]).is_file() and Path(copy["review_mp4"]).is_file()
    notes = Path(copy["handoff_notes"]).read_text(encoding="utf-8")
    assert 'shared project library "Shared Library" as the project ' in notes
    assert '"Synthetic Show - Synthetic Long v1 r1"' in notes
    assert "Saving is all that" in notes and "nothing to export" in notes
    assert "add_media" in notes  # the operations nobody has yet seen open in Resolve

    # Nothing saved yet: the sweep stores nothing.
    sweep = sweep_once(env)
    assert (sweep.checked, sweep.unchanged, sweep.returned) == (1, 1, 0), sweep
    assert returned_edits(env.hub.log, env.hub.project_id, run) == []

    # The owner trims the clip's tail by 12 frames and saves.
    save(db, trim_tail(12))
    sweep = sweep_once(env)
    assert sweep.returned == 1, sweep
    [first] = returned_edits(env.hub.log, env.hub.project_id, run)
    assert first["source"] == "library_save" and first["status"] == "needs_review"
    assert first["base_plan"] == handoff["plan_sha256"] and first["round"] == 1
    assert env.hub.store.read_bytes(first["project_db"]) == db.read_bytes()
    [segment] = _diff(env, first)["segments"]
    old = {"src_in": 27, "src_out": 332, "record_in": 0, "record_out": 305}
    assert segment["status"] == "trimmed" and segment["old"] == old
    assert (segment["head_frames"], segment["tail_frames"]) == (0, 12)
    assert [(p["src_in"], p["src_out"], p["record_in"]) for p in segment["new"]] == [(27, 320, 0)]
    whole = _diff(env, first)
    assert not whole["not_carried"] and not whole["added"] and not whole["foreign"]

    # The same file again: nothing new.
    seq = env.hub.log.latest_seq()
    sweep = sweep_once(env)
    assert sweep.repeated == 1 and sweep.returned == 0
    assert env.hub.log.latest_seq() == seq

    # A second save cuts 60 frames out of the middle; a third swaps the two pieces.
    save(db, blade(0, 100), blade(1, 60), delete(1))
    sweep_once(env)
    save(db, swap_first_two())
    sweep_once(env)
    edits = returned_edits(env.hub.log, env.hub.project_id, run)
    assert len(edits) == 3 and len({e["project_db"] for e in edits}) == 3

    def pieces(ret: dict[str, Any]) -> list[tuple[int, int, int, int]]:
        [seg] = _diff(env, ret)["segments"]
        assert seg["status"] == "split" and seg["old"] == old
        return [(p["src_in"], p["src_out"], p["record_in"], p["record_out"]) for p in seg["new"]]

    # 60 frames (source 127..187) are gone and the rest closed up ...
    assert pieces(edits[1]) == [(27, 127, 0, 100), (187, 320, 100, 233)]
    # ... and then the two pieces swap places.
    assert pieces(edits[2]) == [(187, 320, 0, 133), (27, 127, 133, 233)]
    # Each return's edited plan is the timeline as saved, and the active plan is unchanged.
    plan2 = json.loads(env.hub.store.read_bytes(edits[2]["plan"]))
    assert [(s["src_in"], s["src_out"]) for s in plan2["segments"]] == [(187, 320), (27, 127)]
    assert outputs(env, run, "cut")["plan_sha256"] == handoff["plan_sha256"]

    # The status view shows them as it shows .drt returns, labelled.
    view = recipe_status(env.hub.log, run)
    assert [e["source"] for e in view["returned_edits"]] == ["library_save"] * 3
    assert not list(env.hub.log.iter_events(types=[EVENT_PICKUP]))


def test_a_cut_the_library_cannot_carry_keeps_its_fcpxml_and_says_why(ccn_env: CcnEnv) -> None:
    """Three ranges: the ccn profile puts an audio crossfade on each join, which a library
    project cannot carry yet. The export still succeeds on the FCPXML; the refusal is a
    warning in the step's result and in the notes."""
    env = ccn_env
    run, export = _run(env, keep=THREE)
    handoff = export["library_handoff"]
    assert handoff["status"] == "refused"
    assert handoff["error"]["code"] == "unsupported_plan_feature"
    [warning] = export["warnings"]
    assert warning["code"] == "library_unsupported_plan_feature"
    assert warning["fix_hint"] == handoff["error"]["hint"]
    copy = export["copy"]
    assert Path(copy["fcpxml"]).is_file()
    notes = Path(copy["handoff_notes"]).read_text(encoding="utf-8")
    assert (
        "\nLIBRARY PROJECT: NOT WRITTEN\n  The cut was not written into the project library "
        '"Shared Library":\n' in notes
    )
    assert "[unsupported_plan_feature]" in notes and "import the FCPXML" in notes
    library = Path(handoff["library"]) / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(library.iterdir()) == []
    # Nothing to pick up: a refused hand-off is never watched.
    sweep = sweep_once(env)
    assert (sweep.checked, sweep.returned) == (0, 0)
    assert recipe_status(env.hub.log, run)["status"] == "active"


def test_with_the_opt_ins_off_a_cut_needing_them_gets_the_refusal_in_its_notes(
    ccn_env: CcnEnv,
) -> None:
    env = ccn_env
    _run_id, export = _run(env, allow_unopened_operations=False, allow_unverified_media=False)
    handoff = export["library_handoff"]
    assert handoff["status"] == "refused"
    # the show's media is outside its kind's evidence: refused before anything is written
    assert handoff["error"]["code"] == "media_refused", handoff["error"]
    assert "allow" in handoff["error"]["hint"].lower()
    notes = Path(export["copy"]["handoff_notes"]).read_text(encoding="utf-8")
    assert "LIBRARY PROJECT: NOT WRITTEN" in notes and "[media_refused]" in notes
    assert export["warnings"][0]["code"] == "library_media_refused"


def test_with_media_allowed_the_unopened_operations_are_the_refusal(ccn_env: CcnEnv) -> None:
    env = ccn_env
    _run_id, export = _run(env, allow_unopened_operations=False)
    handoff = export["library_handoff"]
    assert handoff["error"]["code"] == "operation_not_opened", handoff["error"]
    notes = Path(export["copy"]["handoff_notes"]).read_text(encoding="utf-8")
    assert "[operation_not_opened]" in notes
