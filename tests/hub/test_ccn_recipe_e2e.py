"""``ccn-longform`` end to end, offline (S19a): register, create, start, the worker to the agent
step, the cut done as its instruction says, render, QA, export, the review gate rejected once and
then approved, and the report. No Resolve, no network, no model (the transcriber is injected)."""

from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

import pytest
from ccn_recipe_support import PC_ROOT, CcnEnv, agent_cut, latest, make_run, outputs, start, work
from write_support import HUMAN

from aar.hub.reviews import decide, get_review
from aar.media import probe
from aar.plan.export import media_map_from_mapping, verify_plan_fcpxml
from aar.plan.model import EditPlan
from aar.plan.profile import load_profile
from aar.recipes.state import load_state
from ccn_support import needs_ffmpeg

pytestmark = needs_ffmpeg


def _review_id(env: CcnEnv, recipe_node: str) -> str:
    op = latest(env, recipe_node, "review")
    assert op is not None and op.status == "active" and op.review_id is not None
    return op.review_id


def test_ccn_longform_runs_end_to_end(ccn_env: CcnEnv, tmp_path: Path) -> None:
    env = ccn_env
    show = env.show("cam-a.mov")
    run = make_run(env, env.params(show), key="e2e")
    start(env, run, "e2e")
    assert work(env) == 5
    assert latest(env, run, "cut").status == "active"

    # Someone else's file already holds the first export name: it is never overwritten.
    folder = env.exports / "Synthetic Show" / "Synthetic Long v1"
    folder.mkdir(parents=True)
    (folder / "Synthetic Long v1.r1.fcpxml").write_bytes(b"not ours")

    plan_node = agent_cut(env, run, key="r1")
    assert work(env) == 2  # render, export
    render = outputs(env, run, "render")
    assert render["subject_node"] == plan_node
    assert render["artifact_ids"] == [render["review_mp4"]]
    assert render["qa"]["passed"] is True and render["qa_status"] in ("pass", "warn")
    assert {r["name"] for r in render["qa"]["reasons"]} >= {"decode", "loudness", "size-cap"}
    assert "stays on the hub" in render["size_cap"]
    assert set(render["qa_not_required"]) == {"join-retranscribe", "caption-coverage"}
    mp4 = tmp_path / "review.mp4"
    mp4.write_bytes(env.hub.store.read_bytes(render["review_mp4"]))
    info = probe(mp4)
    assert info.video is not None and info.audio is not None
    assert info.video.codec == "h264" and (info.video.width, info.video.height) == (320, 180)
    assert abs((info.video.duration_s or 0) - 160 * 1001 / 30000) < 0.05  # 2 ranges, 160 frames
    assert not (env.work / "aar-render").exists() or not list((env.work / "aar-render").iterdir())

    export = outputs(env, run, "export")
    export_report = json.loads(env.hub.store.read_bytes(export["export_report"]))
    copy = export["copy"]
    assert copy["copied"] is True and copy["round"] == 1 and copy["dir"] == str(folder)
    assert copy["fcpxml"] == str(folder / "Synthetic Long v1.r1-2.fcpxml")
    assert copy["review_mp4"] == str(folder / "Synthetic Long v1.r1-2.review.mp4")
    assert (folder / "Synthetic Long v1.r1.fcpxml").read_bytes() == b"not ours"
    fcpxml = env.hub.store.read_bytes(export["fcpxml"])
    assert Path(copy["fcpxml"]).read_bytes() == fcpxml
    assert Path(copy["review_mp4"]).read_bytes() == mp4.read_bytes()
    # The handoff notes travel with the timeline (the taken .fcpxml name moved the whole set).
    assert copy["handoff_notes"] == str(folder / "Synthetic Long v1.r1-2.notes.txt")
    notes = Path(copy["handoff_notes"]).read_bytes()
    assert notes == env.hub.store.read_bytes(export["handoff_notes"])
    text = notes.decode("utf-8")
    assert text.startswith("HANDOFF NOTES: Synthetic Long v1, round 1 (project Synthetic Show)")
    assert "  - E:\\Shows\\cam-a.mov\r\n" in text and "they are NOT checked" in text
    gain = render["loudness"]["gain_db"]
    assert f"{abs(gain):g} dB to reach {render['loudness']['target_lufs']:g} LUFS" in text
    assert "NOT IN THE FCPXML: REDO IN RESOLVE\r\n  Nothing: the FCPXML carries every" in text
    assert len(export_report["not_expressible"]) == 0  # this plan: nothing to redo
    assert sorted(p.name for p in folder.iterdir()) == [
        "Synthetic Long v1.r1-2.fcpxml",
        "Synthetic Long v1.r1-2.notes.txt",
        "Synthetic Long v1.r1-2.review.mp4",
        "Synthetic Long v1.r1.fcpxml",
    ]
    assert export["assets"] == ["file:///E:/Shows/cam-a.mov"]
    plan_sha = outputs(env, run, "cut")["plan_sha256"]
    plan = EditPlan.model_validate_json(env.hub.store.read_bytes(plan_sha))
    check = verify_plan_fcpxml(
        fcpxml.decode("utf-8"),
        plan,
        load_profile("ccn"),
        media_map=media_map_from_mapping({str(env.media): PC_ROOT}),
    )
    assert check.ok, check.problems
    assert plan.sources[0].start_timecode == 108000  # 01:00:00:00 from the proxy's tmcd track
    assert 'start="108108000/30000s"' in fcpxml.decode()  # the asset at the file's timecode

    # The reviewer is shown the notes with the video: the gate's review carries them.
    review1 = _review_id(env, run)
    shown = get_review(env.hub.log, env.hub.project_id, review1)
    assert shown is not None and shown.renders == (render["review_mp4"],)
    assert shown.attachments == (export["handoff_notes"],)
    # ... and knows what it is of and at what rate, so a note's frame comes from the player
    assert shown.timeline == "Synthetic Long v1"
    assert shown.fps == plan.fps.num / plan.fps.den == 30000 / 1001

    # Round 1 is rejected: the run goes back to the cut, never to the transcript.
    decide(env.hub.log, env.hub.project_id, review1, "reject", "end on the laugh", HUMAN)
    work(env)
    state = load_state(env.hub.log, run)
    assert state.current_round() == 2 and latest(env, run, "cut").status == "active"
    keep2 = {"ranges": [{"start_s": 0.9, "end_s": 4.1}, {"start_s": 4.9, "end_s": 8.1}]}
    agent_cut(env, run, keep=keep2, derives_from=review1, key="r2")
    assert work(env) == 2
    export2 = outputs(env, run, "export")
    copy2 = export2["copy"]
    assert copy2["round"] == 2
    # Round 2's review carries round 2's notes, not round 1's.
    assert export2["handoff_notes"] != export["handoff_notes"]
    assert b", round 2 (" in env.hub.store.read_bytes(export2["handoff_notes"])
    shown2 = get_review(env.hub.log, env.hub.project_id, _review_id(env, run))
    assert shown2 is not None and shown2.attachments == (export2["handoff_notes"],)
    assert copy2["fcpxml"] == str(folder / "Synthetic Long v1.r2.fcpxml")
    assert Path(copy["fcpxml"]).exists()  # round 1's copy stays

    # Round 2 is approved: the report closes the run.
    decide(env.hub.log, env.hub.project_id, _review_id(env, run), "approve", "ship it", HUMAN)
    assert work(env) == 1
    state = load_state(env.hub.log, run)
    assert state.recipe_status == "done"
    report = env.hub.store.read_bytes(outputs(env, run, "report")["report_sha256"]).decode()
    assert "ccn-longform" in report and "approve" in report
    assert len(env.calls) == 1  # transcribed once for the whole run
    # the rejection went back to the cut only: one attempt each of the steps before it
    for step in ("stub", "sync", "transcript", "laughs", "guide"):
        assert len(state.attempts_for(step)) == 1, step
    assert len(state.attempts_for("cut")) == 2


@pytest.mark.parametrize("with_script", [True, False], ids=["aar-script", "python-m-aar"])
def test_the_guide_commands_run_as_printed_from_an_interpreter_dir_with_a_space(
    ccn_env: CcnEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_script: bool
) -> None:
    """MED-1 (r2): the worker's interpreter lives in a folder whose name has a space and ``aar``
    is not on PATH. ``read_guide`` and every guide command name that interpreter's own ``aar``
    (or ``python -m aar`` when there is no script beside it), quoted, and run in ``sh`` as
    printed; the wrappers log that they were the programs run."""
    env = ccn_env
    bin_dir = tmp_path / "py env" / "bin"
    bin_dir.mkdir(parents=True)
    log = tmp_path / "ran.log"
    real_aar = Path(sys.executable).with_name("aar")
    wrappers = {"python": sys.executable, **({"aar": str(real_aar)} if with_script else {})}
    for name, target in wrappers.items():
        wrapper = bin_dir / name
        wrapper.write_text(
            f'#!/bin/sh\necho {name} >> {shlex.quote(str(log))}\nexec {shlex.quote(target)} "$@"\n'
        )
        wrapper.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    monkeypatch.setenv("PATH", "/usr/bin:/bin")  # no aar anywhere on PATH

    show = env.show("cam-a.mov")
    run = make_run(env, env.params(show), key="space")
    start(env, run, "space")
    assert work(env) == 5
    read = outputs(env, run, "guide")["read_guide"]
    want = [str(bin_dir / "aar")] if with_script else [str(bin_dir / "python"), "-m", "aar"]
    assert shlex.split(read)[: len(want)] == want
    assert read.startswith(shlex.quote(want[0]) + " ")  # quoted: the folder has a space
    plan_node = agent_cut(env, run, key="r1")
    ran = log.read_text().split()
    assert ran.count("aar" if with_script else "python") >= 8  # guide, fetches, produce...
    assert work(env) == 2  # render, export
    assert outputs(env, run, "render")["subject_node"] == plan_node
