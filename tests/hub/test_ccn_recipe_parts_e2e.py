"""``ccn-longform`` end to end with a reference camera of three files (S24a): two meeting with
no gap, then a 24-frame gap. Stub, transcript, the agent's cut as the guide says (one range
across the frame-contiguous seam, one across the gap), render, QA and export. No Resolve, no
network, no model (the transcriber is injected)."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

from ccn_recipe_support import PC_ROOT, CcnEnv, agent_cut, latest, make_run, outputs, start, work

from aar.hub.reviews import get_review
from aar.media import probe
from aar.plan.export import media_map_from_mapping, verify_plan_fcpxml
from aar.plan.inputs import Transcript
from aar.plan.model import EditPlan
from aar.plan.profile import load_profile
from ccn_support import WORDS, make_parts, needs_ffmpeg

pytestmark = needs_ffmpeg

#: 360 frames a file at 30000/1001: part 1 is recording 0-360, part 2 360-720 (no gap), then
#: 24 frames with no file, part 3 744-1104.
GAPS = [0, 24]
#: One range across the seam of parts 1 and 2 (8.9-16.2 s), one across the gap (20.9-29.0 s).
KEEP = {"ranges": [{"start_s": 8.9, "end_s": 16.2}, {"start_s": 20.9, "end_s": 29.0}]}


def parts_params(env: CcnEnv, paths: list[Path], **changes: Any) -> dict[str, Any]:
    params = env.params(*paths, **changes)
    for n, source in enumerate(params["sources"]):
        source.update(camera="A", part=n + 1)
    return params


def _seconds(frames: int) -> float:
    return float(Fraction(frames * 1001, 30000))


def test_a_reference_camera_of_three_files_runs_end_to_end(ccn_env: CcnEnv, tmp_path: Path) -> None:
    env = ccn_env
    files = make_parts(env.media, GAPS)
    paths = [path for path, _ in files]
    run = make_run(env, parts_params(env, paths), key="parts")
    start(env, run, "parts")
    assert work(env) == 5
    assert latest(env, run, "cut").status == "active"

    stub = outputs(env, run, "stub")
    assert stub["cameras"] == 1 and stub["camera_labels"] == ["A"]
    assert stub["pause_trim_flag"] == "--no-pause-trim"  # one camera's room audio
    assert stub["reference"] == "cam-a-p1" and stub["part_placement"] == "timecode"
    assert [(p["id"], p["start_s"], p["end_s"], p["gap_s"]) for p in stub["parts"]] == [
        ("cam-a-p1", 0.0, round(_seconds(360), 3), 0.0),
        ("cam-a-p2", round(_seconds(360), 3), round(_seconds(720), 3), 0.0),
        ("cam-a-p3", round(_seconds(744), 3), round(_seconds(1104), 3), round(_seconds(24), 3)),
    ]
    plan_stub = EditPlan.model_validate_json(env.hub.store.read_bytes(stub["stub"]))
    assert [(s.id, s.part, s.recording_origin) for s in plan_stub.sources] == [
        ("cam-a-p1", 1, 0),
        ("cam-a-p2", 2, 360),
        ("cam-a-p3", 3, 744),
    ]
    assert outputs(env, run, "sync")["synced"] is False  # one camera: nothing to sync

    # Every part transcribed at its own offset, merged into one recording-time transcript.
    assert sorted(c["media"] for c in env.calls) == sorted(str(p) for p in paths)
    tx = outputs(env, run, "transcript")
    assert [p["source"] for p in tx["parts"]] == ["cam-a-p1", "cam-a-p2", "cam-a-p3"]
    assert [p["cached"] for p in tx["parts"]] == [False, False, False]
    merged = Transcript.model_validate_json(env.hub.store.read_bytes(tx["transcript"]))
    assert len(merged.words) == 3 * len(WORDS) == tx["words"]
    starts = [w.start_s for w in merged.words if w.text == "So"]
    assert starts == [1.0, 1.0 + _seconds(360), 1.0 + _seconds(744)]
    text = env.hub.store.read_bytes(tx["text"]).decode()
    assert "-- part 2 begins (no gap) --" in text
    assert "-- gap 0.8 s (no file) between part 2 and part 3 --" in text

    # The guide lays out the files and the gap for the agent.
    from ccn_recipe_support import read_guide

    guide = read_guide(env, run)
    assert "cameras: A = `cam-a-p1`, `cam-a-p2`, `cam-a-p3`" in guide
    assert "Camera A is 3 files" in guide and "- gap 0.801 s: no file" in guide
    assert "--schedule" not in guide  # one camera, no schedule

    agent_cut(env, run, keep=KEEP, key="r1")
    assert work(env) == 2  # render, export
    render = outputs(env, run, "render")
    assert render["qa"]["passed"] is True, render["qa"]

    plan = EditPlan.model_validate_json(
        env.hub.store.read_bytes(outputs(env, run, "cut")["plan_sha256"])
    )
    segs = [(s.source, s.src_in, s.src_out, s.transform) for s in plan.segments]
    # Range 1 crosses the seam of parts 1 and 2: two segments that play on unbroken (no
    # transition, the same framing).
    assert segs[0][0] == "cam-a-p1" and segs[0][2] == 360
    assert segs[1][0] == "cam-a-p2" and segs[1][1] == 0
    assert segs[0][3] == segs[1][3]
    assert all(t.after != 0 for t in plan.transitions)
    # Range 2 crosses the gap: cut there, the frames with no file left out (and the crossfade's
    # handle at each file edge), then part 3.
    assert segs[2][0] == "cam-a-p2" and segs[2][2] < 720 - 360
    assert segs[3][0] == "cam-a-p3" and 0 < segs[3][1] < 30
    assert len(segs) == 4
    produced = json.loads((env.agent / "plan.json").read_text())
    assert produced["segments"] == [
        s.model_dump(mode="json", exclude_none=True) for s in plan.segments
    ]

    # The render plays the plan frame for frame.
    mp4 = tmp_path / "review.mp4"
    mp4.write_bytes(env.hub.store.read_bytes(render["review_mp4"]))
    info = probe(mp4)
    frames = sum(s.src_out - s.src_in for s in plan.segments)
    assert info.video is not None
    assert abs((info.video.duration_s or 0) - _seconds(frames)) < 0.05

    # The FCPXML has one asset per file, each at its own start timecode, and every segment links
    # its own file; each file's PC receipt was checked.
    export = outputs(env, run, "export")
    assert export["assets"] == [
        "file:///E:/Shows/cam-a_C001.mov",
        "file:///E:/Shows/cam-a_C002.mov",
        "file:///E:/Shows/cam-a_C003.mov",
    ]
    fcpxml = env.hub.store.read_bytes(export["fcpxml"]).decode()
    for _, tc in files:
        assert f'start="{tc * 1001}/30000s"' in fcpxml
    check = verify_plan_fcpxml(
        fcpxml,
        plan,
        load_profile("ccn"),
        media_map=media_map_from_mapping({str(env.media): PC_ROOT}),
    )
    assert check.ok, check.problems
    report = json.loads(env.hub.store.read_bytes(export["export_report"]))
    evidence = report["media_evidence"]
    assert evidence["status"] == "verified_snapshot"
    assert sorted(f["path"] for f in evidence["files"]) == [
        f"{PC_ROOT}\\cam-a_C00{n}.mov" for n in (1, 2, 3)
    ]
    review = latest(env, run, "review")
    assert get_review(env.hub.log, env.hub.project_id, review.review_id) is not None
