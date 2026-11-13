"""The ``ccn-longform`` recipe's cold steps (S19a), run by the real worker on synthetic media with
an injected transcriber. The end-to-end run is in ``test_ccn_recipe_e2e.py``."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from ccn_recipe_support import (
    CcnEnv,
    agent_cut,
    latest,
    make_run,
    outputs,
    start,
    work,
)
from write_support import AGENT

from aar.analyze.errors import AnalyzeError
from aar.hub.facts import facts_view, record_fact
from aar.recipes import registry
from aar.recipes.ccn import CCN_LONGFORM, CcnLongformParams, set_laughs_runner, set_transcriber
from aar.recipes.ccn.instruction import CUT_INSTRUCTION
from aar.recipes.models import MAX_INSTRUCTION_BYTES
from aar.recipes.state import load_state
from ccn_support import delayed_copy, needs_ffmpeg

pytestmark = needs_ffmpeg


def _problems(env: CcnEnv, params: dict[str, Any]) -> list[str]:
    from write_support import brief_data

    from aar.hub.writes import Brief

    brief = Brief.model_validate(brief_data(env.hub.project_id, kind="longform"))
    return [
        f"{loc}: {msg}"
        for loc, msg in registry.registry_params_validator("ccn-longform", params, brief)
    ]


def test_the_module_registers_the_recipe_from_the_env(tmp_path: Path) -> None:
    run = subprocess.run(
        [sys.executable, "-m", "aar", "recipe", "list"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"AAR_RECIPES": "aar.recipes.ccn", "PATH": "/usr/bin:/bin"},
        check=True,
    )
    names = [r["name"] for r in json.loads(run.stdout)["recipes"]]
    assert "ccn-longform" in names
    assert [s.id for s in CCN_LONGFORM.steps] == [
        "stub", "sync", "transcript", "laughs", "guide", "cut", "render", "export", "review",
        "report",
    ]  # fmt: skip
    assert len(CUT_INSTRUCTION.encode()) <= MAX_INSTRUCTION_BYTES
    for words in ("hub artifact get", "read_guide", "guide.md", "exactly as printed",
                  "not on PATH", "plan produce", "plan lint", "step_complete",
                  "never splice dialogue", "chronological",
                  "end on the laugh", "not silence", "pause_trim_flag", "derives_from",
                  "AAR_RECIPES=aar.recipes.ccn", "transcript.json word times"):  # fmt: skip
        assert words in CUT_INSTRUCTION


def test_params_are_validated_at_the_boundary(ccn_env: CcnEnv, tmp_path: Path) -> None:
    show = ccn_env.show()
    assert _problems(ccn_env, ccn_env.params(show)) == []
    outside = tmp_path / "elsewhere.mov"
    outside.write_bytes(b"x")
    cases: list[tuple[dict[str, Any], str]] = [
        (ccn_env.params(Path("cam-a.mov")), "not an absolute hub path"),
        (ccn_env.params(ccn_env.media / "missing.mov"), "does not exist"),
        (ccn_env.params(outside), "not under the media root"),
        (ccn_env.params(show, media_map={"/elsewhere": "E:\\X"}), "media map does not give"),
        (ccn_env.params(show, media_map={"relative": "E:\\X"}), "media_map"),
        (ccn_env.params(show, whisper_model="../models/x"), "whisper_model"),
        (ccn_env.params(show, extra=1), "extra"),
        (ccn_env.params(show, pause_trim="maybe"), "pause_trim"),
    ]
    for params, expected in cases:
        problems = _problems(ccn_env, params)
        assert problems and any(expected in p for p in problems), (expected, problems)
    one = CcnLongformParams.model_validate(ccn_env.params(show))
    two = CcnLongformParams.model_validate(ccn_env.params(show, ccn_env.show("cam-b.mov")))
    assert (one.pause_trim_on(), two.pause_trim_on()) == (False, True)
    forced = CcnLongformParams.model_validate(ccn_env.params(show, pause_trim="on"))
    assert forced.pause_trim_on() is True


def test_cold_steps_one_camera_then_the_transcript_is_reused(ccn_env: CcnEnv) -> None:
    show = ccn_env.show()
    run1 = make_run(ccn_env, ccn_env.params(show), key="1")
    start(ccn_env, run1, "1")
    assert work(ccn_env) == 5  # stub, sync, transcript, laughs, guide
    stub = outputs(ccn_env, run1, "stub")
    assert stub["fps"] == "30000/1001" and stub["recording_id"] == "show-1"
    assert stub["pause_trim"] == "off" and stub["pause_trim_flag"] == "--no-pause-trim"
    st = show.stat()
    assert stub["sources"] == [
        {
            "id": "cam-a",
            "camera": "A",
            "duration": 360,
            "start_timecode": "01:00:00:00",
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
    ]
    assert outputs(ccn_env, run1, "sync") == {
        "synced": False,
        "report": None,
        "reason": "one source: nothing to sync",
    }
    assert outputs(ccn_env, run1, "laughs") == {"skipped": "not requested", "map": None}
    transcript = outputs(ccn_env, run1, "transcript")
    assert transcript["cached"] is False and transcript["words"] == 18
    assert transcript["model"] == "medium.en" and transcript["normalize"] is True
    assert ccn_env.calls == [{"media": str(show), "model": "medium.en", "normalize": True}]
    # the head, the two one-second pauses and the tail (at the fake's 1 s minimum)
    assert transcript["wordless"]["count"] == 4
    assert [s["start_s"] for s in transcript["wordless"]["stretches"]] == [0.0, 4.0, 8.0, 11.0]
    text = ccn_env.hub.store.read_bytes(transcript["text"]).decode()
    assert text.startswith("[0001.0-0004.0] So I went to the store yesterday\n")
    key = f"transcript.{transcript['cache_key']}"
    fact = facts_view(ccn_env.hub.log, ccn_env.hub.project_id).get(key)
    assert fact is not None and fact.current.value["transcript"] == transcript["transcript"]
    assert not list((ccn_env.work / "aar-transcripts").glob("*/*"))  # scratch removed
    state = load_state(ccn_env.hub.log, run1)
    assert latest(ccn_env, run1, "cut").status == "active"
    assert state.recipe_status == "active"

    # A second run on the same file and settings: the transcript comes from the fact.
    run2 = make_run(ccn_env, ccn_env.params(show, timeline="Second Cut"), key="2")
    start(ccn_env, run2, "2")
    work(ccn_env)
    again = outputs(ccn_env, run2, "transcript")
    assert again["cached"] is True and len(ccn_env.calls) == 1
    for name in ("transcript", "text", "report", "cache_key"):
        assert again[name] == transcript[name]
    assert again["wordless"] == transcript["wordless"]

    # Other settings are another key: normalisation off transcribes again.
    run3 = make_run(ccn_env, ccn_env.params(show, normalize_audio=False), key="3")
    start(ccn_env, run3, "3")
    work(ccn_env)
    third = outputs(ccn_env, run3, "transcript")
    assert third["cached"] is False and third["cache_key"] != transcript["cache_key"]
    assert ccn_env.calls[-1]["normalize"] is False


def test_a_forged_transcript_fact_is_not_trusted(ccn_env: CcnEnv) -> None:
    show = ccn_env.show()
    run1 = make_run(ccn_env, ccn_env.params(show), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    real = outputs(ccn_env, run1, "transcript")
    # An agent records the key with bytes it stored itself (no transcript step made them).
    fake = ccn_env.hub.store.put(b'{"recording":"show-1","words":[]}', actor=AGENT)
    record_fact(
        ccn_env.hub.log,
        ccn_env.hub.project_id,
        f"transcript.{real['cache_key']}",
        {
            "media_sha256": real["media_sha256"],
            "model": "medium.en",
            "recording": "show-1",
            "transcript": fake,
            "text": fake,
            "report": fake,
        },
        AGENT,
        idempotency_key="forge",
    )
    run2 = make_run(ccn_env, ccn_env.params(show), key="2")
    start(ccn_env, run2, "2")
    work(ccn_env)
    again = outputs(ccn_env, run2, "transcript")
    assert again["cached"] is False and again["transcript"] == real["transcript"]
    assert len(ccn_env.calls) == 2


def test_typed_failures_of_the_transcript_and_stub_steps(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    show = ccn_env.show()
    monkeypatch.delenv("AAR_WORK_DIR")
    run1 = make_run(ccn_env, ccn_env.params(show), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    op = latest(ccn_env, run1, "transcript")
    assert op.status == "failed" and op.result_error["code"] == "work_dir_unset"
    assert "AAR_WORK_DIR" in op.result_error["fix_hint"]

    monkeypatch.setenv("AAR_WORK_DIR", str(ccn_env.work))

    def missing_model(*_args: Any, **_kw: Any) -> dict[str, Any]:
        raise AnalyzeError("model-missing", "no model medium.en under the model dir", hint="x")

    set_transcriber(missing_model)
    run2 = make_run(ccn_env, ccn_env.params(show), key="2")
    start(ccn_env, run2, "2")
    work(ccn_env)
    op = latest(ccn_env, run2, "transcript")
    assert op.result_error["code"] == "transcript_model-missing"
    assert not list((ccn_env.work / "aar-transcripts").glob("*/*"))

    silent = ccn_env.show("silent.mov", audio=False)
    run3 = make_run(ccn_env, ccn_env.params(silent), key="3")
    start(ccn_env, run3, "3")
    work(ccn_env)
    assert latest(ccn_env, run3, "stub").result_error["code"] == "stub_no_audio"

    run4 = make_run(ccn_env, ccn_env.params(show), key="4", kind="short")
    start(ccn_env, run4, "4")
    work(ccn_env)
    assert latest(ccn_env, run4, "stub").result_error["code"] == "kind_mismatch"


def test_two_cameras_sync_and_a_failed_pair_fails_the_step(ccn_env: CcnEnv) -> None:
    a = ccn_env.show()
    b = delayed_copy(a, ccn_env.media / "cam-b.mov", delay_s=0.5)
    run1 = make_run(ccn_env, ccn_env.params(a, b), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    stub = outputs(ccn_env, run1, "stub")
    assert stub["cameras"] == 2 and stub["pause_trim_flag"] == "--pause-trim"
    assert [s["start_timecode"] for s in stub["sources"]] == ["01:00:00:00", "01:00:00:15"]
    sync = outputs(ccn_env, run1, "sync")
    assert sync["synced"] is True
    assert [p["status"] for p in sync["pairs"]] == ["ok"]
    report = json.loads(ccn_env.hub.store.read_bytes(sync["report"]))
    assert report["pairs"][0]["file"] == str(b) and report["pairs"][0]["offset_s"] == -0.5
    assert ccn_env.calls[0]["media"] == str(a)  # the reference is transcribed

    # The agent's cut with both cameras: --sync from the step, a camera schedule.
    schedule = {"spans": [{"start_s": 0.0, "end_s": 6.0, "camera": "A"}]}
    schedule["spans"].append({"start_s": 6.0, "end_s": 12.0, "camera": "B"})
    agent_cut(ccn_env, run1, schedule=schedule)
    assert work(ccn_env) == 2
    assert outputs(ccn_env, run1, "render")["qa"]["passed"] is True
    assert outputs(ccn_env, run1, "export")["assets"] == [
        "file:///E:/Shows/cam-a.mov",
        "file:///E:/Shows/cam-b.mov",
    ]

    quiet = ccn_env.show("cam-q.mov", gain=0.0)
    run2 = make_run(ccn_env, ccn_env.params(a, quiet), key="2")
    start(ccn_env, run2, "2")
    work(ccn_env)
    op = latest(ccn_env, run2, "sync")
    assert op.status == "failed" and op.result_error["code"] == "sync_not_ok"
    assert "cam-q.mov (low-confidence)" in op.result_error["message"]


def test_laughs_run_only_when_asked_and_feed_the_cut(ccn_env: CcnEnv) -> None:
    show = ccn_env.show()
    seen: list[dict[str, Any]] = []

    def laughs(media: Any, out: Any, **kw: Any) -> dict[str, Any]:
        seen.append({"media": str(media), "allow_paid": kw["allow_paid"], **kw})
        Path(out).write_text(json.dumps({"schema": "aar.laughs/1"}), encoding="utf-8")
        laugh = {"start_s": 10.65, "end_s": 11.0, "strength": 1.0}
        Path(kw["map_out"]).write_text(
            json.dumps({"recording": kw["recording"], "laughs": [laugh]}), encoding="utf-8"
        )
        return {"ok": True, "model": "gemini-x", "counts": {"laughs": 1}}

    set_laughs_runner(laughs)
    run1 = make_run(ccn_env, ccn_env.params(show, laughs=True), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    out = outputs(ccn_env, run1, "laughs")
    assert out["count"] == 1 and out["model"] == "gemini-x"
    assert seen[0]["media"] == str(show) and seen[0]["allow_paid"] is True
    assert str(ccn_env.work) in str(seen[0]["work_dir"])
    assert not list((ccn_env.work / "aar-laughs").iterdir())  # scratch removed
    plan_node = agent_cut(ccn_env, run1)
    assert latest(ccn_env, run1, "cut").status == "done"
    assert plan_node


def test_a_render_that_fails_qa_names_the_checks(ccn_env: CcnEnv) -> None:
    silent = ccn_env.show("silent-words.mov", gain=0.0, room=False)
    run1 = make_run(ccn_env, ccn_env.params(silent), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    agent_cut(ccn_env, run1)
    work(ccn_env)
    op = latest(ccn_env, run1, "render")
    assert op.status == "failed" and op.result_error["code"] == "qa_failed"
    assert "loudness (fail)" in op.result_error["message"]
    # S19f: the failure is the cut's (marked upstream), so the hint and the status view name the
    # reopen of the cut instead of a new run.
    assert op.upstream is True and op.result_error["upstream"] is True
    assert "recipe_reopen cut" in op.result_error["fix_hint"]
    from aar.recipes.status import status_view

    actions = status_view(ccn_env.hub.log, run1)["next_actions"]
    assert [a.get("tool") for a in actions if a["step_id"] == "render"] == ["recipe_reopen"]
    # A retry renders the same plan again and fails the same way (which is why the reopen exists:
    # tests/hub/test_ccn_recipe_reopen.py).
    from aar.recipes.run import recipe_retry

    recipe_retry(ccn_env.hub.log, run1, "render", AGENT, idempotency_key="retry-render")
    work(ccn_env)
    again = latest(ccn_env, run1, "render")
    assert again.attempt == 2 and again.result_error["code"] == "qa_failed"
    assert latest(ccn_env, run1, "cut").status == "done"
    assert not (ccn_env.work / "aar-render").exists() or not list(
        (ccn_env.work / "aar-render").rglob("*")
    )
    assert latest(ccn_env, run1, "export") is None


def test_without_an_export_dir_the_copy_is_skipped_and_said(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AAR_EXPORT_DIR")
    show = ccn_env.show()
    run1 = make_run(ccn_env, ccn_env.params(show), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    agent_cut(ccn_env, run1)
    work(ccn_env)
    export = outputs(ccn_env, run1, "export")
    assert export["copy"] == {
        "copied": False,
        "reason": "AAR_EXPORT_DIR is not set: nothing was copied",
    }
    assert ccn_env.hub.store.has(export["fcpxml"])
    assert not list(ccn_env.exports.iterdir())
    assert latest(ccn_env, run1, "review").review_id is not None


def test_a_sigterm_during_the_render_leaves_the_claim_to_lapse_and_it_is_reclaimed(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MED-3: a stopped render uses no attempt: the worker exits with the claim held, the
    lease lapses, and the next worker claims the (idempotent) render again and finishes it."""
    import signal
    from datetime import timedelta

    from aar.hub.common import utcnow
    from aar.plan.render import RenderTerminated, render_plan
    from aar.recipes.ccn import steps
    from aar.recipes.ccn.transcript import StepInterrupted
    from aar.recipes.worker import run_worker

    show = ccn_env.show()
    run1 = make_run(ccn_env, ccn_env.params(show), key="1")
    start(ccn_env, run1, "1")
    work(ccn_env)
    agent_cut(ccn_env, run1)
    scratch: list[Path] = []

    def terminated(plan: Any, profile: Any, *, out: Path, **kw: Any) -> Any:
        scratch.append(out.parent)
        (out.parent / "partial.mp4").write_bytes(b"half a render")
        raise RenderTerminated(signal.SIGTERM)  # what render_plan raises on SIGTERM

    monkeypatch.setattr(steps, "render_plan", terminated)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with pytest.raises(StepInterrupted):
            run_worker(ccn_env.hub.log, ccn_env.hub.store, AGENT, install_signals=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    op = latest(ccn_env, run1, "render")
    assert op.status == "active" and op.result_error is None  # no result, no attempt used
    assert not scratch[0].exists()
    # SIGKILL looks the same to the hub: a claim whose lease lapses. The next worker re-claims.
    monkeypatch.setattr(steps, "render_plan", render_plan)
    later = utcnow() + timedelta(seconds=300)
    assert run_worker(ccn_env.hub.log, ccn_env.hub.store, AGENT, once=True, clock=lambda: later)
    again = latest(ccn_env, run1, "render")
    assert again.op_id == op.op_id and again.attempt == 1 and again.status == "done"
