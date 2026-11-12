"""``ccn-longform`` round 2 (S19a validation findings): transcript reuse bound to runner evidence,
the agent's plan checked against the stub, verified cached artifacts, scratch and lock hygiene,
the export share's names and symlinks, and the mutants the first round's tests let survive."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from ccn_recipe_support import (
    CcnEnv,
    _json_main,
    agent_cut,
    fetch,
    latest,
    make_run,
    outputs,
    start,
    work,
)
from write_support import AGENT

from aar.analyze.errors import AnalyzeError
from aar.analyze.whisper import CCN_PRESET
from aar.cli import main
from aar.host.transport import sha256_file
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import EventLog
from aar.hub.graph import WorkGraph
from aar.qa.report import QACheck, build_report
from aar.recipes.ccn import scratch, set_transcriber, transcript
from aar.recipes.ccn.export_files import safe_name
from aar.recipes.ccn.params import CcnLongformParams
from aar.recipes.ccn.steps import review_qa_gate
from aar.recipes.ccn.transcript import transcript_cache_key, whisper_settings
from aar.recipes.run import step_complete
from aar.recipes.worker import run_worker
from ccn_support import KEEP, WORDS, make_show, needs_ffmpeg

pytestmark = needs_ffmpeg

SHORT_WORDS = WORDS[:7]  # a "different" transcript: only the first sentence


def _agent_fact(env: CcnEnv, key: str, value: dict[str, Any], idem: str) -> None:
    """Record a fact the way an agent can: ``aar hub write fact`` with an agent actor."""
    req = env.agent / f"fact-{idem}.json"
    req.write_text(
        json.dumps({"project_id": env.hub.project_id, "key": key, "value": value}),
        encoding="utf-8",
    )
    argv = ["hub", "write", "fact", "--db", env.db]
    _json_main([*argv, "--actor", "agent:agent-1", "--idempotency-key", idem, str(req)])


def _key_for(env: CcnEnv, show: Path, **changes: Any) -> tuple[str, str]:
    params = CcnLongformParams.model_validate(env.params(show, **changes))
    sha = sha256_file(str(show)).lower()
    return sha, transcript_cache_key(sha, whisper_settings(params), "show-1", 0.0)


def _point(env: CcnEnv, show: Path, made: dict[str, Any], idem: str, **changes: Any) -> None:
    sha, key = _key_for(env, show, **changes)
    value = {"media_sha256": sha, "model": "medium.en", "recording": "show-1"}
    value |= {name: made[name] for name in ("transcript", "text", "report")}
    _agent_fact(env, f"transcript.{key}", value, idem)


def _cold(env: CcnEnv, params: dict[str, Any], key: str, **kw: Any) -> str:
    run = make_run(env, params, key=key, **kw)
    start(env, run, key)
    work(env)
    return run


# -- HIGH-1: transcript reuse is bound to runner evidence ------------------------------------


def test_an_agent_cannot_repoint_the_cache_at_an_unnormalised_transcript(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show = env.show()
    env.words = SHORT_WORDS  # the raw-audio transcript misses words
    raw = outputs(env, _cold(env, env.params(show, normalize_audio=False), "raw"), "transcript")
    assert raw["cached"] is False and raw["words"] == 7
    env.words = WORDS
    _point(env, show, raw, "poison-1")  # the normalised key -> the raw-audio transcript
    got = outputs(env, _cold(env, env.params(show), "norm"), "transcript")
    assert got["cached"] is False and got["normalize"] is True
    assert got["words"] == 18 and got["transcript"] != raw["transcript"]
    assert len(env.calls) == 2


def test_an_agent_cannot_repoint_the_cache_at_another_files_transcript(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show_x = env.show("x.mov")
    show_y = env.show("y.mov", timecode="02:00:00:00", gain=0.25)
    env.words = SHORT_WORDS
    tx = outputs(env, _cold(env, env.params(show_x), "x"), "transcript")
    env.words = WORDS
    _point(env, show_y, tx, "poison-2")
    got = outputs(env, _cold(env, env.params(show_y), "y"), "transcript")
    assert got["cached"] is False and got["transcript"] != tx["transcript"]
    assert got["words"] == 18 and len(env.calls) == 2


def test_another_projects_transcript_of_the_same_file_is_not_reused(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show = env.show()
    other = outputs(
        env, _cold(env, env.params(show), "o", project=env.hub.other_project_id), "transcript"
    )
    _point(env, show, other, "cross")
    mine = outputs(env, _cold(env, env.params(show), "m"), "transcript")
    assert mine["cached"] is False and len(env.calls) == 2


def test_bytes_stored_under_an_op_the_runner_did_not_create_are_not_reused(
    ccn_env: CcnEnv,
) -> None:
    env = ccn_env
    show = env.show()
    real = outputs(env, _cold(env, env.params(show), "1"), "transcript")
    graph = WorkGraph(env.hub.log, env.hub.project_id)
    fake_op = graph.add_node(
        "op",
        AGENT,
        title="transcript",
        status="done",
        data={"schema": "aar.step/1", "step_id": "transcript", "recipe_node": "nd_x"},
        idempotency_key="fake-op",
    )
    # another transcription of "the same file", stored as if that op had made it
    forged = env.hub.store.put(
        b'{"recording":"show-1","words":[]}',
        actor=AGENT,
        project_id=env.hub.project_id,
        producing_op=fake_op.id,
    )
    _point(env, show, {"transcript": forged, "text": real["text"], "report": real["report"]}, "f")
    again = outputs(env, _cold(env, env.params(show), "2"), "transcript")
    assert again["cached"] is False and len(env.calls) == 2


def test_the_cache_key_covers_every_setting_the_recording_and_the_offset() -> None:
    base = transcript_cache_key("a" * 64, CCN_PRESET, "show-1", 0.0)
    variants = [
        transcript_cache_key("b" * 64, CCN_PRESET, "show-1", 0.0),
        transcript_cache_key("a" * 64, replace(CCN_PRESET, model="small.en"), "show-1", 0.0),
        transcript_cache_key("a" * 64, replace(CCN_PRESET, normalize=True), "show-1", 0.0),
        transcript_cache_key("a" * 64, replace(CCN_PRESET, wordless_min_s=30.0), "show-1", 0.0),
        transcript_cache_key("a" * 64, replace(CCN_PRESET, beam_size=3), "show-1", 0.0),
        transcript_cache_key("a" * 64, replace(CCN_PRESET, avg_logprob_min=-2.0), "show-1", 0.0),
        transcript_cache_key("a" * 64, CCN_PRESET, "show-2", 0.0),
        transcript_cache_key("a" * 64, CCN_PRESET, "show-1", 2.002),
    ]
    assert base not in variants and len(set(variants)) == len(variants)
    # threads and caps do not change the words
    assert base == transcript_cache_key("a" * 64, replace(CCN_PRESET, cpu_threads=8), "show-1", 0.0)


# -- MED-2: a rotted cached artifact is a miss, and the store is repaired -----------------------


@pytest.mark.parametrize("which", ["report", "transcript", "text"])
@pytest.mark.parametrize("how", ["corrupt", "missing"])
def test_a_rotted_cached_artifact_is_transcribed_again(
    ccn_env: CcnEnv, which: str, how: str
) -> None:
    env = ccn_env
    show = env.show()
    first = outputs(env, _cold(env, env.params(show), "c1"), "transcript")
    path = next(Path(env.artifacts).rglob(first[which]))
    if how == "corrupt":
        os.chmod(path, 0o644)
        path.write_bytes(path.read_bytes()[:-3] + b"zzz")
    else:
        path.unlink()
    second = outputs(env, _cold(env, env.params(show, timeline="Again"), "c2"), "transcript")
    assert second["cached"] is False and len(env.calls) == 2
    assert second[which] == first[which]  # the same bytes again: the store copy is repaired
    env.hub.store.read_bytes(second[which])
    # the next run reuses it again
    third = outputs(env, _cold(env, env.params(show, timeline="Third"), "c3"), "transcript")
    assert third["cached"] is True and len(env.calls) == 2


# -- the transcriber's checkpoint and the per-key lock ------------------------------------------


def test_the_transcriber_resumes_and_a_foreign_checkpoint_is_discarded(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show = env.show()
    real = transcript._TRANSCRIBE[0]
    calls: list[bool] = []

    def mismatch_once(media: Any, out: Any, *, resume: bool, **kw: Any) -> dict[str, Any]:
        calls.append(resume)
        if len(calls) == 1:
            Path(out).with_name("transcript.json.ckpt.jsonl").write_text("{}", encoding="utf-8")
            raise AnalyzeError("checkpoint-mismatch", "made from other bytes")
        assert not any(Path(out).parent.iterdir()), "the foreign checkpoint was removed"
        return real(media, out, resume=resume, **kw)

    set_transcriber(mismatch_once)
    run = _cold(env, env.params(show), "1")
    assert calls == [True, False]  # resume first, then once more from the start
    assert outputs(env, run, "transcript")["cached"] is False
    assert not list((env.work / "aar-transcripts").iterdir())  # folder and lock removed


def test_a_second_worker_waits_for_the_first_and_reuses_its_transcript(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    show = env.show()
    monkeypatch.setattr(scratch, "LOCK_POLL_S", 0.05)
    real = transcript._TRANSCRIBE[0]
    entered, release = threading.Event(), threading.Event()

    def slow(media: Any, out: Any, **kw: Any) -> dict[str, Any]:
        entered.set()
        assert release.wait(30)
        return real(media, out, **kw)

    set_transcriber(slow)
    run1 = make_run(env, env.params(show), key="1")
    start(env, run1, "1")
    errors: list[BaseException] = []

    def first_worker() -> None:
        log = EventLog(env.db)
        try:
            store = ArtifactStore(env.artifacts, log)
            for _ in range(10):
                if run_worker(log, store, AGENT, once=True) == 0:
                    break
        except BaseException as exc:  # reported below
            errors.append(exc)
        finally:
            log.close()

    thread = threading.Thread(target=first_worker)
    thread.start()
    assert entered.wait(30)
    run2 = make_run(env, env.params(show, timeline="Second"), key="2")
    start(env, run2, "2")
    threading.Timer(0.5, release.set).start()
    work(env)  # its transcript step waits on the lock, then finds run 1's result
    thread.join(60)
    assert not errors
    assert outputs(env, run2, "transcript")["cached"] is True
    assert len(env.calls) == 1


def test_a_lock_held_past_the_wait_bound_fails_busy(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = ccn_env
    show = env.show()
    monkeypatch.setattr(scratch, "LOCK_POLL_S", 0.05)
    monkeypatch.setattr(transcript, "LOCK_WAIT_S", 0.3)
    _sha, key = _key_for(env, show)
    lock = env.work / "aar-transcripts" / f"{key}.lock"
    with scratch.key_lock(lock, wait_s=0):
        run = _cold(env, env.params(show), "1")
        op = latest(env, run, "transcript")
        assert op.status == "failed" and op.result_error["code"] == "transcript_busy"
    assert env.calls == []


def test_lock_files_nobody_holds_are_swept_and_held_ones_kept(ccn_env: CcnEnv) -> None:
    env = ccn_env
    base = env.work / "aar-transcripts"
    base.mkdir()
    (base / "dead.lock").write_text("")
    held = base / "held.lock"
    with scratch.key_lock(held, wait_s=0):
        assert scratch.sweep_locks(base) == 1
        assert held.exists() and not (base / "dead.lock").exists()
    assert not held.exists()  # its holder unlinked it


# -- MED-3: render scratch per op, swept, never a live op's ------------------------------------


def test_render_scratch_of_finished_runs_and_dead_ops_is_swept_but_never_a_live_ops(
    ccn_env: CcnEnv,
) -> None:
    env = ccn_env
    show = env.show()
    run1 = _cold(env, env.params(show), "1")
    agent_cut(env, run1)
    work(env)  # render, export: run 1 now waits at its gate (active)
    run2 = _cold(env, env.params(show, timeline="Second"), "2")  # cut is live in run 2
    live_cut = latest(env, run2, "cut").op_id
    done_stub = latest(env, run2, "stub").op_id
    area = env.work / "aar-render"
    for name in (live_cut, done_stub):
        (area / run2 / name).mkdir(parents=True)
    (area / "nd_gone").mkdir()  # a recipe this hub does not know: kept (round 3, LOW-2)
    # a third run renders: its sweep runs over the area
    run3 = _cold(env, env.params(show, timeline="Third"), "3")
    agent_cut(env, run3, key="r3")
    work(env)
    assert latest(env, run3, "render").status == "done"
    assert (area / run2 / live_cut).is_dir()
    assert not (area / run2 / done_stub).exists()
    assert (area / "nd_gone").is_dir()
    assert not (area / run3).exists()


# -- HIGH-2: the agent's plan is checked against the stub ------------------------------------


def _submit_edited_stub(env: CcnEnv, run: str, edit: Any, key: str) -> None:
    stub = fetch(env, outputs(env, run, "stub")["stub"], "stub.json")
    data = json.loads(stub.read_text())
    edit(data)
    stub.write_text(json.dumps(data))
    tr = fetch(env, outputs(env, run, "transcript")["transcript"], "transcript.json")
    keep = env.agent / "keep.json"
    keep.write_text(json.dumps({"recording": "show-1", **KEEP}))
    plan = env.agent / "plan.json"
    argv = ["plan", "produce", str(stub), "--transcript", str(tr), "--keep", str(keep)]
    assert main([*argv, "--out", str(plan), "--no-pause-trim"]) == 0
    req = env.agent / "submit.json"
    body = {
        "project_id": env.hub.project_id,
        "derives_from": run,
        "plan": json.loads(plan.read_text()),
    }
    req.write_text(json.dumps(body))
    write = ["hub", "write", "plan", "--db", env.db, "--artifacts", env.artifacts]
    node = _json_main([*write, "--actor", "agent:a", "--idempotency-key", key, str(req)])
    step_complete(
        env.hub.log, env.hub.store, run, "cut", None, {"plan_node": node["node"]["id"]}, AGENT,
        idempotency_key=f"cut-{key}",
    )  # fmt: skip


def test_a_plan_whose_media_is_outside_the_media_root_is_refused(
    ccn_env: CcnEnv, tmp_path: Path
) -> None:
    env = ccn_env
    show = env.show()
    (tmp_path / "outside").mkdir()
    outside = make_show(tmp_path / "outside" / "private.mov")
    run = _cold(env, env.params(show), "esc")

    def edit(data: dict[str, Any]) -> None:
        data["sources"][0]["media"] = str(outside)
        data["project"] = "Some Other Show"

    _submit_edited_stub(env, run, edit, "p1")
    work(env)
    op = latest(env, run, "render")
    assert op.status == "failed" and op.result_error["code"] == "plan_not_from_stub"
    message = op.result_error["message"]
    assert "project is 'Some Other Show'" in message and "not under the media root" in message
    assert latest(env, run, "export") is None
    assert not any(env.exports.iterdir())


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        (lambda d: d["sources"][0].update(id="cam-z"), "source 'cam-z' is not one of the stub's"),
        (lambda d: d.update(timeline="Other Timeline"), "timeline is 'Other Timeline'"),
        (lambda d: d["sources"][0].update(start_timecode=1), "start_timecode is 1"),
    ],
)
def test_a_plan_that_differs_from_the_stub_names_the_difference(
    ccn_env: CcnEnv, edit: Any, expected: str
) -> None:
    env = ccn_env
    run = _cold(env, env.params(env.show()), "d")
    _submit_edited_stub(env, run, edit, "p1")
    work(env)
    op = latest(env, run, "render")
    assert op.result_error["code"] == "plan_not_from_stub"
    assert expected in op.result_error["message"]


def test_a_plan_naming_another_file_in_the_media_root_is_refused(ccn_env: CcnEnv) -> None:
    env = ccn_env
    show = env.show()
    other = env.show("someone-else.mov")
    run = _cold(env, env.params(show), "o")
    _submit_edited_stub(env, run, lambda d: d["sources"][0].update(media=str(other)), "p1")
    work(env)
    op = latest(env, run, "render")
    assert (
        op.result_error["code"] == "plan_not_from_stub" and "media is" in op.result_error["message"]
    )


# -- the QA gate ---------------------------------------------------------------------------


def _qa(**statuses: str) -> Any:
    ids = ["decode", "loudness", "clipping", "longest-silence", "duration", "join-quiet"]
    checks = [QACheck(id=i, status=statuses.get(i, "pass"), message=i) for i in ids]
    checks.append(
        QACheck(id="size-cap", status="skipped", skip_reason="not-applicable", message="n/a")
    )
    return build_report(file="r.mp4", kind="longform", profile="ccn", media=None, checks=checks)


def test_the_review_gate_allows_only_a_longest_silence_warning() -> None:
    assert review_qa_gate(_qa()).passed
    assert review_qa_gate(_qa(**{"longest-silence": "warn"})).passed
    for check in ("loudness", "clipping", "join-quiet", "duration", "decode"):
        assert not review_qa_gate(_qa(**{check: "warn"})).passed, check


# -- export names and the share --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("..", "untitled"),
        (" .hidden. ", "hidden"),
        ("CON", "_CON"),
        ("con.txt", "_con.txt"),
        ("con .txt", "_con .txt"),
        ("LPT1 ", "_LPT1"),
        ("Nul.tar.gz", "_Nul.tar.gz"),
        ("conin$", "conin_"),
        ("CONOUT$.log", "CONOUT_.log"),
        ("clock$", "clock_"),
        ("COM¹", "COM_"),
        ("a/b\\c", "a_b_c"),
        ("trailing. . ", "trailing"),
        ("Café ☕", "Caf_ _"),
        ("x" * 300, "x" * 100),
        ("Fakes 5-23-26", "Fakes 5-23-26"),
    ],
)
def test_safe_name(text: str, want: str) -> None:
    assert safe_name(text) == want


def _export_run(env: CcnEnv, key: str = "1") -> str:
    run = _cold(env, env.params(env.show()), key)
    agent_cut(env, run, key=f"r{key}")
    work(env)
    return run


def test_a_symlink_at_an_export_name_is_never_followed_or_replaced(ccn_env: CcnEnv) -> None:
    env = ccn_env
    folder = env.exports / "Synthetic Show" / "Synthetic Long v1"
    folder.mkdir(parents=True)
    dangling = folder / "Synthetic Long v1.r1.fcpxml"
    dangling.symlink_to(folder / "nowhere")
    run = _export_run(env)
    copy = outputs(env, run, "export")["copy"]
    assert copy["fcpxml"].endswith("Synthetic Long v1.r1-2.fcpxml")
    assert dangling.is_symlink() and not (folder / "nowhere").exists()


def test_a_symlinked_folder_in_the_share_is_refused(ccn_env: CcnEnv, tmp_path: Path) -> None:
    env = ccn_env
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (env.exports / "Synthetic Show").symlink_to(elsewhere)
    run = _export_run(env)
    op = latest(env, run, "export")
    assert op.status == "failed" and op.result_error["code"] == "export_path_symlink"
    assert not any(elsewhere.iterdir())


def test_an_export_dir_that_is_a_file_is_refused(
    ccn_env: CcnEnv, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = ccn_env
    not_a_dir = tmp_path / "exports.txt"
    not_a_dir.write_text("x")
    monkeypatch.setenv("AAR_EXPORT_DIR", str(not_a_dir))
    run = _export_run(env)
    assert latest(env, run, "export").result_error["code"] == "export_dir_unusable"


# -- params ----------------------------------------------------------------------------------


def test_params_refuse_a_symlink_out_of_the_media_root_and_one_file_twice(
    ccn_env: CcnEnv, tmp_path: Path
) -> None:
    env = ccn_env
    show = env.show()
    outside = tmp_path / "outside.mov"
    outside.write_bytes(b"x")
    link = env.media / "looks-inside.mov"
    link.symlink_to(outside)
    with pytest.raises(ValueError, match="not under the media root"):
        CcnLongformParams.model_validate(env.params(link))
    alias = env.media / "alias.mov"
    alias.symlink_to(show)
    for twice in ((show, show), (show, alias)):
        with pytest.raises(ValueError, match="a file already given"):
            CcnLongformParams.model_validate(env.params(*twice))
