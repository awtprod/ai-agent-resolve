"""The plan producers end to end on the synthetic shows: real ffmpeg audio, golden plans.

The long-form has two cameras (B synced by a real ``aar analyze sync`` report), pauses that
straddle a camera change, a mumble inside a pause, a word Whisper timed 80 ms late at a keep
edge, a one-frame camera flash and a final laugh. The short has a face track that sways, walks
and lunges. Both goldens lint clean under ``ccn``, and every cut passes QA's join-quiet test.
"""

from __future__ import annotations

import json
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from produce_support import FIXTURES, fixture, needs_ffmpeg, stage

from aar.cli import main
from aar.media import probe
from aar.plan.model import EditPlan
from aar.plan.produce import (
    LintGateError,
    load_faces,
    load_keep,
    load_schedule,
    load_stub,
    load_sync,
    produce,
    read_laughs,
    read_transcript,
)
from aar.plan.produce.audio import SourceAudio, pause_runs
from aar.plan.produce.timeline import Piece
from aar.plan.profile import load_profile, parse_profile
from aar.qa import cuts, join_quiet_window, resolve_media, scan_join_quiet

pytestmark = needs_ffmpeg

LONG_PARTS = ("stub", "transcript", "keep", "schedule", "sync", "laughs")
SHORT_PARTS = ("stub", "transcript", "keep", "faces", "laughs")


@pytest.fixture(scope="module")
def longform(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("longform")
    return {"dir": folder, **stage("longform", folder, LONG_PARTS)}


@pytest.fixture(scope="module")
def short(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("short")
    return {"dir": folder, **stage("short", folder, SHORT_PARTS)}


def produce_longform(paths: dict[str, Path], **kw: Any) -> Any:
    stub = load_stub(paths["stub"])
    return produce(
        stub,
        transcript=read_transcript(paths["transcript"]),
        keep=load_keep(paths["keep"]),
        profile=kw.pop("profile", load_profile("ccn")),
        schedule=load_schedule(paths["schedule"]),
        sync=load_sync(paths["sync"]),
        laughs=read_laughs(paths["laughs"]),
        media_root=paths["dir"],
        **kw,
    )


def produce_short(paths: dict[str, Path], **kw: Any) -> Any:
    return produce(
        load_stub(paths["stub"]),
        transcript=read_transcript(paths["transcript"]),
        keep=load_keep(paths["keep"]),
        profile=kw.pop("profile", load_profile("ccn")),
        faces=[load_faces(paths["faces"])],
        laughs=read_laughs(paths["laughs"]),
        media_root=paths["dir"],
        **kw,
    )


def dumped(plan: EditPlan) -> str:
    return plan.model_dump_json(indent=2, exclude_none=True) + "\n"


def golden(show: str) -> str:
    return fixture(show, "plan").read_text(encoding="utf-8")


def test_the_longform_matches_its_golden_plan(longform: dict[str, Path]) -> None:
    result = produce_longform(longform)
    assert dumped(result.plan) == golden("longform")
    # the tag after the final laugh ("and another thing") is dropped, and said so
    assert [(w.code, w.at_s) for w in result.warnings] == [("end-dropped-speech", 35.202)]
    assert "3 transcribed word(s)" in result.warnings[0].message
    assert result.lint.ok and result.lint.counts["error"] == 0
    log = result.log()
    assert len(log["trims"]) == 3
    # the 24.5-25.6 s pause spans the B -> A camera change and is trimmed as one pause
    assert log["trims"][1]["pause_s"][0] < 25.0 < log["trims"][1]["pause_s"][1]
    # the mumble at 8.0 s splits its 1.2 s quiet run, so that pause is not trimmed
    assert not any(7.5 < t["pause_s"][0] < 8.9 for t in log["trims"])
    # keep 2 starts at 19.40 s inside the real tail of "later" (Whisper ended it at 19.30, the
    # sound runs to 19.45): the in point moves later, inward only, until the run holds the
    # crossfade's handle too. Never earlier, into material the human excluded.
    assert [(n["join"], n["side"], n["from_frame"], n["to_frame"]) for n in log["nudges"]] == [
        (2, "in", 582, 584)
    ]
    # camera B's half-second flash at 31.0 s folds into the A shot before it
    assert [(f["source"], f["into"], f["retrimmed"]) for f in log["folds"]] == [
        ("cam-b", "previous", False)
    ]
    # the content ends 200 ms after the final laugh's onset and drops the tag after it
    assert log["ending"]["onset_s"] == 35.0 and log["ending"]["out_s"] == 35.202
    # the sync report put camera B at its measured origin
    assert result.plan.source_map()["cam-b"].recording_origin == 60
    # the stub's template, head, outro and audio tracks pass through
    stub = load_stub(FIXTURES / "longform.stub.json")
    for name in ("template", "head", "outro", "audio_tracks", "end_fade_frames"):
        assert getattr(result.plan, name) == getattr(stub, name)


def test_the_short_matches_its_golden_plan(short: dict[str, Path]) -> None:
    result = produce_short(short)
    assert dumped(result.plan) == golden("short")
    assert result.lint.ok
    plan = result.plan
    # the jump cuts alternate the A and B presets, so no same-camera jump cut
    assert [seg.reframe[0].zoom for seg in plan.segments] == [1.0, 1.22, 1.0]
    assert [(w.code, w.at_s) for w in result.warnings] == [("end-dropped-speech", 23.49)]
    # the face sways inside the dead zone for the first shot: the camera does not move
    assert len({key.x for key in plan.segments[0].reframe}) == 1
    # then it follows the walk and the lunge
    assert len({key.x for key in plan.segments[1].reframe}) > 3
    assert result.lint.duration_s < 37


def test_the_short_takes_its_source_size_from_the_face_track(
    short: dict[str, Path], longform: dict[str, Path]
) -> None:
    # S18 F1: the FCPXML export frames a reframed short from Source.frame_size.
    faces = json.loads(short["faces"].read_text(encoding="utf-8"))
    plan = produce_short(short).plan
    assert plan.sources[0].frame_size == faces["header"]["frame_size"] == [640, 360]
    # No face track, no size: the long-form's sources keep none.
    assert all(s.frame_size is None for s in produce_longform(longform).plan.sources)


@pytest.mark.parametrize("show", ["longform", "short"])
def test_the_golden_plans_lint_clean_under_ccn(
    show: str, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(
        [
            "plan",
            "lint",
            str(fixture(show, "plan")),
            "--transcript",
            str(fixture(show, "transcript")),
            "--laughs",
            str(fixture(show, "laughs")),
            "--json",
        ]
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0 and report["ok"] is True and report["counts"]["error"] == 0
    rules = {f["rule"] for f in report["findings"]}
    assert not rules & {"no-dialogue-splice", "end-on-laugh", "no-whip-pan", "min-shot"}


@pytest.mark.parametrize("show", ["longform", "short"])
def test_every_golden_cut_starts_in_quiet_by_qas_own_test(
    show: str, longform: dict[str, Path], short: dict[str, Path]
) -> None:
    folder = (longform if show == "longform" else short)["dir"]
    plan = EditPlan.model_validate_json(golden(show))
    qa = load_profile("ccn").qa
    found = cuts(plan, qa.join_continuity_tolerance_frames)
    assert found
    for cut in found:
        path = resolve_media(plan.source_map()[cut.source_id].media, folder)
        assert path is not None
        info = probe(path)
        assert info.audio is not None
        window = join_quiet_window(cut.src_in_s, qa.join_quiet_search_s, info.audio.duration_s)
        measured = scan_join_quiet(
            path, channels=1, src_in_s=cut.src_in_s, window=window, qa=qa, timeout_s=60
        )
        assert measured.judged.status == "pass", (show, cut.number)


def test_the_cli_writes_the_golden_and_a_json_report(
    longform: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    out = longform["dir"] / "cli.plan.json"
    p = longform
    code = main(
        ["plan", "produce", str(p["stub"]), "--transcript", str(p["transcript"]),
         "--keep", str(p["keep"]), "--schedule", str(p["schedule"]), "--sync", str(p["sync"]),
         "--laughs", str(p["laughs"]), "--out", str(out), "--json"]
    )  # fmt: skip
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out.read_text(encoding="utf-8") == golden("longform")
    assert report["ok"] is True and report["segments"] == 6 and report["cuts"] == 4
    assert [w["code"] for w in report["warnings"]] == ["end-dropped-speech"]
    assert report["lint"]["counts"]["error"] == 0
    assert {f["severity"] for f in report["lint"]["findings"]} <= {"warn", "info"}


def test_the_cli_text_report(short: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    p, out = short, short["dir"] / "cli.short.json"
    code = main(
        ["plan", "produce", str(p["stub"]), "--transcript", str(p["transcript"]),
         "--keep", str(p["keep"]), "--faces", str(p["faces"]), "--laughs", str(p["laughs"]),
         "--out", str(out)]
    )  # fmt: skip
    text = capsys.readouterr().out
    assert code == 0 and out.read_text(encoding="utf-8") == golden("short")
    assert text.startswith(f"wrote {out}: short, 3 segment(s), 2 cut(s)")
    assert "1 pause trim(s)" in text and "laugh onset 23.300 s" in text


def _profile(tmp_path: Path, edit: Any) -> Path:
    data = tomllib.loads(
        resources.files("aar.plan").joinpath("profiles", "ccn.toml").read_text(encoding="utf-8")
    )
    edit(data)
    lines: list[str] = []

    def emit(prefix: str, table: dict[str, Any]) -> None:
        scalars = {k: v for k, v in table.items() if not isinstance(v, (dict, list))}
        lists = {k: v for k, v in table.items() if isinstance(v, list)}
        if prefix:
            lines.append(f"[{prefix}]")
        lines.extend(f"{k} = {json.dumps(v)}" for k, v in scalars.items())
        for k, items in lists.items():
            for item in items:
                lines.append(f"[[{prefix + '.' if prefix else ''}{k}]]")
                lines.extend(f"{a} = {json.dumps(b)}" for a, b in item.items())
        for k, v in table.items():
            if isinstance(v, dict):
                emit(f"{prefix}.{k}" if prefix else k, v)

    emit("", data)
    path = tmp_path / "profile.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parse_profile(tomllib.loads(path.read_text(encoding="utf-8")))  # it round-trips
    return path


def test_without_room_to_nudge_the_join_is_named_and_qa_agrees(
    longform: dict[str, Path], tmp_path: Path
) -> None:
    path = _profile(tmp_path, lambda d: d["produce"].update(join_nudge_ms=0))
    result = produce_longform(longform, profile=load_profile(path))
    assert [(w.code, w.source, w.at_s) for w in result.warnings] == [
        ("end-dropped-speech", None, 35.202),
        ("join-not-quiet", "cam-b", 19.419),  # 19.40 s rounded inward to frame 582
    ]
    assert result.lint.ok  # lint cannot hear the audio: the warning is the planner's job
    qa = load_profile("ccn").qa
    second = cuts(result.plan, 1)[1]
    window = join_quiet_window(second.src_in_s, qa.join_quiet_search_s, None)
    measured = scan_join_quiet(
        longform["dir"] / "longform_b.wav",
        channels=1,
        src_in_s=second.src_in_s,
        window=window,
        qa=qa,
        timeout_s=60,
    )
    assert measured.judged.status == "fail"


def test_a_plan_the_profile_refuses_is_never_written(
    short: dict[str, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _profile(tmp_path, lambda d: d["rules"]["short-max-duration"].update(max_seconds=10))
    with pytest.raises(LintGateError) as exc:
        produce_short(short, profile=load_profile(path))
    assert exc.value.code == "lint-errors"
    assert [d["rule"] for d in exc.value.details] == ["short-max-duration"]
    out = tmp_path / "refused.json"
    p = short
    code = main(
        ["plan", "produce", str(p["stub"]), "--transcript", str(p["transcript"]),
         "--keep", str(p["keep"]), "--faces", str(p["faces"]), "--laughs", str(p["laughs"]),
         "--profile", str(path), "--out", str(out), "--json"]
    )  # fmt: skip
    body = json.loads(capsys.readouterr().out)
    assert code == 1 and not out.exists()
    assert body["ok"] is False and body["error"]["type"] == "lint-errors"


def test_bad_input_exits_2(longform: dict[str, Path], capsys: pytest.CaptureFixture[str]) -> None:
    p = longform
    code = main(
        ["plan", "produce", str(p["stub"]), "--transcript", str(p["transcript"]),
         "--keep", str(p["dir"] / "missing.json"), "--out", str(p["dir"] / "x.json")]
    )  # fmt: skip
    assert code == 2
    assert "cannot read the keep list" in capsys.readouterr().err


def test_a_loud_venue_hears_room_laughs_as_pauses(tmp_path: Path) -> None:
    """A room laugh at -38 dBFS RMS is speech-loud at -42 dB but a pause at the venue's -36."""
    from produce_support import amp, ffmpeg

    voice = f"{amp(-20)}*sin(2*PI*220*t)"
    room_laugh = f"{amp(-35)}*sin(2*PI*600*t)"  # a sine: RMS 3 dB under its peak
    expr = f"if(between(t\\,1\\,2.2)\\,{room_laugh}\\,{voice})"
    path = tmp_path / "laugh.wav"
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s=16000:d=3.4", "-c:a", "pcm_s16le", str(path))
    audio = {"a": SourceAudio(path, 1, 3.4)}
    profile = load_profile("ccn")
    s = profile.produce
    fps = load_stub(FIXTURES / "short.stub.json").fps
    whole = [Piece("a", 0, fps.frames(3.4), 0)]

    def runs(venue: str | None) -> list[tuple[float, float]]:
        return pause_runs(
            whole,
            audio,
            {"a": 0},
            fps,
            threshold_db=s.silence_for(venue),
            window_s=s.scan_window_ms / 1000,
            rate=s.scan_rate_hz,
            chunk_s=s.scan_chunk_s,
        )

    assert runs(None) == []
    [(start, end)] = runs("loud")
    assert start == pytest.approx(1.0, abs=0.02) and end == pytest.approx(2.2, abs=0.02)


def test_the_pause_scan_joins_runs_across_its_chunks(tmp_path: Path) -> None:
    from produce_support import amp, ffmpeg

    expr = f"if(between(t\\,1\\,2.6)\\,0\\,{amp(-20)}*sin(2*PI*220*t))"
    path = tmp_path / "gap.wav"
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s=16000:d=4", "-c:a", "pcm_s16le", str(path))
    fps = load_stub(FIXTURES / "short.stub.json").fps
    for chunk in (0.7, 30.0):  # 0.7 s chunks put boundaries at 1.4 and 2.1 s, inside the gap
        [(start, end)] = pause_runs(
            [Piece("a", 0, fps.frames(4.0), 0)],
            {"a": SourceAudio(path, 1, 4.0)},
            {"a": 0},
            fps,
            threshold_db=-42.0,
            window_s=0.01,
            rate=16000,
            chunk_s=chunk,
        )
        assert start == pytest.approx(1.0, abs=0.002) and end == pytest.approx(2.6, abs=0.002)


def test_real_sync_output_drives_the_origins(longform: dict[str, Path]) -> None:
    pytest.importorskip("numpy")
    from fractions import Fraction

    from aar.analyze.sync import sync_files

    folder = longform["dir"]
    report = sync_files(
        folder / "longform_a.wav", [folder / "longform_b.wav"], fps=Fraction(30000, 1001)
    )
    [pair] = report["pairs"]
    assert pair["status"] == "ok" and pair["suggested_recording_origin"] == 60
    committed = load_sync(longform["sync"])
    assert committed.pairs[0].suggested_recording_origin == pair["suggested_recording_origin"]


# ---------------------------------------------------------------- round 2: small rendered shows

FPS2997 = "30000/1001"


def _mini(
    tmp_path: Path,
    bursts: list[tuple[float, float]],
    words: list[tuple[str, float, float]],
    keeps: list[tuple[float, float]],
    *,
    seconds: float,
    silent: bool = False,
) -> Any:
    """One camera; ``bursts`` are where the voice really sounds, ``words`` what Whisper says."""
    from produce_support import VOICES, amp, ffmpeg

    if silent:
        source = f"anullsrc=r=16000:cl=mono:d={seconds}"
    else:
        gate = "+".join(f"between(t\\,{a}\\,{b})" for a, b in bursts) or "0"
        expr = f"{amp(-60)}*sin(2*PI*120*t)+({gate})*{amp(-20)}*{VOICES[0]}"
        source = f"aevalsrc={expr}:s=16000:d={seconds}"
    ffmpeg("-f", "lavfi", "-i", source, "-c:a", "pcm_s16le", str(tmp_path / "mini.wav"))
    frames = int(seconds * 30000 / 1001) - 1
    stub = EditPlan.model_validate(
        {
            "project": "p",
            "timeline": "t",
            "kind": "longform",
            "fps": FPS2997,
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [
                {"id": "cam", "media": "mini.wav", "camera": "A", "shot_size": "close",
                 "recording_id": "r", "duration": frames}
            ],
            "segments": [],
        }
    )  # fmt: skip
    from aar.plan.inputs import Transcript
    from aar.plan.produce import KeepList

    return produce(
        stub,
        transcript=Transcript.model_validate(
            {"recording": "r",
             "words": [{"text": t, "start_s": a, "end_s": b} for t, a, b in words]}
        ),
        keep=KeepList.model_validate(
            {"ranges": [{"start_s": a, "end_s": b} for a, b in keeps]}
        ),
        profile=load_profile("ccn"),
        media_root=tmp_path,
    )  # fmt: skip


def test_the_next_removed_words_onset_never_leaks_into_the_out_handle(tmp_path: Path) -> None:
    """F4 (the 10k run's out-handle leaks): the human's out at 2.24 s is quiet before, but the
    removed "next" sounds from 2.25 s, inside the crossfade's outgoing frame. The out point
    moves earlier (inward) until quiet covers 60 ms before it and the handle after it."""
    result = _mini(
        tmp_path,
        bursts=[(1.0, 2.0), (2.25, 2.7), (4.0, 5.0)],
        words=[("kept", 1.0, 2.0), ("next", 2.3, 2.7), ("after", 4.0, 5.0)],
        keeps=[(0.5, 2.24), (3.8, 5.5)],
        seconds=6.0,
    )
    out = [n for n in result.nudges if n.side == "out"]
    assert len(out) == 1 and out[0].to_frame < out[0].from_frame  # earlier only
    assert not [w for w in result.warnings if w.code == "join-out-not-quiet"]
    first = result.plan.segments[0]
    assert first.src_out * 1001 / 30000 + 1001 / 30000 <= 2.25  # the handle ends before "next"


def test_a_real_word_tail_past_whispers_end_is_named(tmp_path: Path) -> None:
    """F4 (the 10k run's audio splices): Whisper ends "kept" at 2.05 s but it sounds to 2.20 s,
    and the human's out at 2.10 s cuts that tail. Moving earlier would cut more of the word,
    and later is the human's cut, so the producer warns, naming the join."""
    result = _mini(
        tmp_path,
        bursts=[(1.0, 2.2), (4.0, 5.0)],
        words=[("kept", 1.0, 2.05), ("after", 4.0, 5.0)],
        keeps=[(0.5, 2.1), (3.8, 5.5)],
        seconds=6.0,
    )
    warned = [w for w in result.warnings if w.code == "join-out-not-quiet"]
    assert len(warned) == 1 and "join 1" in warned[0].message
    assert result.plan.segments[0].src_out == 62  # 2.1 s floored to frame 62, never moved outward


@pytest.mark.parametrize("silent", [True, False])
@pytest.mark.parametrize("kept", [60.8, 20.0])
def test_a_long_source_with_no_pause_at_all_is_a_warning(
    tmp_path: Path, silent: bool, kept: float
) -> None:
    """F10: a source over 60 s of silence, or of unbroken sound, has no quiet stretch at all,
    not even between words: suspicious, however little of it is kept."""
    result = _mini(
        tmp_path,
        bursts=[] if silent else [(0.0, 61.0)],
        words=[("drone", 1.0, kept - 0.5)],
        keeps=[(0.5, kept)],
        seconds=61.0,
        silent=silent,
    )
    assert [(w.code, w.source) for w in result.warnings] == [("no-pauses", "cam")]


def test_ordinary_speech_gives_no_no_pauses_warning(longform: dict[str, Path]) -> None:
    assert "no-pauses" not in [w.code for w in produce_longform(longform).warnings]


def test_a_long_form_with_a_face_track_warns_that_it_is_ignored(
    longform: dict[str, Path],
) -> None:
    """V31."""
    from aar.plan.produce import FaceTrack

    track = FaceTrack.model_validate(
        {
            "schema": "aar.faces/1",
            "header": {
                "frame_size": [640, 360],
                "input": {"path": str(longform["dir"] / "longform_a.wav")},
            },
            "frames": [{"file_s": 0.0, "faces": [{"box": [0.4, 0.2, 0.1, 0.2]}]}],
        }
    )
    result = produce_longform(longform, faces=[track])
    assert "faces-ignored" in [w.code for w in result.warnings]


def test_the_pause_scan_reads_the_last_chunk_of_a_part(tmp_path: Path) -> None:
    """V34: with 0.7 s chunks the last one starts at 3.5 s, within a second of the end."""
    from produce_support import amp, ffmpeg

    expr = f"if(between(t\\,3.2\\,3.9)\\,0\\,{amp(-20)}*sin(2*PI*220*t))"
    path = tmp_path / "tail.wav"
    ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s=16000:d=4", "-c:a", "pcm_s16le", str(path))
    fps = load_stub(FIXTURES / "short.stub.json").fps
    [(start, end)] = pause_runs(
        [Piece("a", 0, fps.frames(4.0), 0)],
        {"a": SourceAudio(path, 1, 4.0)},
        {"a": 0},
        fps,
        threshold_db=-42.0,
        window_s=0.01,
        rate=16000,
        chunk_s=0.7,
    )
    assert start == pytest.approx(3.2, abs=0.002) and end == pytest.approx(3.9, abs=0.002)


def test_no_join_is_quiet_at_or_past_the_end_of_the_audio(longform: dict[str, Path]) -> None:
    """V35: the source audio ends at 40 s; a cut there has no audio to start in."""
    from aar.plan.produce.audio import JoinJudge

    stub = load_stub(FIXTURES / "longform.stub.json")
    path = longform["dir"] / "longform_a.wav"
    judge = JoinJudge({"a": SourceAudio(path, 1, 40.0)}, load_profile("ccn").qa, stub.fps)
    end = stub.fps.frames(40.0)
    assert judge.in_quiet("a", stub.fps.frames(39.9), handle=0)  # room tone before the end
    # QA's window is clipped at the end, so 0.17 s past it there is nothing left to read
    assert not judge.in_quiet("a", end + 5, handle=0)
    assert not judge.out_quiet("a", end, handle=1)


def test_the_out_side_pre_roll_is_its_own_profile_key(tmp_path: Path) -> None:
    """R2-2 (kills W08): the out point needs quiet from ``out_quiet_preroll_ms`` (20 ms) before
    it, not none and not the in side's 60 ms."""
    from produce_support import amp, ffmpeg

    from aar.plan.model import Fps
    from aar.plan.produce.audio import JoinJudge

    fps = Fps(num=25)
    qa = load_profile("ccn").qa
    assert load_profile("ccn").produce.out_quiet_preroll_ms == 20.0
    for loud_until, passes_20, passes_60, passes_0 in (
        (0.990, False, False, True),  # tone ends 10 ms before the cut: no pre-roll would pass it
        (0.970, True, False, True),  # 30 ms of quiet before the cut: 20 ms passes, 60 ms fails
    ):
        expr = f"if(lt(t\\,{loud_until})\\,{amp(-20)}*sin(2*PI*220*t)\\,0)"
        path = tmp_path / f"out_{loud_until}.wav"
        ffmpeg("-f", "lavfi", "-i", f"aevalsrc={expr}:s=16000:d=3", "-c:a", "pcm_s16le", str(path))
        audio = {"a": SourceAudio(path, 1, 3.0)}
        frame = fps.frames(1.0)
        for preroll, expected in ((20.0, passes_20), (60.0, passes_60), (0.0001, passes_0)):
            judge = JoinJudge(audio, qa, fps, preroll)
            assert judge.out_quiet("a", frame, handle=0) is expected, (loud_until, preroll)
