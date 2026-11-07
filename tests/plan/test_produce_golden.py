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
    assert result.warnings == []
    assert result.lint.ok and result.lint.counts["error"] == 0
    log = result.log()
    assert len(log["trims"]) == 3
    # the 24.5-25.6 s pause spans the B -> A camera change and is trimmed as one pause
    assert log["trims"][1]["pause_s"][0] < 25.0 < log["trims"][1]["pause_s"][1]
    # the mumble at 8.0 s splits its 1.2 s quiet run, so that pause is not trimmed
    assert not any(7.5 < t["pause_s"][0] < 8.9 for t in log["trims"])
    # the keep edge at 19.95 s sits in "and", which sounds from 19.92 s: one frame earlier
    assert [(n["join"], n["from_frame"], n["to_frame"]) for n in log["nudges"]] == [(2, 598, 597)]
    # camera B's half-second flash at 31.0 s folds into the A shot before it
    assert [(f["source"], f["into"]) for f in log["folds"]] == [("cam-b", "previous")]
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
    assert [seg.reframe[0].zoom for seg in plan.segments] == [1.0, 1.25, 1.0]
    # the face sways inside the dead zone for the first shot: the camera does not move
    assert len({key.x for key in plan.segments[0].reframe}) == 1
    # then it follows the walk and the lunge
    assert len({key.x for key in plan.segments[1].reframe}) > 3
    assert result.lint.duration_s < 37


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
    assert report["warnings"] == [] and report["lint"]["counts"]["error"] == 0
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
        ("join-not-quiet", "cam-b", 19.953)
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
