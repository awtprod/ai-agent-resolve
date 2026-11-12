"""The plan stub builder (S19a): ``aar.plan.stub`` and ``aar plan stub`` on synthetic media."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aar.cli import main
from aar.plan.lint import lint
from aar.plan.produce.build import produce
from aar.plan.produce.inputs import KeepList, load_stub
from aar.plan.profile import load_profile
from aar.plan.stub import StubError, StubSource, build_stub
from ccn_support import KEEP, make_show, needs_ffmpeg, transcript, write_json

pytestmark = needs_ffmpeg

HOUR_2997 = 108000  # 01:00:00:00 at 30000/1001, non-drop


def _stub(sources: list[StubSource], **kw: object) -> object:
    args: dict[str, object] = {
        "recording_id": "show-1",
        "project": "Synthetic Show",
        "timeline": "Synthetic v1",
        "kind": "longform",
        "profile": "ccn",
    }
    args.update(kw)
    return build_stub(sources, **args)  # type: ignore[arg-type]


def _code(sources: list[StubSource], **kw: object) -> str:
    with pytest.raises(StubError) as caught:
        _stub(sources, **kw)
    return caught.value.code


def test_one_camera_stub_records_media_frames_size_and_timecode(tmp_path: Path) -> None:
    media = make_show(tmp_path / "cam a.mov")
    result = _stub([StubSource(media, camera="A", shot_size="wide")])
    plan = result.plan  # type: ignore[attr-defined]
    assert (plan.fps.num, plan.fps.den) == (30000, 1001)
    assert (plan.resolution.width, plan.resolution.height) == (320, 180)
    assert plan.segments == [] and plan.transitions == []
    (source,) = plan.sources
    assert source.id == "cam-a"
    assert source.media == str(media.resolve())
    assert source.camera == "A" and source.shot_size == "wide"
    assert source.recording_id == "show-1" and source.recording_origin == 0
    assert source.duration == 360  # 12 s at 29.97
    assert list(source.frame_size) == [320, 180]
    assert source.start_timecode == HOUR_2997
    assert (source.start_timecode_rate.num, source.start_timecode_rate.den) == (30000, 1001)
    report = result.report  # type: ignore[attr-defined]
    assert report["sources"][0]["start_timecode"]["timecode"] == "01:00:00:00"
    assert report["unknown_timecode"] == []


def test_the_stub_round_trips_through_load_stub_and_produce_lint_clean(tmp_path: Path) -> None:
    media = make_show(tmp_path / "a.mov")
    result = _stub([StubSource(media, camera="A", shot_size="wide")])
    path = tmp_path / "stub.json"
    path.write_text(result.plan.model_dump_json(exclude_none=True), encoding="utf-8")  # type: ignore[attr-defined]
    stub = load_stub(path)
    profile = load_profile("ccn")
    words = transcript("show-1")
    from aar.plan.inputs import Transcript

    produced = produce(
        stub,
        transcript=Transcript.model_validate(words),
        keep=KeepList.model_validate({"recording": "show-1", **KEEP}),
        profile=profile,
        pause_trim=False,
    )
    assert len(produced.plan.segments) == 2
    report = lint(produced.plan, profile, transcript=Transcript.model_validate(words))
    assert report.ok, [f.message for f in report.findings if f.severity == "error"]
    # produce keeps the stub's timecode and size (no mismatch warnings)
    assert produced.plan.sources[0].start_timecode == HOUR_2997
    assert not [w for w in produced.warnings if "timecode" in w.code]


def test_two_cameras_with_ids_and_a_base_that_passes_through(tmp_path: Path) -> None:
    a = make_show(tmp_path / "a.mov")
    b = make_show(tmp_path / "b.mov", timecode="01:00:02:00")
    outro = {"id": "outro", "media": "ccn_outro.mov", "recording_id": "outro", "duration": 30}
    base = {
        "template": "CCN Long Template",
        "head": {"video": "dissolve", "video_frames": 15, "audio_fade_frames": 15},
        "outro": {"source": "outro", "src_out": 30, "transition": {"video": "cut"}},
        "audio_tracks": [{"name": "A1"}],
        "sources": [outro],
        "segments": [{"ignored": True}],
        "project": "not this one",
    }
    result = _stub(
        [
            StubSource(a, camera="A", shot_size="wide", id="wide"),
            StubSource(b, camera="B", shot_size="close", id="close"),
        ],
        base=base,
    )
    plan = result.plan  # type: ignore[attr-defined]
    assert [s.id for s in plan.sources] == ["wide", "close", "outro"]
    assert plan.sources[1].start_timecode == HOUR_2997 + 60
    assert plan.template == "CCN Long Template"
    assert plan.head is not None and plan.outro is not None and plan.outro.source == "outro"
    assert [t.name for t in plan.audio_tracks] == ["A1"]
    assert plan.project == "Synthetic Show"
    assert result.report["base_ignored"] == ["project", "segments"]  # type: ignore[attr-defined]


def test_refusals_are_typed(tmp_path: Path) -> None:
    good = make_show(tmp_path / "good.mov")
    assert _code([StubSource(tmp_path / "missing.mov")]) == "unreadable"
    text = tmp_path / "notes.mov"
    text.write_text("not media", encoding="utf-8")
    assert _code([StubSource(text)]) == "unreadable"
    silent = make_show(tmp_path / "silent.mov", audio=False)
    assert _code([StubSource(silent)]) == "no_audio"
    vfr = make_show(tmp_path / "vfr.mov", drop_frame_at=100)
    assert _code([StubSource(vfr)]) == "vfr"
    pal = make_show(tmp_path / "pal.mov", rate="25", timecode="01:00:00:00")
    assert _code([StubSource(good), StubSource(pal)]) == "rates_differ"
    assert _code([StubSource(good, id="x"), StubSource(good, id="x")]) == "duplicate_source_id"
    assert _code([StubSource(good, camera="A"), StubSource(good, camera="a")]) == (
        "duplicate_source_id"
    )
    assert _code([StubSource(good)], kind="short") == "resolution_required"
    bad_base = {"outro": {"source": "nowhere", "src_out": 3}}
    assert _code([StubSource(good)], base=bad_base) == "bad_base"


def test_an_unknown_timecode_is_refused_unless_the_caller_opts_out(tmp_path: Path) -> None:
    bare = make_show(tmp_path / "bare.mov", timecode=None)
    assert _code([StubSource(bare)]) == "timecode_unknown"
    result = _stub([StubSource(bare)], allow_unknown_timecode=True)
    assert result.plan.sources[0].start_timecode is None  # type: ignore[attr-defined]
    assert result.report["unknown_timecode"] == ["cam-a"]  # type: ignore[attr-defined]


def test_cli_writes_the_stub_and_refuses_with_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    media = make_show(tmp_path / "a.mov")
    out = tmp_path / "stub.json"
    argv = ["plan", "stub", str(media), "--out", str(out), "--recording", "show-1"]
    argv += ["--project", "P", "--timeline", "T", "--camera", "A", "--shot-size", "wide", "--json"]
    assert main(argv) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True and report["sources"][0]["duration"] == 360
    assert load_stub(out).sources[0].camera == "A"
    assert main(argv) == 2  # exists, no --force
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "invalid_argument"
    assert main([*argv, "--force"]) == 0
    capsys.readouterr()
    missing = ["plan", "stub", str(tmp_path / "nope.mov"), "--out", str(tmp_path / "s2.json")]
    missing += ["--recording", "r", "--project", "P", "--timeline", "T", "--json"]
    assert main(missing) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "unreadable"
    assert not (tmp_path / "s2.json").exists()
    two = [*argv[:3], str(media), "--out", str(tmp_path / "s3.json"), "--recording", "r"]
    two += ["--project", "P", "--timeline", "T", "--camera", "A", "--json"]
    assert main(two) == 2  # one --camera for two files
    assert "--camera" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_keep_ranges_from_the_written_stub_file_lint_clean(tmp_path: Path) -> None:
    media = make_show(tmp_path / "a.mov")
    stub = tmp_path / "stub.json"
    argv = ["plan", "stub", str(media), "--out", str(stub), "--recording", "show-1"]
    argv += ["--project", "P", "--timeline", "T", "--shot-size", "wide"]
    assert main(argv) == 0
    t = write_json(tmp_path / "t.json", transcript("show-1"))
    keep = write_json(tmp_path / "keep.json", {"recording": "show-1", **KEEP})
    plan = tmp_path / "plan.json"
    produce_argv = ["plan", "produce", str(stub), "--transcript", str(t), "--keep", str(keep)]
    assert main([*produce_argv, "--out", str(plan), "--no-pause-trim"]) == 0
    assert main(["plan", "lint", str(plan), "--transcript", str(t)]) == 0


def test_a_fifo_a_non_utf8_path_and_one_file_twice_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import os

    fifo = tmp_path / "pipe.mov"
    os.mkfifo(fifo)
    assert _code([StubSource(fifo)]) == "unreadable"  # never handed to ffprobe
    good = make_show(tmp_path / "a.mov")
    link = tmp_path / "same.mov"
    link.symlink_to(good)
    assert _code([StubSource(good), StubSource(good, camera="B")]) == "duplicate_media"
    assert _code([StubSource(good), StubSource(link, camera="B")]) == "duplicate_media"
    odd = Path(os.fsdecode(bytes(tmp_path) + b"/caf\xe9.mov"))
    make_show(odd)
    argv = ["plan", "stub", str(odd), "--out", str(tmp_path / "s.json"), "--recording", "r"]
    assert main([*argv, "--project", "P", "--timeline", "T", "--json"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "unreadable"


def test_the_resolution_is_the_first_sources_display_size(tmp_path: Path) -> None:
    small = make_show(tmp_path / "small.mov")
    big = make_show(tmp_path / "big.mov", size="640x360")
    plan = _stub([StubSource(small), StubSource(big, camera="B")]).plan  # type: ignore[attr-defined]
    assert (plan.resolution.width, plan.resolution.height) == (320, 180)
    plan = _stub([StubSource(big), StubSource(small, camera="B")]).plan  # type: ignore[attr-defined]
    assert (plan.resolution.width, plan.resolution.height) == (640, 360)
