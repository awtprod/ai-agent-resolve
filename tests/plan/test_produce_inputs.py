"""Plan producer inputs: every malformed input fails with a typed, coded error."""

from __future__ import annotations

import copy
import json
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any

import pytest
from produce_support import FIXTURES, load
from pydantic import ValidationError

from aar.plan.inputs import LaughMap, Transcript
from aar.plan.model import EditPlan
from aar.plan.produce import (
    CameraSchedule,
    FaceTrack,
    KeepList,
    ProduceInputError,
    SyncReport,
    apply_sync,
    bind_faces,
    check_profile,
    load_faces,
    load_keep,
    load_schedule,
    load_stub,
    load_sync,
    produce,
    read_laughs,
    read_transcript,
)
from aar.plan.profile import (
    FramingPreset,
    ProduceSettings,
    Profile,
    ProfileError,
    load_profile,
    parse_profile,
)

V1_COMPAT = Path(__file__).parent / "fixtures" / "v1_compat" / "ccn_v1.toml"


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def raw_ccn() -> dict[str, Any]:
    text = resources.files("aar.plan").joinpath("profiles", "ccn.toml").read_text(encoding="utf-8")
    return tomllib.loads(text)


def stub() -> EditPlan:
    return EditPlan.model_validate(load("longform", "stub"))


def stub_with(edit: Any) -> EditPlan:
    data = load("longform", "stub")
    edit(data)
    return EditPlan.model_validate(data)


def check_code(info: pytest.ExceptionInfo[Any], code: str) -> None:
    assert info.value.code == code


# ---------------------------------------------------------------- loaders: keep


def test_keep_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        load_keep(tmp_path / "nope.json")
    check_code(info, "input-missing")


def test_keep_not_json(tmp_path: Path) -> None:
    path = tmp_path / "keep.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ProduceInputError) as info:
        load_keep(path)
    check_code(info, "invalid-json")


@pytest.mark.parametrize(
    "data",
    [
        {"ranges": [{"start_s": 5.0, "end_s": 5.0}]},
        {"ranges": [{"start_s": 5.0, "end_s": 4.0}]},
        {"ranges": [{"start_s": 1.0, "end_s": 5.0}, {"start_s": 4.0, "end_s": 8.0}]},
        {"ranges": []},
        {"ranges": [{"start_s": 1.0, "end_s": 2.0}], "bogus": 1},
        {"ranges": [{"start_s": 1.0, "end_s": 2.0, "bogus": 1}]},
    ],
    ids=["empty-range", "reversed", "overlap", "no-ranges", "unknown-key", "unknown-range-key"],
)
def test_keep_invalid(tmp_path: Path, data: dict[str, Any]) -> None:
    path = write_json(tmp_path / "keep.json", data)
    with pytest.raises(ProduceInputError) as info:
        load_keep(path)
    check_code(info, "invalid-keep")
    assert isinstance(info.value.details, list)
    assert info.value.details


def test_keep_empty_bare_list(tmp_path: Path) -> None:
    path = write_json(tmp_path / "keep.json", [])
    with pytest.raises(ProduceInputError) as info:
        load_keep(path)
    check_code(info, "invalid-keep")


def test_keep_nan_start(tmp_path: Path) -> None:
    path = tmp_path / "keep.json"
    path.write_text('{"ranges": [{"start_s": NaN, "end_s": 4.0}]}', encoding="utf-8")
    with pytest.raises(ProduceInputError) as info:
        load_keep(path)
    check_code(info, "invalid-keep")


def test_keep_bare_list_and_touching_ranges_accepted(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "keep.json",
        [{"start_s": 1.0, "end_s": 4.0}, {"start_s": 4.0, "end_s": 6.0, "label": "x"}],
    )
    keep = load_keep(path)
    assert [(r.start_s, r.end_s) for r in keep.ranges] == [(1.0, 4.0), (4.0, 6.0)]


def test_committed_keep_loads() -> None:
    assert len(load_keep(FIXTURES / "longform.keep.json").ranges) == 2


# ---------------------------------------------------------------- loaders: schedule


@pytest.mark.parametrize(
    "spans",
    [
        [
            {"start_s": 0.0, "end_s": 5.0, "camera": "A"},
            {"start_s": 4.0, "end_s": 8.0, "camera": "B"},
        ],
        [{"start_s": 0.0, "end_s": 5.0, "camera": ""}],
    ],
    ids=["overlap", "empty-camera"],
)
def test_schedule_invalid(tmp_path: Path, spans: list[dict[str, Any]]) -> None:
    path = write_json(tmp_path / "schedule.json", {"spans": spans})
    with pytest.raises(ProduceInputError) as info:
        load_schedule(path)
    check_code(info, "invalid-schedule")
    assert info.value.details


def test_schedule_bare_list_accepted(tmp_path: Path) -> None:
    path = write_json(
        tmp_path / "schedule.json",
        [{"start_s": 0.0, "end_s": 5.0, "camera": "A"}],
    )
    assert load_schedule(path).spans[0].camera == "A"


# ---------------------------------------------------------------- loaders: faces


def faces_data() -> dict[str, Any]:
    return copy.deepcopy(load("short", "faces"))


def test_faces_wrong_schema(tmp_path: Path) -> None:
    data = faces_data()
    data["schema"] = "aar.faces/2"
    with pytest.raises(ProduceInputError) as info:
        load_faces(write_json(tmp_path / "f.json", data))
    check_code(info, "invalid-faces")


def test_faces_frames_not_increasing(tmp_path: Path) -> None:
    data = faces_data()
    data["frames"] = [data["frames"][0], copy.deepcopy(data["frames"][0])]
    with pytest.raises(ProduceInputError) as info:
        load_faces(write_json(tmp_path / "f.json", data))
    check_code(info, "invalid-faces")


@pytest.mark.parametrize("box", [[0.3, 0.2, 0.0, 0.3], [0.3, 0.2, 1.5, 0.3]], ids=["w0", "over1"])
def test_faces_bad_box(tmp_path: Path, box: list[float]) -> None:
    data = faces_data()
    data["frames"][0]["faces"][0]["box"] = box
    with pytest.raises(ProduceInputError) as info:
        load_faces(write_json(tmp_path / "f.json", data))
    check_code(info, "invalid-faces")


def test_faces_zero_frame_size(tmp_path: Path) -> None:
    data = faces_data()
    data["header"]["frame_size"] = [0, 360]
    with pytest.raises(ProduceInputError) as info:
        load_faces(write_json(tmp_path / "f.json", data))
    check_code(info, "invalid-faces")


def test_faces_committed_fixture_resolves_relative_path() -> None:
    track = load_faces(FIXTURES / "short.faces.json")
    assert track.header.input.path == str(FIXTURES / "short_master.mov")
    assert track.samples()


# ---------------------------------------------------------------- loaders: sync / stub


def test_sync_missing_reference(tmp_path: Path) -> None:
    data = copy.deepcopy(load("longform", "sync"))
    del data["reference"]
    with pytest.raises(ProduceInputError) as info:
        load_sync(write_json(tmp_path / "s.json", data))
    check_code(info, "invalid-sync")


def test_sync_committed_fixture_resolves_relative_files() -> None:
    report = load_sync(FIXTURES / "longform.sync.json")
    assert report.reference.file == str(FIXTURES / "longform_a.wav")
    assert report.pairs[0].file == str(FIXTURES / "longform_b.wav")
    assert report.pairs[0].suggested_recording_origin == 60


def test_stub_with_segments(tmp_path: Path) -> None:
    data = load("longform", "stub")
    data["segments"] = [{"track": 1, "source": "cam-a", "src_in": 0, "src_out": 60}]
    with pytest.raises(ProduceInputError) as info:
        load_stub(write_json(tmp_path / "stub.json", data))
    check_code(info, "stub-has-segments")


def test_stub_invalid_plan(tmp_path: Path) -> None:
    data = load("longform", "stub")
    data["kind"] = "movie"
    with pytest.raises(ProduceInputError) as info:
        load_stub(write_json(tmp_path / "stub.json", data))
    check_code(info, "invalid-stub")


def test_committed_stub_loads() -> None:
    assert load_stub(FIXTURES / "longform.stub.json").kind == "longform"


# ---------------------------------------------------------------- transcript / laughs


def test_transcript_word_ends_before_start(tmp_path: Path) -> None:
    data = {"words": [{"text": "hi", "start_s": 2.0, "end_s": 1.0}]}
    with pytest.raises(ProduceInputError) as info:
        read_transcript(write_json(tmp_path / "t.json", data))
    check_code(info, "invalid-transcript")


def test_laughs_bad(tmp_path: Path) -> None:
    data = {"laughs": [{"start_s": 2.0, "end_s": 1.0, "strength": 1.0}]}
    with pytest.raises(ProduceInputError) as info:
        read_laughs(write_json(tmp_path / "l.json", data))
    check_code(info, "invalid-laughs")


def test_transcript_and_laughs_missing(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        read_transcript(tmp_path / "t.json")
    check_code(info, "input-missing")
    with pytest.raises(ProduceInputError) as info:
        read_laughs(tmp_path / "l.json")
    check_code(info, "input-missing")


def test_transcript_not_json(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text("nope", encoding="utf-8")
    with pytest.raises(ProduceInputError) as info:
        read_transcript(path)
    check_code(info, "invalid-json")


# ---------------------------------------------------------------- apply_sync


def sync_report(tmp_path: Path, **change: Any) -> SyncReport:
    (tmp_path / "longform_a.wav").write_bytes(b"")
    (tmp_path / "longform_b.wav").write_bytes(b"")
    data: dict[str, Any] = {
        "fps": "30000/1001",
        "reference": {"file": str(tmp_path / "longform_a.wav"), "recording_origin": 0},
        "pairs": [
            {
                "file": str(tmp_path / "longform_b.wav"),
                "status": "ok",
                "suggested_recording_origin": 60,
            }
        ],
    }
    data.update(change)
    return SyncReport.model_validate(data)


def test_sync_valid_sets_origin(tmp_path: Path) -> None:
    plan = apply_sync(stub(), sync_report(tmp_path), tmp_path)
    origins = {s.id: s.recording_origin for s in plan.sources}
    assert origins["cam-b"] == 60
    assert origins["cam-a"] == 0


def test_sync_fps_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        apply_sync(stub(), sync_report(tmp_path, fps="25/1"), tmp_path)
    check_code(info, "sync-fps-mismatch")


def test_sync_reference_unknown(tmp_path: Path) -> None:
    ref = {"file": str(tmp_path / "other.wav"), "recording_origin": 0}
    with pytest.raises(ProduceInputError) as info:
        apply_sync(stub(), sync_report(tmp_path, reference=ref), tmp_path)
    check_code(info, "sync-reference-unknown")


def test_sync_origin_mismatch(tmp_path: Path) -> None:
    ref = {"file": str(tmp_path / "longform_a.wav"), "recording_origin": 5}
    with pytest.raises(ProduceInputError) as info:
        apply_sync(stub(), sync_report(tmp_path, reference=ref), tmp_path)
    check_code(info, "sync-origin-mismatch")


def test_sync_not_ok(tmp_path: Path) -> None:
    pairs = [
        {
            "file": str(tmp_path / "longform_b.wav"),
            "status": "low-confidence",
            "suggested_recording_origin": 60,
        }
    ]
    with pytest.raises(ProduceInputError) as info:
        apply_sync(stub(), sync_report(tmp_path, pairs=pairs), tmp_path)
    check_code(info, "sync-not-ok")


def test_sync_file_unknown(tmp_path: Path) -> None:
    pairs = [
        {
            "file": str(tmp_path / "stranger.wav"),
            "status": "ok",
            "suggested_recording_origin": 60,
        }
    ]
    with pytest.raises(ProduceInputError) as info:
        apply_sync(stub(), sync_report(tmp_path, pairs=pairs), tmp_path)
    check_code(info, "sync-file-unknown")


# ---------------------------------------------------------------- bind_faces


def track(path: Path | str, recording: str | None = None) -> FaceTrack:
    data = faces_data()
    data["header"]["input"]["path"] = str(path)
    data["header"]["recording"] = recording
    return FaceTrack.model_validate(data)


def test_faces_bind_valid(tmp_path: Path) -> None:
    bound = bind_faces(stub(), [track(tmp_path / "longform_a.wav")], tmp_path)
    assert list(bound) == ["cam-a"]


def test_faces_unmatched(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        bind_faces(stub(), [track(tmp_path / "nothing.mov")], tmp_path)
    check_code(info, "faces-unmatched")


def test_faces_duplicate(tmp_path: Path) -> None:
    both = [track(tmp_path / "longform_a.wav"), track(tmp_path / "longform_a.wav")]
    with pytest.raises(ProduceInputError) as info:
        bind_faces(stub(), both, tmp_path)
    check_code(info, "faces-duplicate")


def test_faces_recording_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        bind_faces(stub(), [track(tmp_path / "longform_a.wav", "other")], tmp_path)
    check_code(info, "recording-mismatch")


# ---------------------------------------------------------------- check_profile


def ccn_with(edit: Any) -> Profile:
    data = raw_ccn()
    edit(data)
    return parse_profile(data)


def test_check_profile_committed_passes() -> None:
    profile = load_profile("ccn")
    view = check_profile(profile, "longform")
    assert view.max_pan_speed is not None
    check_profile(profile, "short")


def test_check_end_after_outside_window() -> None:
    profile = ccn_with(lambda d: d["produce"].update(end_after_ms=450))
    with pytest.raises(ProfileError, match="end_after_ms"):
        check_profile(profile, "longform")


def test_check_punch_in_too_small() -> None:
    profile = ccn_with(lambda d: d["produce"].update(punch_in_zoom=1.1))
    with pytest.raises(ProfileError, match="punch_in_zoom"):
        check_profile(profile, "longform")


def test_check_single_framing_preset_short() -> None:
    profile = ccn_with(lambda d: d["produce"].update(framing=[{"name": "A", "zoom": 1.0}]))
    with pytest.raises(ProfileError, match="framing"):
        check_profile(profile, "short")


def test_check_crossfade_too_long() -> None:
    profile = ccn_with(lambda d: d["produce"].update(crossfade_frames=31))
    with pytest.raises(ProfileError, match="crossfade_frames"):
        check_profile(profile, "longform")


def test_check_whip_pan_rule_needed_for_short_only() -> None:
    profile = ccn_with(lambda d: d["rules"].pop("no-whip-pan"))
    with pytest.raises(ProfileError, match="no-whip-pan"):
        check_profile(profile, "short")
    check_profile(profile, "longform")


# ---------------------------------------------------------------- ProduceSettings


@pytest.mark.parametrize(
    "kwargs",
    [
        {"beat_s": 0.9},
        {"beat_s": 1.0},
        {"beat_lead_s": 0.5},
        {"punch_in_zoom": 1.0},
        {"framing": [{"name": "A", "zoom": 1.0}, {"name": "A", "zoom": 1.25}]},
        {"silence_db": 0},
        {"silence_db": float("nan")},
        {"min_pause_s": float("nan")},
    ],
    ids=[
        "beat-eq-pause",
        "beat-gt-pause",
        "lead-gt-beat",
        "zoom-1",
        "dup-framing",
        "silence-0",
        "nan-silence",
        "nan-pause",
    ],
)
def test_produce_settings_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ProduceSettings(**kwargs)


def test_framing_crop_outside_picture() -> None:
    with pytest.raises(ValidationError, match="outside the picture"):
        FramingPreset(name="X", zoom=1.0, y=0.4)


def test_toml_nan_refused() -> None:
    data = raw_ccn()
    data["produce"].update(tomllib.loads("min_pause_s = nan"))
    with pytest.raises(ProfileError):
        parse_profile(data)


def test_toml_invalid_produce_is_profile_error() -> None:
    data = raw_ccn()
    data["produce"]["beat_s"] = 2.0
    with pytest.raises(ProfileError, match="beat_s"):
        parse_profile(data)


def test_silence_for_venues() -> None:
    s = load_profile("ccn").produce
    assert s.silence_for("loud") == -36.0
    assert s.silence_for(None) == -42.0
    with pytest.raises(ProfileError, match="nope"):
        s.silence_for("nope")


def test_profile_without_produce_block_gets_defaults() -> None:
    s = load_profile(V1_COMPAT).produce
    assert s.min_pause_s == 0.9
    assert s.beat_s == 0.45
    assert s.silence_db == -42.0
    assert s.join_nudge_ms == 150
    assert s.min_shot_s == 1.0
    assert s.end_after_ms == 200


def test_committed_ccn_produce_values() -> None:
    s = load_profile("ccn").produce
    d = ProduceSettings()
    for name in (
        "default_camera",
        "min_pause_s",
        "beat_s",
        "beat_lead_s",
        "silence_db",
        "scan_window_ms",
        "scan_rate_hz",
        "scan_chunk_s",
        "join_nudge_ms",
        "min_shot_s",
        "end_after_ms",
        "crossfade_frames",
        "punch_in_zoom",
    ):
        assert getattr(s, name) == getattr(d, name), name
    assert {k: v.silence_db for k, v in s.venues.items()} == {"loud": -36.0}
    assert [p.name for p in s.framing] == ["A", "B"]
    assert [p.zoom for p in s.framing] == [1.0, 1.25]


# ---------------------------------------------------------------- produce(): before any audio


def run(tmp_path: Path, plan: EditPlan | None = None, **change: Any) -> Any:
    args: dict[str, Any] = {
        "transcript": Transcript.model_validate(load("longform", "transcript")),
        "keep": load_keep(FIXTURES / "longform.keep.json"),
        "profile": load_profile("ccn"),
        "media_root": tmp_path,
    }
    args.update(change)
    return produce(plan or stub(), **args)


def keep_of(*ranges: tuple[float, float]) -> Any:
    path = FIXTURES / "longform.keep.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["ranges"] = [{"start_s": a, "end_s": b} for a, b in ranges]
    return KeepList.model_validate(data)


def sched_of(*spans: tuple[float, float, str]) -> Any:
    return CameraSchedule.model_validate(
        {
            "recording": "show-1",
            "spans": [{"start_s": a, "end_s": b, "camera": c} for a, b, c in spans],
        }
    )


def test_produce_keep_too_short(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, keep=keep_of((3.0, 3.5)))
    check_code(info, "keep-too-short")


def test_produce_unknown_camera(tmp_path: Path) -> None:
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, schedule=sched_of((0.0, 40.0, "Z")))
    check_code(info, "unknown-camera")


def test_produce_ambiguous_camera(tmp_path: Path) -> None:
    def both_a(data: dict[str, Any]) -> None:
        data["sources"][1]["camera"] = "A"

    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, stub_with(both_a))
    check_code(info, "ambiguous-camera")


def test_produce_schedule_gap(tmp_path: Path) -> None:
    schedule = sched_of((0.0, 10.0, "A"), (11.0, 40.0, "B"))
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, schedule=schedule, keep=keep_of((1.0, 14.0)))
    check_code(info, "schedule-gap")


def test_produce_camera_does_not_cover(tmp_path: Path) -> None:
    def short_a(data: dict[str, Any]) -> None:
        data["sources"][0]["duration"] = 100

    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, stub_with(short_a), keep=keep_of((1.0, 14.0)))
    check_code(info, "camera-does-not-cover")


def test_produce_transcript_other_recording(tmp_path: Path) -> None:
    data = load("longform", "transcript")
    data["recording"] = "other"
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, transcript=Transcript.model_validate(data))
    check_code(info, "recording-mismatch")


def test_produce_laughs_other_recording(tmp_path: Path) -> None:
    data = load("longform", "laughs")
    data["recording"] = "other"
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, laughs=LaughMap.model_validate(data))
    check_code(info, "recording-mismatch")


def test_produce_mixed_recordings(tmp_path: Path) -> None:
    def big_outro(data: dict[str, Any]) -> None:
        data["sources"][2]["duration"] = 100000

    schedule = sched_of((0.0, 8.0, "A"), (8.0, 40.0, "outro"))
    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, stub_with(big_outro), schedule=schedule, keep=keep_of((1.0, 14.0)))
    check_code(info, "mixed-recordings")


def test_produce_unknown_venue(tmp_path: Path) -> None:
    # The venue is checked before any media is opened.
    with pytest.raises(ProfileError, match="nope"):
        run(tmp_path, venue="nope")


def test_produce_stub_with_segments(tmp_path: Path) -> None:
    def segment(data: dict[str, Any]) -> None:
        data["segments"] = [{"track": 1, "source": "cam-a", "src_in": 0, "src_out": 60}]

    with pytest.raises(ProduceInputError) as info:
        run(tmp_path, stub_with(segment))
    check_code(info, "stub-has-segments")
