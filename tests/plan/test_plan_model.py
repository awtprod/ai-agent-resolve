"""EditPlan schema, timebase helpers and the transcript / laugh-map inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from plan_support import FIXTURES, PlanData
from pydantic import ValidationError

from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.model import EditPlan, Fps


def test_clean_plan_round_trips_through_json() -> None:
    text = (FIXTURES / "short_clean.plan.json").read_text(encoding="utf-8")
    plan = EditPlan.model_validate_json(text)
    again = EditPlan.model_validate_json(plan.model_dump_json())
    assert again == plan
    # Only what the author wrote comes back out: defaults are not invented into the file.
    assert json.loads(plan.model_dump_json(exclude_unset=True)) == json.loads(text)


def test_schema_lists_the_plan_fields_and_the_required_ones() -> None:
    schema = EditPlan.model_json_schema()
    assert schema["title"] == "EditPlan"
    assert set(schema["properties"]) == {
        "version",
        "project",
        "timeline",
        "kind",
        "fps",
        "resolution",
        "profile",
        "sources",
        "segments",
        "transitions",
        "titles",
        "captions",
        "caption_style",
        "markers",
    }
    assert set(schema["required"]) == {
        "project",
        "timeline",
        "kind",
        "fps",
        "resolution",
        "profile",
        "sources",
        "segments",
    }
    assert schema["properties"]["kind"]["enum"] == ["longform", "short"]
    assert schema["additionalProperties"] is False


@pytest.mark.parametrize(
    ("raw", "num", "den"),
    [("24000/1001", 24000, 1001), ("25", 25, 1), (30, 30, 1), ({"num": 60, "den": 1}, 60, 1)],
)
def test_fps_accepts_string_int_and_object(raw: object, num: int, den: int) -> None:
    fps = Fps.model_validate(raw)
    assert (fps.num, fps.den) == (num, den)


@pytest.mark.parametrize("raw", ["abc", "24/0", 0, {"num": 24, "den": 0}])
def test_fps_rejects_nonsense(raw: object) -> None:
    with pytest.raises(ValidationError):
        Fps.model_validate(raw)


def test_timecode_uses_the_nominal_rate_and_handles_negatives() -> None:
    fps = Fps(num=24000, den=1001)
    assert fps.nominal == 24
    assert fps.timecode(240) == "00:00:10:00"
    assert fps.timecode(24 * 3600 + 24 * 60 + 24 + 5) == "01:01:01:05"
    assert fps.timecode(-25) == "-00:00:01:01"
    assert Fps(num=30, den=1).timecode(31) == "00:00:01:01"


def test_seconds_and_frames_are_inverse_at_a_fractional_rate() -> None:
    fps = Fps(num=24000, den=1001)
    assert fps.seconds(24000) == pytest.approx(1001.0)
    assert fps.frames(1.001) == 24
    assert fps.frames(fps.seconds(1234)) == 1234


def test_unknown_keys_are_rejected(clean_data: PlanData) -> None:
    clean_data["segments"][0]["speed"] = 2.0
    with pytest.raises(ValidationError, match="speed"):
        EditPlan.model_validate(clean_data)


def test_out_of_order_reframe_keys_are_rejected(clean_data: PlanData) -> None:
    clean_data["segments"][0]["reframe"] = [{"time": 10}, {"time": 5}]
    with pytest.raises(ValidationError, match="non-decreasing"):
        EditPlan.model_validate(clean_data)


def test_reframe_centre_must_stay_in_the_frame(clean_data: PlanData) -> None:
    clean_data["segments"][0]["reframe"] = [{"time": 0, "x": 1.2}]
    with pytest.raises(ValidationError):
        EditPlan.model_validate(clean_data)


def test_record_ranges_chain_per_track_and_honour_record_in(clean_data: PlanData) -> None:
    plan = EditPlan.model_validate(clean_data)
    assert plan.record_ranges() == [(0, 240), (240, 480), (480, 720)]
    assert plan.total_frames() == 720
    clean_data["segments"][1]["track"] = 2
    clean_data["segments"][1]["record_in"] = 100
    plan = EditPlan.model_validate(clean_data)
    assert plan.record_ranges() == [(0, 240), (100, 340), (240, 480)]
    assert plan.joins() == [(0, 2)]
    assert plan.total_frames() == 480


def test_recording_key_falls_back_to_the_source_id(clean_data: PlanData) -> None:
    clean_data["sources"][0]["recording_id"] = None
    plan = EditPlan.model_validate(clean_data)
    sources = plan.source_map()
    assert plan.recording_key(sources["cam-a"]) == "source:cam-a"
    assert plan.recording_key(sources["cam-b"]) == "take-1"


def test_transcript_file_may_be_a_bare_list_and_is_sorted(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps(
            [
                {"text": "b", "start_s": 2.0, "end_s": 2.5},
                {"text": "a", "start_s": 1.0, "end_s": 1.5, "speaker": "x"},
            ]
        ),
        encoding="utf-8",
    )
    transcript = load_transcript(path)
    assert [w.text for w in transcript.words] == ["a", "b"]
    assert transcript.recording is None
    assert transcript.words[0].speaker == "x"


def test_transcript_object_form_names_a_recording(tmp_path: Path) -> None:
    path = tmp_path / "t.json"
    path.write_text(
        json.dumps({"recording": "take-1", "words": [{"text": "a", "start_s": 0, "end_s": 1}]}),
        encoding="utf-8",
    )
    assert load_transcript(path).recording == "take-1"


def test_word_that_ends_before_it_starts_is_rejected() -> None:
    with pytest.raises(ValidationError, match="ends before it starts"):
        Transcript.model_validate({"words": [{"text": "a", "start_s": 2, "end_s": 1}]})


def test_laugh_map_defaults_strength_and_sorts(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    path.write_text(
        json.dumps([{"start_s": 5, "end_s": 6}, {"start_s": 1, "end_s": 2, "strength": 0.4}]),
        encoding="utf-8",
    )
    laughs = load_laughs(path)
    assert [(x.start_s, x.strength) for x in laughs.laughs] == [(1, 0.4), (5, 1.0)]
    with pytest.raises(ValidationError):
        LaughMap.model_validate({"laughs": [{"start_s": 3, "end_s": 1}]})
