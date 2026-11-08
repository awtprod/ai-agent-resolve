"""Source.start_timecode (S18 round 4): an optional, additive EditPlan v2 field, the media's start
timecode as a frame count at the source's rate. Validated by the model and the JSON schema, left
out of plan_hash while unset."""

from __future__ import annotations

from typing import Any

import pytest
from apply_builders import longform_data
from pydantic import ValidationError

from aar.plan.apply.inputs import plan_hash
from aar.plan.model import EditPlan


def _with(value: Any) -> dict[str, Any]:
    data = longform_data()
    data["sources"][0]["start_timecode"] = value
    return data


@pytest.mark.parametrize("value", [0, 86400, 108000, 107892])
def test_a_whole_frame_count_is_accepted(value: int) -> None:
    assert EditPlan.model_validate(_with(value)).sources[0].start_timecode == value


@pytest.mark.parametrize("value", [-1, 1.5, 86400.0, True, "01:00:00:00", "86400"])
def test_anything_else_is_refused(value: Any) -> None:
    with pytest.raises(ValidationError) as caught:
        EditPlan.model_validate(_with(value))
    assert caught.value.errors()[0]["loc"] == ("sources", 0, "start_timecode")


def test_the_schema_says_a_whole_number_from_zero() -> None:
    source = EditPlan.model_json_schema()["$defs"]["Source"]["properties"]["start_timecode"]
    whole = next(option for option in source["anyOf"] if option.get("type") == "integer")
    assert whole["minimum"] == 0
    assert {"type": "null"} in source["anyOf"]
    assert "00:00:00:00" in source["description"]


@pytest.mark.parametrize("value", [-1, 1.5, "01:00:00:00"])
def test_the_schema_refuses_what_the_model_refuses(value: Any) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(EditPlan.model_json_schema())
    assert list(validator.iter_errors(_with(86400))) == []
    assert list(validator.iter_errors(_with(value))) != []


def test_plan_hash_leaves_it_out_while_unset() -> None:
    """Like frame_size: every plan written before the field keeps its hash (and the phase
    results stamped with it); a set value changes the hash."""
    plain = EditPlan.model_validate(longform_data())
    unset = EditPlan.model_validate(_with(None))
    assert plan_hash(unset) == plan_hash(plain)
    assert plan_hash(EditPlan.model_validate(_with(86400))) != plan_hash(plain)
    assert plan_hash(EditPlan.model_validate(_with(0))) != plan_hash(plain)


def test_a_version_1_plan_may_not_use_it() -> None:
    data = _with(86400)
    data["version"] = 1
    for key in ("template", "head", "outro", "audio_tracks", "end_fade_frames"):
        data.pop(key, None)
    with pytest.raises(ValidationError, match=r"sources\[0\]\.start_timecode"):
        EditPlan.model_validate(data)


# -- Source.start_timecode_rate (fix-check Q3) -------------------------------------------------


def _rate(value: Any) -> dict[str, Any]:
    data = _with(90000)
    data["sources"][0]["start_timecode_rate"] = value
    return data


@pytest.mark.parametrize(
    ("value", "want"),
    [
        ({"num": 25}, (25, 1)),
        ({"num": 30000, "den": 1001}, (30000, 1001)),
        ("25", (25, 1)),
        (24, (24, 1)),
    ],
)
def test_the_rate_takes_the_fps_forms(value: Any, want: tuple[int, int]) -> None:
    rate = EditPlan.model_validate(_rate(value)).sources[0].start_timecode_rate
    assert rate is not None and (rate.num, rate.den) == want


@pytest.mark.parametrize("value", [0, -25, {"num": 0}, {"num": 25, "den": 0}, "x", True])
def test_a_bad_rate_is_refused(value: Any) -> None:
    with pytest.raises(ValidationError):
        EditPlan.model_validate(_rate(value))


def test_plan_hash_leaves_the_rate_out_while_unset() -> None:
    assert plan_hash(EditPlan.model_validate(_with(90000))) == plan_hash(
        EditPlan.model_validate(_rate(None))
    )
    assert plan_hash(EditPlan.model_validate(_rate(25))) != plan_hash(
        EditPlan.model_validate(_with(90000))
    )
