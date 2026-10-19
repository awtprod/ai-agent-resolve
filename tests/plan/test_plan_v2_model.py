"""EditPlan version 2: the version field, the build-from-template fields and their checks."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from plan_support import FIXTURES, PlanData
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from aar.plan.model import EditPlan

#: A path into plan data: ("sources", 1, "color") is data["sources"][1]["color"].
Loc = tuple[str | int, ...]

#: One version 2 field, each as (dotted name in the error, where it sits, an example value).
V2_USES: list[tuple[str, Loc, Any]] = [
    ("template", ("template",), "Example Template"),
    ("head", ("head",), {"video": "dissolve", "video_frames": 15}),
    ("outro", ("outro",), {"source": "cam-a", "src_out": 10}),
    ("audio_tracks", ("audio_tracks",), [{"name": "A1"}]),
    ("end_fade_frames", ("end_fade_frames",), 1),
    ("sources[1].color", ("sources", 1, "color"), {"lut": "match.cube"}),
    ("segments[2].transform", ("segments", 2, "transform"), {"zoom": 1.25}),
]
V2_IDS = [use[0] for use in V2_USES]

#: The same places holding nothing: null, or an empty list. That is not a use.
V2_EMPTY: list[tuple[Loc, Any]] = [
    (("template",), None),
    (("head",), None),
    (("outro",), None),
    (("audio_tracks",), []),
    (("end_fade_frames",), None),
    (("sources", 0, "color"), None),
    (("segments", 0, "transform"), None),
]

CDL_PATH = ("sources", 0, "color", "cdl")


def v2(data: PlanData) -> PlanData:
    """Make the plan version 2 and give it an outro source of 302 frames."""
    data["version"] = 2
    data["sources"].append({"id": "outro", "media": "media/outro.mov", "duration": 302})
    return data


def put(data: PlanData, loc: Loc, value: object) -> None:
    """Set the value at ``loc`` in plan data."""
    target: Any = data
    for part in loc[:-1]:
        target = target[part]
    target[loc[-1]] = copy.deepcopy(value)


def errors_of(data: PlanData) -> list[ErrorDetails]:
    """The validation errors of plan data, which must not be valid."""
    with pytest.raises(ValidationError) as exc:
        EditPlan.model_validate(data)
    return exc.value.errors()


def one_error(data: PlanData) -> ErrorDetails:
    """The only validation error of plan data."""
    found = errors_of(data)
    assert len(found) == 1, found
    return found[0]


def with_outro(data: PlanData, outro: dict[str, Any], *, duration: int | None = 302) -> PlanData:
    """A version 2 plan with this outro, on an outro source of ``duration`` frames (or none)."""
    v2(data)
    if duration is None:
        del data["sources"][-1]["duration"]
    data["outro"] = {"source": "outro", **outro}
    return data


def with_cdl(data: PlanData, cdl: dict[str, Any]) -> PlanData:
    """A version 2 plan whose first source has this CDL."""
    data["version"] = 2
    data["sources"][0]["color"] = {"cdl": cdl}
    return data


def with_transform(data: PlanData, transform: dict[str, Any]) -> PlanData:
    """A version 2 plan whose first segment has this transform."""
    data["version"] = 2
    data["segments"][0]["transform"] = transform
    return data


# --- the version field ---------------------------------------------------------------------


def test_a_plan_without_a_version_is_version_2(clean_data: PlanData) -> None:
    del clean_data["version"]
    assert EditPlan.model_validate(clean_data).version == 2


def test_the_clean_fixture_says_version_1_and_stays_version_1(clean_data: PlanData) -> None:
    assert clean_data["version"] == 1
    assert EditPlan.model_validate(clean_data).version == 1


@pytest.mark.parametrize("name", ["short_clean.plan.json", "short_broken.plan.json"])
def test_the_plan_fixtures_validate_unchanged_as_version_1(name: str) -> None:
    text = (FIXTURES / name).read_text(encoding="utf-8")
    plan = EditPlan.model_validate_json(text)
    assert plan.version == 1
    assert json.loads(plan.model_dump_json(exclude_unset=True)) == json.loads(text)


@pytest.mark.parametrize("version", [0, 3, "2", None])
def test_only_versions_1_and_2_exist(clean_data: PlanData, version: object) -> None:
    clean_data["version"] = version
    assert one_error(clean_data)["loc"] == ("version",)


# --- a version 1 plan may not use a version 2 field -----------------------------------------


@pytest.mark.parametrize(("field", "loc", "value"), V2_USES, ids=V2_IDS)
def test_a_version_1_plan_using_a_v2_field_is_an_error_naming_it(
    clean_data: PlanData, field: str, loc: Loc, value: object
) -> None:
    put(clean_data, loc, value)
    error = one_error(clean_data)
    assert error["type"] == "plan_version"
    assert error["loc"] == loc
    assert f"{field} is a version 2 field" in error["msg"]
    assert 'set "version": 2' in error["msg"]


def test_the_error_for_a_nested_field_reads_in_full(clean_data: PlanData) -> None:
    put(clean_data, ("sources", 1, "color"), {"lut": "match.cube"})
    error = one_error(clean_data)
    assert error["loc"] == ("sources", 1, "color")
    assert "sources[1].color is a version 2 field, but this plan says version 1" in error["msg"]


def test_the_error_for_a_segment_field_reads_in_full(clean_data: PlanData) -> None:
    put(clean_data, ("segments", 2, "transform"), {"zoom": 1.25})
    error = one_error(clean_data)
    assert error["loc"] == ("segments", 2, "transform")
    assert error["msg"].startswith("segments[2].transform is a version 2 field")


def test_the_same_message_reaches_a_caller_that_only_matches_text(clean_data: PlanData) -> None:
    clean_data["template"] = "Example Template"
    with pytest.raises(ValidationError, match="template is a version 2 field"):
        EditPlan.model_validate(clean_data)


@pytest.mark.parametrize(
    "pair",
    [
        ("template", "head"),
        ("sources[1].color", "segments[2].transform"),
        ("template", "segments[2].transform"),
        ("end_fade_frames", "sources[1].color"),
    ],
)
def test_two_v2_fields_in_a_version_1_plan_give_two_errors(
    clean_data: PlanData, pair: tuple[str, str]
) -> None:
    uses = {field: (loc, value) for field, loc, value in V2_USES}
    for field in pair:
        put(clean_data, *uses[field])
    found = errors_of(clean_data)
    assert len(found) == 2
    assert {e["type"] for e in found} == {"plan_version"}
    assert {e["loc"] for e in found} == {uses[field][0] for field in pair}
    by_loc = {e["loc"]: e["msg"] for e in found}
    for field in pair:
        assert f"{field} is a version 2 field" in by_loc[uses[field][0]]


@pytest.mark.parametrize(
    ("loc", "value"), V2_EMPTY, ids=[".".join(map(str, e[0])) for e in V2_EMPTY]
)
def test_null_or_empty_v2_fields_are_not_a_use(
    clean_data: PlanData, loc: Loc, value: object
) -> None:
    put(clean_data, loc, value)
    assert EditPlan.model_validate(clean_data).version == 1


def test_all_the_null_and_empty_v2_fields_together_are_still_version_1(
    clean_data: PlanData,
) -> None:
    for loc, value in V2_EMPTY:
        put(clean_data, loc, value)
    assert EditPlan.model_validate(clean_data).version == 1


def test_a_full_dump_of_a_version_1_plan_reads_back_as_version_1(clean_data: PlanData) -> None:
    plan = EditPlan.model_validate(clean_data)
    dumped = plan.model_dump_json()
    written = json.loads(dumped)
    # A full dump writes every field, the version 2 ones as null or empty.
    assert written["template"] is None
    assert written["head"] is None
    assert written["outro"] is None
    assert written["audio_tracks"] == []
    assert written["end_fade_frames"] is None
    assert written["sources"][0]["color"] is None
    assert written["segments"][0]["transform"] is None
    again = EditPlan.model_validate_json(dumped)
    assert again == plan
    assert again.version == 1


# --- the same fields are valid in version 2 -------------------------------------------------


@pytest.mark.parametrize(("field", "loc", "value"), V2_USES, ids=V2_IDS)
def test_each_v2_field_is_valid_in_a_version_2_plan(
    clean_data: PlanData, field: str, loc: Loc, value: object
) -> None:
    v2(clean_data)
    put(clean_data, loc, value)
    plan = EditPlan.model_validate(clean_data)
    assert plan.version == 2


def test_all_the_v2_fields_together_are_valid_in_a_version_2_plan(clean_data: PlanData) -> None:
    v2(clean_data)
    for _, loc, value in V2_USES:
        put(clean_data, loc, value)
    clean_data["outro"]["source"] = "outro"
    plan = EditPlan.model_validate(clean_data)
    assert plan.version == 2
    assert plan.template == "Example Template"
    assert plan.head is not None
    assert (plan.head.video, plan.head.video_frames) == ("dissolve", 15)
    assert plan.outro is not None
    assert (plan.outro.source, plan.outro.src_out) == ("outro", 10)
    assert [t.name for t in plan.audio_tracks] == ["A1"]
    assert plan.end_fade_frames == 1
    assert plan.sources[1].color is not None
    assert plan.sources[1].color.lut == "match.cube"
    assert plan.segments[2].transform is not None
    assert plan.segments[2].transform.zoom == 1.25


def test_a_plan_without_a_version_may_use_v2_fields(clean_data: PlanData) -> None:
    del clean_data["version"]
    clean_data["template"] = "Example Template"
    plan = EditPlan.model_validate(clean_data)
    assert (plan.version, plan.template) == (2, "Example Template")


# --- template -------------------------------------------------------------------------------


@pytest.mark.parametrize("template", ["", "   "])
def test_a_blank_template_is_an_error(clean_data: PlanData, template: str) -> None:
    v2(clean_data)
    clean_data["template"] = template
    error = one_error(clean_data)
    assert error["loc"] == ("template",)
    assert "must not be empty or blank" in error["msg"]


# --- head and the outro's transition --------------------------------------------------------


def place_join(data: PlanData, where: str, join: dict[str, Any]) -> Loc:
    """Put a head or an outro transition in a version 2 plan; return where its errors start."""
    v2(data)
    if where == "head":
        data["head"] = join
        return ("head",)
    data["outro"] = {"source": "outro", "transition": join}
    return ("outro", "transition")


@pytest.mark.parametrize("where", ["head", "outro"])
def test_a_cut_with_a_length_is_an_error(clean_data: PlanData, where: str) -> None:
    start = place_join(clean_data, where, {"video_frames": 15})
    error = one_error(clean_data)
    assert error["loc"][: len(start)] == start
    assert "a cut has no length" in error["msg"]


@pytest.mark.parametrize("where", ["head", "outro"])
def test_a_dissolve_without_a_length_is_an_error(clean_data: PlanData, where: str) -> None:
    start = place_join(clean_data, where, {"video": "dissolve"})
    error = one_error(clean_data)
    assert error["loc"][: len(start)] == start
    assert "a dissolve needs video_frames above 0" in error["msg"]


@pytest.mark.parametrize("where", ["head", "outro"])
def test_a_dissolve_with_zero_frames_is_the_same_error(clean_data: PlanData, where: str) -> None:
    place_join(clean_data, where, {"video": "dissolve", "video_frames": 0})
    assert "a dissolve needs video_frames above 0" in one_error(clean_data)["msg"]


def test_a_head_dissolve_with_an_audio_fade_is_valid(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["head"] = {"video": "dissolve", "video_frames": 15, "audio_fade_frames": 15}
    head = EditPlan.model_validate(clean_data).head
    assert head is not None
    assert (head.video, head.video_frames, head.audio_fade_frames) == ("dissolve", 15, 15)


def test_a_head_with_nothing_set_is_a_hard_start(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["head"] = {}
    head = EditPlan.model_validate(clean_data).head
    assert head is not None
    assert (head.video, head.video_frames, head.audio_fade_frames) == ("cut", 0, 0)


def test_an_outro_dissolve_with_an_audio_cross_fade_is_valid(clean_data: PlanData) -> None:
    place_join(
        clean_data,
        "outro",
        {"video": "dissolve", "video_frames": 15, "audio": {"frames": 15}},
    )
    outro = EditPlan.model_validate(clean_data).outro
    assert outro is not None
    assert (outro.transition.video, outro.transition.video_frames) == ("dissolve", 15)
    assert outro.transition.audio is not None
    assert outro.transition.audio.frames == 15


def test_an_outro_without_a_transition_cuts_in(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["outro"] = {"source": "outro"}
    outro = EditPlan.model_validate(clean_data).outro
    assert outro is not None
    assert (outro.transition.video, outro.transition.video_frames) == ("cut", 0)
    assert outro.transition.audio is None


# --- the outro: source and range ------------------------------------------------------------


def test_an_outro_source_that_is_not_in_the_plan_is_an_error(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["outro"] = {"source": "nope"}
    error = one_error(clean_data)
    assert error["type"] == "plan_reference"
    assert error["loc"] == ("outro", "source")
    assert error["msg"] == (
        "outro source 'nope' is not one of the plan's sources ('cam-a', 'cam-b', 'cam-c', 'outro')"
    )


def test_a_blank_outro_source_is_an_error(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["outro"] = {"source": "  "}
    error = one_error(clean_data)
    assert error["loc"] == ("outro", "source")
    assert "must not be empty or blank" in error["msg"]


def test_an_outro_that_plays_to_the_end_needs_a_source_duration(clean_data: PlanData) -> None:
    with_outro(clean_data, {}, duration=None)
    error = one_error(clean_data)
    assert error["type"] == "plan_reference"
    assert error["loc"] == ("outro", "src_out")
    assert "has no duration" in error["msg"]


def test_an_outro_with_src_out_needs_no_source_duration(clean_data: PlanData) -> None:
    with_outro(clean_data, {"src_out": 50}, duration=None)
    plan = EditPlan.model_validate(clean_data)
    assert plan.outro_range() == (720, 770)


def test_an_outro_that_starts_where_the_source_ends_is_an_error(clean_data: PlanData) -> None:
    with_outro(clean_data, {"src_in": 302})
    error = one_error(clean_data)
    assert error["loc"] == ("outro", "src_out")
    assert "not after its start 302" in error["msg"]


def test_an_outro_with_src_out_not_after_src_in_is_an_error(clean_data: PlanData) -> None:
    with_outro(clean_data, {"src_in": 20, "src_out": 20})
    error = one_error(clean_data)
    assert error["loc"] == ("outro", "src_out")
    assert "not after its start 20" in error["msg"]


def test_an_outro_past_the_end_of_its_source_is_an_error(clean_data: PlanData) -> None:
    with_outro(clean_data, {"src_out": 303})
    error = one_error(clean_data)
    assert error["type"] == "plan_reference"
    assert error["loc"] == ("outro", "src_out")
    assert "past the end of source 'outro' (302 frames)" in error["msg"]


def test_an_outro_ending_exactly_at_the_end_of_its_source_is_valid(clean_data: PlanData) -> None:
    with_outro(clean_data, {"src_out": 302})
    assert EditPlan.model_validate(clean_data).outro_range() == (720, 1022)


# --- the outro: sting -----------------------------------------------------------------------

STING = {"lines": ["Pat Example"], "start": 2, "frames": 300}


def test_a_sting_that_fits_the_outro_exactly_is_valid(clean_data: PlanData) -> None:
    with_outro(clean_data, {"sting": STING})
    outro = EditPlan.model_validate(clean_data).outro
    assert outro is not None
    assert outro.sting is not None
    assert (outro.sting.lines, outro.sting.start, outro.sting.frames) == (["Pat Example"], 2, 300)


def test_a_sting_that_runs_past_the_outro_is_an_error(clean_data: PlanData) -> None:
    with_outro(clean_data, {"sting": {**STING, "frames": 301}})
    error = one_error(clean_data)
    assert error["type"] == "plan_reference"
    assert error["loc"] == ("outro", "sting")
    assert error["msg"] == ("the sting runs to frame 303 of the outro, past its end (302 frames)")


@pytest.mark.parametrize(("frames", "fits"), [(203, False), (202, True)])
def test_the_sting_is_measured_against_the_part_of_the_outro_that_plays(
    clean_data: PlanData, frames: int, fits: bool
) -> None:
    sting = {"lines": ["Pat Example"], "start": 0, "frames": frames}
    with_outro(clean_data, {"src_in": 100, "sting": sting})
    if fits:
        assert EditPlan.model_validate(clean_data).outro_range() == (720, 922)
        return
    error = one_error(clean_data)
    assert error["loc"] == ("outro", "sting")
    assert error["msg"] == "the sting runs to frame 203 of the outro, past its end (202 frames)"


@pytest.mark.parametrize("sting", [{"lines": [], "frames": 10}, {"lines": ["x"], "frames": 0}])
def test_a_sting_needs_lines_and_a_length(clean_data: PlanData, sting: dict[str, Any]) -> None:
    with_outro(clean_data, {"sting": sting})
    assert one_error(clean_data)["loc"][:2] == ("outro", "sting")


# --- the outro: audio track and the audio_tracks list ---------------------------------------


def test_an_outro_audio_track_must_be_one_of_the_plans_audio_tracks(clean_data: PlanData) -> None:
    with_outro(clean_data, {"audio_track": "A5"})
    error = one_error(clean_data)
    assert error["type"] == "plan_reference"
    assert error["loc"] == ("outro", "audio_track")
    assert "is not one of the plan's audio_tracks (none)" in error["msg"]


def test_the_missing_audio_track_error_lists_the_ones_there_are(clean_data: PlanData) -> None:
    with_outro(clean_data, {"audio_track": "A5"})
    clean_data["audio_tracks"] = [{"name": "A1"}, {"name": "A2"}]
    error = one_error(clean_data)
    assert "is not one of the plan's audio_tracks ('A1', 'A2')" in error["msg"]


def test_an_outro_audio_track_that_the_plan_declares_is_valid(clean_data: PlanData) -> None:
    with_outro(clean_data, {"audio_track": "A5"})
    clean_data["audio_tracks"] = [{"name": "A5", "bus": "Bus 1"}]
    outro = EditPlan.model_validate(clean_data).outro
    assert outro is not None
    assert outro.audio_track == "A5"


def test_duplicate_audio_track_names_are_an_error(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["audio_tracks"] = [{"name": "A1"}, {"name": "A2"}, {"name": "A1"}]
    error = one_error(clean_data)
    assert error["loc"] == ("audio_tracks",)
    assert "audio track name 'A1' is used by entries 0 and 2" in error["msg"]


def test_a_blank_audio_track_name_is_an_error(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["audio_tracks"] = [{"name": "  "}]
    error = one_error(clean_data)
    assert error["loc"] == ("audio_tracks", 0, "name")
    assert "must not be empty or blank" in error["msg"]


def test_a_full_audio_topology_is_valid_and_keeps_its_order(clean_data: PlanData) -> None:
    v2(clean_data)
    clean_data["audio_tracks"] = [
        {"name": "A1", "mute": True},
        {"name": "A2", "mute": True},
        {"name": "A3", "mute": True},
        {"name": "A4", "mute": True},
        {"name": "A5", "bus": "Bus 1", "gain_db": -3.5},
        {"name": "Bounce"},
    ]
    tracks = EditPlan.model_validate(clean_data).audio_tracks
    assert [t.name for t in tracks] == ["A1", "A2", "A3", "A4", "A5", "Bounce"]
    assert [t.mute for t in tracks] == [True, True, True, True, False, False]
    assert (tracks[4].bus, tracks[4].gain_db) == ("Bus 1", -3.5)
    assert (tracks[5].bus, tracks[5].gain_db) == (None, 0.0)


# --- colour: CDL, colour space and LUT ------------------------------------------------------


@pytest.mark.parametrize(
    ("cdl", "part", "message"),
    [
        ({"slope": [1, 1]}, "slope", "slope must be 3 numbers (red, green, blue), not 2"),
        ({"offset": [0, 0, 0, 0]}, "offset", "offset must be 3 numbers (red, green, blue), not 4"),
        ({"power": []}, "power", "power must be 3 numbers (red, green, blue), not 0"),
        ({"slope": [1, -0.1, 1]}, "slope", "slope green must be at least 0, not -0.1"),
        ({"power": [1, 0, 1]}, "power", "power green must be more than 0, not 0.0"),
        ({"power": [1, -2, 1]}, "power", "power green must be more than 0, not -2.0"),
        ({"slope": [1, 1, float("nan")]}, "slope", "slope blue must be finite"),
        ({"offset": [0, float("inf"), 0]}, "offset", "offset green must be finite"),
    ],
)
def test_a_bad_cdl_triple_is_an_error(
    clean_data: PlanData, cdl: dict[str, Any], part: str, message: str
) -> None:
    error = one_error(with_cdl(clean_data, cdl))
    assert error["loc"] == (*CDL_PATH, part)
    assert message in error["msg"]


def test_a_negative_offset_is_fine(clean_data: PlanData) -> None:
    cdl = EditPlan.model_validate(with_cdl(clean_data, {"offset": [-0.1, 0, 0.2]})).sources[0].color
    assert cdl is not None
    assert cdl.cdl is not None
    assert cdl.cdl.offset == [-0.1, 0.0, 0.2]


def test_a_negative_saturation_is_an_error(clean_data: PlanData) -> None:
    error = one_error(with_cdl(clean_data, {"saturation": -0.5}))
    assert error["loc"] == (*CDL_PATH, "saturation")
    assert error["type"] == "greater_than_equal"


def test_an_empty_cdl_is_the_identity(clean_data: PlanData) -> None:
    color = EditPlan.model_validate(with_cdl(clean_data, {})).sources[0].color
    assert color is not None
    assert color.cdl is not None
    assert color.cdl.slope == [1.0, 1.0, 1.0]
    assert color.cdl.offset == [0.0, 0.0, 0.0]
    assert color.cdl.power == [1.0, 1.0, 1.0]
    assert color.cdl.saturation == 1.0


def test_a_cdl_keeps_the_values_it_is_given(clean_data: PlanData) -> None:
    cdl = {
        "slope": [1.1, 1, 0.9],
        "offset": [0.01, 0, -0.01],
        "power": [1, 1.2, 1],
        "saturation": 0,
    }
    color = EditPlan.model_validate(with_cdl(clean_data, cdl)).sources[0].color
    assert color is not None
    assert color.cdl is not None
    assert color.cdl.model_dump() == {
        "slope": [1.1, 1.0, 0.9],
        "offset": [0.01, 0.0, -0.01],
        "power": [1.0, 1.2, 1.0],
        "saturation": 0.0,
    }


@pytest.mark.parametrize("part", ["input_color_space", "lut"])
@pytest.mark.parametrize("blank", ["", "  "])
def test_a_blank_colour_space_or_lut_is_an_error(
    clean_data: PlanData, part: str, blank: str
) -> None:
    clean_data["version"] = 2
    clean_data["sources"][0]["color"] = {part: blank}
    error = one_error(clean_data)
    assert error["loc"] == ("sources", 0, "color", part)
    assert "must not be empty or blank" in error["msg"]


def test_a_colour_space_with_a_lut_is_valid(clean_data: PlanData) -> None:
    clean_data["version"] = 2
    clean_data["sources"][0]["color"] = {
        "input_color_space": "Rec.709 Gamma 2.4",
        "lut": "match.cube",
    }
    color = EditPlan.model_validate(clean_data).sources[0].color
    assert color is not None
    assert (color.input_color_space, color.lut, color.cdl) == (
        "Rec.709 Gamma 2.4",
        "match.cube",
        None,
    )


# --- transform ------------------------------------------------------------------------------


def test_an_empty_transform_changes_nothing(clean_data: PlanData) -> None:
    transform = EditPlan.model_validate(with_transform(clean_data, {})).segments[0].transform
    assert transform is not None
    assert transform.zoom == 1.0
    assert transform.pan == 0
    assert transform.tilt == 0
    assert transform.rotation_deg == 0
    assert transform.flip_h is False
    assert transform.flip_v is False


def test_a_transform_keeps_the_values_it_is_given(clean_data: PlanData) -> None:
    given = {"zoom": 1.5, "pan": -120.5, "tilt": 40, "rotation_deg": 3, "flip_h": True}
    transform = EditPlan.model_validate(with_transform(clean_data, given)).segments[0].transform
    assert transform is not None
    assert transform.model_dump() == {
        "zoom": 1.5,
        "pan": -120.5,
        "tilt": 40.0,
        "rotation_deg": 3.0,
        "flip_h": True,
        "flip_v": False,
    }


def test_a_zero_zoom_is_an_error(clean_data: PlanData) -> None:
    error = one_error(with_transform(clean_data, {"zoom": 0}))
    assert error["loc"] == ("segments", 0, "transform", "zoom")
    assert error["type"] == "greater_than"


def test_an_infinite_zoom_is_an_error(clean_data: PlanData) -> None:
    error = one_error(with_transform(clean_data, {"zoom": float("inf")}))
    assert error["loc"] == ("segments", 0, "transform", "zoom")
    assert error["type"] == "finite_number"


@pytest.mark.parametrize("field", ["pan", "tilt", "rotation_deg"])
def test_a_transform_number_that_is_not_finite_is_an_error(
    clean_data: PlanData, field: str
) -> None:
    error = one_error(with_transform(clean_data, {field: float("nan")}))
    assert error["loc"] == ("segments", 0, "transform", field)
    assert error["type"] == "finite_number"


@pytest.mark.parametrize("degrees", [361, -361])
def test_a_rotation_beyond_one_turn_is_an_error(clean_data: PlanData, degrees: int) -> None:
    error = one_error(with_transform(clean_data, {"rotation_deg": degrees}))
    assert error["loc"] == ("segments", 0, "transform", "rotation_deg")


@pytest.mark.parametrize("degrees", [-360, 360])
def test_a_rotation_of_exactly_one_turn_is_valid(clean_data: PlanData, degrees: int) -> None:
    transform = (
        EditPlan.model_validate(with_transform(clean_data, {"rotation_deg": degrees}))
        .segments[0]
        .transform
    )
    assert transform is not None
    assert transform.rotation_deg == degrees


# --- lengths --------------------------------------------------------------------------------


def test_without_an_outro_the_total_is_the_content(clean_data: PlanData) -> None:
    plan = EditPlan.model_validate(v2(clean_data))
    assert plan.content_frames() == 720
    assert plan.outro_range() is None
    assert plan.total_frames() == 720


def test_the_outro_starts_where_the_content_ends(clean_data: PlanData) -> None:
    clean_data["outro"] = {"source": "outro"}
    plan = EditPlan.model_validate(v2(clean_data))
    assert plan.content_frames() == 720
    assert plan.outro_range() == (720, 1022)
    assert plan.total_frames() == 1022


def test_a_part_of_the_outro_counts_only_its_own_frames(clean_data: PlanData) -> None:
    clean_data["outro"] = {"source": "outro", "src_in": 2, "src_out": 32}
    plan = EditPlan.model_validate(v2(clean_data))
    assert plan.outro_range() == (720, 750)
    assert plan.total_frames() == 750
    assert plan.content_frames() == 720


# --- schema ---------------------------------------------------------------------------------


def test_the_schema_describes_version_2() -> None:
    schema = EditPlan.model_json_schema()
    properties = schema["properties"]
    assert properties["version"]["enum"] == [1, 2]
    assert properties["version"]["default"] == 2
    for name in ("template", "head", "outro", "audio_tracks", "end_fade_frames"):
        assert name in properties
    defs = schema["$defs"]
    for name in (
        "Head",
        "Outro",
        "OutroTransition",
        "Sting",
        "AudioTrack",
        "SourceColor",
        "Cdl",
        "Transform",
    ):
        assert name in defs
    assert "color" in defs["Source"]["properties"]
    assert "transform" in defs["Segment"]["properties"]
    for part in ("slope", "offset", "power"):
        triple = defs["Cdl"]["properties"][part]
        assert (triple["minItems"], triple["maxItems"]) == (3, 3)
