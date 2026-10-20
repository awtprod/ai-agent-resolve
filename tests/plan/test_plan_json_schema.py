"""F4: the published JSON Schema names its dialect and admits every fps form the model accepts."""

from __future__ import annotations

import json
from typing import Any

import pytest
from plan_support import FIXTURES, PlanData
from pydantic import ValidationError

from aar.cli import main
from aar.plan import model
from aar.plan.model import EditPlan, Fps

DIALECT = "https://json-schema.org/draft/2020-12/schema"


def fps_def(mode: str = "validation") -> dict[str, Any]:
    return EditPlan.model_json_schema(mode=mode)["$defs"]["Fps"]  # type: ignore[arg-type]


def with_fps(data: PlanData, fps: Any) -> PlanData:
    data["fps"] = fps
    return data


def model_accepts(data: PlanData) -> bool:
    try:
        EditPlan.model_validate(data)
    except ValidationError:
        return False
    return True


def schema_accepts(data: PlanData) -> bool:
    jsonschema = pytest.importorskip("jsonschema")
    validator = jsonschema.Draft202012Validator(EditPlan.model_json_schema())
    return validator.is_valid(data)


# --- the dialect stamp -------------------------------------------------------------------------


def test_the_dialect_constant_is_draft_2020_12() -> None:
    assert model.JSON_SCHEMA_DIALECT == DIALECT


@pytest.mark.parametrize("mode", ["validation", "serialization"])
def test_the_root_schema_names_its_dialect(mode: str) -> None:
    schema = EditPlan.model_json_schema(mode=mode)  # type: ignore[arg-type]
    assert schema["$schema"] == DIALECT


def test_only_the_root_carries_the_dialect() -> None:
    defs = EditPlan.model_json_schema()["$defs"]
    assert [name for name, body in defs.items() if "$schema" in body] == []


def test_the_schema_is_a_valid_draft_2020_12_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = EditPlan.model_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert jsonschema.validators.validator_for(schema) is jsonschema.Draft202012Validator


def test_the_cli_schema_output_names_its_dialect(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["plan", "schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["$schema"] == DIALECT
    assert [form.get("type") for form in schema["$defs"]["Fps"]["anyOf"]] == [
        "object",
        "integer",
        "string",
    ]


# --- the fps shorthand in the schema -----------------------------------------------------------


def test_the_input_fps_schema_lists_the_object_the_whole_number_and_the_text_form() -> None:
    definition = fps_def()
    assert definition["title"] == "Fps"
    forms = definition["anyOf"]
    assert [form["type"] for form in forms] == ["object", "integer", "string"]
    obj, whole, text = forms
    assert set(obj["properties"]) == {"num", "den"}
    assert obj["properties"]["num"]["exclusiveMinimum"] == 0
    assert obj["properties"]["den"]["exclusiveMinimum"] == 0
    assert obj["required"] == ["num"]
    assert whole["exclusiveMinimum"] == 0
    assert text["pattern"].startswith("^")


def test_the_serialized_fps_is_still_the_object_only() -> None:
    definition = fps_def("serialization")
    assert "anyOf" not in definition
    assert definition["type"] == "object"
    assert set(definition["properties"]) == {"num", "den"}


def test_the_fps_description_mentions_the_shorthand() -> None:
    description = " ".join(fps_def()["description"].split())
    assert "24000/1001" in description
    assert "whole number" in description


GOOD_FPS: list[Any] = [
    {"num": 24000, "den": 1001},
    {"num": 25},
    25,
    "25",
    "24000/1001",
    " 24 / 1 ",
    "030",
]
BAD_FPS: list[Any] = [
    0,
    -25,
    "0",
    "24/0",
    "0/1",
    "abc",
    "24/",
    "/24",
    "1.5",
    "24.0",
    "-24",
    "",
    "   ",
    True,
    False,
    25.5,
    [25],
    None,
    {"num": 0},
    {"num": 24, "den": 0},
]


@pytest.mark.parametrize("fps", GOOD_FPS, ids=repr)
def test_a_plan_with_this_fps_passes_the_schema_and_the_model(
    clean_data: PlanData, fps: Any
) -> None:
    assert model_accepts(with_fps(clean_data, fps))
    assert schema_accepts(clean_data)


@pytest.mark.parametrize("fps", BAD_FPS, ids=repr)
def test_a_plan_with_this_fps_fails_the_schema_and_the_model(
    clean_data: PlanData, fps: Any
) -> None:
    assert not model_accepts(with_fps(clean_data, fps))
    assert not schema_accepts(clean_data)


@pytest.mark.parametrize("fps", ["25", "24000/1001", " 24 / 1 ", "030"], ids=repr)
def test_the_text_form_means_what_the_pattern_says(fps: str) -> None:
    value = Fps.model_validate(fps)
    num, _, den = fps.partition("/")
    assert (value.num, value.den) == (int(num), int(den or 1))


@pytest.mark.parametrize("name", ["short_clean.plan.json", "short_broken.plan.json"])
def test_the_fixture_plans_pass_the_schema(name: str) -> None:
    data = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    assert schema_accepts(data)
