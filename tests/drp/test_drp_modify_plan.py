"""Tests for the DRP edit plan in ``aar.drp.modify.plan``."""

from __future__ import annotations

import json
from typing import Any

import pytest

from aar.drp.modify.plan import (
    DEFERRED_OPS,
    MAX_PLAN_BYTES,
    PLAN_VERSION,
    WHITELISTED_OPS,
    DrpEditPlan,
    MarkerAdd,
    MarkerDelete,
    MarkerUpdate,
    PathMap,
    PlanProblem,
    Rename,
    parse_plan,
)

SHA = "0123456789abcdef" * 4

# One valid edit per whitelisted op, with the model class it must parse into.
VALID_EDITS: dict[str, tuple[dict[str, Any], type]] = {
    "rename": (
        {"op": "rename", "target": "clip", "id": "c1", "new_name": "Hero"},
        Rename,
    ),
    "path_map": (
        {
            "op": "path_map",
            "direction": "windows_to_mac",
            "table": [{"windows": "E:", "mac": "/Volumes/Media"}],
        },
        PathMap,
    ),
    "marker_add": (
        {"op": "marker_add", "owner": "timeline", "owner_id": "t1", "frame": 5},
        MarkerAdd,
    ),
    "marker_update": (
        {
            "op": "marker_update",
            "owner": "item",
            "owner_id": "i1",
            "frame": 5,
            "set": {"name": "x"},
        },
        MarkerUpdate,
    ),
    "marker_delete": (
        {"op": "marker_delete", "owner": "item", "owner_id": "i1", "frame": 5},
        MarkerDelete,
    ),
}


def _plan(*edits: dict[str, Any], **top: Any) -> str:
    return json.dumps({**top, "edits": list(edits)})


def _problem(text: str | bytes) -> PlanProblem:
    with pytest.raises(PlanProblem) as info:
        parse_plan(text)
    return info.value


def _marker_add(**extra: Any) -> dict[str, Any]:
    return {"op": "marker_add", "owner": "timeline", "owner_id": "t1", "frame": 1, **extra}


def _marker_update(set_: dict[str, Any]) -> dict[str, Any]:
    return {"op": "marker_update", "owner": "item", "owner_id": "i1", "frame": 1, "set": set_}


def _rename(**fields: Any) -> dict[str, Any]:
    return {"op": "rename", "new_name": "New", **fields}


# --------------------------------------------------------------------------- accepted plans


def test_minimal_plan_has_no_edits_and_the_defaults() -> None:
    plan = parse_plan('{"edits": []}')
    assert isinstance(plan, DrpEditPlan)
    assert plan.edits == []
    assert plan.plan_version == PLAN_VERSION == 1
    assert plan.input_sha256 is None
    assert plan.max_project_version is None


def test_an_empty_object_is_an_empty_plan() -> None:
    assert parse_plan("{}") == parse_plan('{"edits": []}')


def test_str_and_bytes_parse_to_the_same_plan() -> None:
    text = _plan(*(edit for edit, _ in VALID_EDITS.values()))
    assert parse_plan(text.encode("utf-8")) == parse_plan(text)


def test_the_fixture_covers_exactly_the_whitelisted_ops() -> None:
    assert set(VALID_EDITS) == set(WHITELISTED_OPS)
    assert WHITELISTED_OPS == ("rename", "path_map", "marker_add", "marker_update", "marker_delete")
    assert set(DEFERRED_OPS) == {"metadata", "fader_keyframes"}
    assert not set(DEFERRED_OPS) & set(WHITELISTED_OPS)


@pytest.mark.parametrize("op", WHITELISTED_OPS)
def test_each_whitelisted_op_parses_to_its_own_model(op: str) -> None:
    edit, model = VALID_EDITS[op]
    plan = parse_plan(_plan(edit))
    assert len(plan.edits) == 1
    assert type(plan.edits[0]) is model
    assert plan.edits[0].op == op


def test_edits_keep_their_order() -> None:
    ops = ["marker_delete", "rename", "path_map", "marker_add", "marker_update", "rename"]
    plan = parse_plan(_plan(*(VALID_EDITS[op][0] for op in ops)))
    assert [e.op for e in plan.edits] == ops


def test_rename_project_needs_no_selector() -> None:
    plan = parse_plan(_plan(_rename(target="project")))
    edit = plan.edits[0]
    assert isinstance(edit, Rename)
    assert (edit.target, edit.id, edit.name, edit.new_name, edit.expect_name) == (
        "project",
        None,
        None,
        "New",
        None,
    )


@pytest.mark.parametrize("target", ["timeline", "bin", "clip", "item"])
@pytest.mark.parametrize("selector", [{"id": "x1"}, {"name": "Old"}])
def test_rename_other_targets_take_exactly_one_selector(
    target: str, selector: dict[str, str]
) -> None:
    plan = parse_plan(_plan(_rename(target=target, expect_name="Old", **selector)))
    edit = plan.edits[0]
    assert isinstance(edit, Rename)
    assert edit.target == target
    assert edit.expect_name == "Old"
    assert {"id": edit.id, "name": edit.name} == {"id": None, "name": None, **selector}


def test_path_map_keeps_direction_and_table_rows_in_order() -> None:
    table = [
        {"windows": "E:", "mac": "/Volumes/Media"},
        {"windows": "\\\\nas\\share", "mac": "/Volumes/share"},
    ]
    plan = parse_plan(_plan({"op": "path_map", "direction": "mac_to_windows", "table": table}))
    edit = plan.edits[0]
    assert isinstance(edit, PathMap)
    assert edit.direction == "mac_to_windows"
    assert [row.model_dump() for row in edit.table] == table


def test_marker_add_defaults() -> None:
    plan = parse_plan(_plan(_marker_add()))
    edit = plan.edits[0]
    assert isinstance(edit, MarkerAdd)
    assert (edit.owner, edit.owner_id, edit.frame) == ("timeline", "t1", 1)
    assert edit.color == "Blue"
    assert edit.duration == 1
    assert edit.name == ""
    assert edit.note == ""
    assert edit.custom_data == ""


def test_marker_add_keeps_every_given_field() -> None:
    given = {"color": "Cream", "name": "n", "note": "z", "duration": 7, "custom_data": "cd"}
    edit = parse_plan(_plan(_marker_add(owner="item", **given))).edits[0]
    assert isinstance(edit, MarkerAdd)
    assert edit.owner == "item"
    assert {k: getattr(edit, k) for k in given} == given


@pytest.mark.parametrize(
    "color",
    [
        "Blue",
        "Cyan",
        "Green",
        "Yellow",
        "Red",
        "Pink",
        "Purple",
        "Fuchsia",
        "Rose",
        "Lavender",
        "Sky",
        "Mint",
        "Lemon",
        "Sand",
        "Cocoa",
        "Cream",
    ],
)
def test_marker_add_accepts_every_resolve_marker_colour(color: str) -> None:
    edit = parse_plan(_plan(_marker_add(color=color))).edits[0]
    assert isinstance(edit, MarkerAdd)
    assert edit.color == color


def test_marker_update_records_only_the_fields_that_were_given() -> None:
    edit = parse_plan(_plan(_marker_update({"name": "x"}))).edits[0]
    assert isinstance(edit, MarkerUpdate)
    assert edit.set.model_fields_set == {"name"}
    assert edit.set.name == "x"
    assert edit.set.frame is None
    assert edit.set.color is None


def test_marker_update_accepts_all_six_fields_and_empty_text() -> None:
    fields = {
        "frame": 0,
        "color": "Red",
        "name": "",
        "note": "",
        "duration": 3,
        "custom_data": "",
    }
    edit = parse_plan(_plan(_marker_update(fields))).edits[0]
    assert isinstance(edit, MarkerUpdate)
    assert edit.set.model_fields_set == set(fields)
    assert edit.set.model_dump() == fields


def test_marker_frame_bounds() -> None:
    top = 2**31 - 1
    for frame in (0, top):
        edit = parse_plan(_plan(_marker_add(frame=frame))).edits[0]
        assert isinstance(edit, MarkerAdd)
        assert edit.frame == frame
    assert _problem(_plan(_marker_add(frame=top + 1))).details[0]["type"] == "less_than_equal"


def test_top_level_fields_are_kept() -> None:
    plan = parse_plan(_plan(input_sha256=SHA, max_project_version=17))
    assert plan.input_sha256 == SHA
    assert plan.max_project_version == 17


@pytest.mark.parametrize("version", [0, 10_000])
def test_max_project_version_bounds_are_inclusive(version: int) -> None:
    assert parse_plan(_plan(max_project_version=version)).max_project_version == version


def test_ten_thousand_edits_are_accepted_and_one_more_is_refused() -> None:
    edit = {"op": "marker_delete", "owner": "item", "owner_id": "x", "frame": 0}
    assert len(parse_plan(_plan(*([edit] * 10_000))).edits) == 10_000
    problem = _problem(_plan(*([edit] * 10_001)))
    assert problem.code == "plan-invalid"
    assert (problem.details[0]["loc"], problem.details[0]["type"]) == (["edits"], "too_long")


def test_a_plan_round_trips_through_json_when_unset_fields_are_left_out() -> None:
    plan = parse_plan(_plan(*(edit for edit, _ in VALID_EDITS.values()), input_sha256=SHA))
    assert parse_plan(plan.model_dump_json(exclude_unset=True)) == plan
    assert parse_plan(plan.model_dump_json(exclude_none=True)) == plan


# --------------------------------------------------------------------------- plan-too-large


def test_a_plan_over_the_byte_limit_is_too_large() -> None:
    problem = _problem("x" * (MAX_PLAN_BYTES + 1))
    assert problem.code == "plan-too-large"
    assert problem.details == []
    assert str(MAX_PLAN_BYTES + 1) in str(problem)
    assert _problem(b"x" * (MAX_PLAN_BYTES + 1)).code == "plan-too-large"


def test_the_size_limit_counts_bytes_not_characters() -> None:
    text = "é" * (MAX_PLAN_BYTES // 2 + 1)  # 524_289 characters, 1_048_578 bytes
    assert len(text) < MAX_PLAN_BYTES < len(text.encode("utf-8"))
    assert _problem(text).code == "plan-too-large"


def test_a_plan_of_exactly_the_byte_limit_is_read() -> None:
    body = b'{"edits": []}'
    plan = parse_plan(body + b" " * (MAX_PLAN_BYTES - len(body)))
    assert plan.edits == []


def test_the_size_is_checked_before_the_json_is_read() -> None:
    # Not JSON at all, yet reported as too large: nothing was parsed.
    assert _problem("}" * (MAX_PLAN_BYTES + 1)).code == "plan-too-large"


# --------------------------------------------------------------------------- plan-not-json


@pytest.mark.parametrize(
    "text",
    ["not json", "", "   ", "{", '{"edits": [}', "{'edits': []}", b"\xff\xfe\x00", b"\xc3\x28"],
)
def test_text_that_is_not_json_is_plan_not_json(text: str | bytes) -> None:
    problem = _problem(text)
    assert problem.code == "plan-not-json"
    assert problem.details == []
    assert "not JSON" in str(problem)


@pytest.mark.parametrize("text", ["[]", "[1, 2]", "5", "null", '"edits"', "true"])
def test_json_that_is_not_an_object_is_plan_invalid(text: str) -> None:
    problem = _problem(text)
    assert problem.code == "plan-invalid"
    assert problem.details[0]["loc"] == []
    assert problem.details[0]["type"] == "model_type"


# --------------------------------------------------------------------------- edit-deferred


@pytest.mark.parametrize("op", ["metadata", "fader_keyframes"])
def test_deferred_ops_are_refused_with_their_reason(op: str) -> None:
    problem = _problem(_plan({"op": op}))
    assert problem.code == "edit-deferred"
    assert len(problem.details) == 1
    detail = problem.details[0]
    assert (detail["edit"], detail["op"]) == (0, op)
    assert isinstance(detail["reason"], str)
    assert detail["reason"].startswith("deferred: ")
    assert detail["reason"] == DEFERRED_OPS[op]
    assert set(detail) == {"edit", "op", "reason"}


def test_the_two_deferred_ops_have_different_reasons() -> None:
    assert DEFERRED_OPS["metadata"] != DEFERRED_OPS["fader_keyframes"]
    assert "BtMetadataLockableBlob" in DEFERRED_OPS["metadata"]
    assert "FLStudioModelBA" in DEFERRED_OPS["fader_keyframes"]


def test_every_deferred_edit_is_listed_with_its_index() -> None:
    problem = _problem(
        _plan(
            VALID_EDITS["rename"][0],
            {"op": "metadata", "clip": "c1"},
            VALID_EDITS["marker_add"][0],
            {"op": "fader_keyframes"},
            {"op": "metadata"},
        )
    )
    assert problem.code == "edit-deferred"
    assert [(d["edit"], d["op"]) for d in problem.details] == [
        (1, "metadata"),
        (3, "fader_keyframes"),
        (4, "metadata"),
    ]
    assert "3 edit(s)" in str(problem)


def test_a_deferred_edit_is_refused_even_though_its_fields_would_not_validate() -> None:
    assert _problem(_plan({"op": "metadata", "bogus": [1, 2, 3]})).code == "edit-deferred"


# --------------------------------------------------------------------------- edit-not-whitelisted


def test_an_unknown_op_is_not_whitelisted() -> None:
    problem = _problem(_plan({"op": "delete_timeline", "id": "t1"}))
    assert problem.code == "edit-not-whitelisted"
    assert problem.details == [{"edit": 0, "op": "delete_timeline"}]
    for op in WHITELISTED_OPS:
        assert op in str(problem)


def test_an_edit_with_no_op_is_not_whitelisted() -> None:
    problem = _problem(_plan({"target": "clip", "id": "c1", "new_name": "x"}))
    assert problem.code == "edit-not-whitelisted"
    assert problem.details == [{"edit": 0, "op": None}]


@pytest.mark.parametrize("edit", ["rename", 3, None, [1], []])
def test_an_edit_that_is_not_an_object_is_not_whitelisted(edit: Any) -> None:
    problem = _problem(json.dumps({"edits": [edit]}))
    assert problem.code == "edit-not-whitelisted"
    assert problem.details == [{"edit": 0, "op": None}]


@pytest.mark.parametrize("op", ["Rename", "RENAME", " rename", "rename ", "", 5, True, None])
def test_op_names_are_matched_exactly(op: Any) -> None:
    problem = _problem(_plan({"op": op, "target": "project", "new_name": "x"}))
    assert problem.code == "edit-not-whitelisted"
    assert problem.details == [{"edit": 0, "op": op}]


def test_every_unwhitelisted_edit_is_listed_with_its_index() -> None:
    problem = _problem(
        _plan(
            VALID_EDITS["rename"][0],
            {"op": "delete_timeline"},
            VALID_EDITS["path_map"][0],
            {},
            {"op": "set_color"},
        )
    )
    assert problem.code == "edit-not-whitelisted"
    assert problem.details == [
        {"edit": 1, "op": "delete_timeline"},
        {"edit": 3, "op": None},
        {"edit": 4, "op": "set_color"},
    ]
    assert "3 edit(s)" in str(problem)


# --------------------------------------------------------------------------- plan-invalid

BAD_PLANS: dict[str, tuple[str, list[Any], str, str]] = {
    # id: (plan text, expected loc, expected pydantic error type, fragment of the message)
    "unknown top-level key": (_plan(bogus=1), ["bogus"], "extra_forbidden", "not permitted"),
    "unknown key in an edit": (
        _plan(_rename(target="project", bogus=1)),
        ["edits", 0, "rename", "bogus"],
        "extra_forbidden",
        "not permitted",
    ),
    "unknown key in a marker set": (
        _plan(_marker_update({"bogus": 1})),
        ["edits", 0, "marker_update", "set", "bogus"],
        "extra_forbidden",
        "not permitted",
    ),
    "unknown key in a table row": (
        _plan(
            {
                "op": "path_map",
                "direction": "windows_to_mac",
                "table": [{"windows": "E:", "mac": "/V/M", "extra": 1}],
            }
        ),
        ["edits", 0, "path_map", "table", 0, "extra"],
        "extra_forbidden",
        "not permitted",
    ),
    "project rename with an id": (
        _plan(_rename(target="project", id="p1")),
        ["edits", 0, "rename"],
        "value_error",
        "a project rename takes no id or name",
    ),
    "project rename with a name": (
        _plan(_rename(target="project", name="P")),
        ["edits", 0, "rename"],
        "value_error",
        "a project rename takes no id or name",
    ),
    "clip rename with neither id nor name": (
        _plan(_rename(target="clip")),
        ["edits", 0, "rename"],
        "value_error",
        "a clip rename needs exactly one of id and name",
    ),
    "timeline rename with neither id nor name": (
        _plan(_rename(target="timeline", expect_name="Old")),
        ["edits", 0, "rename"],
        "value_error",
        "a timeline rename needs exactly one of id and name",
    ),
    "clip rename with both id and name": (
        _plan(_rename(target="clip", id="c1", name="Old")),
        ["edits", 0, "rename"],
        "value_error",
        "a clip rename needs exactly one of id and name",
    ),
    "rename with an empty new_name": (
        _plan(_rename(target="clip", id="c1", new_name="")),
        ["edits", 0, "rename", "new_name"],
        "string_too_short",
        "at least 1 character",
    ),
    "rename without new_name": (
        _plan({"op": "rename", "target": "clip", "id": "c1"}),
        ["edits", 0, "rename", "new_name"],
        "missing",
        "required",
    ),
    "rename with an empty id": (
        _plan(_rename(target="clip", id="")),
        ["edits", 0, "rename", "id"],
        "string_too_short",
        "at least 1 character",
    ),
    "rename with an empty name": (
        _plan(_rename(target="clip", name="")),
        ["edits", 0, "rename", "name"],
        "string_too_short",
        "at least 1 character",
    ),
    "rename of an unknown target": (
        _plan(_rename(target="widget", id="w1")),
        ["edits", 0, "rename", "target"],
        "literal_error",
        "'project', 'timeline', 'bin', 'clip' or 'item'",
    ),
    "rename with an over-long new_name": (
        _plan(_rename(target="project", new_name="n" * 1025)),
        ["edits", 0, "rename", "new_name"],
        "string_too_long",
        "at most 1024 characters",
    ),
    "marker_update with an empty set": (
        _plan(_marker_update({})),
        ["edits", 0, "marker_update", "set"],
        "value_error",
        "marker_update needs at least one field in 'set'",
    ),
    "marker_update with a null name": (
        _plan(_marker_update({"name": None})),
        ["edits", 0, "marker_update", "set"],
        "value_error",
        "'name' cannot be null",
    ),
    "marker_update with a null note beside a good field": (
        _plan(_marker_update({"color": "Red", "note": None})),
        ["edits", 0, "marker_update", "set"],
        "value_error",
        "'note' cannot be null",
    ),
    "marker_update with a negative frame": (
        _plan(_marker_update({"frame": -1})),
        ["edits", 0, "marker_update", "set", "frame"],
        "greater_than_equal",
        "greater than or equal to 0",
    ),
    "marker_update with a zero duration": (
        _plan(_marker_update({"duration": 0})),
        ["edits", 0, "marker_update", "set", "duration"],
        "greater_than_equal",
        "greater than or equal to 1",
    ),
    "marker_update with a bad colour": (
        _plan(_marker_update({"color": "Mauve"})),
        ["edits", 0, "marker_update", "set", "color"],
        "literal_error",
        "'Blue'",
    ),
    "marker_update without set": (
        _plan({"op": "marker_update", "owner": "item", "owner_id": "i1", "frame": 1}),
        ["edits", 0, "marker_update", "set"],
        "missing",
        "required",
    ),
    "marker_add with a negative frame": (
        _plan(_marker_add(frame=-1)),
        ["edits", 0, "marker_add", "frame"],
        "greater_than_equal",
        "greater than or equal to 0",
    ),
    "marker_add with a zero duration": (
        _plan(_marker_add(duration=0)),
        ["edits", 0, "marker_add", "duration"],
        "greater_than_equal",
        "greater than or equal to 1",
    ),
    "marker_add with a bad colour": (
        _plan(_marker_add(color="Mauve")),
        ["edits", 0, "marker_add", "color"],
        "literal_error",
        "'Blue'",
    ),
    "marker_add with a lower-case colour": (
        _plan(_marker_add(color="blue")),
        ["edits", 0, "marker_add", "color"],
        "literal_error",
        "'Blue'",
    ),
    "marker_add with a frame that is not an integer": (
        _plan(_marker_add(frame=1.5)),
        ["edits", 0, "marker_add", "frame"],
        "int_from_float",
        "integer",
    ),
    "marker_add with an unknown owner": (
        _plan(_marker_add(owner="clip")),
        ["edits", 0, "marker_add", "owner"],
        "literal_error",
        "'timeline' or 'item'",
    ),
    "marker_add with an empty owner_id": (
        _plan(_marker_add(owner_id="")),
        ["edits", 0, "marker_add", "owner_id"],
        "string_too_short",
        "at least 1 character",
    ),
    "marker_add without a frame": (
        _plan({"op": "marker_add", "owner": "timeline", "owner_id": "t1"}),
        ["edits", 0, "marker_add", "frame"],
        "missing",
        "required",
    ),
    "marker_delete without an owner_id": (
        _plan({"op": "marker_delete", "owner": "timeline", "frame": 1}),
        ["edits", 0, "marker_delete", "owner_id"],
        "missing",
        "required",
    ),
    "uppercase input_sha256": (
        _plan(input_sha256=SHA.upper()),
        ["input_sha256"],
        "string_pattern_mismatch",
        "match pattern",
    ),
    "short input_sha256": (
        _plan(input_sha256=SHA[:63]),
        ["input_sha256"],
        "string_pattern_mismatch",
        "match pattern",
    ),
    "long input_sha256": (
        _plan(input_sha256=SHA + "0"),
        ["input_sha256"],
        "string_pattern_mismatch",
        "match pattern",
    ),
    "non-hex input_sha256": (
        _plan(input_sha256="g" * 64),
        ["input_sha256"],
        "string_pattern_mismatch",
        "match pattern",
    ),
    "input_sha256 with a trailing newline": (
        _plan(input_sha256=SHA[:63] + "\n"),
        ["input_sha256"],
        "string_pattern_mismatch",
        "match pattern",
    ),
    "negative max_project_version": (
        _plan(max_project_version=-1),
        ["max_project_version"],
        "greater_than_equal",
        "greater than or equal to 0",
    ),
    "over-large max_project_version": (
        _plan(max_project_version=10_001),
        ["max_project_version"],
        "less_than_equal",
        "less than or equal to 10000",
    ),
    "path_map with an empty table": (
        _plan({"op": "path_map", "direction": "windows_to_mac", "table": []}),
        ["edits", 0, "path_map", "table"],
        "too_short",
        "at least 1 item",
    ),
    "path_map with 257 rows": (
        _plan(
            {
                "op": "path_map",
                "direction": "windows_to_mac",
                "table": [{"windows": f"E:\\d{i}", "mac": f"/V/{i}"} for i in range(257)],
            }
        ),
        ["edits", 0, "path_map", "table"],
        "too_long",
        "at most 256 items",
    ),
    "path_map with an unknown direction": (
        _plan(
            {"op": "path_map", "direction": "sideways", "table": [{"windows": "E:", "mac": "/V"}]}
        ),
        ["edits", 0, "path_map", "direction"],
        "literal_error",
        "'windows_to_mac' or 'mac_to_windows'",
    ),
    "path_map row with a one-character windows prefix": (
        _plan(
            {
                "op": "path_map",
                "direction": "windows_to_mac",
                "table": [{"windows": "E", "mac": "/V"}],
            }
        ),
        ["edits", 0, "path_map", "table", 0, "windows"],
        "string_too_short",
        "at least 2 characters",
    ),
    "path_map row with an over-long mac prefix": (
        _plan(
            {
                "op": "path_map",
                "direction": "windows_to_mac",
                "table": [{"windows": "E:", "mac": "/" + "a" * 4096}],
            }
        ),
        ["edits", 0, "path_map", "table", 0, "mac"],
        "string_too_long",
        "at most 4096 characters",
    ),
    "plan_version 2": (
        _plan(plan_version=2),
        ["plan_version"],
        "literal_error",
        "Input should be 1",
    ),
    "plan_version as a string": (
        _plan(plan_version="1"),
        ["plan_version"],
        "literal_error",
        "Input should be 1",
    ),
    "edits that is not a list": (
        '{"edits": 5}',
        ["edits"],
        "list_type",
        "valid list",
    ),
}


@pytest.mark.parametrize("case", list(BAD_PLANS))
def test_invalid_plans_are_refused_with_the_failing_location(case: str) -> None:
    text, loc, kind, fragment = BAD_PLANS[case]
    problem = _problem(text)
    assert problem.code == "plan-invalid"
    assert str(problem) == "the plan failed validation"
    assert len(problem.details) == 1, problem.details
    detail = problem.details[0]
    assert (detail["loc"], detail["type"]) == (loc, kind)
    assert fragment in detail["msg"]


def test_every_problem_is_reported_at_once() -> None:
    problem = _problem(
        _plan(
            _marker_add(frame=-1),
            VALID_EDITS["rename"][0],
            _marker_add(duration=0, color="Mauve"),
            input_sha256="nope",
        )
    )
    assert problem.code == "plan-invalid"
    assert sorted((tuple(d["loc"]) for d in problem.details), key=str) == sorted(
        [
            ("input_sha256",),
            ("edits", 0, "marker_add", "frame"),
            ("edits", 2, "marker_add", "duration"),
            ("edits", 2, "marker_add", "color"),
        ],
        key=str,
    )


def test_details_name_the_edit_position_among_valid_edits() -> None:
    problem = _problem(_plan(*[VALID_EDITS["marker_delete"][0]] * 3, _marker_add(frame=-1)))
    assert [d["loc"] for d in problem.details] == [["edits", 3, "marker_add", "frame"]]


def test_details_never_echo_the_plan_input_or_a_documentation_url() -> None:
    problem = _problem(_plan(input_sha256="SECRET-VALUE", max_project_version=-5))
    dumped = json.dumps(problem.details)
    assert "SECRET-VALUE" not in dumped
    assert "-5" not in dumped
    for detail in problem.details:
        assert set(detail) == {"type", "loc", "msg"}


def test_details_are_json_serialisable_even_for_a_value_error() -> None:
    problem = _problem(_plan(_marker_update({})))
    assert json.loads(json.dumps(problem.details)) == problem.details


# --------------------------------------------------------------------------- PlanProblem


def test_plan_problem_is_a_value_error_with_code_message_and_details() -> None:
    problem = PlanProblem("some-code", "the message", [{"edit": 0}])
    assert isinstance(problem, ValueError)
    assert problem.code == "some-code"
    assert str(problem) == "the message"
    assert problem.details == [{"edit": 0}]


def test_plan_problem_details_default_to_a_fresh_empty_list() -> None:
    a = PlanProblem("c", "m")
    b = PlanProblem("c", "m", None)
    assert a.details == [] == b.details
    a.details.append("x")
    assert b.details == []
    assert PlanProblem("c", "m", []).details == []


def test_parse_plan_problems_can_be_caught_as_value_error() -> None:
    with pytest.raises(ValueError, match="not JSON"):
        parse_plan("nope")


# --------------------------------------------------------------------------- JSON schema


def _resolve(schema: dict[str, Any], ref: str) -> dict[str, Any]:
    assert ref.startswith("#/$defs/")
    return schema["$defs"][ref.removeprefix("#/$defs/")]


def test_the_json_schema_is_json_serialisable() -> None:
    schema = DrpEditPlan.model_json_schema()
    assert json.loads(json.dumps(schema)) == schema


def test_the_schema_lists_exactly_the_five_whitelisted_ops() -> None:
    schema = DrpEditPlan.model_json_schema()
    items = schema["properties"]["edits"]["items"]
    one_of = [_resolve(schema, entry["$ref"]) for entry in items["oneOf"]]
    ops = [d["properties"]["op"]["const"] for d in one_of]
    assert len(ops) == len(set(ops)) == 5
    assert set(ops) == set(WHITELISTED_OPS)
    assert not set(ops) & set(DEFERRED_OPS)


def test_the_schema_discriminator_maps_each_op_to_its_definition() -> None:
    schema = DrpEditPlan.model_json_schema()
    discriminator = schema["properties"]["edits"]["items"]["discriminator"]
    assert discriminator["propertyName"] == "op"
    mapping = discriminator["mapping"]
    assert set(mapping) == set(WHITELISTED_OPS)
    for op, ref in mapping.items():
        assert _resolve(schema, ref)["properties"]["op"]["const"] == op


def test_the_schema_forbids_unknown_keys_everywhere() -> None:
    schema = DrpEditPlan.model_json_schema()
    assert schema["additionalProperties"] is False
    for name in ("Rename", "PathMap", "PathPair", "MarkerAdd", "MarkerUpdate", "MarkerDelete"):
        assert schema["$defs"][name]["additionalProperties"] is False, name
    assert schema["$defs"]["MarkerFields"]["additionalProperties"] is False


def test_the_schema_has_a_bounded_edit_count() -> None:
    schema = DrpEditPlan.model_json_schema()
    assert schema["properties"]["edits"]["maxItems"] == 10_000


# --------------------------------------------------------------------------- hostile input
# parse_plan's contract: whatever the bytes, it returns a plan or raises PlanProblem.


@pytest.mark.parametrize("op", [[1], {"a": 1}])
def test_an_unhashable_op_is_not_whitelisted(op: Any) -> None:
    problem = _problem(_plan({"op": op}))
    assert problem.code == "edit-not-whitelisted"


def test_an_integer_over_the_python_digit_limit_is_a_plan_problem() -> None:
    problem = _problem('{"edits": [], "max_project_version": ' + "9" * 5000 + "}")
    assert problem.code in {"plan-not-json", "plan-invalid"}


def test_deep_nesting_is_a_plan_problem() -> None:
    assert _problem("[" * 100_000).code == "plan-not-json"
