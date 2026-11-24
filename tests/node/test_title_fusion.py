"""Tests for Fusion MacroOperator support in ``title.place`` and ``plan.apply``."""

from __future__ import annotations

from aar.plan.apply import compile_finish
from aar.plan.model import EditPlan

from .ops_support import Show


def _macro_show() -> tuple[Show, str]:
    show = Show()
    show.main.AddTrack("video")
    tpl = show.project.media_pool.add_clip(
        ("Footage",),
        "CCN LowerThird",
        kind="video",
        fps=29.97,
        frames=300,
        title_tool="MacroOperator",
        title_inputs={"Input1": "NAME", "Input2": "ROLE", "Input3": 1.0},
    )
    return show, tpl.GetUniqueId()


def test_title_place_macro_operator_inputs_and_idempotency() -> None:
    show, tpl_id = _macro_show()
    params = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 30,
                "record_out": 150,
                "inputs": {"Input1": "Jerry Morrison", "Input2": "Comedian", "Input3": 0.75},
            }
        ],
    }
    first = show.ok("title.place", params)
    placed = first.titles[0]
    assert placed.status == "placed"
    assert placed.inputs == {"Input1": "Jerry Morrison", "Input2": "Comedian", "Input3": 0.75}
    assert "Jerry Morrison" in placed.text

    # Idempotent rerun reports "present"
    again = show.ok("title.place", params)
    assert again.titles[0].status == "present"

    # Changing one macro input reports "updated"
    params_updated = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 30,
                "record_out": 150,
                "inputs": {"Input1": "Jane Doe", "Input2": "Comedian", "Input3": 0.75},
            }
        ],
    }
    updated = show.ok("title.place", params_updated)
    assert updated.titles[0].status == "updated"
    assert updated.titles[0].inputs["Input1"] == "Jane Doe"

    # SetInput:no-op fault must be caught by read-back verification
    show.fake.faults.add("SetInput:no-op")
    params_fail = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 30,
                "record_out": 150,
                "inputs": {"Input1": "Should Fail"},
            }
        ],
    }
    err = show.fails("title.place", params_fail)
    assert err.code == "title_text_not_applied"


def test_title_place_macro_lines_fallback_and_setting_load() -> None:
    show, tpl_id = _macro_show()
    params = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 0,
                "record_out": 90,
                "lines": ["Alice", "Guest"],
                "setting": "C:/Templates/LowerThird.setting",
            }
        ],
    }
    res = show.ok("title.place", params)
    assert res.titles[0].inputs == {"Input1": "Alice", "Input2": "Guest"}
    assert any(call[0] == "LoadSettings" for call in show.fake.calls)


def test_compile_finish_forwards_fusion_inputs_and_setting() -> None:
    plan = EditPlan.model_validate(
        {
            "version": 2,
            "project": "CCN",
            "timeline": "Test",
            "kind": "longform",
            "fps": {"num": 30000, "den": 1001},
            "resolution": {"width": 1920, "height": 1080},
            "profile": "longform",
            "sources": [{"id": "a", "media": "X:/show/a.mp4", "duration": 300}],
            "segments": [{"source": "a", "src_in": 0, "src_out": 200, "track": 1}],
            "titles": [
                {
                    "kind": "graphic",
                    "template": "CCN LowerThird",
                    "setting": "CCN_LowerThird.setting",
                    "inputs": {"Input1": "Jerry", "Input2": "Host"},
                    "record_in": 10,
                    "record_out": 100,
                }
            ],
        }
    )
    comp = compile_finish(plan)
    title_steps = [s for s in comp.steps if s.op == "title.place"]
    assert len(title_steps) == 1
    t = title_steps[0].params["titles"][0]
    assert t["setting"] == "CCN_LowerThird.setting"
    assert t["inputs"] == {"Input1": "Jerry", "Input2": "Host"}
    assert t["lines"] == []
