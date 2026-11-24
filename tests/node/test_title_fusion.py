"""Tests for Fusion MacroOperator support in ``title.place`` and ``plan.apply``."""

from __future__ import annotations

from pathlib import Path

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
                    "setting": "C:/Templates/CCN_LowerThird.setting",
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
    assert t["setting"] == "C:/Templates/CCN_LowerThird.setting"
    assert t["inputs"] == {"Input1": "Jerry", "Input2": "Host"}
    assert t["lines"] == []


def test_compile_finish_rejects_non_edit_pc_setting_and_blocks_setting_only() -> None:
    import pytest

    from aar.plan.apply.errors import CompileError
    from aar.plan.apply.refs import find_unresolved

    base = {
        "version": 2,
        "project": "CCN",
        "timeline": "Test",
        "kind": "longform",
        "fps": {"num": 30000, "den": 1001},
        "resolution": {"width": 1920, "height": 1080},
        "profile": "longform",
        "sources": [{"id": "a", "media": "X:/show/a.mp4", "duration": 300}],
        "segments": [{"source": "a", "src_in": 0, "src_out": 200, "track": 1}],
    }

    # Bare filename or hub path for setting must raise setting_not_on_edit_pc
    bad_plan = EditPlan.model_validate(
        {
            **base,
            "titles": [
                {
                    "kind": "graphic",
                    "template": "CCN LowerThird",
                    "setting": "/hub/templates/LowerThird.setting",
                    "record_in": 0,
                    "record_out": 60,
                }
            ],
        }
    )
    with pytest.raises(CompileError) as exc:
        compile_finish(bad_plan)
    assert exc.value.code == "setting_not_on_edit_pc"

    # Setting-only title AND setting-only sting must NOT be silently dropped: emit blocking
    # unresolved title_template markers instead.
    setting_only_plan = EditPlan.model_validate(
        {
            **base,
            "sources": [
                {"id": "a", "media": "X:/show/a.mp4", "duration": 300},
                {"id": "outro_s", "media": "X:/show/outro.mp4", "duration": 100},
            ],
            "titles": [
                {
                    "kind": "graphic",
                    "setting": "C:/Templates/Badge.setting",
                    "inputs": {"Label": "LIVE"},
                    "record_in": 10,
                    "record_out": 60,
                }
            ],
            "outro": {
                "source": "outro_s",
                "src_in": 0,
                "src_out": 60,
                "sting": {
                    "setting": "C:/Templates/Sting.setting",
                    "start": 0,
                    "frames": 40,
                },
            },
        }
    )
    comp = compile_finish(setting_only_plan)
    title_steps = [s for s in comp.steps if s.op == "title.place"]
    assert len(title_steps) == 1
    emitted = title_steps[0].params["titles"]
    assert len(emitted) == 2
    assert {t["setting"] for t in emitted} == {
        "C:/Templates/Badge.setting",
        "C:/Templates/Sting.setting",
    }
    unresolved_refs = find_unresolved(title_steps[0].params)
    assert len(unresolved_refs) >= 2
    assert all(not intent.resolved for intent in comp.intents if intent.op == "title.place")


def test_title_place_setting_import_appends_comp_and_idempotent_retry() -> None:
    show, tpl_id = _macro_show()
    show.fake.faults.add("ImportFusionComp:append")
    params = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 0,
                "record_out": 90,
                "setting": "C:/Templates/LowerThird.setting",
                "inputs": {"Input1": "Alice", "Input2": "Host"},
            }
        ],
    }
    first = show.ok("title.place", params)
    assert first.titles[0].status == "placed"
    assert first.titles[0].inputs == {"Input1": "Alice", "Input2": "Host"}

    # Verify comp 2 (the imported comp) received the writes and comp 1 remained untouched.
    item = show.main.GetItemListInTrack("video", 2)[0]
    assert item.GetFusionCompCount() == 2
    assert item.GetFusionCompByIndex(1).FindToolByID("MacroOperator").GetInput("Input1") == "NAME"
    assert item.GetFusionCompByIndex(2).FindToolByID("MacroOperator").GetInput("Input1") == "Alice"
    import_calls_1 = [c for c in show.fake.calls if c[0] == "ImportFusionComp"]
    assert len(import_calls_1) == 1

    # Second placement with the exact same setting must NOT re-import and must return "present".
    second = show.ok("title.place", params)
    assert second.titles[0].status == "present"
    assert item.GetFusionCompCount() == 2
    assert len([c for c in show.fake.calls if c[0] == "ImportFusionComp"]) == 1

    # Updating an input on the reused item edits comp 2 without re-importing.
    params_upd = {
        "timeline": show.ref("Main"),
        "titles": [{**params["titles"][0], "inputs": {"Input1": "Bob", "Input2": "Host"}}],
    }
    third = show.ok("title.place", params_upd)
    assert third.titles[0].status == "updated"
    assert item.GetFusionCompByIndex(2).FindToolByID("MacroOperator").GetInput("Input1") == "Bob"
    assert len([c for c in show.fake.calls if c[0] == "ImportFusionComp"]) == 1


def test_title_place_setting_load_failure_raises_instead_of_succeeding() -> None:
    show, tpl_id = _macro_show()
    show.fake.faults.add("LoadSettings:false")
    params = {
        "timeline": show.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl_id},
                "track": 2,
                "record_in": 0,
                "record_out": 90,
                "setting": "C:/Templates/Broken.setting",
            }
        ],
    }
    err = show.fails("title.place", params)
    assert err.code == "title_setting_not_loaded"


def test_title_place_rejects_non_edit_pc_setting_path() -> None:
    import pytest
    from pydantic import ValidationError

    from aar.node.ops.build import TitleSpec

    with pytest.raises(ValidationError):
        TitleSpec.model_validate(
            {
                "template": {"unique_id": "mpi-1"},
                "track": 2,
                "record_in": 0,
                "record_out": 60,
                "setting": "/etc/passwd",
            }
        )


def test_title_place_applies_both_lines_and_inputs_with_precedence() -> None:
    # 1. TextPlus with both lines and inputs (non-overlapping)
    show = Show()
    show.main.AddTrack("video")
    tp = show.project.media_pool.add_clip(
        ("Footage",),
        "Text+ With Size",
        kind="video",
        fps=29.97,
        frames=300,
        title_tool="TextPlus",
        title_inputs={"StyledText": "DEFAULT", "Size": 0.05},
    )
    res = show.ok(
        "title.place",
        {
            "timeline": show.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tp.GetUniqueId()},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 90,
                    "lines": ["Alice", "Host"],
                    "inputs": {"Size": 0.09},
                }
            ],
        },
    )
    assert res.titles[0].text == "Alice\nHost"
    assert res.titles[0].inputs == {"Size": 0.09}
    tool = (
        show.main.GetItemListInTrack("video", 2)[0].GetFusionCompByIndex(1).FindToolByID("TextPlus")
    )
    assert tool.GetInput("StyledText") == "Alice\nHost"
    assert tool.GetInput("Size") == 0.09

    # 2. MacroOperator with both lines and overlapping/non-overlapping inputs
    show2, macro_id = _macro_show()
    res2 = show2.ok(
        "title.place",
        {
            "timeline": show2.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": macro_id},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 90,
                    "lines": ["Fallback1", "Subtitle"],
                    "inputs": {"Input1": "Override1", "Input3": 0.5},
                }
            ],
        },
    )
    assert res2.titles[0].text == "Override1\nSubtitle"
    assert res2.titles[0].inputs == {
        "Input2": "Subtitle",
        "Input1": "Override1",
        "Input3": 0.5,
    }
    macro = (
        show2.main.GetItemListInTrack("video", 2)[0]
        .GetFusionCompByIndex(1)
        .FindToolByID("MacroOperator")
    )
    assert macro.GetInput("Input1") == "Override1"
    assert macro.GetInput("Input2") == "Subtitle"
    assert macro.GetInput("Input3") == 0.5


def test_fusion_apply_helpers_display_names_point_dicts_and_manifest_validation(
    tmp_path: Path,
) -> None:
    import pytest

    from aar.fusion.apply import (
        apply_fusion_inputs_to_item,
        load_fusion_manifest,
        resolve_tool_input_id,
        values_match,
    )
    from aar.fusion.errors import FusionError

    from .fake_resolve import FakeFusionTool, FakeResolve

    # Resolve 1-based Point table dicts and bool/numeric semantics
    assert values_match({1: 0.5, 2: 0.25, 3: 0.0}, [0.5, 0.25]) is True
    assert values_match({1: 0.5, 2: 0.9}, [0.5, 0.25]) is False
    assert values_match(0.0, False) is True
    assert values_match(1.0, True) is True
    assert values_match(False, 0) is False

    # Display-name resolution via GetInputList()
    fake = FakeResolve({})
    tool = FakeFusionTool(
        fake,
        "ti-1",
        "MacroOperator",
        {"Input1": "Old", "Input2": 0.5},
        input_names={"Input1": "SpeakerName", "Input2": "BarOpacity"},
    )
    assert resolve_tool_input_id(tool, "SpeakerName") == "Input1"
    assert resolve_tool_input_id(tool, "Input1") == "Input1"

    # Malformed manifest schema wraps Pydantic ValidationError as FusionError("invalid_setting")
    bad_manifest = tmp_path / "bad.fusion.json"
    bad_manifest.write_text('{"schema_version": 1, "unexpected_key": true}\n', encoding="utf-8")
    with pytest.raises(FusionError) as exc:
        load_fusion_manifest(bad_manifest)
    assert exc.value.code == "invalid_setting"

    # apply_fusion_inputs_to_item requires non-None verified read-back
    class _StubTool:
        ID = "MacroOperator"
        Name = "M"

        def SetInput(self, _k: str, _v: object) -> None:
            return None

        def GetInput(self, _k: str) -> None:
            return None

    class _StubComp:
        def GetToolList(self, _sel: bool = False) -> dict[int, object]:
            return {1: _StubTool()}

    class _StubItem:
        def GetFusionCompCount(self) -> int:
            return 1

        def GetFusionCompByIndex(self, _idx: int) -> object:
            return _StubComp()

    with pytest.raises(FusionError) as exc2:
        apply_fusion_inputs_to_item(_StubItem(), inputs={"Input1": "Hello"})
    assert exc2.value.code == "apply_failed"


def test_title_place_setting_identity_distinct_paths_and_unrelated_comp() -> None:
    """Retry matches only the exact canonical setting path, never a same-stem path or an unrelated
    pre-existing Composition 2. Plain titles without a setting still target comp 1.
    """
    from .fake_resolve import FakeFusionComp, FakeFusionTool

    # 1. Two different full paths sharing the same filename stem ("LowerThird.setting").
    show, tpl_id = _macro_show()
    show.fake.faults.add("ImportFusionComp:append")
    setting_a = "D:\\Templates\\PackA\\LowerThird.setting"
    setting_b = "D:\\Templates\\PackB\\LowerThird.setting"

    first = show.ok(
        "title.place",
        {
            "timeline": show.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "setting": setting_a,
                    "inputs": {"Input1": "From A"},
                }
            ],
        },
    )
    assert first.titles[0].status == "placed"
    item = show.main.GetItemListInTrack("video", 2)[0]
    assert item.GetFusionCompCount() == 2
    assert item.active_comp_name == "LowerThird"

    # Retrying with the same path (different slash/case spelling) reuses comp 2 without reload.
    retry_a = show.ok(
        "title.place",
        {
            "timeline": show.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "setting": "d:/Templates/PackA/lowerthird.setting",
                    "inputs": {"Input1": "From A"},
                }
            ],
        },
    )
    assert retry_a.titles[0].status == "present"
    assert item.GetFusionCompCount() == 2

    # Retrying with a DIFFERENT full path that has the same stem must NOT match PackA's comp.
    second = show.ok(
        "title.place",
        {
            "timeline": show.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "setting": setting_b,
                    "inputs": {"Input1": "From B"},
                }
            ],
        },
    )
    assert second.titles[0].status == "updated"
    assert item.GetFusionCompCount() == 3
    assert item.GetFusionCompByIndex(2).FindToolByID("MacroOperator").GetInput("Input1") == "From A"
    assert item.GetFusionCompByIndex(3).FindToolByID("MacroOperator").GetInput("Input1") == "From B"
    assert item.active_comp_name == "LowerThird 1"

    # 2. Pre-existing unrelated "Composition 2" with a MacroOperator must never match as loaded.
    show2, tpl_id2 = _macro_show()
    show2.ok(
        "title.place",
        {
            "timeline": show2.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id2},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "inputs": {"Input1": "Comp1 Text"},
                }
            ],
        },
    )
    item2 = show2.main.GetItemListInTrack("video", 2)[0]
    unrelated_macro = FakeFusionTool(
        show2.fake, item2.GetUniqueId(), "MacroOperator", {"Input1": "UNRELATED"}, name="OtherMacro"
    )
    item2.comps.append(
        FakeFusionComp(show2.fake, item2.GetUniqueId(), [unrelated_macro], name="Composition 2")
    )

    # Without ImportFusionComp succeeding, must refuse instead of selecting or clobbering Comp 2.
    err = show2.fails(
        "title.place",
        {
            "timeline": show2.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id2},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "setting": setting_a,
                    "inputs": {"Input1": "Should Not Clobber"},
                }
            ],
        },
    )
    assert err.code == "title_setting_not_loaded"
    assert unrelated_macro.GetInput("Input1") == "UNRELATED"

    # Plain title (setting=None) still targets comp 1, leaving Composition 2 untouched.
    plain = show2.ok(
        "title.place",
        {
            "timeline": show2.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl_id2},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "inputs": {"Input1": "Still Comp1"},
                }
            ],
        },
    )
    assert plain.titles[0].status == "updated"
    assert (
        item2.GetFusionCompByIndex(1).FindToolByID("MacroOperator").GetInput("Input1")
        == "Still Comp1"
    )
    assert unrelated_macro.GetInput("Input1") == "UNRELATED"


def test_title_place_setting_durability_activation_and_posix_case() -> None:
    """Setting identity requires durable SetData/GetData read-back, multi-comp items require
    LoadFusionCompByName activation to succeed, and POSIX paths stay case-sensitive.
    """
    from aar.fusion.apply import _norm_setting_path

    # Windows paths fold case; POSIX paths keep case sensitivity.
    assert _norm_setting_path("D:\\PackA\\LowerThird.setting") == _norm_setting_path(
        "d:/packa/lowerthird.setting"
    )
    assert _norm_setting_path("/mnt/Templates/LowerThird.setting") != _norm_setting_path(
        "/mnt/Templates/lowerthird.setting"
    )

    # 1. If SetData does not persist (GetData read-back fails), title.place must fail loudly.
    show1, tpl1 = _macro_show()
    show1.fake.faults.add("ImportFusionComp:append")
    show1.fake.faults.add("SetData:no-op")
    err_data = show1.fails(
        "title.place",
        {
            "timeline": show1.ref("Main"),
            "titles": [
                {
                    "template": {"unique_id": tpl1},
                    "track": 2,
                    "record_in": 0,
                    "record_out": 24,
                    "setting": "D:\\Templates\\LowerThird.setting",
                    "inputs": {"Input1": "Alice"},
                }
            ],
        },
    )
    assert err_data.code == "title_setting_not_loaded"

    # 2. If LoadFusionCompByName fails on a multi-comp item (initial load or retry), fail loudly.
    show2, tpl2 = _macro_show()
    show2.fake.faults.add("ImportFusionComp:append")
    show2.fake.faults.add("LoadFusionCompByName:fail")
    params = {
        "timeline": show2.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl2},
                "track": 2,
                "record_in": 0,
                "record_out": 24,
                "setting": "D:\\Templates\\LowerThird.setting",
                "inputs": {"Input1": "Alice"},
            }
        ],
    }
    err_act_initial = show2.fails("title.place", params)
    assert err_act_initial.code == "title_setting_not_loaded"

    # On retry of a previously loaded multi-comp item, activation failure also surfaces an error.
    show3, tpl3 = _macro_show()
    show3.fake.faults.add("ImportFusionComp:append")
    params3 = {
        "timeline": show3.ref("Main"),
        "titles": [
            {
                "template": {"unique_id": tpl3},
                "track": 2,
                "record_in": 0,
                "record_out": 24,
                "setting": "D:\\Templates\\LowerThird.setting",
                "inputs": {"Input1": "Alice"},
            }
        ],
    }
    assert show3.ok("title.place", params3).titles[0].status == "placed"
    show3.fake.faults.add("LoadFusionCompByName:fail")
    err_act_retry = show3.fails("title.place", params3)
    assert err_act_retry.code == "title_setting_not_loaded"
