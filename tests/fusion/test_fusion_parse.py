"""Tests for ``aar.fusion.parse`` and ``aar.fusion.customize`` and inspect/customize CLI."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from aar.cli import main
from aar.fusion.customize import apply_inputs_to_ast, customize_setting
from aar.fusion.errors import FusionError
from aar.fusion.parse import (
    LuaTable,
    dump_setting,
    load_template,
    map_lines_to_inputs,
    parse_setting,
    resolve_template_path,
)

SAMPLE_MACRO = """{
\tTools = ordered() {
\t\tTestLowerThird = MacroOperator {
\t\t\tInputs = ordered() {
\t\t\t\tInput1 = InstanceInput {
\t\t\t\t\tSourceOp = "NameText",
\t\t\t\t\tSource = "StyledText",
\t\t\t\t\tName = "Name",
\t\t\t\t},
\t\t\t\tInput2 = InstanceInput {
\t\t\t\t\tSourceOp = "TitleText",
\t\t\t\t\tSource = "StyledText",
\t\t\t\t\tName = "Title",
\t\t\t\t},
\t\t\t\tInput3 = InstanceInput {
\t\t\t\t\tSourceOp = "BgBox",
\t\t\t\t\tSource = "TopLeftRed",
\t\t\t\t\tName = "AccentRed",
\t\t\t\t\tDefault = 0.8,
\t\t\t\t},
\t\t\t},
\t\t\tOutputs = {
\t\t\t\tMainOutput1 = InstanceOutput {
\t\t\t\t\tSourceOp = "OutMerge",
\t\t\t\t\tSource = "Output",
\t\t\t\t},
\t\t\t},
\t\t\tTools = ordered() {
\t\t\t\tBgBox = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 1920, },
\t\t\t\t\t\tHeight = Input { Value = 1080, },
\t\t\t\t\t\tTopLeftRed = Input { Value = 0.8, },
\t\t\t\t\t\tTopLeftGreen = Input { Value = 0.1, },
\t\t\t\t\t\tTopLeftBlue = Input { Value = 0.1, },
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tLogoLoader = Loader {
\t\t\t\t\tClips = {
\t\t\t\t\t\tClip {
\t\t\t\t\t\t\tID = "Clip1",
\t\t\t\t\t\t\tFilename = "Templates:\\\\Edit\\\\Titles\\\\logo.png",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tNameText = TextPlus {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tStyledText = Input { Value = "DEFAULT NAME", },
\t\t\t\t\t\tFont = Input { Value = "DejaVu Sans", },
\t\t\t\t\t\tStyle = Input { Value = "Bold", },
\t\t\t\t\t\tSize = Input { Value = 0.05, },
\t\t\t\t\t\tCenter = Input {
\t\t\t\t\t\t\tValue = { 0.2, 0.2 },
\t\t\t\t\t\t\tExpression = "Point(0.2, 0.2)",
\t\t\t\t\t\t},
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tTitleText = TextPlus {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tStyledText = Input { Value = "DEFAULT TITLE", },
\t\t\t\t\t\tFont = Input { Value = "DejaVu Sans", },
\t\t\t\t\t\tSize = Input { Value = 0.03, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tOutMerge = Merge {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tBackground = Input { SourceOp = "BgBox", Source = "Output", },
\t\t\t\t\t\tForeground = Input { SourceOp = "NameText", Source = "Output", },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""


def test_parse_and_dump_round_trip() -> None:
    ast = parse_setting(SAMPLE_MACRO)
    assert isinstance(ast, LuaTable)
    dumped = dump_setting(ast)
    reparsed = parse_setting(dumped)
    assert dump_setting(reparsed) == dumped


def test_load_template_extracts_inputs_assets_and_fonts(tmp_path: Path) -> None:
    setting = tmp_path / "TestLowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")
    tpl = load_template(setting)
    assert tpl.name == "TestLowerThird"
    assert tpl.operator_type == "MacroOperator"
    assert tpl.output_tool == "OutMerge"
    assert len(tpl.inputs) == 3
    assert [i.key for i in tpl.inputs] == ["Input1", "Input2", "Input3"]
    assert [i.name for i in tpl.inputs] == ["Name", "Title", "AccentRed"]
    assert tpl.inputs[0].default == "DEFAULT NAME"
    assert tpl.inputs[2].default == pytest.approx(0.8)
    assert len(tpl.assets) == 1
    assert tpl.assets[0].tool == "LogoLoader"
    assert "logo.png" in tpl.assets[0].filename
    assert ("DejaVu Sans", "Bold") in tpl.fonts


def test_load_template_from_drfx_zip(tmp_path: Path) -> None:
    drfx = tmp_path / "Pack.drfx"
    with zipfile.ZipFile(drfx, "w") as zf:
        zf.writestr("Edit/Titles/TestLowerThird.setting", SAMPLE_MACRO)
        zf.writestr("Edit/Titles/logo.png", b"\x89PNG\r\n\x1a\n")
    tpl = load_template(drfx)
    assert tpl.name == "TestLowerThird"
    assert tpl.member == "Edit/Titles/TestLowerThird.setting"
    assert tpl.assets[0].archive_member == "Edit/Titles/logo.png"


def test_customize_setting_updates_inputs_and_preserves_expressions(tmp_path: Path) -> None:
    setting = tmp_path / "TestLowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")
    out = tmp_path / "custom.setting"
    tpl = load_template(setting)

    text = customize_setting(
        tpl,
        inputs={"Name": "Jane Doe", "AccentRed": 0.25},
        lines=["IgnoredBecauseInputWins", "Host & Producer"],
    )
    out.write_text(text, encoding="utf-8")

    reloaded = load_template(out)
    assert reloaded.inputs[0].default == "Jane Doe"
    assert reloaded.inputs[1].default == "Host & Producer"
    assert reloaded.inputs[2].default == pytest.approx(0.25)
    # Expression on NameText.Center must survive untouched
    assert 'Expression = "Point(0.2, 0.2)"' in out.read_text(encoding="utf-8")


def test_map_lines_and_unknown_input_raises(tmp_path: Path) -> None:
    setting = tmp_path / "TestLowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")
    tpl = load_template(setting)
    mapped = map_lines_to_inputs(tpl, ["Line One", "Line Two"])
    assert mapped == {"Name": "Line One", "Title": "Line Two"}

    with pytest.raises(FusionError) as exc:
        apply_inputs_to_ast(tpl, inputs={"DoesNotExist": "x"})
    assert exc.value.code == "unknown_input"


def test_resolve_template_path_env_and_search_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tpl_dir = tmp_path / "templates"
    tpl_dir.mkdir()
    setting = tpl_dir / "LowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")

    assert resolve_template_path("LowerThird", search_dirs=[tpl_dir]) == setting.resolve()
    assert resolve_template_path("Missing", search_dirs=[tpl_dir]) is None

    monkeypatch.setenv("AAR_FUSION_TEMPLATES_DIR", str(tpl_dir))
    assert resolve_template_path("LowerThird") == setting.resolve()


def test_fusion_cli_inspect_and_customize(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    setting = tmp_path / "TestLowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")
    out = tmp_path / "out.setting"

    assert main(["fusion", "inspect", str(setting), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["name"] == "TestLowerThird"
    assert len(inspected["inputs"]) == 3

    assert (
        main(
            [
                "fusion",
                "customize",
                str(setting),
                "--input",
                "Name=Alex",
                "--line",
                "Line1",
                "--line",
                "Guest",
                "--out",
                str(out),
                "--json",
            ]
        )
        == 0
    )
    custom = json.loads(capsys.readouterr().out)
    assert custom["ok"] is True
    assert custom["template"] == "TestLowerThird"
    assert out.is_file()
    reloaded = load_template(out)
    assert reloaded.inputs[0].default == "Alex"
    assert reloaded.inputs[1].default == "Guest"
