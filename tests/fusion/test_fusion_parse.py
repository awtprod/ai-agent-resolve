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


def test_customize_bool_numeric_and_expression_override(tmp_path: Path) -> None:
    expr_setting = """{
\tTools = ordered() {
\t\tExprMacro = MacroOperator {
\t\t\tInputs = ordered() {
\t\t\t\tInput1 = InstanceInput { SourceOp = "Bg", Source = "TopLeftAlpha", Name = "Enabled", },
\t\t\t\tInput2 = InstanceInput { SourceOp = "Rect", Source = "Width", Name = "BarWidth", },
\t\t\t\tInput3 = InstanceInput { SourceOp = "Rect", Source = "Center", Name = "CenterPt", },
\t\t\t},
\t\t\tOutputs = { MainOutput1 = InstanceOutput { SourceOp = "Bg", Source = "Output", }, },
\t\t\tTools = ordered() {
\t\t\t\tBg = Background {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tTopLeftAlpha = Input { Value = 1, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t\tRect = RectangleMask {
\t\t\t\t\tInputs = {
\t\t\t\t\t\tWidth = Input { Value = 0.5, Expression = "time / 100", },
\t\t\t\t\t\tHeight = Input { Value = 0.2, Expression = "0.1 + time / 200", },
\t\t\t\t\t\tCenter = Input { Value = { 0.5, 0.5 }, },
\t\t\t\t\t},
\t\t\t\t},
\t\t\t},
\t\t},
\t},
}
"""
    setting = tmp_path / "ExprMacro.setting"
    setting.write_text(expr_setting, encoding="utf-8")
    tpl = load_template(setting)

    # False on numeric input must write 0 (not 1!), and overriding BarWidth must strip its
    # competing Expression while keeping Rect.Height's Expression intact.
    out = tmp_path / "out_bool.setting"
    text = customize_setting(tpl, inputs={"Enabled": False, "BarWidth": 0.75})
    out.write_text(text, encoding="utf-8")
    assert "TopLeftAlpha = Input { Value = 0" in text
    assert "time / 100" not in text
    assert "0.1 + time / 200" in text
    reloaded = load_template(out)
    by_name = {i.name: i for i in reloaded.inputs}
    assert by_name["Enabled"].default == 0
    assert by_name["BarWidth"].default == pytest.approx(0.75)

    # True writes 1 on numeric input
    text_true = customize_setting(tpl, inputs={"Enabled": True})
    assert "TopLeftAlpha = Input { Value = 1" in text_true

    # Bool inside 2D point tuple must be rejected
    with pytest.raises(FusionError) as exc:
        apply_inputs_to_ast(tpl, inputs={"CenterPt": [True, 0.5]})
    assert exc.value.code == "invalid_input_value"


def test_parse_bounds_bare_hex_depth_and_excess_lines(tmp_path: Path) -> None:
    from aar.fusion.parse import MAX_SETTING_BYTES, read_zip_member_bounded

    with pytest.raises(FusionError) as exc_hex:
        parse_setting("{ Value = 0x }")
    assert exc_hex.value.code == "invalid_setting"

    deep_lua = "{" * 200 + "}" * 200
    with pytest.raises(FusionError) as exc_depth:
        parse_setting(deep_lua)
    assert exc_depth.value.code == "invalid_setting"

    # Bounded zip member read rejects oversized archive entries
    zpath = tmp_path / "big.drfx"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Edit/Titles/Huge.setting", "{" + " " * 1024 + "}")
    with zipfile.ZipFile(zpath, "r") as zf, pytest.raises(FusionError) as exc_zip:
        read_zip_member_bounded(zf, "Edit/Titles/Huge.setting", max_bytes=100)
    assert exc_zip.value.code == "invalid_setting"
    assert MAX_SETTING_BYTES > 0

    # Excess lines on a 2-text-input macro must raise invalid_input_value (never silently drop)
    setting = tmp_path / "TestLowerThird.setting"
    setting.write_text(SAMPLE_MACRO, encoding="utf-8")
    tpl = load_template(setting)
    with pytest.raises(FusionError) as exc_lines:
        map_lines_to_inputs(tpl, ["Line 1", "Line 2", "Line 3"])
    assert exc_lines.value.code == "invalid_input_value"


def test_template_and_asset_confinement(tmp_path: Path) -> None:
    from aar.fusion.parse import resolve_asset_path

    allowed_dir = tmp_path / "allowed"
    outside_dir = tmp_path / "outside"
    allowed_dir.mkdir()
    outside_dir.mkdir()

    tpl_a = allowed_dir / "A.setting"
    tpl_b = allowed_dir / "B.setting"
    tpl_out = outside_dir / "Secret.setting"
    for p in (tpl_a, tpl_b, tpl_out):
        p.write_text(SAMPLE_MACRO, encoding="utf-8")

    # allowed_files strictly restricts to listed templates, blocking siblings in same directory
    assert resolve_template_path("A.setting", allowed_files=[tpl_a]) == tpl_a.resolve()
    assert resolve_template_path("B.setting", allowed_files=[tpl_a]) is None
    assert resolve_template_path(str(tpl_out), allowed_files=[tpl_a]) is None
    assert resolve_template_path("../outside/Secret.setting", search_dirs=[allowed_dir]) is None

    # allow_direct=False rejects paths outside search_dirs and symlinks escaping search_dirs
    assert (
        resolve_template_path(str(tpl_out), search_dirs=[allowed_dir], allow_direct=False) is None
    )
    escape_link = allowed_dir / "Escape.setting"
    escape_link.symlink_to(tpl_out)
    assert (
        resolve_template_path("Escape.setting", search_dirs=[allowed_dir], allow_direct=False)
        is None
    )

    # resolve_asset_path must not climb out of confined_roots or follow escaping symlinks
    parent_secret = tmp_path / "secret.png"
    parent_secret.write_bytes(b"secret")
    outside_img = outside_dir / "outside.png"
    outside_img.write_bytes(b"out")
    img_link = allowed_dir / "link.png"
    img_link.symlink_to(outside_img)

    assert resolve_asset_path("secret.png", template_path=tpl_a, confined_roots=[allowed_dir]) == (
        None,
        None,
    )
    assert resolve_asset_path(
        str(outside_img), template_path=tpl_a, confined_roots=[allowed_dir]
    ) == (None, None)
    assert resolve_asset_path(
        "Setting:../secret.png", template_path=tpl_a, confined_roots=[allowed_dir]
    ) == (None, None)
    assert resolve_asset_path(
        "Setting:link.png", template_path=tpl_a, confined_roots=[allowed_dir]
    ) == (None, None)
    ok_img = allowed_dir / "logo.png"
    ok_img.write_bytes(b"ok")
    assert resolve_asset_path(
        "Setting:logo.png", template_path=tpl_a, confined_roots=[allowed_dir]
    ) == (ok_img.resolve(), None)
