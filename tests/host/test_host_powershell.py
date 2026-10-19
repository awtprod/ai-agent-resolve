"""Quoting, -EncodedCommand building, the command-line limit, and output cleanup."""

from __future__ import annotations

import base64

import pytest

from aar.host.errors import ScriptTooLong
from aar.host.powershell import (
    WINDOWS_CMDLINE_MAX,
    encode_command,
    normalize_newlines,
    powershell_argv,
    ps_quote,
    strip_clixml,
)

_PREFIX = "powershell -NoProfile -NonInteractive -EncodedCommand "


def test_ps_quote_wraps_in_single_quotes() -> None:
    assert ps_quote("plain") == "'plain'"


def test_ps_quote_doubles_single_quotes() -> None:
    assert ps_quote("O'Brien") == "'O''Brien'"
    assert ps_quote("''") == "''''''"


@pytest.mark.parametrize(
    "value",
    ["$()", "$(Get-Process)", "`n`t", "a;b;c", "line1\nline2", "mixed'$(`;"],
)
def test_ps_quote_keeps_metacharacters_literal(value: str) -> None:
    # Inside single quotes everything but ' is literal, so quoting only doubles ' and wraps.
    assert ps_quote(value) == "'" + value.replace("'", "''") + "'"


def test_ps_quote_preserves_non_bmp() -> None:
    treble = "\U0001d11e"  # U+1D11E MUSICAL SYMBOL G CLEF (a surrogate pair in UTF-16)
    quoted = ps_quote(f"name {treble}")
    assert quoted == f"'name {treble}'"
    # survives the UTF-16LE encode/decode used for -EncodedCommand
    assert base64.b64decode(encode_command(quoted)).decode("utf-16-le") == quoted


def test_ps_quote_refuses_nul() -> None:
    with pytest.raises(ValueError, match="NUL"):
        ps_quote("a\x00b")


def test_encode_command_is_utf16le_base64() -> None:
    script = "Write-Output 'hi'"
    encoded = encode_command(script)
    assert base64.b64decode(encoded) == script.encode("utf-16-le")
    assert base64.b64decode(encoded).decode("utf-16-le") == script


def test_powershell_argv_pins_flags_and_payload() -> None:
    script = "Get-Date"
    argv = powershell_argv(script)
    assert argv == [
        "powershell",
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        encode_command(script),
    ]


def test_powershell_argv_accepts_a_script_under_the_limit() -> None:
    # ~10k chars -> well under the limit; must not raise.
    powershell_argv("a" * 10_000)


def test_powershell_argv_refuses_over_the_limit() -> None:
    script = "a" * 20_000  # UTF-16LE then base64 blows past 32767
    with pytest.raises(ScriptTooLong) as info:
        powershell_argv(script)
    encoded = encode_command(script)
    assert info.value.length == len(_PREFIX + encoded)
    assert info.value.limit == WINDOWS_CMDLINE_MAX
    assert "32767" in str(info.value)


def test_normalize_newlines() -> None:
    assert normalize_newlines("a\r\nb\rc\n") == "a\nb\nc\n"


def test_strip_clixml_removes_progress_records() -> None:
    stderr = (
        "#< CLIXML\n"
        '<Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/powershell/2004/04">'
        '<Obj S="progress" RefId="0"><TN RefId="0"><T>System.Management.Automation.Host'
        '.ProgressRecord</T></TN><MS><I64 N="SourceId">1</I64><PR N="Record">'
        "<AV>Preparing modules for first use.</AV><AI>0</AI><PI>-1</PI><PC>-1</PC>"
        "<T>Completed</T><SR>-1</SR><SD> </SD></PR></MS></Obj></Objs>\n"
    )
    cleaned = strip_clixml(stderr)
    assert "CLIXML" not in cleaned
    assert "progress" not in cleaned
    assert cleaned.strip() == ""


def test_strip_clixml_keeps_error_text() -> None:
    stderr = (
        "#< CLIXML\n"
        '<Objs Version="1.1.0.1" xmlns="x"><S S="Error">Get-Item : Cannot find path '
        "'E:\\Show\\missing' because it does not exist._x000D__x000A_</S></Objs>\n"
    )
    cleaned = strip_clixml(stderr)
    assert "Cannot find path" in cleaned
    assert "CLIXML" not in cleaned and "<Objs" not in cleaned


def test_strip_clixml_passthrough_without_marker() -> None:
    assert strip_clixml("plain stderr line\n") == "plain stderr line\n"
