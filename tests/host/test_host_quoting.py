"""ps_quote keeps every embedded value a single literal *as PowerShell parses it* (HIGH-1).

Adapted from the validator's probe_quoting. No PowerShell runs here: ``ps_tokenize_single_quoted``
models PowerShell's single-quote tokenizer (Tokenizer.ScanStringLiteral with
CharExtensions.IsSingleQuote): U+0027, U+2018, U+2019, U+201A and U+201B all open and close a
single-quoted string, and two consecutive ones are an escaped quote. The invariant: quoting a
value must not change the code skeleton (text outside literals), and the literal read back must
equal the value.
"""

from __future__ import annotations

import pytest
from host_support import FakeRunner, decode_encoded_command, make_config, sha256_upper

from aar.host.powershell import ps_quote
from aar.host.transport import EditHost, exists_script, move_script, remove_script, stat_script
from aar.host.winpath import normalize_root, validate_windows_path, windows_literal

PS_SINGLE_QUOTES = "".join(map(chr, (0x27, 0x2018, 0x2019, 0x201A, 0x201B)))


def ps_tokenize_single_quoted(script: str) -> tuple[str, list[str]]:
    """Split ``script`` into (code skeleton with literals replaced by <S>, literal values)."""
    skeleton: list[str] = []
    literals: list[str] = []
    i, n = 0, len(script)
    while i < n:
        ch = script[i]
        if ch in PS_SINGLE_QUOTES:
            i += 1
            value: list[str] = []
            while True:
                if i >= n:
                    raise ValueError("unterminated single-quoted string (PowerShell parse error)")
                c = script[i]
                i += 1
                if c in PS_SINGLE_QUOTES:
                    if i < n and script[i] in PS_SINGLE_QUOTES:
                        value.append(script[i])  # the second quote of a doubled pair
                        i += 1
                        continue
                    break
                value.append(c)
            literals.append("".join(value))
            skeleton.append("<S>")
        else:
            skeleton.append(ch)
            i += 1
    return "".join(skeleton), literals


ROOTS = (normalize_root("E:\\Show"),)
BENIGN = "E:\\Show\\benign.mp4"


def _scripts(literal: str) -> dict[str, str]:
    return {
        "stat": stat_script(literal),
        "exists": exists_script(literal),
        "move-src": move_script(literal, BENIGN, False),
        "move-dst": move_script(BENIGN, literal, True),
        "remove": remove_script(literal),
    }


# Names that are legal on Windows and pass validate_windows_path.
SAFE_NAMES = [
    "O'Brien.mp4",
    "cut [v2].mp4",
    "tick`n.mp4",
    "$(Get-Process).mp4",
    "${env_TEMP}.mp4",
    "a;b & c.mp4",
    "caf\u00e9 \u00fcber.mp4",
    "clef \U0001d11e.mp4",
    "it''s.mp4",
]


@pytest.mark.parametrize("name", SAFE_NAMES)
def test_ascii_quote_and_metachar_names_stay_one_literal(name: str) -> None:
    win = validate_windows_path(f"E:\\Show\\{name}", ROOTS)
    literal = windows_literal(win)
    for label, script in _scripts(literal).items():
        skel_ok, _ = ps_tokenize_single_quoted(_scripts(BENIGN)[label])
        skel, lits = ps_tokenize_single_quoted(script)
        assert skel == skel_ok, f"{label}: code skeleton changed for {name!r}"
        assert literal in lits, f"{label}: literal not read back verbatim"


SMART_QUOTES = list(map(chr, (0x2018, 0x2019, 0x201A, 0x201B)))


@pytest.mark.parametrize("q", SMART_QUOTES, ids=["U+2018", "U+2019", "U+201A", "U+201B"])
def test_ps_quote_doubles_every_powershell_single_quote(q: str) -> None:
    value = f"Don{q}t Tell.mp4"
    skel, lits = ps_tokenize_single_quoted(ps_quote(value))
    assert (skel, lits) == ("<S>", [value]), (
        f"ps_quote({value!r}) = {ps_quote(value)!r} parses as skeleton={skel!r} literals={lits!r}"
    )


def test_smart_apostrophe_name_is_a_valid_path_and_stays_one_literal() -> None:
    # A very common real filename (macOS/Word autocorrect apostrophe).
    win = validate_windows_path(f"E:\\Show\\Don{chr(0x2019)}t Tell Mom.mp4", ROOTS)  # accepted
    skel, _ = ps_tokenize_single_quoted(stat_script(windows_literal(win)))
    benign_skel, _ = ps_tokenize_single_quoted(stat_script(BENIGN))
    assert skel == benign_skel, f"stat script skeleton changed: {skel!r}"


INJECT = f"E:\\Show\\x{chr(0x2019)}; Stop-Computer -Force; {chr(0x2019)}.mp4"


def test_injection_path_validates_inside_root_but_runs_no_code(tmp_path) -> None:
    """End to end through EditHost.pull: the decoded -EncodedCommand carries no live code."""
    validate_windows_path(INJECT, ROOTS)  # passes every lexical check (no <>:"|?*/\\, no ctl)
    runner = FakeRunner(stat_result=(1, sha256_upper(b"x")), scp_payload=b"x")
    host = EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")
    host.pull(INJECT, str(tmp_path / "out.mp4"))
    sent = decode_encoded_command(runner.calls[0])
    skel, _ = ps_tokenize_single_quoted(sent)
    benign_skel, _ = ps_tokenize_single_quoted(stat_script(BENIGN))
    assert "Stop-Computer" not in skel, f"code outside any literal:\n{skel}"
    assert skel == benign_skel


def test_push_injection_via_destination_runs_no_code(tmp_path) -> None:
    src = tmp_path / "m.mov"
    src.write_bytes(b"m")
    runner = FakeRunner(stat_result=(1, sha256_upper(b"m")))
    host = EditHost(make_config(tmp_path), runner, token_factory=lambda: "T")
    host.push(str(src), INJECT)
    for script in runner.scripts:
        skel, _ = ps_tokenize_single_quoted(script)
        assert "Stop-Computer" not in skel, f"code outside any literal:\n{skel}"


def test_model_agrees_with_ascii_doubling_rule() -> None:
    for value in ["", "'", "''", "a'b", "$()`;\n", "\U0001d11e"]:
        assert ps_tokenize_single_quoted(ps_quote(value)) == ("<S>", [value])
