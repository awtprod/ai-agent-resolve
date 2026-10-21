"""PowerShell invocation helpers: quoting, ``-EncodedCommand`` building, output cleanup.

Scripts are sent to the edit PC as ``powershell -NoProfile -NonInteractive -EncodedCommand
<base64 of UTF-16LE>``. That avoids a second layer of shell quoting on the Windows side: the
script bytes travel verbatim. Values embedded *inside* a script still need quoting, and
:func:`ps_quote` is the only sanctioned way to do it (a single-quoted PowerShell literal, with
``'`` doubled). Nothing here contacts a host.
"""

from __future__ import annotations

import base64
import re

from aar.host.errors import ScriptTooLong

__all__ = [
    "CMD_EXE_CMDLINE_MAX",
    "DEFAULT_REMOTE_CMDLINE_MAX",
    "POWERSHELL_CMDLINE_MAX",
    "encode_command",
    "normalize_newlines",
    "powershell_argv",
    "ps_quote",
    "strip_clixml",
]

# Win32-OpenSSH runs a remote command through the sshd DefaultShell. The default DefaultShell is
# cmd.exe, whose command line caps near 8191 characters; a PowerShell DefaultShell uses the
# CreateProcess cap of 32767. sshd wraps the command roughly as `cmd.exe /c <command>`, so the
# usable budget is the shell cap minus that wrapper. The cap is a host-config value
# (AAR_EDIT_CMDLINE_MAX); the default below is conservative (cmd.exe).
CMD_EXE_CMDLINE_MAX = 8191
POWERSHELL_CMDLINE_MAX = 32767
_CMD_C_OVERHEAD = len("cmd.exe /c ")  # the sshd command wrapper for a cmd.exe DefaultShell
DEFAULT_REMOTE_CMDLINE_MAX = CMD_EXE_CMDLINE_MAX - _CMD_C_OVERHEAD

# Fixed tokens of the remote command. Order and spelling are pinned by tests.
_PS_PREFIX: tuple[str, ...] = ("powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand")


# PowerShell's tokenizer (CharExtensions.IsSingleQuote) treats all of U+0027 and the four
# "smart" single quotes U+2018, U+2019, U+201A, U+201B as string delimiters, and its own escaper
# (CodeGeneration.EscapeSingleQuotedStringContent) doubles all of them. Doubling only U+0027 would
# let a name such as a right-single-quote followed by "; Stop-Computer; " break out of the literal.
# None of the four is illegal in a Windows filename, so validate_windows_path accepts them.
_PS_SINGLE_QUOTES = frozenset(map(chr, (0x27, 0x2018, 0x2019, 0x201A, 0x201B)))


def ps_quote(value: str) -> str:
    """Return ``value`` as a single-quoted PowerShell literal.

    Every PowerShell single-quote character (``'`` and the smart quotes U+2018, U+2019, U+201A,
    U+201B) is doubled; everything else -- ``$()``, backticks, ``;``, newlines -- is literal inside
    single quotes. A NUL is refused: it cannot be carried safely and has no place in a path or
    argument. All embedded values in this package's scripts use single-quoted literals, so the
    double-quoted smart quotes (U+201C/U+201D/U+201E) do not arise.
    """
    if "\x00" in value:
        raise ValueError("a PowerShell literal cannot contain NUL")
    return "'" + "".join(ch * 2 if ch in _PS_SINGLE_QUOTES else ch for ch in value) + "'"


def encode_command(script: str) -> str:
    """Base64 of ``script`` encoded as UTF-16LE, the argument for ``-EncodedCommand``."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def powershell_argv(script: str, *, cmdline_max: int = DEFAULT_REMOTE_CMDLINE_MAX) -> list[str]:
    """Build the remote ``powershell ... -EncodedCommand <b64>`` argument list.

    Refuses (with :class:`ScriptTooLong`) when the whole command line would not fit ``cmdline_max``
    (the usable budget for the remote DefaultShell), so the failure happens here, not as a
    truncated command on the host. A NUL in the script is refused with :class:`ValueError`: it
    cannot be carried (a BOM-less UTF-16 ``.ps1`` decoded as UTF-8 is full of NULs).
    """
    if "\x00" in script:
        raise ValueError("the PowerShell script contains a NUL character")
    encoded = encode_command(script)
    command_line = " ".join((*_PS_PREFIX, encoded))
    if len(command_line) > cmdline_max:
        raise ScriptTooLong(len(command_line), cmdline_max)
    return [*_PS_PREFIX, encoded]


def normalize_newlines(text: str) -> str:
    """Normalise CRLF and lone CR to LF."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


_CLIXML_MARKER = "#< CLIXML"
_OBJS = re.compile(r"<Objs\b[^>]*>(.*?)</Objs>", re.S)
# A CLIXML <S> string carries real text (errors, warnings, verbose); progress is <Obj S="progress">.
_S_STRING = re.compile(r'<S(?:\s+S="[^"]*")?>(.*?)</S>', re.S)
_XML_ENTITY = {"&lt;": "<", "&gt;": ">", "&quot;": '"', "&apos;": "'", "&amp;": "&"}
_CLIXML_ESCAPE = re.compile(r"_x([0-9A-Fa-f]{4})_")


def _unescape_clixml(text: str) -> str:
    text = _CLIXML_ESCAPE.sub(lambda m: chr(int(m.group(1), 16)), text)
    for entity, char in _XML_ENTITY.items():
        text = text.replace(entity, char)
    return text


def strip_clixml(stderr: str) -> str:
    """Remove CLIXML progress records from PowerShell stderr, keeping real message text.

    PowerShell serialises its non-output streams as a ``#< CLIXML`` header followed by an
    ``<Objs>`` document. The progress stream is pure noise under ``-NonInteractive``; error and
    warning text is carried as ``<S>`` strings and is preserved (decoded to plain text) so a
    genuine failure is never hidden. Text outside the CLIXML document is left untouched.
    """
    if _CLIXML_MARKER not in stderr:
        return stderr
    head, _, rest = stderr.partition(_CLIXML_MARKER)

    def replace(match: re.Match[str]) -> str:
        strings = _S_STRING.findall(match.group(1))
        return "".join(_unescape_clixml(s) for s in strings)

    cleaned_rest = _OBJS.sub(replace, rest)
    # Drop a lone leading newline left where the marker used to be.
    return (head + cleaned_rest.lstrip("\n")) if not head else head + cleaned_rest
