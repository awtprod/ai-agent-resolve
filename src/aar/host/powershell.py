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
    "WINDOWS_CMDLINE_MAX",
    "encode_command",
    "normalize_newlines",
    "powershell_argv",
    "ps_quote",
    "strip_clixml",
]

# CreateProcess caps lpCommandLine at 32767 UTF-16 units including the terminating NUL, so the
# usable text is one shorter. The whole remote command line (prefix + encoded command) must fit.
WINDOWS_CMDLINE_MAX = 32767

# Fixed tokens of the remote command. Order and spelling are pinned by tests.
_PS_PREFIX: tuple[str, ...] = ("powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand")


def ps_quote(value: str) -> str:
    """Return ``value`` as a single-quoted PowerShell literal.

    In a single-quoted PowerShell string every character is literal except ``'``, which is
    written ``''``. So ``$()``, backticks, ``;`` and newlines need no other handling. A NUL is
    refused: it cannot be carried safely and has no place in a path or argument.
    """
    if "\x00" in value:
        raise ValueError("a PowerShell literal cannot contain NUL")
    return "'" + value.replace("'", "''") + "'"


def encode_command(script: str) -> str:
    """Base64 of ``script`` encoded as UTF-16LE, the argument for ``-EncodedCommand``."""
    return base64.b64encode(script.encode("utf-16-le")).decode("ascii")


def powershell_argv(script: str) -> list[str]:
    """Build the remote ``powershell ... -EncodedCommand <b64>`` argument list.

    Refuses (with :class:`ScriptTooLong`) when the whole command line would not fit the Windows
    command-line limit, so the failure happens here, not as a truncated command on the host.
    """
    encoded = encode_command(script)
    command_line = " ".join((*_PS_PREFIX, encoded))
    if len(command_line) > WINDOWS_CMDLINE_MAX - 1:  # leave room for the NUL terminator
        raise ScriptTooLong(len(command_line), WINDOWS_CMDLINE_MAX)
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
