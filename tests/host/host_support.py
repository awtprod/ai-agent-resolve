"""Shared helpers for the ``aar.host`` tests: a fake runner and small builders.

The fake runner never touches a real host. It records every argv, decodes the PowerShell
``-EncodedCommand`` so a test can inspect or classify the script, and answers from attributes the
test sets. For a ``pull`` it writes ``scp_payload`` to the local temp the real scp would have
created, so size/hash verification runs end to end.
"""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field

from aar.host.config import EditHostConfig
from aar.host.runner import RunResult

ALIAS = "edit-pc"


def sha256_upper(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def decode_encoded_command(argv: list[str]) -> str:
    """Decode the UTF-16LE base64 payload following ``-EncodedCommand`` in an ssh argv."""
    index = argv.index("-EncodedCommand")
    return base64.b64decode(argv[index + 1]).decode("utf-16-le")


def make_config(tmp_path, **overrides) -> EditHostConfig:
    """Build an EditHostConfig for tests (alias ``edit-pc``, local root = ``tmp_path``)."""
    params = {
        "alias": ALIAS,
        "ssh_config": str(tmp_path / "ssh.cfg"),
        "remote_roots": ("E:\\Show",),
        "local_roots": (str(tmp_path),),
        "path_form": "drive",
        "connect_timeout": 10,
    }
    params.update(overrides)
    return EditHostConfig(**params)


@dataclass
class FakeRunner:
    """A programmable, host-free :class:`~aar.host.runner.Runner`."""

    calls: list[list[str]] = field(default_factory=list)
    scripts: list[str] = field(default_factory=list)
    timeouts: list[float | None] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    # Remote file facts (what the stat script reports): (size, sha256-upper).
    stat_result: tuple[int, str] = (0, "")
    exists: bool = False
    # What a pull's scp writes locally.
    scp_payload: bytes = b""
    # Failure switches.
    scp_timeout: bool = False
    scp_returncode: int = 0
    scp_stderr: bytes = b""
    stat_timeout: bool = False
    stat_returncode: int = 0
    stat_stderr: bytes = b""
    move_returncode: int = 0
    move_stderr: bytes = b""
    ps_timeout: bool = False
    ps_returncode: int = 0
    ps_stdout: bytes = b""
    ps_stderr: bytes = b""

    def run(
        self, argv: list[str], *, stdin: bytes | None = None, timeout: float | None = None
    ) -> RunResult:
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        if argv[0] == "ssh":
            script = decode_encoded_command(argv)
            self.scripts.append(script)
            return self._ssh(script)
        if argv[0] == "scp":
            return self._scp(argv)
        raise AssertionError(f"unexpected argv: {argv!r}")

    def _ssh(self, script: str) -> RunResult:
        if "Get-FileHash" in script:
            if self.stat_timeout:
                return RunResult(-1, b"", b"", timed_out=True)
            if self.stat_returncode:
                return RunResult(self.stat_returncode, b"", self.stat_stderr)
            size, digest = self.stat_result
            out = f"AARSTAT {size} {digest}\r\n".encode()
            return RunResult(0, out, self.stat_stderr)
        if "Test-Path" in script:
            answer = "yes" if self.exists else "no"
            return RunResult(0, f"AAREXISTS {answer}\r\n".encode(), b"")
        if "Move-Item" in script:
            return RunResult(self.move_returncode, b"", self.move_stderr)
        if "Remove-Item" in script:
            self.removed.append(script)
            return RunResult(0, b"", b"")
        if self.ps_timeout:
            return RunResult(-1, b"", b"", timed_out=True)
        return RunResult(self.ps_returncode, self.ps_stdout, self.ps_stderr)

    @staticmethod
    def _is_remote(token: str) -> bool:
        return ":" in token and not token.startswith("/")

    def _scp(self, argv: list[str]) -> RunResult:
        if self.scp_timeout:
            return RunResult(-1, b"", b"", timed_out=True)
        if self.scp_returncode:
            return RunResult(self.scp_returncode, b"", self.scp_stderr)
        source, dest = argv[-2], argv[-1]
        if self._is_remote(source) and not self._is_remote(dest):  # a pull: write the local temp
            with open(dest, "wb") as handle:
                handle.write(self.scp_payload)
        return RunResult(0, b"", b"")
