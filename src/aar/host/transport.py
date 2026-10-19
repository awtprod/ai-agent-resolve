"""The edit-PC transport: PowerShell over ssh, and verified scp transfers.

:class:`EditHost` ties a :class:`~aar.host.config.EditHostConfig` to a
:class:`~aar.host.runner.Runner`. It runs PowerShell through ``ssh`` and moves files through
``scp`` in SFTP mode (the OpenSSH 9 default; ``-O`` is never passed). Every transfer checks the
size and the SHA-256 on both ends, writes to a temporary name beside the destination, and moves
it into place only when both match. A partial transfer is removed and a typed error raised; an
existing destination is never overwritten unless the caller asks.

An upload's move fails closed if the remote destination is a directory or if its immediate parent
is a reparse point (junction/symlink); a reparse point higher in the remote tree is a known
lexical-validation limit (see :mod:`aar.host.winpath` and DESIGN "Edit-PC transport").
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass, field

from aar.host.config import EditHostConfig
from aar.host.errors import (
    DestinationExists,
    HashMismatch,
    HostError,
    LocalIOError,
    PathRefused,
    RemoteCommandError,
    SizeMismatch,
    TransportTimeout,
)
from aar.host.powershell import normalize_newlines, powershell_argv, ps_quote, strip_clixml
from aar.host.runner import Runner, RunResult
from aar.host.winpath import (
    _Win,
    remote_spec,
    scp_remote_dest,
    scp_remote_source,
    validate_windows_path,
    windows_literal,
    with_suffix,
)

__all__ = [
    "EditHost",
    "TransferResult",
    "exists_script",
    "move_script",
    "remove_script",
    "stat_script",
]

_READ_CHUNK = 1024 * 1024
_STAT_TAG = "AARSTAT"
_EXISTS_TAG = "AAREXISTS"


def stat_script(literal: str) -> str:
    """PowerShell that prints ``AARSTAT <size> <sha256>`` for the file at ``literal``."""
    quoted = ps_quote(literal)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        f"$p = {quoted}\n"
        "$i = Get-Item -LiteralPath $p\n"
        "$h = Get-FileHash -LiteralPath $p -Algorithm SHA256\n"
        f"Write-Output ('{_STAT_TAG} ' + $i.Length + ' ' + $h.Hash)\n"
    )


def exists_script(literal: str) -> str:
    """PowerShell that prints ``AAREXISTS container|file|none`` for ``literal``.

    A container (directory) is reported distinctly so a transfer is never aimed at a directory.
    """
    quoted = ps_quote(literal)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        f"$p = {quoted}\n"
        "$k = if (Test-Path -LiteralPath $p -PathType Container) { 'container' }"
        " elseif (Test-Path -LiteralPath $p) { 'file' } else { 'none' }\n"
        f"Write-Output ('{_EXISTS_TAG} ' + $k)\n"
    )


def move_script(source: str, destination: str, overwrite: bool) -> str:
    """PowerShell that moves ``source`` onto ``destination`` (``-Force`` only if ``overwrite``).

    Fails closed before the move if the destination is a directory (so ``-Force`` cannot drop the
    file *into* a container), or if the destination's immediate parent is a reparse point
    (junction/symlink), which could redirect the write outside the allowed root. A reparse point
    higher in the tree is a known residual limit (DESIGN "Edit-PC transport").
    """
    force = " -Force" if overwrite else ""
    dst = ps_quote(destination)
    return (
        "$ErrorActionPreference = 'Stop'\n"
        f"$dst = {dst}\n"
        "if (Test-Path -LiteralPath $dst -PathType Container) "
        "{ throw 'aar: destination is a directory' }\n"
        "$parent = Split-Path -LiteralPath $dst -Parent\n"
        "$pitem = Get-Item -LiteralPath $parent -Force\n"
        "if ($pitem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) "
        "{ throw 'aar: destination parent is a reparse point' }\n"
        f"Move-Item -LiteralPath {ps_quote(source)} -Destination $dst{force}\n"
    )


def remove_script(literal: str) -> str:
    """Best-effort PowerShell that removes ``literal`` (used to clean up a partial transfer)."""
    return f"Remove-Item -LiteralPath {ps_quote(literal)} -Force -ErrorAction SilentlyContinue\n"


@dataclass(frozen=True)
class TransferResult:
    """What a completed operation moved or measured, plus the primary argv it ran."""

    op: str
    size: int
    sha256: str
    remote: str | None = None
    local: str | None = None
    argv: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "op": self.op,
            "size": self.size,
            "sha256": self.sha256,
            "remote": self.remote,
            "local": self.local,
            "argv": list(self.argv),
        }


def sha256_file(path: str) -> str:
    """The SHA-256 of a local file as upper-case hex (matching PowerShell's ``Get-FileHash``)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _within_local(child: str, roots: tuple[str, ...]) -> bool:
    return any(child == root or child.startswith(root.rstrip(os.sep) + os.sep) for root in roots)


def _validate_local(path: str, roots: tuple[str, ...], *, is_dest: bool) -> str:
    """Resolve a local path (following symlinks) and confirm it stays inside an allowed root."""
    if not isinstance(path, str) or not path:
        raise PathRefused(str(path), "empty")
    if "\x00" in path:
        raise PathRefused(path, "control-character")
    real = os.path.realpath(path)  # resolves symlinks and .. ; catches escapes out of the root
    if not _within_local(real, roots):
        raise PathRefused(path, "outside-allowed-roots")
    if is_dest:
        if not os.path.isdir(os.path.dirname(real)):
            raise PathRefused(path, "no-parent-directory")
        if os.path.isdir(real):
            raise PathRefused(path, "destination-is-a-directory")
    elif not os.path.isfile(real):
        raise PathRefused(path, "not-a-file")
    return real


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _parse_tagged(stdout: str, tag: str) -> list[str]:
    for line in stdout.splitlines():
        parts = line.strip().split()
        if parts and parts[0] == tag:
            return parts[1:]
    raise RemoteCommandError(0, f"remote output had no {tag} line")


class EditHost:
    """Runs PowerShell and verified transfers against one configured edit PC."""

    def __init__(
        self,
        config: EditHostConfig,
        runner: Runner,
        *,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self.config = config
        self.runner = runner
        self._token = token_factory or (lambda: secrets.token_hex(8))

    # -- argv construction (pinned by tests) --------------------------------------------------

    def _ssh_base(self) -> list[str]:
        cfg = self.config
        return [
            "ssh",
            "-F",
            cfg.ssh_config,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={cfg.connect_timeout}",
            cfg.alias,
        ]

    def _scp_base(self) -> list[str]:
        cfg = self.config
        return [
            "scp",
            "-F",
            cfg.ssh_config,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={cfg.connect_timeout}",
        ]

    def powershell_argv(self, script: str) -> list[str]:
        """The full ssh+powershell argv for ``script`` (also used to pin argv in tests).

        The command-line cap is the host's ``cmdline_max`` (default sized for a cmd.exe
        DefaultShell), so an over-long script is refused here rather than truncated on the host.
        """
        return self._ssh_base() + powershell_argv(script, cmdline_max=self.config.cmdline_max)

    # -- PowerShell ---------------------------------------------------------------------------

    def run_powershell(self, script: str, *, timeout: float | None = None) -> str:
        """Run ``script`` on the edit PC and return its cleaned stdout.

        Raises :class:`TransportTimeout` on timeout and :class:`RemoteCommandError` on a non-zero
        exit. Output newlines are normalised to LF and CLIXML progress is stripped from stderr.
        """
        result = self.runner.run(self.powershell_argv(script), timeout=self._timeout(timeout))
        return self._finish_powershell(result)

    def _finish_powershell(self, result: RunResult) -> str:
        if result.timed_out:
            raise TransportTimeout("the remote PowerShell command timed out")
        if result.returncode != 0:
            raise RemoteCommandError(result.returncode, self._clean(result.stderr))
        return normalize_newlines(result.stdout.decode("utf-8", "replace"))

    @staticmethod
    def _clean(stderr: bytes) -> str:
        return strip_clixml(normalize_newlines(stderr.decode("utf-8", "replace")))

    def _timeout(self, override: float | None) -> float:
        return override if override is not None else self.config.command_timeout

    def remote_stat(self, win: _Win) -> tuple[int, str]:
        """``(size, sha256)`` of a remote file, from ``Get-Item`` and ``Get-FileHash``."""
        fields = _parse_tagged(self.run_powershell(stat_script(windows_literal(win))), _STAT_TAG)
        if len(fields) != 2 or not fields[0].isdigit():
            raise RemoteCommandError(0, f"could not parse {_STAT_TAG} output: {fields!r}")
        return int(fields[0]), fields[1].upper()

    def remote_exists(self, win: _Win) -> bool:
        """Whether a remote path exists at all (a file or a directory)."""
        return self.remote_kind(win) in ("file", "container")

    def remote_kind(self, win: _Win) -> str:
        """``'file'``, ``'container'`` (directory) or ``'none'`` for a remote path."""
        fields = _parse_tagged(
            self.run_powershell(exists_script(windows_literal(win))), _EXISTS_TAG
        )
        return fields[0] if fields else "none"

    def _remote_remove_quiet(self, win: _Win) -> None:
        with contextlib.suppress(HostError):
            self.run_powershell(remove_script(windows_literal(win)))

    # -- transfers ----------------------------------------------------------------------------

    def pull(self, remote: str, local: str, *, overwrite: bool = False) -> TransferResult:
        """Download ``remote`` (an edit-PC path) to ``local`` (a hub path), verified."""
        win = validate_windows_path(remote, self.config.remote_roots_parsed)
        dest = _validate_local(local, self.config.local_roots, is_dest=True)
        if os.path.exists(dest) and not overwrite:
            raise DestinationExists(dest)
        remote_size, remote_hash = self.remote_stat(win)
        tmp = f"{dest}.aarpart-{self._token()}"
        spec = scp_remote_source(self.config.alias, win, self.config.path_form)
        argv = [*self._scp_base(), spec, tmp]
        result = self.runner.run(argv, timeout=self._timeout(None))
        try:
            self._check_scp(result, "download")
            size = os.path.getsize(tmp)
            digest = sha256_file(tmp)
            if size != remote_size:
                raise SizeMismatch(remote_size, size)
            if digest != remote_hash:
                raise HashMismatch(remote_hash, digest)
            self._place(tmp, dest, overwrite)
        except HostError:
            _unlink_quiet(tmp)
            raise
        except OSError as exc:  # a local filesystem failure while verifying or placing
            _unlink_quiet(tmp)
            raise LocalIOError(f"local I/O error completing the download: {exc}") from exc
        except BaseException:
            _unlink_quiet(tmp)
            raise
        return TransferResult(
            "pull",
            size,
            digest,
            remote=remote_spec(win, self.config.path_form),
            local=dest,
            argv=argv,
        )

    def push(self, local: str, remote: str, *, overwrite: bool = False) -> TransferResult:
        """Upload ``local`` (a hub path) to ``remote`` (an edit-PC path), verified."""
        win = validate_windows_path(remote, self.config.remote_roots_parsed)
        src = _validate_local(local, self.config.local_roots, is_dest=False)
        size = os.path.getsize(src)
        digest = sha256_file(src)
        if not overwrite and self.remote_exists(win):
            raise DestinationExists(remote)
        tmp_win = with_suffix(win, f".aarpart-{self._token()}")
        spec = scp_remote_dest(self.config.alias, tmp_win, self.config.path_form)
        argv = [*self._scp_base(), src, spec]
        result = self.runner.run(argv, timeout=self._timeout(None))
        try:
            self._check_scp(result, "upload")
            remote_size, remote_hash = self.remote_stat(tmp_win)
            if remote_size != size:
                raise SizeMismatch(size, remote_size)
            if remote_hash != digest:
                raise HashMismatch(digest, remote_hash)
            self.run_powershell(
                move_script(windows_literal(tmp_win), windows_literal(win), overwrite)
            )
        except BaseException:
            self._remote_remove_quiet(tmp_win)
            raise
        return TransferResult(
            "push",
            size,
            digest,
            remote=remote_spec(win, self.config.path_form),
            local=src,
            argv=argv,
        )

    def file_hash(self, remote: str) -> TransferResult:
        """Size and SHA-256 of a remote file (no transfer)."""
        win = validate_windows_path(remote, self.config.remote_roots_parsed)
        size, digest = self.remote_stat(win)
        argv = self.powershell_argv(stat_script(windows_literal(win)))
        return TransferResult(
            "hash", size, digest, remote=remote_spec(win, self.config.path_form), argv=argv
        )

    def _check_scp(self, result: RunResult, direction: str) -> None:
        if result.timed_out:
            raise TransportTimeout(f"the scp {direction} timed out")
        if result.returncode != 0:
            raise RemoteCommandError(result.returncode, self._clean(result.stderr))

    @staticmethod
    def _place(tmp: str, dest: str, overwrite: bool) -> None:
        if overwrite:
            os.replace(tmp, dest)
            return
        try:
            os.link(tmp, dest)  # atomic no-clobber: fails if dest exists
        except FileExistsError:
            raise DestinationExists(dest) from None
        except OSError:
            # No hard links on this filesystem (FAT/exFAT, many SMB/CIFS and some FUSE mounts, or a
            # cross-device dir): EPERM/EOPNOTSUPP/EXDEV. Fall back to an O_EXCL claim then replace.
            _place_no_hardlink(tmp, dest)
            return
        os.unlink(tmp)


def _place_no_hardlink(tmp: str, dest: str) -> None:
    """No-clobber placement without hard links: claim ``dest`` with ``O_EXCL``, then replace it.

    ``O_EXCL`` guarantees we never overwrite a pre-existing destination. The residual window is
    small: between the empty claim and the replace, ``dest`` exists as a zero-byte file, so a
    crash there can leave an empty file (the verified data is still in ``tmp``, which the caller
    removes). ``aar.drp.modify.write._place`` solves the same problem with a check-then-rename;
    the ``O_EXCL`` claim here is tighter against a concurrent creator.
    """
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise DestinationExists(dest) from None
    os.close(fd)
    try:
        os.replace(tmp, dest)
    except OSError:
        _unlink_quiet(dest)  # drop the empty claim so a failed placement leaves nothing behind
        raise
