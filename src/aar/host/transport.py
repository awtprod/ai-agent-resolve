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

import hashlib
import os
import re
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
_SIZE_RE = re.compile(r"[0-9]+")  # ASCII digits only (str.isdigit accepts superscripts etc.)
_SHA256_RE = re.compile(r"[0-9A-Fa-f]{64}")


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
        "$parent = [System.IO.Path]::GetDirectoryName($dst)\n"
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


def _unlink_quiet(path: str, *, note_to: BaseException | None = None) -> None:
    """Best-effort unlink. A FileNotFoundError is normal; any other OSError is swallowed so it
    never replaces the original error, and is recorded as a note on it (F2)."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as exc:
        if note_to is not None:
            note_to.add_note(f"cleanup: could not remove {path}: {exc}")


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
        # Always -s: force SFTP mode so the glob-escaping and no-remote-shell assumptions never
        # depend on the client's default. A client older than OpenSSH 8.7 rejects -s and fails
        # closed with a typed transport error; legacy SCP (which parses the remote path through the
        # remote shell) is unsupported because it is unsafe. The hub is OpenSSH 9.6p1.
        return [
            "scp",
            "-s",
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
        exit. A failure to even spawn ssh (an ``OSError`` from the runner, e.g. ssh not on PATH or
        ``BlockingIOError`` on fork) is mapped once to :class:`LocalIOError` so the stat/exists
        pre-checks never escape raw (F2). Output newlines are normalised to LF and CLIXML progress
        is stripped from stderr.
        """
        try:
            result = self.runner.run(self.powershell_argv(script), timeout=self._timeout(timeout))
        except OSError as exc:
            raise LocalIOError(f"could not run ssh: {exc}") from exc
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
        """``(size, sha256)`` of a remote file, from ``Get-Item`` and ``Get-FileHash``.

        The size must be ASCII digits and the hash exactly 64 hex characters; anything else is a
        :class:`RemoteCommandError` (``str.isdigit`` would accept e.g. superscript digits).
        """
        fields = _parse_tagged(self.run_powershell(stat_script(windows_literal(win))), _STAT_TAG)
        if (
            len(fields) != 2
            or not _SIZE_RE.fullmatch(fields[0])
            or not _SHA256_RE.fullmatch(fields[1])
        ):
            raise RemoteCommandError(0, f"could not parse {_STAT_TAG} output: {fields!r}")
        return int(fields[0]), fields[1].upper()

    def remote_exists(self, win: _Win) -> bool:
        """Whether a remote path exists at all (a file or a directory)."""
        return self.remote_kind(win) in ("file", "container")

    def remote_kind(self, win: _Win) -> str:
        """``'file'``, ``'container'`` (directory) or ``'none'`` for a remote path.

        Any other output (a bare tag, an unknown word, extra fields) is a
        :class:`RemoteCommandError`, so the push pre-check fails closed rather than open.
        """
        fields = _parse_tagged(
            self.run_powershell(exists_script(windows_literal(win))), _EXISTS_TAG
        )
        if len(fields) != 1 or fields[0] not in ("file", "container", "none"):
            raise RemoteCommandError(0, f"could not parse {_EXISTS_TAG} output: {fields!r}")
        return fields[0]

    def _remote_remove_quiet(self, win: _Win, *, note_to: BaseException | None = None) -> None:
        """Best-effort remote cleanup of a partial upload. Suppresses ``(HostError, OSError)`` so a
        cleanup failure never replaces the original error, and records it as a note on it (F2)."""
        try:
            self.run_powershell(remove_script(windows_literal(win)))
        except (HostError, OSError) as exc:
            if note_to is not None:
                note_to.add_note(f"cleanup: remote Remove-Item failed: {exc}")

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
        try:
            # inside the try: a KeyboardInterrupt or a runner OSError during scp must still remove
            # the partial scp wrote in place (SFTP mode writes directly to the temp).
            result = self.runner.run(argv, timeout=self._timeout(None))
            self._check_scp(result, "download")
            size = os.path.getsize(tmp)
            digest = sha256_file(tmp)
            if size != remote_size:
                raise SizeMismatch(remote_size, size)
            if digest != remote_hash:
                raise HashMismatch(remote_hash, digest)
            self._place(tmp, dest, overwrite)
        except OSError as exc:  # a runner spawn or local filesystem failure (not a HostError)
            typed = LocalIOError(f"local I/O error completing the download: {exc}")
            _unlink_quiet(tmp, note_to=typed)
            raise typed from exc
        except BaseException as exc:  # HostError, KeyboardInterrupt, ... : the original wins
            _unlink_quiet(tmp, note_to=exc)
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
        try:
            # inside the try: an interrupt or runner error during scp must still attempt the
            # remote Remove-Item for the partial upload.
            result = self.runner.run(argv, timeout=self._timeout(None))
            self._check_scp(result, "upload")
            remote_size, remote_hash = self.remote_stat(tmp_win)
            if remote_size != size:
                raise SizeMismatch(size, remote_size)
            if remote_hash != digest:
                raise HashMismatch(digest, remote_hash)
            self.run_powershell(
                move_script(windows_literal(tmp_win), windows_literal(win), overwrite)
            )
        except OSError as exc:  # a runner spawn failure during the upload path (not a HostError)
            typed = LocalIOError(f"I/O error during the upload: {exc}")
            self._remote_remove_quiet(tmp_win, note_to=typed)
            raise typed from exc
        except BaseException as exc:  # HostError, KeyboardInterrupt, ... : the original wins
            self._remote_remove_quiet(tmp_win, note_to=exc)
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

    ``O_EXCL`` guarantees we never overwrite a pre-existing destination, and it does not follow a
    symlink planted at the name. On any failure or interrupt before the replace lands, ``dest`` is
    removed only if it is still our zero-byte claim, matched by device, inode, ``ctime_ns`` and
    size 0. A file that is not provably ours (written into, a symlink, or a different inode) is
    left in place and reported via the raised error (F4), never deleted. If the file cannot be
    proven ours, it is kept.
    """
    try:
        fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise DestinationExists(dest) from None
    try:
        claim = os.fstat(fd)
    except BaseException as exc:
        # The fresh claim cannot be identified, so it cannot be proven ours: leave it and report it.
        os.close(fd)
        raise _report_kept_claim(exc, dest) from None
    os.close(fd)
    try:
        os.replace(tmp, dest)
    except BaseException as exc:
        if not _unlink_if_claim(dest, claim):
            raise _report_kept_claim(exc, dest) from None
        raise


def _report_kept_claim(exc: BaseException, dest: str) -> BaseException:
    """Record that ``dest`` was left in place. Returns the exception to re-raise: an ``OSError`` is
    wrapped in a :class:`LocalIOError` whose message names the path; any other (e.g.
    ``KeyboardInterrupt``) keeps its type with the fact added as a note (F4)."""
    note = f"left in place (not provably ours): {dest}"
    if isinstance(exc, OSError):
        return LocalIOError(f"local I/O error placing the download ({exc}); {note}")
    exc.add_note(note)
    return exc


def _unlink_if_claim(dest: str, claim: os.stat_result) -> bool:
    """Remove ``dest`` only if it is still our zero-byte O_EXCL claim; return whether it was gone.

    Identity is device, inode and ``ctime_ns``, plus size 0. ``ctime_ns`` narrows inode reuse but
    does not close it within one timestamp tick (ext4 timestamps are coarse: a reused inode often
    carries the same ``ctime_ns``), so another writer's *empty* file created in the same tick is
    indistinguishable and would be removed; a non-empty file never is. Our own
    ``fstat -> close -> failed replace`` never changes the identity, so a genuine claim is still
    recognised.
    """
    try:
        now = os.lstat(dest)  # lstat: never follow a symlink that replaced the claim
    except OSError:
        return False
    same = (now.st_dev, now.st_ino, now.st_ctime_ns) == (
        claim.st_dev,
        claim.st_ino,
        claim.st_ctime_ns,
    )
    if same and now.st_size == 0:
        try:
            os.unlink(dest)
            return True
        except OSError:
            return False
    return False
