"""``edit_file``: plan -> verified new file, written atomically next to an ``unverified`` label.

Order of work, and what stops it (every stop writes nothing):

1. The input must read without warnings or errors: the post-write gate requires both files to
   read cleanly, so an input that does not (for example one with a ``blob-partial`` keyed dict)
   can never pass it. Such inputs are refused up front (``input-has-warnings``) rather than by
   weakening the gate.
2. ``input_sha256`` and ``max_project_version`` from the plan, and the output path: never the
   input itself, never an existing file unless ``force``.
3. The edits are resolved and spliced (:mod:`aar.drp.modify.engine`) and the zip is rewritten
   with every untouched byte copied (:mod:`aar.drp.modify.zipraw`).
4. The result is written to a temporary file in the output directory and verified
   (:func:`aar.drp.modify.verify.verify_write`): gate, exact change list, version stamps.
5. Only then are the sidecar ``<out>.aar.json`` (status ``unverified``) and the file moved into
   place, the sidecar first, so a written file never lacks its label. Without ``--force`` each
   move hard-links onto the name, or on a filesystem with no hard links (FAT/exFAT) claims the
   name with ``O_CREAT|O_EXCL`` and ``os.replace``s onto its own placeholder. That stops an
   exclusive or check-then-create writer; a plain non-exclusive writer racing in the window is
   still replaced (:func:`_place_no_hardlink` documents the window).

A run stopped hard between those two moves leaves a label without its file; the next run for
the same ``--out`` refuses with ``orphan-sidecar`` and lists what to remove. Temporary files a
stopped run left behind are listed in ``leftovers``. Nothing is ever deleted that this run did
not create, nor anything it cannot prove it created: what this run leaves in place for that
reason (an empty placeholder after a failed replace, a snapshot of a label that is not aar's) is
named in ``left_behind`` (:class:`LeftBehind`), on the outcome or on the raised
:class:`PlacementError`.

Nothing here talks to Resolve. Outputs stay ``unverified`` until the sandbox round trip (D3).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from aar import __version__
from aar.drp.errors import ContainerError
from aar.drp.model import DrpFile
from aar.drp.modify import engine, zipraw
from aar.drp.modify.engine import EditRefused, Refusal
from aar.drp.modify.plan import DrpEditPlan
from aar.drp.modify.verify import verify_write
from aar.drp.reader import read_drp

__all__ = [
    "DEFAULT_MAX_INPUT_BYTES",
    "LABEL",
    "SIDECAR_SUFFIX",
    "SNAPSHOT_SUFFIX",
    "EditOutcome",
    "LeftBehind",
    "PlacementError",
    "edit_file",
    "leftovers",
]

LABEL = "unverified"
SIDECAR_SUFFIX = ".aar.json"
UNVERIFIED_WHY = (
    "checked offline only (S2 gate plus the exact intended change list); not yet imported into "
    "a Resolve sandbox and read back, which waits on approval D3"
)
# The largest corpus file is 11.6 MB; the writer holds the input and the output in memory.
DEFAULT_MAX_INPUT_BYTES = 1024 * 1024 * 1024
# A label we write embeds the plan (<= MAX_PLAN_BYTES, 1 MiB) plus the change list, expected set
# and report, each bounded by the edit count. 16 MiB is ample headroom for a real label and, read
# on every --force, refuses a multi-hundred-MB read bomb (a 512 MiB cap cost ~13 GiB of RSS).
_MAX_SIDECAR_BYTES = 16 * 1024 * 1024
# A label snapshot that is kept (it does not bear our stamp) is renamed to end in this, so the
# temporary-file pattern of leftovers() can never match it (see _keep_snapshot).
SNAPSHOT_SUFFIX = ".label-snapshot.json"
# A path recorded on Windows starts with a drive and a separator (``E:\``, ``E:/``) or a
# backslash (``\\server\share``, ``\dir``). A path recorded on POSIX starts with ``/`` or is
# relative (see _recorded_name).
_WINDOWS_ROOT = re.compile(r"^(?:[A-Za-z]:[\\/]|\\)")


@dataclass(frozen=True, slots=True)
class LeftBehind:
    """A file this run left on disk on purpose, because deleting it might destroy someone else's.

    ``kind`` is one of:

    * ``placeholder``: an empty regular file at the output (or label) name after ``os.replace``
      failed on a filesystem without hard links. It is usually aar's own claim on the name, but
      another program may have truncated it or put its own empty file there, and aar cannot tell,
      so it is not deleted. Nothing was placed (an output whose placement failed has its new
      label removed again): delete the file once nothing else is writing to it, then run again.
    * ``label-snapshot``: a hard link (else a copy) of the label ``--force`` found at
      ``<out>.aar.json``, kept because it does not bear aar's stamp. Check it against that label,
      then delete it. It is named ``.<out>.aar.json.<8 chars>.label-snapshot.json`` (see
      :data:`SNAPSHOT_SUFFIX`), which :func:`leftovers` never lists, so a later run never asks for
      it to be removed as a stale temporary file.
    * ``own-label``: aar's own label at ``<out>.aar.json``, describing an output this run did not
      place: placing the output failed and removing (or restoring) the label afterwards failed
      too. The next run for that ``--out`` reports it as ``orphan-sidecar`` (``out-exists`` if a
      file is at the output name); delete it, or pass ``--force``.
    """

    path: str
    kind: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "kind": self.kind, "message": self.message}


class PlacementError(OSError):
    """Placing a file failed, and the failed placement left ``left_behind`` on disk.

    Raised instead of the original ``OSError`` (chained as ``__cause__``, same ``errno``,
    ``strerror`` and ``filename``) only when something was left behind.
    """

    def __init__(self, cause: OSError, left_behind: list[LeftBehind]) -> None:
        super().__init__(*cause.args)
        self.filename = cause.filename
        self.filename2 = cause.filename2
        self.left_behind = left_behind


@dataclass(slots=True)
class EditOutcome:
    """What ``aar drp edit`` prints. ``status`` is ``written``, ``dry-run`` or ``refused``."""

    status: str
    input: dict[str, Any]
    plan: dict[str, Any]
    output: dict[str, Any] | None = None
    changes: list[dict[str, Any]] = field(default_factory=list)
    expected: list[dict[str, Any]] = field(default_factory=list)
    report: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[Refusal] = field(default_factory=list)
    gate: dict[str, Any] | None = None
    leftovers: list[str] = field(default_factory=list)
    left_behind: list[LeftBehind] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.status == "refused"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "label": LABEL,
            "label_reason": UNVERIFIED_WHY,
            "tool": f"aar {__version__}",
            "input": self.input,
            "output": self.output,
            "plan": self.plan,
            "gate": self.gate,
            "changes": self.changes,
            "expected": self.expected,
            "report": self.report,
            "refusals": [r.to_dict() for r in self.refusals],
            "leftovers": self.leftovers,
            "left_behind": [x.to_dict() for x in self.left_behind],
        }


def _input_info(path: Path, drp: DrpFile | None) -> dict[str, Any]:
    v = drp.versions if drp is not None else None
    return {
        "path": str(path),
        "sha256": drp.sha256 if drp is not None else None,
        "kind": drp.kind if drp is not None else None,
        "app_version": v.app_version if v else None,
        "project_version": v.project_version if v else None,
    }


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (b.exists() and os.path.samefile(a, b))
    except OSError:
        return False


def _precheck(
    drp: DrpFile, path: Path, plan: DrpEditPlan, out: Path | None, force: bool, dry_run: bool
) -> list[Refusal]:
    refusals: list[Refusal] = []
    noisy = sorted({i.code for i in drp.issues if i.severity in ("warning", "error")})
    if noisy:
        refusals.append(
            Refusal(
                "input-has-warnings",
                "the input reads with warnings or errors, so the post-write gate can never pass "
                "for it and no edit can be verified: " + ", ".join(noisy),
                details={"codes": noisy},
            )
        )
    if drp.kind not in ("drp", "drt") or not drp.is_zip:
        refusals.append(
            Refusal("unsupported-kind", f"only .drp and .drt archives are edited, not {drp.kind}")
        )
    if plan.input_sha256 is not None and plan.input_sha256 != drp.sha256:
        refusals.append(
            Refusal(
                "input-sha-mismatch",
                "the input is not the file the plan was made for",
                details={"plan": plan.input_sha256, "input": drp.sha256},
            )
        )
    prj = drp.versions.project_version if drp.versions else None
    if plan.max_project_version is not None and (prj is None or prj > plan.max_project_version):
        refusals.append(
            Refusal(
                "version-gate",
                f"the input is stamped DbPrjVer {prj}, newer than the target host's "
                f"{plan.max_project_version}; the writer never restamps a file",
                details={"input": prj, "max": plan.max_project_version},
            )
        )
    if out is None:
        if not dry_run:
            refusals.append(Refusal("no-output", "--out is required unless --dry-run is given"))
        return refusals
    if out.suffix.lower() != f".{drp.kind}":
        refusals.append(
            Refusal(
                "out-suffix",
                f"the output must end in .{drp.kind} like the input; Resolve and the reader "
                "decide what a file is from its suffix",
            )
        )
    sidecar = Path(str(out) + SIDECAR_SUFFIX)
    if _same_file(path, out) or _same_file(path, sidecar):
        refusals.append(Refusal("out-is-input", "the output must be a new file, not the input"))
    elif not force and _orphan_sidecar(out, sidecar):
        remove = [str(sidecar), *(str(p) for p in leftovers(out))]
        refusals.append(
            Refusal(
                "orphan-sidecar",
                f"an earlier aar drp edit was stopped after it wrote the label {sidecar} and "
                f"before it moved {out.name} into place, so the label describes a file that does "
                "not exist; remove the files listed in details.remove (or pass --force) and run "
                "again",
                details={"remove": remove},
            )
        )
    elif not force and (out.exists() or sidecar.exists()):
        existing = [str(p) for p in (out, sidecar) if p.exists()]
        refusals.append(
            Refusal(
                "out-exists",
                "the output exists; pass --force to replace it",
                details={"paths": existing},
            )
        )
    elif force and os.path.lexists(sidecar) and not _is_our_sidecar(sidecar, out):
        # lexists, not is_file: any entry at the label name that is not our own stamp (a foreign
        # regular file, a directory, a FIFO, a dangling symlink, a symlink to a directory) is
        # refused, never replaced by --force.
        refusals.append(
            Refusal(
                "foreign-sidecar",
                f"a label {sidecar} that aar drp edit did not write is already here, so --force "
                "leaves it and the output untouched; move or delete it, or choose another --out",
                details={"sidecar": str(sidecar)},
            )
        )
    if not out.parent.is_dir():
        refusals.append(
            Refusal("no-output-dir", f"the output directory {out.parent} does not exist")
        )
    return refusals


def _temp_pattern(name: str, suffix: str) -> re.Pattern[str]:
    # _temp names files "." + final name + 8 random mkstemp characters + ".tmp" + suffix.
    return re.compile(rf"\.{re.escape(name)}\.[a-z0-9_]{{8}}\.tmp{re.escape(suffix)}")


def leftovers(out: Path) -> list[Path]:
    """Temporary files an interrupted run for ``out`` left in its directory (ours by name).

    A label snapshot a run kept on purpose (``left_behind`` kind ``label-snapshot``) ends in
    :data:`SNAPSHOT_SUFFIX` and is never listed. A snapshot an older aar kept still has a
    temporary-file name, and so does one a stopped run never settled: both are listed.
    """
    sidecar = out.name + SIDECAR_SUFFIX
    patterns = (_temp_pattern(out.name, out.suffix), _temp_pattern(sidecar, ".json"))
    try:
        with os.scandir(out.parent) as entries:
            names = [e.name for e in entries if e.name.startswith("." + out.name)]
    except OSError:
        return []
    return sorted(out.parent / n for n in names if any(p.fullmatch(n) for p in patterns))


def _read_label(path: Path) -> bytes | None:
    """The bytes of the regular file at ``path``, or ``None`` if it is not one or is over the cap.

    One open; the size check is on that open file (``fstat``) and the read itself stops at the
    cap plus one byte, so a label that grows after the check, or a symlink retargeted in between,
    is refused, never read whole. ``O_NONBLOCK`` keeps a FIFO from blocking the open.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > _MAX_SIDECAR_BYTES:
            return None
        with open(fd, "rb", closefd=False) as fh:
            data = fh.read(_MAX_SIDECAR_BYTES + 1)
    except OSError:
        return None
    finally:
        os.close(fd)
    return data if len(data) <= _MAX_SIDECAR_BYTES else None


def _recorded_name(recorded: str) -> str:
    """The file name in a path a label records, read with the rules of the OS that wrote it.

    A label written on Windows records ``E:\\dir\\out.drp`` (or a UNC path); on POSIX
    ``Path(...).name`` would take that whole string as the name. A path that starts with ``/``
    was written on POSIX (Windows writes its separators as backslashes), so it is read with POSIX
    rules even on Windows, and a POSIX name that contains a backslash keeps it. A relative path
    is read with this host's rules: ``sub\\out.drp`` is a valid POSIX name, so a label recorded
    with a relative Windows path is still refused on POSIX. (A relative POSIX name that itself
    starts with a backslash or with ``X:\\`` is read the Windows way, so its own label is refused
    too.) Both err on the safe side: a refused label is left alone, never replaced.
    """
    if recorded.startswith("/"):
        return PurePosixPath(recorded).name
    if _WINDOWS_ROOT.match(recorded):
        return PureWindowsPath(recorded).name
    return Path(recorded).name


def _is_our_sidecar(sidecar: Path, out: Path, read: Path | None = None) -> bool:
    """``sidecar`` is a label this tool wrote for this exact ``out`` (see DESIGN "Modify").

    Recognised by the stamp every ``aar drp edit`` sidecar carries: a ``tool`` field of
    ``aar <version>``, a ``label`` and ``status`` of ``unverified`` (the only states a sidecar on
    disk is ever written in), and an ``output`` object whose ``path`` and ``sidecar`` name this
    output file and this label (:func:`_recorded_name`, so a label aar wrote on Windows matches
    on POSIX). ``--force`` replaces only a sidecar that bears this stamp; any other file at the
    path is left untouched. ``read`` is the file to read the stamp from, when that is a snapshot
    of the label rather than the label itself.
    """
    data = _read_label(read if read is not None else sidecar)
    if data is None:
        return False
    try:
        record = json.loads(data)
    except (ValueError, RecursionError):
        return False
    if not isinstance(record, dict):
        return False
    output = record.get("output")
    return (
        isinstance(output, dict)
        and str(record.get("tool", "")).startswith("aar ")
        and record.get("label") == LABEL
        and record.get("status") == LABEL
        and _recorded_name(str(output.get("path"))) == out.name
        and _recorded_name(str(output.get("sidecar"))) == sidecar.name
    )


def _orphan_sidecar(out: Path, sidecar: Path) -> bool:
    """``sidecar`` is a label this tool wrote for ``out``, and ``out`` itself does not exist."""
    if out.exists() or out.is_symlink():
        return False
    return _is_our_sidecar(sidecar, out)


def _read_input(path: Path, max_bytes: int) -> bytes:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise ContainerError(f"{path.name} is {size} bytes, over the {max_bytes}-byte limit")
        with path.open("rb") as fh:
            data = fh.read(max_bytes + 1)
    except OSError as exc:
        raise ContainerError(f"cannot read {path}: {exc}") from exc
    if len(data) > max_bytes:
        raise ContainerError(f"{path.name} grew past the {max_bytes}-byte limit while read")
    return data


def _temp(directory: Path, name: str, data: bytes, mode: int | None = None) -> Path:
    """Write ``data`` to a new temporary file next to its final ``name``.

    The temporary file keeps the final suffix, so it reads exactly as the final file will.
    ``mode`` sets its permission bits (mkstemp creates files 0600).
    """
    fd, tmp = tempfile.mkstemp(prefix=f".{name}.", suffix=f".tmp{Path(name).suffix}", dir=directory)
    try:
        if mode is not None:
            os.chmod(tmp, mode)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return Path(tmp)


def _backup(directory: Path, sidecar: Path) -> Path | None:
    """A hard link (else a copy) of an existing label, named like our temporary files."""
    if not sidecar.is_file():
        return None
    fd, name = tempfile.mkstemp(prefix=f".{sidecar.name}.", suffix=".tmp.json", dir=directory)
    os.close(fd)
    backup = Path(name)
    try:
        backup.unlink()
        try:
            os.link(sidecar, backup)
        except OSError:
            shutil.copy2(sidecar, backup)
    except BaseException:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _place(tmp: Path, dest: Path, force: bool) -> None:
    """Move ``tmp`` to ``dest`` atomically; without ``force`` never replace an existing file."""
    if force:
        os.replace(tmp, dest)
        return
    try:
        os.link(tmp, dest)  # fails if dest exists: no window for a race
    except FileExistsError:
        raise
    except OSError:
        _place_no_hardlink(tmp, dest)  # no hard links on this filesystem (FAT/exFAT)
        return
    tmp.unlink()


def _place_no_hardlink(tmp: Path, dest: Path) -> None:
    """Place ``tmp`` at ``dest`` on a filesystem without hard links (FAT/exFAT).

    ``os.link`` is unavailable, so claim the name with ``O_CREAT|O_EXCL`` -- which fails with
    ``FileExistsError`` if the name is already taken -- then ``os.replace`` our verified temporary
    onto the placeholder we just made. ``os.replace`` (not ``os.rename``) is required because on
    Windows ``os.rename`` fails when the destination exists, which here it always does.

    The window between the claim and the replace is NOT closed for every writer, only narrowed:

    * An exclusive creator (``O_CREAT|O_EXCL``) or a check-then-create writer loses: the name is
      already held, so its create fails. This is the common case (another ``aar`` run, or any
      tool that refuses to clobber) and is what the pre-S8c check-then-rename window let through.
    * A plain writer that does not create exclusively -- ``open(dest, "wb")``, ``cp``, Windows
      ``CREATE_ALWAYS`` -- opens and truncates our placeholder (it exists) and writes into it; our
      ``os.replace`` then discards those bytes. Its data is silently replaced. For this class of
      writer the window runs from the exclusive create through its close to the replace (the
      temporary's identity is read before the claim, so nothing else sits in between).

    On a failed replace the cleanup deletes ``dest`` only when it is our finished output: the
    temporary we were placing, matched by device+inode on ``lstat`` (an interrupt that lands after
    a good replace). It deletes nothing else, not even an *empty* regular file. That is usually
    our untouched placeholder, but it may be a file another writer created after deleting the
    placeholder, or the placeholder a plain writer has truncated and not yet written to, and no
    identity aar can record tells these apart: the inode number is reused, and a truncating open
    of an empty file can leave its size and timestamps unchanged (a coarse kernel clock, or
    FAT/exFAT's 2 s/10 ms resolution). So the empty file is left in place and named in the
    raised :class:`PlacementError`'s ``left_behind`` (kind ``placeholder``). Holding the
    placeholder's fd open across the replace would settle it, but on Windows that fails every
    exFAT placement. A crash in the window leaves our own empty placeholder or our finished
    output, never a file that predated the claim.
    """
    try:
        tmp_id = _dev_ino(os.stat(tmp))  # after a good replace, dest carries this inode
    except OSError:
        tmp_id = None
    fd = os.open(dest, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)  # FileExistsError if taken
    os.close(fd)
    try:
        os.replace(tmp, dest)  # cross-platform atomic replace of the placeholder we just created
    except OSError as exc:
        left = _unlink_if_ours(dest, tmp_id)
        if left:
            raise PlacementError(exc, left) from exc
        raise
    except BaseException:
        _unlink_if_ours(dest, tmp_id)
        raise


def _dev_ino(st: os.stat_result) -> tuple[int, int]:
    return (st.st_dev, st.st_ino)


def _unlink_if_ours(dest: Path, tmp_id: tuple[int, int] | None) -> list[LeftBehind]:
    """Remove ``dest`` only if it is our finished output (``tmp_id``); report an empty file.

    ``lstat``, so a symlink an adversary put at ``dest`` (even one pointing at our temporary) is
    never followed or deleted; only a regular file is ever ours.
    """
    try:
        now = os.lstat(dest)
    except OSError:
        return []
    if not stat.S_ISREG(now.st_mode):
        return []
    if _dev_ino(now) == tmp_id:
        with contextlib.suppress(OSError):
            os.unlink(dest)
        return []
    if now.st_size != 0:
        return []  # someone else's bytes
    return [
        LeftBehind(
            str(dest),
            "placeholder",
            f"os.replace failed after aar claimed {dest.name} with an empty placeholder; the "
            "empty file there now is probably that placeholder, but another program may have "
            "truncated it or created its own, so aar did not delete it. Nothing was placed (not "
            "the output, not its label): delete the empty file once nothing else writes to it, "
            "then run again",
        )
    ]


def _settle_backup(backup: Path, sidecar: Path, out: Path, left: list[LeftBehind]) -> None:
    """Delete the ``--force`` snapshot of the label, unless it no longer bears our stamp.

    Runs when the edit ends, on every path. A snapshot moved back over the label (a failed
    placement) is gone and needs nothing. A hard-linked snapshot shares the old label's file, so a
    label written in place after the check shows up here: such a snapshot, or one that was never
    ours (a refused run), is kept and named in ``left``, never deleted.
    """
    if not os.path.lexists(backup):
        return
    if _is_our_sidecar(sidecar, out, read=backup):
        backup.unlink(missing_ok=True)
        return
    kept = _keep_snapshot(backup, sidecar)
    left.append(
        LeftBehind(
            str(kept),
            "label-snapshot",
            f"a hard link (or copy) of the label that was at {sidecar.name} when --force checked "
            "it; that label does not bear aar's stamp, so this snapshot of it is kept: compare it "
            "with the label, then delete it",
        )
    )


def _keep_snapshot(backup: Path, sidecar: Path) -> Path:
    """Move a kept snapshot off its temporary-file name, so leftovers() never lists it.

    The new name, ``.<label>.<8 chars>.label-snapshot.json``, is claimed with ``mkstemp``
    (exclusive, never an existing file) and the snapshot ``os.replace``s that empty claim. If the
    move fails the snapshot stays under its temporary name, where it is still kept and reported,
    and a later run lists it among the leftovers.
    """
    fd, name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=SNAPSHOT_SUFFIX, dir=backup.parent
    )
    os.close(fd)
    try:
        os.replace(backup, name)
    except OSError:
        with contextlib.suppress(OSError):
            os.unlink(name)  # our own empty claim on a fresh random name
        return backup
    return Path(name)


def _roll_back_label(
    exc: BaseException, backup: Path | None, sidecar: Path, left: list[LeftBehind]
) -> BaseException:
    """Undo the label after placing the output failed; return the exception to raise.

    The label goes back to its snapshot (``--force``) or is removed. If that cleanup fails too,
    the placement error ``exc`` still wins: the cleanup failure is added to it as a note, and the
    label left over (aar's own) is reported as ``own-label`` in ``left`` and on the raised error
    (a :class:`PlacementError` with the same errno). A ``FileExistsError`` is returned as it is,
    because ``edit_file`` turns it into the ``out-exists`` refusal, whose outcome carries ``left``.
    """
    try:
        if backup is not None:
            os.replace(backup, sidecar)
        else:
            sidecar.unlink(missing_ok=True)
    except OSError as cleanup:
        own = LeftBehind(
            str(sidecar),
            "own-label",
            "aar's own label for an output this run did not place: placing the output failed, "
            f"and removing or restoring this label afterwards failed too ({cleanup}). The "
            "next run for this --out reports it as orphan-sidecar (out-exists if a file is at "
            "the output name): delete it, or pass --force",
        )
        left.append(own)
        note = (
            f"aar could not roll back its label {sidecar} after this error: {cleanup!r}; the label "
            "is left over (left_behind kind own-label)"
        )
        if isinstance(exc, PlacementError):
            exc.left_behind.append(own)
        elif isinstance(exc, OSError) and not isinstance(exc, FileExistsError):
            wrapped = PlacementError(exc, [own])  # edit_file raises it "from exc"
            wrapped.add_note(note)
            return wrapped
        exc.add_note(note)
    return exc


def edit_file(
    input_path: str | Path,
    plan: DrpEditPlan,
    out: str | Path | None = None,
    *,
    dry_run: bool = False,
    force: bool = False,
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES,
) -> EditOutcome:
    """Apply ``plan`` to ``input_path`` and write ``out`` (see the module docstring).

    Raises :class:`~aar.drp.errors.DrpError` when the input cannot be read and ``OSError`` when
    the output cannot be written; every refusal is returned in the outcome instead.
    """
    path = Path(input_path)
    out_path = Path(out) if out is not None else None
    plan_dict = plan.model_dump(mode="json", exclude_none=True)  # parse_plan reads it back
    drp = read_drp(path)
    outcome = EditOutcome("refused", _input_info(path, drp), plan_dict)
    if out_path is not None:
        outcome.leftovers = [str(p) for p in leftovers(out_path)]
    outcome.refusals = _precheck(drp, path, plan, out_path, force, dry_run)
    if outcome.refusals:
        return outcome
    original = _read_input(path, max_input_bytes)
    # Outputs get the input's permission bits (execute and special bits dropped).
    mode = stat.S_IMODE(path.stat().st_mode) & 0o666
    if hashlib.sha256(original).hexdigest() != drp.sha256:
        outcome.refusals = [Refusal("input-changed", "the input changed on disk while it was read")]
        return outcome
    try:
        zipraw.parse_layout(original)
        prepared = engine.prepare(drp, path, plan)
        new_members = engine.apply(prepared.member_data, prepared.splices)
        new_bytes = zipraw.rewrite(original, new_members)
    except EditRefused as exc:
        outcome.refusals = exc.refusals
        return outcome
    except zipraw.ZipLayoutError as exc:
        outcome.refusals = [Refusal("zip-layout", str(exc))]
        return outcome
    outcome.expected = [e.to_dict() for e in prepared.expected]
    outcome.report = prepared.report
    if not prepared.expected:
        # The plan has no intended change (every rename is to the current name, a path_map
        # matched no path, a marker edit nets out, an empty plan, ...). Refuse rather than write
        # an "unverified" copy with zero changes. The guard is on the expected set, not on the
        # bytes: a touched marker list whose BlobData is a compressed zstd frame is re-framed to a
        # raw-block frame, so the bytes can differ while nothing the gate sees changed. The report
        # says what each edit matched.
        outcome.refusals = [
            Refusal(
                "no-op",
                "the plan changes nothing: every edit leaves the file as it is (no intended "
                "change), so no output was written (see the report for what each edit matched)",
            )
        ]
        return outcome
    with contextlib.ExitStack() as stack:
        if dry_run or out_path is None:
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="aar-edit-")))
            name = (out_path.name if out_path else path.name) or "out.drp"
        else:
            directory, name = out_path.parent, out_path.name
        tmp = _temp(directory, name, new_bytes, mode)
        stack.callback(tmp.unlink, missing_ok=True)
        check = verify_write(path, tmp, prepared.expected, prepared.members())
        if check.diff is not None:
            outcome.changes = [c.to_dict() for c in check.diff.changes]
            outcome.gate = {
                "passed": check.diff.gate.passed,
                "reasons": list(check.diff.gate.reasons),
            }
        if not check.ok:
            outcome.refusals = check.refusals
            return outcome
        digest = hashlib.sha256(new_bytes).hexdigest()
        outcome.output = {
            "path": str(out_path) if out_path else None,
            "sha256": digest,
            "size": len(new_bytes),
            "sidecar": str(out_path) + SIDECAR_SUFFIX if out_path else None,
        }
        if dry_run or out_path is None:
            outcome.status = "dry-run"
            return outcome
        outcome.status = "written"
        sidecar = Path(str(out_path) + SIDECAR_SUFFIX)
        record = outcome.to_dict()
        record["status"] = LABEL
        label = json.dumps(record, indent=1, ensure_ascii=True, allow_nan=False) + "\n"
        side_tmp = _temp(directory, sidecar.name, label.encode("ascii"), mode)
        stack.callback(side_tmp.unlink, missing_ok=True)
        backup = None
        if force and os.path.lexists(sidecar):
            # The label passed the pre-check, but the edit took time. Snapshot it (a hard link,
            # else a copy) and check the SNAPSHOT for our stamp: a label that is not ours by the
            # time of the snapshot is refused, and the snapshot is kept (_settle_backup). This
            # narrows the window, it does not close it: rename has no compare-and-swap, so a label
            # written as a new file after this check and before the os.replace below is lost.
            backup = _backup(directory, sidecar)
            if backup is not None:
                stack.callback(_settle_backup, backup, sidecar, out_path, outcome.left_behind)
            if backup is None or not _is_our_sidecar(sidecar, out_path, read=backup):
                outcome.status = "refused"
                outcome.output = None
                outcome.refusals = [
                    Refusal(
                        "foreign-sidecar",
                        f"the label {sidecar} changed to one aar did not write while the edit "
                        "ran; nothing was placed",
                        details={"sidecar": str(sidecar)},
                    )
                ]
                return outcome
        try:
            _place(side_tmp, sidecar, force)
            try:
                _place(tmp, out_path, force)
            except BaseException as exc:
                surfaced = _roll_back_label(exc, backup, sidecar, outcome.left_behind)
                if surfaced is exc:
                    raise
                raise surfaced from exc
        except FileExistsError as exc:
            outcome.status = "refused"
            outcome.output = None
            outcome.refusals = [Refusal("out-exists", f"the output appeared while writing: {exc}")]
        return outcome
