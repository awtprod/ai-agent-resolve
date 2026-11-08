"""Write the FCPXML and its report as one pair: both land, or neither does (S18 round 3).

Both files are written to temporary files beside their destinations and fsynced first. With
``overwrite``, each existing destination is kept as a hard-link backup (a copy where the file
system has no hard links). Then both are placed; if anything fails, every destination already
placed is put back (its backup restored, or the new file removed when there was none) and every
temporary file and backup is removed. A previous pair is left byte-identical; a fresh export
leaves nothing. The temporary, placement and backup helpers are the DRP writer's.

Each destination is recorded before it is placed, with the identity of its temporary file
(device, inode, size, ``mtime_ns``: a rename or hard link keeps all four), so a failure inside the
placement itself (the temporary's unlink, a signal) still rolls it back. The rollback touches a
destination only while it is still that file. If another writer replaced or changed it after our
placement, it is left alone, the backup (if any) is kept beside it as
``.aar-backup-<name>.<token>``, and the failure is raised as ``output_conflict``.

The rollback is best effort per destination: a step that fails for one (a restore, renaming its
backup, removing the new file) is recorded and the next destination is still rolled back, and
every temporary file is still removed. Then one error lists every destination's final state
(``restored``, ``untouched``, ``kept_backup`` with its path, ``other_writer``, ``unknown``):
``rollback_failed`` when any rollback step failed or a temporary file could not be removed,
``output_conflict`` when only another writer got in the way. It is chained to the original
failure, which nothing in the rollback can mask. An interrupt (``KeyboardInterrupt``) is
re-raised as itself, with the states as notes.

aar exports to one destination are serialized: each holds an exclusive lock on
``<name>.aar-lock`` beside each of its destinations, the FCPXML's and the report's (``x.fcpxml``
and ``x.xml`` share ``x.report.json``), taken in sorted order, with the node job store's portable
helper (``fcntl.flock``, or ``msvcrt.locking`` on Windows), for the whole write: temporary files,
backups, both placements and any rollback. A second export to the same destination fails at once
with ``output_locked`` and touches nothing. The lock files stay after release (deleting one
would reopen the race it closes). The limit: a process outside aar that replaces the files
during an export is not serialized; the identity check above is the best effort against it, and
no more.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aar.drp.modify.write import _backup, _place, _temp
from aar.plan.export.errors import ExportError

MODE = 0o644
#: The lock beside each destination that serializes aar exports to it.
LOCK_SUFFIX = ".aar-lock"


def lock_path(dest: Path) -> Path:
    """The lock file that serializes aar exports to ``dest``: ``<dest>.aar-lock`` beside it."""
    return dest.with_name(dest.name + LOCK_SUFFIX)


def _lock_api() -> Any:
    """The node job store's portable exclusive lock (fcntl, or msvcrt on Windows), imported only
    when an export writes."""
    from aar.node.jobs import file_lock

    return file_lock


def _write_temp(dest: Path, data: bytes) -> Path:
    """A fsynced temporary file holding ``data`` beside ``dest``."""
    return _temp(dest.parent, dest.name, data, mode=MODE)


def _place_one(tmp: Path, dest: Path, overwrite: bool) -> None:
    """Move ``tmp`` to ``dest`` (never over an existing file unless ``overwrite``)."""
    _place(tmp, dest, overwrite)


Identity = tuple[int, int, int, int]  # st_dev, st_ino, st_size, st_mtime_ns


def _identity(path: Path) -> Identity:
    st = os.lstat(path)  # lstat: a symlink planted at the name is not the file we placed
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns


def _keep_backup(backup: Path, dest: Path) -> Path:
    """Rename a backup to a clearly named ``.aar-backup-<name>.<token>`` beside ``dest``."""
    while True:
        kept = dest.with_name(f".aar-backup-{dest.name}.{secrets.token_hex(4)}")
        try:
            os.link(backup, kept)  # never over an existing file
        except FileExistsError:
            continue
        except OSError:
            if kept.exists():
                continue
            os.rename(backup, kept)  # no hard links on this file system
            return kept
        backup.unlink()
        return kept


def _refuse_existing(dests: list[Path], overwrite: bool) -> None:
    for dest in dests:
        if dest.is_dir():
            raise ExportError(
                "output_exists", f"{str(dest)!r} is a directory", "give a file path for --out"
            )
        if not overwrite and (dest.exists() or dest.is_symlink()):
            raise ExportError(
                "output_exists",
                f"{str(dest)!r} already exists",
                "choose another --out, or pass --overwrite",
                path=str(dest),
            )


def write_pair(files: list[tuple[Path, bytes]], *, overwrite: bool) -> None:
    """Write every ``(destination, bytes)`` or none of them, holding every destination's export
    lock throughout (see the module docstring)."""
    api = _lock_api()
    held: list[int] = []
    try:
        # Every destination, in one fixed order: x.fcpxml and x.xml both write x.report.json,
        # so two such exports meet on that lock (fix-check Q1), and never deadlock.
        for lock in sorted({lock_path(dest) for dest, _ in files}):
            try:
                held.append(api.acquire_file_lock(str(lock)))
            except api.FileLockUnavailable:
                raise ExportError(
                    "output_locked",
                    f"another aar export holds the lock {str(lock)!r} (or the file system has no "
                    "file locks)",
                    "wait for the other export to finish, or choose another --out",
                    path=str(lock.with_name(lock.name[: -len(LOCK_SUFFIX)])),
                    lock=str(lock),
                ) from None
        _write_pair_locked(files, overwrite)
    finally:
        for fd in reversed(held):
            api.release_file_lock(fd)


def _write_pair_locked(files: list[tuple[Path, bytes]], overwrite: bool) -> None:
    _refuse_existing([dest for dest, _ in files], overwrite)
    temps: list[Path] = []
    backups: dict[Path, Path] = {}
    placed: list[tuple[Path, Path, Identity]] = []  # (destination, its temp, the temp's identity)
    try:
        for dest, data in files:
            temps.append(_write_temp(dest, data))
        if overwrite:
            for dest, _ in files:
                backup = _backup(dest.parent, dest)
                if backup is not None:
                    backups[dest] = backup
        for (dest, _), tmp in zip(files, temps, strict=True):
            # Recorded before placing, so a failure inside the placement still rolls it back.
            placed.append((dest, tmp, _identity(tmp)))
            _place_one(tmp, dest, overwrite)
    except BaseException as exc:
        outcomes = _roll_back(placed, backups)  # never raises: each destination best effort
        left = _discard([*temps, *backups.values()])
        temps.clear()
        backups.clear()
        failure = _failure(exc, [dest for dest, _ in files], outcomes, left)
        if failure is exc:
            raise
        if isinstance(failure, ExportError) and failure.code == "output_exists":
            raise failure from None
        raise failure from exc
    finally:
        _discard([*temps, *backups.values()])


@dataclass(frozen=True)
class Outcome:
    """Where the rollback left one destination."""

    path: Path
    #: restored | untouched | kept_backup | other_writer | unknown
    state: str
    kept_backup: Path | None
    detail: str
    #: A rollback step failed for this destination.
    failed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "state": self.state,
            "kept_backup": None if self.kept_backup is None else str(self.kept_backup),
            "detail": self.detail,
        }

    def line(self) -> str:
        kept = f" (the previous file is at {str(self.kept_backup)!r})" if self.kept_backup else ""
        return f"{str(self.path)!r}: {self.state}{kept}: {self.detail}"


def _discard(paths: list[Path]) -> list[Path]:
    """Remove each file, best effort; the ones that could not be removed."""
    left: list[Path] = []
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            left.append(path)
    return left


def _failure(
    exc: BaseException, dests: list[Path], outcomes: list[Outcome], left: list[Path]
) -> BaseException:
    """The error to raise for ``exc`` after the rollback: ``exc`` itself (``output_exists`` for a
    destination that appeared) when everything was put back; else one error naming every
    destination's final state."""
    base: BaseException = exc
    if isinstance(exc, FileExistsError):
        base = ExportError(
            "output_exists",
            f"{exc.filename!r} appeared while writing",
            "choose another --out, or pass --overwrite",
        )
    failed = left or any(o.failed for o in outcomes)
    if not failed and not any(o.state == "other_writer" for o in outcomes):
        return base
    by_path = {o.path: o for o in outcomes}
    every = [by_path.get(d) or Outcome(d, "untouched", None, "never placed") for d in dests]
    lines = [o.line() for o in every]
    lines += [f"{str(path)!r}: a temporary file that could not be removed" for path in left]
    if not isinstance(exc, Exception):  # an interrupt stays an interrupt, with the states noted
        for line in lines:
            exc.add_note(f"aar export rollback: {line}")
        return exc
    if failed:
        return ExportError(
            "rollback_failed",
            f"writing the export failed ({base}), and the rollback could not put everything "
            f"back: {'; '.join(lines)}",
            "check each file named: keep the right one, put a kept previous file back by hand, "
            "and remove what is left over",
            destinations=[o.as_dict() for o in every],
            left_behind=[str(path) for path in left],
        )
    return ExportError(
        "output_conflict",
        f"writing the export failed ({base}), and another writer changed a destination: "
        f"{'; '.join(lines)}",
        "compare the files, keep the right one and remove the other",
        destinations=[o.as_dict() for o in every],
        left_behind=[],
    )


def _roll_back(
    placed: list[tuple[Path, Path, Identity]], backups: dict[Path, Path]
) -> list[Outcome]:
    """Put back every destination this call placed, best effort: one whose rollback fails is
    recorded and the rest are still rolled back; this never raises an ``Exception``. A backup
    reported as kept is taken out of ``backups`` (so it is not removed); one for a destination
    never placed is left in it, to be removed."""
    outcomes: list[Outcome] = []
    for dest, tmp, identity in reversed(placed):
        backup = backups.pop(dest, None)
        try:
            outcome = _undo(dest, tmp, identity, backup)
        except Exception as exc:
            kept, why = _keep(backup, dest)
            outcome = Outcome(dest, "unknown", kept, f"the rollback failed ({exc}){why}", True)
        if outcome.state == "untouched" and backup is not None:
            backups[dest] = backup  # never placed: the backup is a spare copy
        outcomes.append(outcome)
    return outcomes[::-1]


def _undo(dest: Path, tmp: Path, identity: Identity, backup: Path | None) -> Outcome:
    """Roll one destination back: its backup restored, or the new file removed when there was
    none, but only while it is still the file placed (device, inode, size, mtime_ns)."""
    try:
        ours = _identity(dest) == identity
    except FileNotFoundError:
        ours = False
    if not ours:
        if tmp.exists():
            return Outcome(dest, "untouched", None, "never placed")
        there = dest.exists() or dest.is_symlink()
        if backup is None and not there:
            return Outcome(dest, "restored", None, "the new file is gone; there was none before")
        what = "another writer's file is left in place" if there else "another writer removed it"
        if backup is None:
            return Outcome(dest, "other_writer", None, what)
        kept, why = _keep(backup, dest)
        return Outcome(dest, "other_writer", kept, what + why, kept is None)
    if backup is None:
        dest.unlink()
        return Outcome(dest, "restored", None, "the new file was removed; there was none before")
    try:
        os.replace(backup, dest)
    except OSError as exc:
        kept, why = _keep(backup, dest)
        state = "kept_backup" if kept is not None else "unknown"
        detail = f"could not be restored ({exc}){why}; the new file is in place"
        return Outcome(dest, state, kept, detail, True)
    return Outcome(dest, "restored", None, "the previous file is back")


def _keep(backup: Path | None, dest: Path) -> tuple[Path | None, str]:
    """Keep a backup under a clear name, best effort: (where it is now, a note when renaming it
    failed). When the rename fails the backup stays at its temporary name, which is reported."""
    if backup is None:
        return None, ""
    try:
        return _keep_backup(backup, dest), ""
    except Exception as exc:
        try:
            still = backup.exists()
        except OSError:
            still = False
        return (backup if still else None), f"; keeping the backup failed ({exc})"
