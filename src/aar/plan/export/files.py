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
``.aar-backup-<name>.<token>``, and the failure is raised as ``output_conflict`` naming both; a
backup that cannot be restored is kept and named the same way.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from aar.drp.modify.write import _backup, _place, _temp
from aar.plan.export.errors import ExportError

MODE = 0o644


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
    """Write every ``(destination, bytes)`` or none of them (see the module docstring)."""
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
    except FileExistsError as exc:
        conflicts = _roll_back(placed, backups)
        error = ExportError(
            "output_exists",
            f"{exc.filename!r} appeared while writing",
            "choose another --out, or pass --overwrite",
        )
        raise _conflict(error, conflicts) from None
    except BaseException as exc:
        conflicts = _roll_back(placed, backups)
        if conflicts and isinstance(exc, Exception):
            raise _conflict(exc, conflicts) from exc
        raise
    finally:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _conflict(
    cause: BaseException, conflicts: list[tuple[Path, Path | None, str]]
) -> BaseException:
    """The failure, as ``output_conflict`` when the rollback could not put a destination back
    (another writer replaced it, or the restore failed); else the failure itself."""
    if not conflicts:
        return cause
    lines = [
        f"{str(dest)!r} {why}"
        + (f"; the previous file is kept as {str(kept)!r}" if kept is not None else "")
        for dest, kept, why in conflicts
    ]
    return ExportError(
        "output_conflict",
        f"writing the export failed ({cause}), and {'; '.join(lines)}",
        "compare the files, keep the right one and remove the other",
        conflicts=[
            {"path": str(dest), "kept_backup": None if kept is None else str(kept), "reason": why}
            for dest, kept, why in conflicts
        ],
    )


def _roll_back(
    placed: list[tuple[Path, Path, Identity]], backups: dict[Path, Path]
) -> list[tuple[Path, Path | None, str]]:
    """Put back every destination this call placed: its backup, or nothing when it was new, but
    only while it is still the file placed (device, inode, size, mtime_ns). A destination whose
    temporary file was never consumed was never placed and is left alone. One another writer
    replaced after our placement is left in place and returned, with its backup kept beside it
    under a clear name; so is one whose backup cannot be restored."""
    conflicts: list[tuple[Path, Path | None, str]] = []
    for dest, tmp, identity in reversed(placed):
        backup = backups.pop(dest, None)
        try:
            ours = _identity(dest) == identity
        except FileNotFoundError:
            ours = False
        if not ours:
            if tmp.exists():
                if backup is not None:
                    backups[dest] = backup  # never placed: dest is untouched; drop the backup
                continue
            if backup is not None or dest.exists() or dest.is_symlink():
                kept = _keep_backup(backup, dest) if backup is not None else None
                conflicts.append((dest, kept, "was replaced by another writer and left in place"))
            continue
        if backup is None:
            dest.unlink()
            continue
        try:
            os.replace(backup, dest)
        except OSError as exc:
            kept = _keep_backup(backup, dest)
            conflicts.append(
                (dest, kept, f"could not be restored ({exc}); the new file is in place")
            )
    return conflicts
