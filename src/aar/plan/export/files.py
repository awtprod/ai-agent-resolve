"""Write the FCPXML and its report as one pair: both land, or neither does (S18 round 3).

Both files are written to temporary files beside their destinations and fsynced first. With
``overwrite``, each existing destination is kept as a hard-link backup (a copy where the file
system has no hard links). Then both are placed; if anything fails, every destination already
placed is put back (its backup restored, or the new file removed when there was none) and every
temporary file and backup is removed. A previous pair is left byte-identical; a fresh export
leaves nothing. The temporary, placement and backup helpers are the DRP writer's.
"""

from __future__ import annotations

import os
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


def _identity(path: Path) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


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
    placed: list[tuple[Path, tuple[int, int]]] = []
    try:
        for dest, data in files:
            temps.append(_write_temp(dest, data))
        if overwrite:
            for dest, _ in files:
                backup = _backup(dest.parent, dest)
                if backup is not None:
                    backups[dest] = backup
        for (dest, _), tmp in zip(files, temps, strict=True):
            identity = _identity(tmp)
            _place_one(tmp, dest, overwrite)
            placed.append((dest, identity))
    except FileExistsError as exc:
        _roll_back(placed, backups)
        raise ExportError(
            "output_exists",
            f"{exc.filename!r} appeared while writing",
            "choose another --out, or pass --overwrite",
        ) from None
    except BaseException:
        _roll_back(placed, backups)
        raise
    finally:
        for tmp in temps:
            tmp.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _roll_back(placed: list[tuple[Path, tuple[int, int]]], backups: dict[Path, Path]) -> None:
    """Put back every destination already placed: its backup, or nothing when it was new. A file
    at the destination that is not the one this call placed is left alone."""
    for dest, identity in reversed(placed):
        backup = backups.pop(dest, None)
        if backup is not None:
            os.replace(backup, dest)
            continue
        try:
            if _identity(dest) == identity:
                dest.unlink()
        except FileNotFoundError:
            pass  # already gone: nothing to put back
