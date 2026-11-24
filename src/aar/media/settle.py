"""Copy a file another program may still be writing, only once it has settled.

Shared by the desktop ``.drt`` pickup (:mod:`aar.recipes.desktop_returns`) and the Resolve disk
library reader (:mod:`aar.drp.library`). Stdlib only.

A file counts as settled when its modification time is at least ``settle_seconds`` old when it
is opened, or when the caller has watched its identity (device, inode, size, modification time)
stay the same for that long (``settled_identity``; this also covers a share whose clock runs
ahead of ours). The copy is then accepted only if the open file's identity, change time
included, is the same after the last byte as before the first. Anything else returns ``None``
and the caller tries again later with a fresh, empty target.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from aar.media.receipt import _stat_identity, open_regular

__all__ = ["Copied", "TooLargeError", "copy_settled", "settle_identity"]

_BLOCK = 1024 * 1024


class TooLargeError(ValueError):
    """The file is, or grew while it was copied, past the caller's size limit."""


@dataclass(frozen=True, slots=True)
class Copied:
    """A settled copy: the SHA-256 and size of the bytes written to the target."""

    sha256: str
    size: int


def settle_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    """Device, inode, size and modification time: what must not change while a file settles.

    The change time is left out because Windows reports it differently through ``stat`` and
    ``fstat`` (see :func:`aar.media.receipt.open_regular`); the copy itself still checks it.
    """
    return _stat_identity(value)[:4]  # type: ignore[return-value]


def _mib(limit: int) -> str:
    return f"{limit // (1024 * 1024)} MiB" if limit % (1024 * 1024) == 0 else f"{limit} bytes"


def copy_settled(
    path: Path,
    target: BinaryIO,
    *,
    max_bytes: int,
    settle_seconds: float,
    what: str = "file",
    dir_fd: int | None = None,
    settled_identity: tuple[int, int, int, int] | None = None,
    now: Callable[[], float] = time.time,
) -> Copied | None:
    """Copy ``path`` into ``target`` if it has settled and did not change while copied.

    Returns ``None`` when the file was modified less than ``settle_seconds`` ago (and its
    identity is not ``settled_identity``), or when it changed during the copy; ``target`` then
    holds nothing useful. Raises :class:`TooLargeError` past ``max_bytes``, and
    :class:`~aar.media.receipt.ReceiptError` (a ``ValueError``) for a symlink or a file that is
    not regular. The file is opened read-only and never written.
    """
    with open_regular(path, dir_fd=dir_fd) as source:
        before = os.fstat(source.fileno())
        if before.st_size > max_bytes:
            raise TooLargeError(f"{what} exceeds {_mib(max_bytes)}")
        quiet = settled_identity is not None and settled_identity == settle_identity(before)
        if not quiet and now() - before.st_mtime < settle_seconds:
            return None
        digest = hashlib.sha256()
        size = 0
        while block := source.read(_BLOCK):
            size += len(block)
            if size > max_bytes:
                raise TooLargeError(f"{what} grew past {_mib(max_bytes)}")
            digest.update(block)
            target.write(block)
        if _stat_identity(before) != _stat_identity(os.fstat(source.fileno())):
            return None  # another save or copy is in progress; retry with a fresh target
    return Copied(digest.hexdigest(), size)
