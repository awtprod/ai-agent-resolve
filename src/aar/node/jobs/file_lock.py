"""Cross-platform exclusive lock for the job store sidecar."""

from __future__ import annotations

import os
import sys

_WINDOWS_BACKEND = sys.platform == "win32"

if _WINDOWS_BACKEND:
    import msvcrt
else:
    import fcntl


class FileLockUnavailable(OSError):
    """Another process holds the requested file lock."""


def acquire_file_lock(path: str) -> int:
    """Open ``path`` and take its nonblocking exclusive lock."""
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_BINARY", 0), 0o600)
    try:
        if _WINDOWS_BACKEND:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise FileLockUnavailable(path) from exc
        else:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise FileLockUnavailable(path) from exc
    except BaseException:
        os.close(fd)
        raise
    return fd


def release_file_lock(fd: int) -> None:
    """Release a lock from :func:`acquire_file_lock` and close its handle."""
    try:
        if _WINDOWS_BACKEND:
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
