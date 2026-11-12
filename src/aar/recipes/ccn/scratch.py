"""Scratch space and locks of the ``ccn-longform`` steps, all under ``AAR_WORK_DIR``.

- :func:`work_root`: the configured work directory (a typed failure when unset), never ``/tmp``.
- :func:`op_scratch`: a deterministic directory per step op, ``<area>/<recipe node>/<op node>``
  (an op is one attempt of one round), emptied at the start of every attempt. A worker that
  re-claims a lapsed op after a crash starts from an empty directory, and the result is stored
  content-addressed, so a repeat cannot duplicate anything.
- :func:`sweep`: removes the scratch of finished and cancelled runs and of ops that are no longer
  live, at most :data:`SWEEP_MAX` entries a call, never a live op's directory.
- :func:`key_lock`: an exclusive lock file per key that is unlinked by its holder, so lock files
  do not pile up; a waiter that locked a file someone unlinked meanwhile opens the new one.
  :func:`sweep_locks` removes lock files nobody holds (from a killed worker).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from aar.node.jobs.file_lock import FileLockUnavailable, acquire_file_lock, release_file_lock
from aar.recipes.context import StepContext, StepFailure
from aar.recipes.state import load_state

WORK_DIR_ENV = "AAR_WORK_DIR"
#: At most this many directory entries are looked at by one sweep.
SWEEP_MAX = 200
#: How often a waiter retries a held lock.
LOCK_POLL_S = 2.0
LOCK_SUFFIX = ".lock"


def work_root() -> Path:
    """``AAR_WORK_DIR``: where every step keeps its scratch; a typed failure when unset."""
    raw = os.environ.get(WORK_DIR_ENV, "").strip()
    if not raw or not os.path.isabs(raw):
        raise StepFailure(
            "work_dir_unset",
            f"{WORK_DIR_ENV} is not set to an absolute directory",
            f"set {WORK_DIR_ENV} for the worker (a directory on the media disk, not /tmp)",
        )
    root = Path(raw)
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StepFailure(
            "work_dir_unusable", f"cannot use {WORK_DIR_ENV} {raw!r}: {exc}", "fix its permissions"
        ) from None
    return root


def _remove(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path, ignore_errors=True)


def op_scratch(ctx: StepContext, area: str) -> Path:
    """``<work>/<area>/<recipe>/<op>``, empty: whatever an earlier claim of this op left is
    removed first. Sweeps the area first (:func:`sweep`)."""
    base = work_root() / area
    sweep(ctx, base)
    folder = base / ctx.recipe_node / ctx.op_node
    if folder.exists() or folder.is_symlink():
        _remove(folder)
    folder.mkdir(parents=True, mode=0o700)
    return folder


def drop_scratch(folder: Path) -> None:
    """Remove an op's scratch and its recipe folder when that is left empty."""
    _remove(folder)
    with contextlib.suppress(OSError):  # only when empty; another op's folder keeps it
        folder.parent.rmdir()


def sweep(ctx: StepContext, base: Path) -> int:
    """Remove scratch under ``base`` that no live op owns: a recipe folder whose run is finished
    or unknown, an op folder whose op is not live. Bounded by :data:`SWEEP_MAX`; returns the
    number of entries removed. The current op's folder is never removed here."""
    if not base.is_dir():
        return 0
    removed = 0
    with os.scandir(base) as entries:
        recipes = [Path(e.path) for _, e in zip(range(SWEEP_MAX), entries, strict=False)]
    for folder in recipes:
        if not folder.is_dir() or folder.is_symlink():
            _remove(folder)
            removed += 1
            continue
        try:
            state = load_state(ctx.log, folder.name)
        except Exception:  # not a recipe node of this hub: nobody owns it
            state = None
        finished = (
            state is None
            or state.recipe_data is None
            or state.recipe_status in ("done", "cancelled")
        )
        if finished and folder.name != ctx.recipe_node:
            _remove(folder)
            removed += 1
            continue
        live = {op.op_id for op in state.ops if op.is_live} if state is not None else set()
        with os.scandir(folder) as entries:
            ops = [Path(e.path) for _, e in zip(range(SWEEP_MAX), entries, strict=False)]
        for op_folder in ops:
            if op_folder.name != ctx.op_node and op_folder.name not in live:
                _remove(op_folder)
                removed += 1
        with contextlib.suppress(OSError):
            folder.rmdir()
    return removed


def _try_lock(path: Path) -> int | None:
    """The lock on ``path``, or None when another process holds it. A lock taken on a file that
    was unlinked meanwhile (its holder finished) is dropped and taken again on the new file."""
    for _ in range(10):
        try:
            fd = acquire_file_lock(str(path))
        except FileLockUnavailable:
            return None
        try:
            same = os.path.samestat(os.fstat(fd), os.stat(path))
        except FileNotFoundError:
            same = False
        if same:
            return fd
        release_file_lock(fd)
    return None


@contextlib.contextmanager
def key_lock(
    path: Path,
    *,
    wait_s: float,
    on_wait: Callable[[], object] | None = None,
    poll_s: float | None = None,
) -> Iterator[bool]:
    """Hold the lock file ``path`` for the block; yields True. While another process holds it,
    retry every ``poll_s`` for at most ``wait_s``, calling ``on_wait`` each time: when it returns
    something truthy the wait ends and the block runs **without** the lock (yields False), so a
    waiter can pick up what the holder made. Past ``wait_s`` it raises ``TimeoutError``. The
    holder unlinks the file before releasing it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + max(0.0, wait_s)
    fd = _try_lock(path)
    while fd is None:
        if on_wait is not None and on_wait():
            yield False
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(str(path))
        time.sleep(LOCK_POLL_S if poll_s is None else poll_s)
        fd = _try_lock(path)
    try:
        yield True
    finally:
        try:
            path.unlink(missing_ok=True)
        finally:
            release_file_lock(fd)


def sweep_locks(folder: Path) -> int:
    """Remove lock files in ``folder`` that nobody holds (a worker killed while holding one);
    bounded by :data:`SWEEP_MAX`. Returns the number removed."""
    if not folder.is_dir():
        return 0
    removed = 0
    with os.scandir(folder) as entries:
        names = [e.name for _, e in zip(range(SWEEP_MAX), entries, strict=False)]
    for name in names:
        if not name.endswith(LOCK_SUFFIX):
            continue
        path = folder / name
        fd = _try_lock(path)
        if fd is None:
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        finally:
            release_file_lock(fd)
    return removed


__all__ = [
    "LOCK_POLL_S",
    "SWEEP_MAX",
    "WORK_DIR_ENV",
    "drop_scratch",
    "key_lock",
    "op_scratch",
    "sweep",
    "sweep_locks",
    "work_root",
]
