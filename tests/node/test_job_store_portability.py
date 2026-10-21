"""Portable job-store locking and native node import smoke tests."""

from __future__ import annotations

import multiprocessing
import os
import queue
import runpy
import sys
import types
from pathlib import Path

import pytest

from aar.node.jobs import JobStore, JobStoreLocked, file_lock

WAIT_SECONDS = 15


def _hold_store(path: str, ready: multiprocessing.Queue, release: multiprocessing.Event) -> None:
    try:
        store = JobStore(path)
    except BaseException as exc:
        ready.put(("error", f"{type(exc).__name__}: {exc}"))
        return
    ready.put(("locked", ""))
    try:
        if not release.wait(WAIT_SECONDS):
            raise TimeoutError("parent did not release lock holder")
    finally:
        store.close()


def test_native_node_runtime_import_and_job_store_smoke(tmp_path: Path) -> None:
    from aar.node.runtime import NodeRuntime

    assert NodeRuntime.__name__ == "NodeRuntime"
    store = JobStore(tmp_path / "jobs.db")
    try:
        assert store.load_all() == []
    finally:
        store.close()


def test_spawned_process_contends_then_lock_can_be_reacquired(tmp_path: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    release = ctx.Event()
    db = tmp_path / "jobs.db"
    process = ctx.Process(target=_hold_store, args=(str(db), ready, release))
    process.start()
    try:
        try:
            status, detail = ready.get(timeout=WAIT_SECONDS)
        except queue.Empty:
            pytest.fail("spawned lock holder did not become ready")
        assert (status, detail) == ("locked", "")
        assert db.with_suffix(".db.lock").exists()
        with pytest.raises(JobStoreLocked):
            JobStore(db)
        release.set()
        process.join(WAIT_SECONDS)
        assert process.exitcode == 0
        JobStore(db).close()
        assert db.with_suffix(".db.lock").exists()
    finally:
        release.set()
        process.join(WAIT_SECONDS)
        if process.is_alive():
            process.terminate()
            process.join(WAIT_SECONDS)
        ready.close()
        ready.join_thread()


def test_constructor_failure_releases_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = tmp_path / "jobs.db"

    def fail_init(_store: JobStore) -> None:
        raise RuntimeError("forced constructor failure")

    with monkeypatch.context() as patch:
        patch.setattr(JobStore, "_init_version", fail_init)
        with pytest.raises(RuntimeError, match="forced constructor failure"):
            JobStore(db)

    JobStore(db).close()


def test_close_failure_still_releases_lock(tmp_path: Path) -> None:
    class BrokenClose:
        def close(self) -> None:
            raise RuntimeError("forced close failure")

    db = tmp_path / "jobs.db"
    store = JobStore(db)
    store._conn.close()
    store._conn = BrokenClose()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="forced close failure"):
        store.close()
    JobStore(db).close()


def test_release_uses_imported_backend_after_sys_platform_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sidecar = tmp_path / "native.lock"
    fd = file_lock.acquire_file_lock(str(sidecar))
    changed_platform = "linux" if sys.platform == "win32" else "win32"

    monkeypatch.setattr(sys, "platform", changed_platform)
    file_lock.release_file_lock(fd)
    with pytest.raises(OSError):
        os.fstat(fd)

    reacquired = file_lock.acquire_file_lock(str(sidecar))
    file_lock.release_file_lock(reacquired)


def test_windows_backend_uses_a_real_byte_without_importing_fcntl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    helper = Path(__file__).parents[2] / "src" / "aar" / "node" / "jobs" / "file_lock.py"
    calls: list[tuple[int, int, int]] = []
    fake_msvcrt = types.ModuleType("msvcrt")
    fake_msvcrt.LK_NBLCK = 2  # type: ignore[attr-defined]
    fake_msvcrt.LK_UNLCK = 0  # type: ignore[attr-defined]
    fake_msvcrt.locking = lambda fd, mode, size: calls.append((fd, mode, size))  # type: ignore[attr-defined]

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake_msvcrt)
    monkeypatch.setitem(sys.modules, "fcntl", None)
    namespace = runpy.run_path(str(helper))
    assert namespace["_WINDOWS_BACKEND"] is True

    sidecar = tmp_path / "jobs.db.lock"
    fd = namespace["acquire_file_lock"](str(sidecar))
    assert sidecar.read_bytes() == b"\0"
    namespace["release_file_lock"](fd)
    assert [(mode, size) for _, mode, size in calls] == [(2, 1), (0, 1)]
    with pytest.raises(OSError):
        os.fstat(fd)

    failed_fd: list[int] = []

    def refuse_lock(fd: int, _mode: int, _size: int) -> None:
        failed_fd.append(fd)
        raise OSError("already locked")

    fake_msvcrt.locking = refuse_lock  # type: ignore[attr-defined]
    with pytest.raises(namespace["FileLockUnavailable"]):
        namespace["acquire_file_lock"](str(tmp_path / "contended.lock"))
    with pytest.raises(OSError):
        os.fstat(failed_fd[0])
