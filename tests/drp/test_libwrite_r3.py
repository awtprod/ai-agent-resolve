"""S25c2 fix round 3 (the automated review of PR #40): the placement acts only on what it
created (modes and owners through descriptors, never through a link), a staging folder someone
else prepared is refused, exports of different names never fail each other over the shared
staging folder, and the item gate tells two projects apart by their full path.

Every value is made up (``libwrite_support``)."""

from __future__ import annotations

import os
import random
import shutil
import stat
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from libwrite_support import linked_pair, make_library, mp4_template, write_seed

import aar.drp.libwrite.place as place
from aar.drp.items import gate
from aar.drp.library import LibraryOptions
from aar.drp.libwrite import ProjectWriteError, place_project
from aar.drp.libwrite.place import STAGING

POSIX = sys.platform != "win32"
ROOT = POSIX and os.geteuid() == 0
needs_links = pytest.mark.skipif(not POSIX, reason="symbolic links and modes are POSIX here")


def _projects(library: Path) -> Path:
    return library / "Resolve Projects" / "Users" / "guest" / "Projects"


def _top(library: Path) -> list[str]:
    return sorted(os.listdir(library / "Resolve Projects"))


def _library_with_pattern(tmp_path: Path) -> Path:
    """A library whose one project has modes no new file has (folder 750, file 640)."""
    library = make_library(tmp_path / "lib")
    sibling = _projects(library) / "Existing"
    sibling.mkdir()
    (sibling / "Project.db").write_bytes(b"theirs")
    sibling.chmod(0o750)
    (sibling / "Project.db").chmod(0o640)
    return library


def _outside(tmp_path: Path) -> tuple[Path, os.stat_result]:
    """A file outside the library that the exporting process owns, mode 600."""
    target = tmp_path / "outside" / "secret.txt"
    target.parent.mkdir()
    target.write_bytes(b"not the library's")
    target.chmod(0o600)
    return target, os.stat(target)


def _untouched(target: Path, before: os.stat_result) -> None:
    after = os.stat(target)
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    assert target.read_bytes() == b"not the library's"


# -- 1. permissions and every other act only on what this export created --------------------


@needs_links
def test_a_project_db_swapped_for_a_link_is_refused_and_the_outside_file_untouched(
    tmp_path: Path,
) -> None:
    library = _library_with_pattern(tmp_path)
    target, before = _outside(tmp_path)

    def build(path: Path) -> None:
        path.write_bytes(b"ours")
        path.unlink()  # someone swaps the file for a link before the permission step
        path.symlink_to(target)

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Swapped", build)
    assert info.value.code == "library_unwritable"
    assert "Project.db" in info.value.message
    _untouched(target, before)
    assert sorted(os.listdir(_projects(library))) == ["Existing"]
    assert _top(library) == ["Users"]


@needs_links
def test_a_build_folder_swapped_while_building_is_refused_and_the_outside_file_untouched(
    tmp_path: Path,
) -> None:
    library = _library_with_pattern(tmp_path)
    target, before = _outside(tmp_path)
    moved = tmp_path / "moved-away"

    def build(path: Path) -> None:
        path.write_bytes(b"ours")
        work = path.parent.parent
        # Someone who can write to the staging folder renames the build folder away and puts
        # their own in its place, holding a Project.db that is a link to an outside file.
        os.rename(work, moved)
        (work / "project").mkdir(parents=True)
        (work / "project" / "Project.db").symlink_to(target)

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Swapped", build)
    assert info.value.code == "library_unwritable"
    _untouched(target, before)
    assert sorted(os.listdir(_projects(library))) == ["Existing"]
    # The substitute is not this export's: it is named, never removed.
    assert "not the build folder this export made" in info.value.details["residue"]
    [substitute] = (library / "Resolve Projects" / STAGING).glob("build-*/project/Project.db")
    assert substitute.is_symlink()


@needs_links
def test_a_project_folder_swapped_for_a_link_is_refused(tmp_path: Path) -> None:
    library = _library_with_pattern(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "Project.db").write_bytes(b"theirs")
    elsewhere.chmod(0o700)

    def build(path: Path) -> None:
        path.write_bytes(b"ours")
        shutil.rmtree(path.parent)
        path.parent.symlink_to(elsewhere)

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Swapped", build)
    assert info.value.code == "library_unwritable"
    assert stat.S_IMODE(os.stat(elsewhere).st_mode) == 0o700
    assert (elsewhere / "Project.db").read_bytes() == b"theirs"
    assert sorted(os.listdir(_projects(library))) == ["Existing"]


@needs_links
def test_a_project_db_hard_linked_to_an_outside_file_is_refused(tmp_path: Path) -> None:
    library = _library_with_pattern(tmp_path)
    target, before = _outside(tmp_path)

    def build(path: Path) -> None:
        path.write_bytes(b"ours")
        path.unlink()
        os.link(target, path)  # same file system: tmp_path holds both

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Linked", build)
    assert info.value.code == "library_unwritable"
    _untouched(target, before)
    assert sorted(os.listdir(_projects(library))) == ["Existing"]


@needs_links
def test_modes_are_set_through_descriptors_and_follow_the_library(tmp_path: Path) -> None:
    library = _library_with_pattern(tmp_path)
    placed, _ = place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert stat.S_IMODE(os.stat(placed.folder).st_mode) == 0o750
    assert stat.S_IMODE(os.stat(placed.path).st_mode) == 0o640
    assert placed.file_id == (os.stat(placed.path).st_dev, os.stat(placed.path).st_ino)
    assert _top(library) == ["Users"]


@needs_links
def test_the_placed_file_is_read_only_when_it_is_the_one_created(tmp_path: Path) -> None:
    library = _library_with_pattern(tmp_path)
    placed, _ = place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert place.read_placed(placed) == b"ok"
    target, _ = _outside(tmp_path)
    placed.path.unlink()
    placed.path.symlink_to(target)
    with pytest.raises(ProjectWriteError) as info:
        place.read_placed(placed)
    assert info.value.code == "library_unwritable"


def _replace_after_recording(path: Path) -> place.FileId:
    """A build that records the file it created, then finds another file in its place."""
    path.write_bytes(b"ours")
    made = place.created_file(path)
    other = path.with_name("other")
    other.write_bytes(b"another file")  # made while the first exists: another inode
    os.replace(other, path)
    return made


def test_a_project_db_replaced_after_it_was_recorded_is_refused(tmp_path: Path) -> None:
    library = _library_with_pattern(tmp_path)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Swapped", _replace_after_recording, file_id=lambda made: made)
    assert info.value.code == "library_unwritable"
    assert "another file or folder is in its place" in info.value.message
    assert sorted(os.listdir(_projects(library))) == ["Existing"]
    assert _top(library) == ["Users"]


def test_on_windows_the_identities_are_checked_on_the_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Windows has no ``O_NOFOLLOW``, ``dir_fd`` or ``fchmod``: the identities are checked on
    the paths (``lstat``, never a link or junction), and modes follow the share's."""
    library = _library_with_pattern(tmp_path)
    monkeypatch.setattr(place.sys, "platform", "win32")
    monkeypatch.setattr(place, "_NOREPLACE", [])
    real_open = os.open

    def no_nofollow(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        assert not flags & getattr(os, "O_NOFOLLOW", 0), "O_NOFOLLOW used on Windows"
        assert "dir_fd" not in kwargs, "dir_fd used on Windows"
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(place.os, "open", no_nofollow)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Swapped", _replace_after_recording, file_id=lambda made: made)
    assert info.value.code == "library_unwritable"
    assert sorted(os.listdir(_projects(library))) == ["Existing"]
    placed, _ = place_project(library, "Kept", lambda p: p.write_bytes(b"ok"))
    assert placed.permissions == ["permissions follow the share's (Windows)"]
    assert place.read_placed(placed) == b"ok"
    other = placed.path.with_name("other")
    other.write_bytes(b"another file")
    os.replace(other, placed.path)
    with pytest.raises(ProjectWriteError) as info:
        place.read_placed(placed)
    assert info.value.code == "library_unwritable"


@needs_links
def test_on_windows_a_link_in_place_of_project_db_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = _library_with_pattern(tmp_path)
    target, before = _outside(tmp_path)
    monkeypatch.setattr(place.sys, "platform", "win32")
    monkeypatch.setattr(place, "_NOREPLACE", [])

    def build(path: Path) -> None:
        path.symlink_to(target)

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Linked", build)
    assert info.value.code == "library_unwritable"
    _untouched(target, before)


def _make_project(tmp_path: Path, check: Callable[..., None]) -> Any:
    pytest.importorskip("zstandard")
    from test_libwrite import _make, _seed

    return _make(tmp_path, _seed(tmp_path), check=check)


def test_a_staged_file_replaced_during_the_checks_is_refused(tmp_path: Path) -> None:
    """``make_project`` records the staged file when it is made and checks it after every step
    that opens it by name (here: after the caller's check)."""

    def check(path: Path, seed: Any, authored: Any) -> None:
        other = path.with_name("other")
        other.write_bytes(path.read_bytes())
        os.replace(other, path)

    with pytest.raises(ProjectWriteError) as info:
        _make_project(tmp_path, check)
    assert info.value.code == "library_unwritable"
    assert "after the caller's check" in info.value.message
    assert os.listdir(_projects(tmp_path / "lib")) == []
    assert _top(tmp_path / "lib") == ["Users"]


@needs_links
def test_a_staged_file_swapped_for_a_link_during_the_checks_is_refused(tmp_path: Path) -> None:
    target, before = _outside(tmp_path)

    def check(path: Path, seed: Any, authored: Any) -> None:
        path.unlink()
        path.symlink_to(target)

    with pytest.raises(ProjectWriteError) as info:
        _make_project(tmp_path, check)
    assert info.value.code == "library_unwritable"
    _untouched(target, before)
    assert os.listdir(_projects(tmp_path / "lib")) == []


def test_a_made_project_records_the_file_it_placed(tmp_path: Path) -> None:
    result = _make_project(tmp_path, lambda path, seed, authored: None)
    st = os.stat(result.placed.path)
    assert result.placed.file_id == (st.st_dev, st.st_ino)
    assert place.read_placed(result.placed) == result.placed.path.read_bytes()


# -- 1b. a staging folder someone else prepared ---------------------------------------------


@needs_links
@pytest.mark.parametrize("mode", [0o777, 0o775, 0o757])
def test_a_staging_folder_writable_more_widely_than_projects_is_refused(
    tmp_path: Path, mode: int
) -> None:
    library = make_library(tmp_path / "lib")
    _projects(library).chmod(0o755)
    root = library / "Resolve Projects" / STAGING
    root.mkdir()
    root.chmod(mode)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "library_unwritable"
    assert str(root) in info.value.message
    assert root.is_dir()  # not this export's: left as it is
    assert os.listdir(root) == []
    assert os.listdir(_projects(library)) == []


@needs_links
def test_a_staging_folder_as_open_as_projects_is_accepted(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    _projects(library).chmod(0o775)
    root = library / "Resolve Projects" / STAGING
    root.mkdir()
    root.chmod(0o775)
    placed, _ = place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert placed.path.read_bytes() == b"ok"


@needs_links
def test_a_staging_folder_this_export_makes_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    seen: list[int] = []

    def build(path: Path) -> None:
        root = path.parent.parent.parent
        seen.extend(
            stat.S_IMODE(os.stat(p).st_mode) for p in (root, path.parent.parent, path.parent)
        )
        path.write_bytes(b"ok")

    old = os.umask(0)
    try:
        place_project(library, "Mine", build)
    finally:
        os.umask(old)
    assert seen == [0o700, 0o700, 0o700]


@needs_links
def test_a_staging_folder_of_another_owner_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    root = library / "Resolve Projects" / STAGING
    root.mkdir(mode=0o700)
    if ROOT:
        os.chown(root, 4242, 4242)  # a real other owner
    else:
        real = os.geteuid()
        monkeypatch.setattr(place.os, "geteuid", lambda: real + 1)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "library_unwritable"
    assert str(root) in info.value.message and "uid" in info.value.message
    assert root.is_dir()


@needs_links
def test_a_staging_folder_that_is_a_link_is_refused(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (library / "Resolve Projects" / STAGING).symlink_to(elsewhere)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "library_unwritable"
    assert os.listdir(elsewhere) == []


# -- 3. the shared staging folder and concurrent exports --------------------------------------


def test_a_staging_folder_removed_by_another_export_before_the_lock_is_made_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Export B's cleanup removes the empty staging folder just after export A checked it and
    before A makes its lock: A makes it again and goes on."""
    library = make_library(tmp_path / "lib")
    root = library / "Resolve Projects" / STAGING
    real = place._take_lock
    calls: list[int] = []

    def raced(root_: Path, claim: Any, name: str) -> None:
        calls.append(1)
        if len(calls) == 1:
            assert place._remove_root(root_) is None  # another export's cleanup
            assert not root.exists()
        real(root_, claim, name)

    monkeypatch.setattr(place, "_take_lock", raced)
    placed, _ = place_project(library, "Raced", lambda p: p.write_bytes(b"ok"))
    assert placed.path.read_bytes() == b"ok"
    assert len(calls) == 2
    assert _top(library) == ["Users"]


def test_a_staging_folder_that_keeps_vanishing_is_a_bounded_typed_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    real = place._take_lock
    calls: list[int] = []

    def raced(root_: Path, claim: Any, name: str) -> None:
        calls.append(1)
        place._remove_root(root_)
        real(root_, claim, name)

    monkeypatch.setattr(place, "_take_lock", raced)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Raced", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "library_unwritable"
    assert len(calls) == place._ROOT_TRIES
    assert _top(library) == ["Users"]


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_a_refused_lock_is_tried_again_only_where_a_removed_folder_refuses_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    """On Windows a folder being removed refuses access until it is gone: the lock is tried
    again. Elsewhere a refused lock is a real refusal, reported at once."""
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place.sys, "platform", platform)
    monkeypatch.setattr(place, "_NOREPLACE", [])
    real = place._take_lock
    calls: list[int] = []

    def refused_once(root_: Path, claim: Any, name: str) -> None:
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError(13, "Access is denied", str(root_))
        real(root_, claim, name)

    monkeypatch.setattr(place, "_take_lock", refused_once)
    if platform == "win32":
        placed, _ = place_project(library, "Pending", lambda p: p.write_bytes(b"ok"))
        assert placed.path.read_bytes() == b"ok" and len(calls) == 2
    else:
        with pytest.raises(ProjectWriteError) as info:
            place_project(library, "Pending", lambda p: p.write_bytes(b"ok"))
        assert info.value.code == "library_unwritable" and len(calls) == 1
    assert _top(library) == ["Users"]


def _concurrently(
    library: Path, names: list[str], build: Callable[[Path], None], delays: list[float]
) -> list[Any]:
    barrier = threading.Barrier(len(names))
    results: list[Any] = [None] * len(names)

    def run(i: int) -> None:
        barrier.wait()
        time.sleep(delays[i])
        try:
            results[i] = place_project(library, names[i], build)[0]
        except BaseException as exc:  # collected, asserted below
            results[i] = exc

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(names))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    return results


def test_concurrent_exports_of_different_names_never_fail_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Many rounds of exports of different names started a little apart, with the window
    between checking the staging folder and making the lock widened: one export's cleanup
    often removes the empty staging folder inside another's window (the old code failed it)."""
    library = make_library(tmp_path / "lib")
    real_listdir = place._listdir

    def slow_listdir(folder: Path) -> list[str]:
        out = real_listdir(folder)
        if folder.name == STAGING:
            time.sleep(0.003)
        return out

    def build(path: Path) -> None:
        path.write_bytes(b"ok")

    monkeypatch.setattr(place, "_listdir", slow_listdir)
    rng = random.Random(25)
    failures: list[str] = []
    rounds, width = 60, 4
    for round_ in range(rounds):
        names = [f"R{round_}-{i}" for i in range(width)]
        delays = [rng.uniform(0.0, 0.012) for _ in names]
        for name, result in zip(names, _concurrently(library, names, build, delays), strict=True):
            if not isinstance(result, place.Placed):
                failures.append(f"{name}: {result!r}"[:300])
    assert failures == []
    assert len(os.listdir(_projects(library))) == rounds * width
    assert _top(library) == ["Users"]


# -- 2. the gate tells projects apart by their full path -------------------------------------


def _two_projects(tmp_path: Path) -> list[Path]:
    pytest.importorskip("zstandard")
    clip = mp4_template()
    out = []
    for folder in ("first", "second"):
        items = linked_pair(clip, 0xB000, start=86400, duration=48, src_in=0) + linked_pair(
            clip, 0xB010, start=86448, duration=48, src_in=48, index=1
        )
        path = tmp_path / folder / "Project.db"
        path.parent.mkdir()
        out.append(write_seed(path, clips=[clip], items=items))
    return out


def test_the_cross_project_run_takes_every_template_from_the_other_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    projects = _two_projects(tmp_path)
    assert projects[0].name == projects[1].name == "Project.db"
    pairs: list[tuple[Path, Path]] = []
    real = gate._check

    def spy(group: Any, target: Any, template: Any) -> None:
        pairs.append((target.items.path, template.items.path))
        real(group, target, template)

    monkeypatch.setattr(gate, "_check", spy)
    opts = LibraryOptions(settle_seconds=0.0)
    board = gate.run(projects, options=opts, other_project=True)
    assert board.items_checked == 8 and pairs
    assert all(Path(t).resolve() != Path(s).resolve() for t, s in pairs), pairs
    pairs.clear()
    gate.run(projects, options=opts, other_project=False)
    assert all(Path(t).resolve() == Path(s).resolve() for t, s in pairs), pairs
