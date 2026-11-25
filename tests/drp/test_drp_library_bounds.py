"""S25a fix round 1: what a library read and a library listing may leave out, and their bounds.

Every link the reader does not consume is reported; memory is bounded while rows are fetched;
the listing has one deadline and one byte budget and says what it skipped; names cannot forge
the text output; and the guards a mutation run left untested (the integrity check, the journal
re-check after a copy, the copy's size limits, the truncate between copies, the virtual-table
guard, the listing's caps and the bin-depth guard) each have a test.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sqlite3
import stat
import sys
import tracemalloc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from library_support import OLD, SCHEMA, Tables, find, library_db, library_tree, rows_from_members
from library_support import write_db as _write_db
from test_drp_library import FAST, _Clock, codes, edited, refusal, rich_members
from test_drp_synthetic import members

from aar.cli import main
from aar.drp import library, read_drp
from aar.drp.cli import _library_lines
from aar.drp.describe import describe, quote
from aar.drp.library import (
    LibraryOptions,
    list_library,
)
from aar.media import settle
from aar.media.settle import TooLargeError, copy_settled
from aar.plan.timeline_import import READER_WARNING_CODES

LIBRARY_CODES = (
    "library-dangling-link",
    "library-link-conflict",
    "library-unlinked-row",
    "library-unread-link",
    "library-bad-type",
)


def _projects(root: Path) -> Path:
    return root / "Resolve Projects" / "Users" / "guest" / "Projects"


def _items(drp: Any) -> list[str]:
    return [i.id for t in drp.timelines for tr in t.tracks for i in tr.items]


# ---------------------------------------------------------------------------- 1. unread links


def _move(join: str, child: str, prop: str) -> Any:
    def edit(tables: Tables) -> None:
        row = next(r for r in tables[join] if r["DbAssociate"] == child)
        row["DbPropertyName"] = prop

    return edit


@pytest.mark.parametrize(
    ("join", "child", "prop", "gone"),
    [
        ("Sm2TiItem_Sm2TiTrack", "v1t", "Other", "item"),
        ("Sm2SequenceContainer_Sm2TiTrack", "t2", "Whatever", "track"),
        ("Sm2MpFolder_Sm2MpMedia", "clip2", "OtherVec", "media"),
        ("BtAudioInfo_Sm2MpMedia", "ai1", "OtherAudioVec", "audio"),
        ("SM_Project_Sm2Timeline", "tl2", "OtherHandleVec", "handle"),
    ],
)
def test_a_link_under_a_property_the_reader_does_not_read_is_reported(
    tmp_path: Path, join: str, child: str, prop: str, gone: str
) -> None:
    base = read_drp(library_db(tmp_path / "base.db", rich_members()), library_options=FAST)
    drp = read_drp(edited(tmp_path, rich_members(), _move(join, child, prop)), library_options=FAST)
    [issue] = [i for i in drp.issues if i.code == "library-unread-link" and repr(prop) in i.message]
    assert issue.where is not None and issue.where.member.startswith(f"{join}/")
    if gone == "item":
        assert len(_items(drp)) == len(_items(base)) - 1
    elif gone == "track":
        assert sum(len(t.tracks) for t in drp.timelines) < sum(
            len(t.tracks) for t in base.timelines
        )
    elif gone == "media":
        assert "clip2" not in drp.media_by_id() and "clip2" in base.media_by_id()


def test_a_link_from_an_owner_that_is_not_read_is_reported(tmp_path: Path) -> None:
    def orphan(tables: Tables) -> None:
        tables["Sm2TiItem"].append({**find(tables, "Sm2TiItem", "v1i1"), "Sm2TiItem_id": "lost"})
        find(tables, "Sm2TiItem", "lost")["Sm2TiTrack_id"] = None
        tables["Sm2TiItem_Sm2TiTrack"].append(
            {"DbOwner": "no-track", "DbAssociate": "lost", "DbPropertyName": "Items", "DbIndex": 0}
        )

    drp = read_drp(edited(tmp_path, rich_members(), orphan), library_options=FAST)
    [issue] = [i for i in drp.issues if i.code == "library-unread-link"]
    assert "no-track is not read" in issue.message
    assert "library-unlinked-row" in codes(drp) and "lost" not in _items(drp)


def test_a_row_no_link_places_is_reported_even_without_an_owner(tmp_path: Path) -> None:
    def stray(tables: Tables) -> None:
        tables["Sm2TiItem"].append({**find(tables, "Sm2TiItem", "v1i1"), "Sm2TiItem_id": "stray"})
        find(tables, "Sm2TiItem", "stray")["Sm2TiTrack_id"] = None

    drp = read_drp(edited(tmp_path, rich_members(), stray), library_options=FAST)
    [issue] = [i for i in drp.issues if i.code == "library-unlinked-row"]
    assert "1 Sm2TiItem row(s) are placed by no link (0 of them name an owner" in issue.message


def test_no_library_code_passes_an_import() -> None:
    # A code on the pass-through list would let an import go on with an item missing.
    assert not set(LIBRARY_CODES) & READER_WARNING_CODES


def test_import_refuses_a_timeline_with_an_item_under_an_unread_property(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "plan"))
    from test_timeline_import_cli import CLIPS, call, write_plan
    from timeline_import_support import drt_members, plan_for

    db = library_db(
        tmp_path / "Project.db", drt_members(CLIPS), _move("Sm2TiItem_Sm2TiTrack", "t1i2", "X")
    )
    plan = write_plan(tmp_path / "plan.json", plan_for(CLIPS))
    out = tmp_path / "out"
    out.mkdir()
    code, stdout, stderr = call(out, capsys, db, plan)
    assert code == 2 and "refused (damaged_timeline)" in stdout + stderr
    assert "library-unread-link" in stdout + stderr


# ---------------------------------------------------------------------------- 2. memory


def _with_names(tmp_path: Path, size: int, count: int) -> Path:
    def big(tables: Tables) -> None:
        for row in tables["Sm2TiItem"][:count]:
            row["Name"] = b"n" * size

    return edited(tmp_path, rich_members(), big)


def test_values_are_charged_as_they_are_fetched_not_after(tmp_path: Path) -> None:
    path = _with_names(tmp_path, 1024 * 1024, 8)  # 8 MiB of names
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_load_bytes=2 * 1024 * 1024)
    tracemalloc.start()
    try:
        error = refusal(path, options)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert error.code == "too_large" and "load budget" in error.message
    assert "--max-load-bytes" in error.hint
    assert peak < 5 * 1024 * 1024  # never all 8 MiB: refused within a row of the budget
    assert read_drp(path, library_options=FAST).project is not None  # the defaults read it


def test_one_value_over_max_value_bytes_is_refused_by_sqlite(tmp_path: Path) -> None:
    path = _with_names(tmp_path, 300_000, 1)
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_value_bytes=100_000)
    error = refusal(path, options)
    assert error.code == "too_large" and "--max-value-bytes" in error.hint


def test_rows_past_the_budget_are_refused(tmp_path: Path) -> None:
    def many(tables: Tables) -> None:
        base = find(tables, "Sm2TiItem", "v1i1")
        tables["Sm2TiItem"].extend({**base, "Sm2TiItem_id": f"x{n}"} for n in range(300))

    path = edited(tmp_path, rich_members(), many)
    error = refusal(path, LibraryOptions(settle_seconds=0, deadline_seconds=0, max_rows=200))
    assert error.code == "too_large" and "200-row limit" in error.message


def test_rows_fetched_are_charged_even_if_the_count_is_low(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The count is only an early refusal: the fetch itself is bounded too.
    path = library_db(tmp_path / "Project.db", rich_members())
    real = library._load

    def low_count(conn: sqlite3.Connection, p: Path, opts: LibraryOptions) -> Any:
        class Counting:
            def execute(self, sql: str) -> Any:
                if sql.startswith("SELECT count(*)"):
                    return conn.execute("SELECT 0")
                return conn.execute(sql)

        return real(Counting(), p, opts)  # type: ignore[arg-type]

    monkeypatch.setattr(library, "_load", low_count)
    error = refusal(path, LibraryOptions(settle_seconds=0, deadline_seconds=0, max_rows=10))
    assert error.code == "too_large" and "more than 10 rows" in error.message


# ---------------------------------------------------------------------------- 3 and 4. listing


def _lib(tmp_path: Path, n: int = 2) -> Path:
    return library_tree(tmp_path / "lib", [(f"P{i}", members()) for i in range(n)])


def test_a_listing_has_one_deadline_and_says_what_it_did_not_read(tmp_path: Path) -> None:
    root = _lib(tmp_path, 4)
    for i in range(4):
        (_projects(root) / f"P{i}" / "Project.db-journal").write_bytes(b"")
    fake = _Clock()
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=5, poll_seconds=1, clock=fake.clock, sleep=fake.sleep
    )
    listing = list_library(root, options=options, deadline_seconds=8)
    first, second, third, fourth = listing.projects
    assert first.refusal is not None and first.refusal["code"] == "busy"
    assert second.refusal is None and second.not_read is not None
    assert "deadline passed while it was waited for" in second.not_read
    for p in (third, fourth):
        assert (
            p.refusal is None and p.not_read is not None and "before it was reached" in p.not_read
        )
    assert listing.not_read == 3 and not listing.complete
    assert fake.t - 1000.0 <= 8.5  # the whole listing kept to its deadline, not 4 x 5 s
    text = "\n".join(_library_lines(listing, 1))
    assert "projects: 4 (1 refused, 3 not read)" in text and "INCOMPLETE" in text
    assert '"guest/P3": NOT READ: "the listing\'s 8 s deadline passed before' in text
    assert list_library(root, options=options, deadline_seconds=100).not_read == 0


def test_a_listing_has_a_byte_budget(tmp_path: Path) -> None:
    root = _lib(tmp_path, 3)
    size = (_projects(root) / "P0" / "Project.db").stat().st_size
    listing = list_library(root, options=FAST, max_bytes=size * 2)
    assert [p.not_read is None for p in listing.projects] == [True, True, False]
    assert "byte budget" in (listing.projects[2].not_read or "")
    assert listing.bytes_read == size * 2 and not listing.complete


def test_cli_deadline_cuts_the_listing_short_and_exits_3(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lib(tmp_path, 2)
    assert main(["drp", "library", str(root), "--wait", "0", "--deadline", "0"]) == 3
    text = capsys.readouterr().out
    assert "INCOMPLETE" in text and "search cut short: the listing's deadline passed" in text
    assert main(["drp", "library", str(root), "--deadline", "-1"]) == 2
    capsys.readouterr()
    assert main(["drp", "library", str(root), "--wait", "0"]) == 0


def test_a_symlinked_project_folder_is_not_followed_and_is_listed_as_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lib(tmp_path, 1)
    outside = tmp_path / "outside"
    outside.mkdir()
    library_db(outside / "Project.db", members())
    (_projects(root) / "Link").symlink_to(outside, target_is_directory=True)
    listing = list_library(root, options=FAST)
    assert [p.folder for p in listing.projects] == ["P0"]
    assert listing.skipped["symlink"].count == 1
    assert listing.skipped["symlink"].first == ["guest/Projects/Link"]
    assert not listing.complete
    assert main(["drp", "library", str(root), "--wait", "0"]) == 3
    text = capsys.readouterr().out
    assert 'skipped 1 symbolic link(s), never followed: "guest/Projects/Link"' in text
    assert main(["drp", "library", str(root), "--wait", "0", "--json"]) == 3
    data = json.loads(capsys.readouterr().out)
    assert data["complete"] is False and data["skipped"]["symlink"]["count"] == 1


def test_a_symlinked_projects_folder_or_user_is_skipped(tmp_path: Path) -> None:
    root = _lib(tmp_path, 1)
    users = root / "Resolve Projects" / "Users"
    (users / "linked").symlink_to(users / "guest", target_is_directory=True)
    (users / "other").mkdir()
    (users / "other" / "Projects").symlink_to(_projects(root), target_is_directory=True)
    listing = list_library(root, options=FAST)
    assert [(p.user, p.folder) for p in listing.projects] == [("guest", "P0")]
    assert listing.skipped["symlink"].first == ["linked", "other/Projects"]


def test_a_flood_of_skipped_entries_is_counted_and_the_first_few_shown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lib(tmp_path, 1)
    for i in range(30):
        (_projects(root) / f"L{i:02d}").symlink_to(tmp_path, target_is_directory=True)
    listing = list_library(root, options=FAST)
    assert listing.skipped["symlink"].count == 30 and len(listing.skipped["symlink"].first) == 10
    assert main(["drp", "library", str(root), "--wait", "0"]) == 3
    assert "skipped 30 symbolic link(s)" in capsys.readouterr().out


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a non-root POSIX user")
def test_an_unreadable_folder_is_listed_as_skipped(tmp_path: Path) -> None:
    root = _lib(tmp_path, 1)
    locked = _projects(root) / "Locked"
    (locked / "Inner").mkdir(parents=True)
    locked.chmod(0)
    try:
        listing = list_library(root, options=FAST)
    finally:
        locked.chmod(stat.S_IRWXU)
    assert listing.skipped["unreadable"].count == 1
    assert listing.skipped["unreadable"].first[0].startswith("guest/Projects/Locked (")
    assert not listing.complete


def test_a_user_without_a_projects_folder_is_listed_as_skipped(tmp_path: Path) -> None:
    root = _lib(tmp_path, 1)
    (root / "Resolve Projects" / "Users" / "empty").mkdir()
    listing = list_library(root, options=FAST)
    assert listing.skipped["unreadable"].first[0].startswith("empty/Projects (")


def test_folders_past_the_depth_limit_are_listed_as_skipped(tmp_path: Path) -> None:
    root = _lib(tmp_path, 0)
    deep = _projects(root)
    for n in range(library._MAX_FOLDER_DEPTH):
        deep = deep / f"d{n}"
    deep.mkdir(parents=True)
    library_db(deep / "Project.db", members())  # 8 levels down: listed
    other = deep.parent / "z" / "too-deep"  # 9 levels down: not looked into
    other.mkdir(parents=True)
    library_db(other / "Project.db", members())
    listing = list_library(root, options=FAST)
    assert [p.folder.count("/") + 1 for p in listing.projects] == [library._MAX_FOLDER_DEPTH]
    assert listing.skipped["too_deep"].count == 1
    assert listing.skipped["too_deep"].first[0].endswith("/z/too-deep")


def test_the_project_cap_cuts_the_listing_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _lib(tmp_path, 3)
    monkeypatch.setattr(library, "_MAX_LISTED_PROJECTS", 3)
    assert list_library(root, options=FAST).complete
    monkeypatch.setattr(library, "_MAX_LISTED_PROJECTS", 2)
    listing = list_library(root, options=FAST)
    assert len(listing.projects) == 2 and listing.cut_short == ["projects"]
    assert not listing.complete
    assert main(["drp", "library", str(root), "--wait", "0"]) == 3
    assert "search cut short: more than" in capsys.readouterr().out


def test_the_entry_cap_counts_the_users_folder_and_cuts_the_listing_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _lib(tmp_path, 3)
    users = root / "Resolve Projects" / "Users"
    for i in range(3):
        (users / f"file{i}").write_bytes(b"")
    # Users: 4 entries; Projects: 3; each project folder: 1 (Project.db).
    monkeypatch.setattr(library, "_MAX_SCANNED_ENTRIES", 10)
    assert list_library(root, options=FAST).complete
    monkeypatch.setattr(library, "_MAX_SCANNED_ENTRIES", 9)
    listing = list_library(root, options=FAST)
    assert listing.cut_short == ["entries"] and not listing.complete
    monkeypatch.setattr(library, "_MAX_SCANNED_ENTRIES", 3)
    listing = list_library(root, options=FAST)
    assert listing.cut_short == ["entries"] and listing.projects == []


# ---------------------------------------------------------------------------- 5. forged names


def test_quotes_in_names_cannot_forge_fields_of_a_listing_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    forged = 'A", id "evil", timelines: "fake'

    def name(tables: Tables) -> None:
        tables["SM_Project"][0]["ProjectName"] = forged
        find(tables, "Sm2Timeline", "tl1")["Name"] = 'T"1'

    base = _projects(tmp_path / "lib")
    folder = base / 'F"x'
    folder.mkdir(parents=True)
    library_db(folder / "Project.db", rich_members(), name)
    main(["drp", "library", str(tmp_path / "lib"), "--wait", "0"])
    [line] = [x for x in capsys.readouterr().out.splitlines() if x.startswith("  ")]
    folder_name = 'guest/F"x'
    assert line.startswith(f"  {quote(folder_name)}: {quote(forged)} version 15")
    assert quote('T"1') in line and line.count('"evil"') == 0
    assert quote(forged) == '"A\\", id \\"evil\\", timelines: \\"fake"'


def test_quotes_in_names_are_escaped_by_describe_too(tmp_path: Path) -> None:
    def name(tables: Tables) -> None:
        tables["SM_Project"][0]["ProjectName"] = 'P"x'
        find(tables, "Sm2Timeline", "tl1")["Name"] = 'T"\\1'

    text = describe(read_drp(edited(tmp_path, rich_members(), name), library_options=FAST))
    assert 'project "P\\"x";' in text and 'Timeline "T\\"\\\\1":' in text


# ---------------------------------------------------------------------------- 6. not a file


def test_a_folder_named_project_db_is_refused_as_a_folder(tmp_path: Path) -> None:
    (tmp_path / "Project.db").mkdir()
    error = refusal(tmp_path / "Project.db")
    assert error.code == "unreadable" and "is a folder, not a regular file" in error.message
    assert "link" not in error.hint


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs named pipes")
def test_a_fifo_named_project_db_is_refused_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "Project.db")
    error = refusal(tmp_path / "Project.db")
    assert error.code == "unreadable" and "is a named pipe (FIFO)" in error.message


def test_a_symlink_named_project_db_says_it_is_a_link(tmp_path: Path) -> None:
    db = library_db(tmp_path / "real.db", rich_members())
    (tmp_path / "Project.db").symlink_to(db)
    error = refusal(tmp_path / "Project.db")
    assert "is a symbolic link" in error.message and "not a link to it" in error.hint


# ---------------------------------------------------------------------------- mutation guards


def test_damage_only_in_a_part_the_reader_never_queries_is_still_refused(tmp_path: Path) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE pad(x)")
    conn.executemany("INSERT INTO pad VALUES (?)", [(f"v{i}",) for i in range(50)])
    conn.execute("CREATE INDEX padx ON pad(x)")
    conn.commit()
    (root,) = conn.execute("SELECT rootpage FROM sqlite_master WHERE name = 'padx'").fetchone()
    (page,) = conn.execute("PRAGMA page_size").fetchone()
    conn.close()
    data = bytearray(db.read_bytes())
    end = root * page  # the index's only page: rename keys inside it, nothing else
    data[end - 200 : end] = data[end - 200 : end].replace(b"v1", b"w1")
    db.write_bytes(bytes(data))
    os.utime(db, (OLD, OLD))
    error = refusal(db)
    assert error.code == "damaged" and "missing from index padx" in error.message


def test_a_journal_that_appears_during_the_copy_means_copy_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    db = library_db(project / "Project.db", rich_members())
    journal = project / "Project.db-journal"
    real_open = settle.open_regular
    opened = {"n": 0}

    @contextlib.contextmanager
    def open_and_start_saving(path: Path, *, dir_fd: int | None = None) -> Iterator[Any]:
        with real_open(path, dir_fd=dir_fd) as stream:
            opened["n"] += 1
            if opened["n"] == 1:
                journal.write_bytes(b"")  # Resolve starts a save while the copy runs
            yield stream

    monkeypatch.setattr(settle, "open_regular", open_and_start_saving)
    snaps = tmp_path / "snaps"
    snaps.mkdir()
    error = refusal(db, LibraryOptions(settle_seconds=0, deadline_seconds=0, snapshot_dir=snaps))
    assert error.code == "busy" and "appeared while it was copied" in error.message
    assert list(snaps.iterdir()) == []
    journal.unlink()
    opened["n"] = 0
    fake = _Clock(lambda n: journal.unlink())  # the save ends; the next copy is clean
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=5, poll_seconds=1, clock=fake.clock, sleep=fake.sleep
    )
    assert read_drp(db, library_options=options).project is not None
    assert (opened["n"], fake.sleeps) == (2, 1)


def test_a_shorter_second_copy_does_not_keep_the_first_copys_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = library_db(tmp_path / "Project.db", rich_members())
    good = db.read_bytes()
    calls = {"n": 0}

    def flaky_copy(path: Path, target: Any, **kw: Any) -> settle.Copied | None:
        calls["n"] += 1
        if calls["n"] == 1:
            target.write(b"\xee" * (len(good) + 50_000))  # a longer copy, then a change
            return None
        return copy_settled(path, target, **kw)

    monkeypatch.setattr(library, "copy_settled", flaky_copy)
    fake = _Clock()
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=5, poll_seconds=1, clock=fake.clock, sleep=fake.sleep
    )
    snapshot, copied = library._take_snapshot(db, options)
    try:
        assert snapshot.read_bytes() == good and copied.size == len(good)
    finally:
        snapshot.unlink()


def test_a_virtual_table_standing_in_for_a_table_is_not_a_project(tmp_path: Path) -> None:
    schema = {t: c for t, c in SCHEMA.items() if t != "Sm2TiItem"}
    path = _write_db(tmp_path / "Project.db", rows_from_members(rich_members()), schema=schema)
    conn = sqlite3.connect(path)
    cols = ", ".join(f'"{c}"' for c in SCHEMA["Sm2TiItem"])
    conn.execute(f'CREATE VIRTUAL TABLE "Sm2TiItem" USING fts5({cols})')
    conn.commit()
    conn.close()
    os.utime(path, (OLD, OLD))
    error = refusal(path)
    assert error.code == "not_a_project" and "it has no Sm2TiItem" in error.message


# ---------------------------------------------------------------------------- copy size limits


def _growing(real_open: Any, extra: int) -> Any:
    """``open_regular`` whose file grows by ``extra`` bytes after the copy's first read (after
    the size check at open)."""

    @contextlib.contextmanager
    def open_and_grow(p: Path, *, dir_fd: int | None = None) -> Iterator[Any]:
        with real_open(p, dir_fd=dir_fd) as stream:
            read = stream.read
            grown = {"done": False}

            def read_then_grow(size: int = -1) -> bytes:
                data = read(size)
                if not grown["done"]:
                    grown["done"] = True
                    with p.open("ab") as writer:
                        writer.write(b"\x00" * extra)
                return data

            stream.read = read_then_grow
            yield stream

    return open_and_grow


def _old_file(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    os.utime(path, (OLD, OLD))
    return path


def test_copy_settled_refuses_a_file_over_its_limit_before_copying(tmp_path: Path) -> None:
    path = _old_file(tmp_path / "f", b"x" * 100)
    target = io.BytesIO()
    with pytest.raises(TooLargeError, match="exceeds 99 bytes"):
        copy_settled(path, target, max_bytes=99, settle_seconds=0)
    assert target.getvalue() == b""
    assert copy_settled(path, io.BytesIO(), max_bytes=100, settle_seconds=0) is not None


def test_copy_settled_refuses_a_file_that_grows_past_its_limit_while_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _old_file(tmp_path / "f", b"x" * 100)
    real_open = settle.open_regular

    monkeypatch.setattr(settle, "open_regular", _growing(real_open, 100))
    with pytest.raises(TooLargeError, match="grew past 150 bytes"):
        copy_settled(path, io.BytesIO(), max_bytes=150, settle_seconds=0)


def test_a_library_file_over_or_growing_past_max_db_bytes_is_too_large(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "proj").mkdir()
    db = library_db(tmp_path / "proj" / "Project.db", rich_members())
    size = db.stat().st_size
    snaps = tmp_path / "snaps"
    snaps.mkdir()

    def options(limit: int) -> LibraryOptions:
        return LibraryOptions(
            settle_seconds=0, deadline_seconds=0, max_db_bytes=limit, snapshot_dir=snaps
        )

    error = refusal(db, options(size - 1))
    assert error.code == "too_large" and "exceeds" in error.message
    real_open = settle.open_regular

    monkeypatch.setattr(settle, "open_regular", _growing(real_open, 4096))
    error = refusal(db, options(size + 1000))
    assert error.code == "too_large" and "grew past" in error.message
    assert list(snaps.iterdir()) == []
