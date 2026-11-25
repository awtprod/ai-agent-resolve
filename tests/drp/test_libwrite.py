"""S25c2: the library writer: seeds, the closed list of operations, the read-back and placement.

Hostile inputs first: seeds that are not projects or not of the expected shape, names that are
not folder names, libraries that are read-only or vanish, a crash between staging and
placement. Every operation has a test that fails if it changes."""

from __future__ import annotations

import os
import sqlite3
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from libwrite_support import (
    AUDIO_TRACK,
    PROJECT,
    TIMELINE,
    TIMELINE_CLIP,
    VIDEO_TRACK,
    item_row,
    linked_pair,
    make_library,
    mp4_template,
    seed_tables,
    write_seed,
    write_tables,
)
from mediapool_support import uid

from aar.drp.items.rows import read_items
from aar.drp.library import LibraryOptions
from aar.drp.libwrite import (
    OPENED,
    OPERATIONS,
    Authored,
    ProjectWriteError,
    check_project_name,
    make_project,
    place_project,
    read_seed,
)
from aar.drp.libwrite.cut import Cut, author_cut, pick_item_templates
from aar.drp.libwrite.place import STAGING
from aar.drp.libwrite.write import ItemSpec, sequence_length
from aar.plan.timeline_import import READER_WARNING_CODES

pytest.importorskip("zstandard")

OPTS = LibraryOptions(settle_seconds=0.0)
CLIP = mp4_template()
RATE = 24000 / 1001
POSIX = sys.platform != "win32"
ROOT = POSIX and os.geteuid() == 0


def _seed(tmp: Path, name: str = "seed.db", **kwargs: Any) -> Path:
    kwargs.setdefault("clips", [CLIP])
    kwargs.setdefault("items", linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0))
    return write_seed(tmp / name, **kwargs)


def _cut(seed: Path, cuts_of: Any = None) -> Any:
    picks = pick_item_templates([read_items(seed, options=OPTS)])

    def author(s: Any) -> Authored:
        clip = s.clips[0]
        cuts = cuts_of(clip) if cuts_of else [Cut(clip, 0, 24, 0), Cut(clip, 24, 24, 100)]
        items, specs = author_cut(s, cuts, picks)
        return Authored([], items, specs)

    return author


def _make(tmp: Path, seed: Path, name: str = "New cut", **kwargs: Any) -> Any:
    library = tmp / "lib"
    if not library.exists():
        make_library(library)
    if "author" not in kwargs:
        kwargs["author"] = _cut(seed)
    kwargs.setdefault("allow_unopened", True)
    return make_project(
        library,
        seed,
        project_name=name,
        timeline_name=kwargs.pop("timeline_name", None),
        allowed_issues=READER_WARNING_CODES,
        options=OPTS,
        **kwargs,
    )


def _rows(path: Path, sql: str, *args: Any) -> list[tuple[Any, ...]]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# -- seeds -------------------------------------------------------------------------------------


def test_a_file_that_is_not_a_project_is_not_a_seed(tmp_path: Path) -> None:
    bad = tmp_path / "x.db"
    bad.write_bytes(b"not sqlite" * 50)
    with pytest.raises(ProjectWriteError) as info:
        read_seed(bad, options=OPTS)
    assert info.value.code == "bad_seed"
    assert info.value.details["library_code"] == "not_a_project"


def test_a_seed_with_two_timelines_is_refused(tmp_path: Path) -> None:
    tables = seed_tables(clips=[CLIP])
    tables["Sm2Timeline"].append({**tables["Sm2Timeline"][0], "Sm2Timeline_id": uid(0xA0F0)})
    path = write_tables(tmp_path / "two.db", tables)
    with pytest.raises(ProjectWriteError, match="2 timelines"):
        read_seed(path, options=OPTS)


def test_a_seed_with_an_extra_track_is_refused(tmp_path: Path) -> None:
    path = write_seed(tmp_path / "extra.db", clips=[CLIP], extra_tracks=1)
    with pytest.raises(ProjectWriteError, match="one video and one audio"):
        read_seed(path, options=OPTS)


def test_a_seed_of_another_project_version_is_refused(tmp_path: Path) -> None:
    path = write_seed(tmp_path / "v15.db", clips=[CLIP], version=15)
    with pytest.raises(ProjectWriteError, match="version is 15"):
        read_seed(path, options=OPTS)


@pytest.mark.parametrize(
    ("column", "value"),
    [("pLmVerTable", uid(0xF001)), ("EffectFiltersBA", b"\x01"), ("MarkersBA", b"")],
)
def test_a_seed_item_carrying_later_work_is_not_removed(
    tmp_path: Path, column: str, value: object
) -> None:
    items = linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0)
    items[0] = ({**items[0][0], column: value}, items[0][1])
    path = write_seed(tmp_path / "busy.db", clips=[CLIP], items=items)
    with pytest.raises(ProjectWriteError, match="cannot remove"):
        read_seed(path, options=OPTS)


def test_a_seed_item_named_elsewhere_is_not_removed(tmp_path: Path) -> None:
    tables = seed_tables(
        clips=[CLIP], items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0)
    )
    tables["Sm2Sequence"][0]["FieldsBlob"] = uid(0xB000).encode("utf-16-be")
    path = write_tables(tmp_path / "named.db", tables)
    with pytest.raises(ProjectWriteError, match="named in Sm2Sequence"):
        read_seed(path, options=OPTS)


def test_a_plain_seed_is_read_with_its_settings(tmp_path: Path) -> None:
    seed = read_seed(_seed(tmp_path), options=OPTS)
    assert (seed.timeline_id, seed.video_track, seed.audio_track) == (
        TIMELINE,
        VIDEO_TRACK,
        AUDIO_TRACK,
    )
    assert (seed.start_frame, seed.resolution, seed.resolution_from) == (
        86400,
        (3840, 2160),
        "timeline",
    )
    assert seed.items == (uid(0xB000), uid(0xB001))
    assert seed.folder_count == 2  # the timeline's clip and the template clip
    assert PROJECT in seed.taken and TIMELINE_CLIP in seed.taken


def test_a_timeline_following_the_project_takes_the_stored_project_resolution(
    tmp_path: Path,
) -> None:
    path = _seed(tmp_path, resolution=(0, 0))
    seed = read_seed(path, options=OPTS)
    assert (seed.resolution, seed.resolution_from) == (None, None)


# -- operations ----------------------------------------------------------------------------------


def test_the_operation_list_is_closed_and_its_statuses_are_stable() -> None:
    assert {n: op.status == OPENED for n, op in OPERATIONS.items()} == {
        "new_project_identity": True,
        "rename_timeline": False,
        "remove_items": True,
        "add_media": False,
        "add_linked_items": True,
        "add_picture_only_items": False,
        "add_sound_only_items": False,
        "leave_gaps": False,
        "split_edit": False,
        "set_extents": True,
    }


def test_a_project_of_opened_operations_only_needs_no_flag(tmp_path: Path) -> None:
    result = _make(tmp_path, _seed(tmp_path), allow_unopened=False)
    assert list(result.built.operations) == [
        "new_project_identity",
        "remove_items",
        "add_linked_items",
        "set_extents",
    ]
    assert all(op["status"] == OPENED for op in result.operations())


@pytest.mark.parametrize(
    ("cuts_of", "needs"),
    [
        (lambda c: [Cut(c, 0, 24, 0, sound=False)], "add_picture_only_items"),
        (lambda c: [Cut(c, 0, 24, 0, picture=False)], "add_sound_only_items"),
        (lambda c: [Cut(c, 5, 24, 0)], "leave_gaps"),
        (
            lambda c: [Cut(c, 4, 20, 4, sound_record_in=0, sound_src_in=0, sound_duration=24)],
            "split_edit",
        ),
    ],
)
def test_an_unopened_operation_is_refused_without_the_flag(
    tmp_path: Path, cuts_of: Any, needs: str
) -> None:
    seed = _seed(tmp_path)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, author=_cut(seed, cuts_of), allow_unopened=False)
    assert info.value.code == "operation_not_opened"
    assert needs in info.value.details["operations"]
    assert not any(
        (tmp_path / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects").iterdir()
    )
    assert not (tmp_path / "lib" / "Resolve Projects" / STAGING).exists()
    result = _make(tmp_path, seed, author=_cut(seed, cuts_of))
    assert needs in result.built.operations


def test_new_project_identity_replaces_every_text_copy_of_the_project_id(tmp_path: Path) -> None:
    result = _make(tmp_path, _seed(tmp_path), name="Fresh")
    path = result.placed.path
    new = result.built.project_id
    assert _rows(path, "SELECT SM_Project_id, ProjectName FROM SM_Project") == [(new, "Fresh")]
    for table in ("SM_Config", "BtLockableBlobMap"):
        assert _rows(path, f'SELECT SM_Project_id FROM "{table}"') == [(new,)]
    assert _rows(path, "SELECT DbOwner FROM SM_Project_Sm2Timeline") == [(new,)]
    conn = sqlite3.connect(path)
    try:
        dump = "\n".join(conn.iterdump())
    finally:
        conn.close()
    assert PROJECT not in dump


def test_rename_timeline_renames_the_timeline_and_its_pool_clip(tmp_path: Path) -> None:
    result = _make(tmp_path, _seed(tmp_path), timeline_name="Cut two")
    path = result.placed.path
    assert _rows(path, "SELECT Name FROM Sm2Timeline") == [("Cut two",)]
    assert _rows(path, "SELECT Name FROM Sm2MpMedia WHERE Sm2MpMedia_id = ?", TIMELINE_CLIP) == [
        ("Cut two",)
    ]
    assert "rename_timeline" in result.built.operations


def test_remove_items_deletes_the_seed_items_and_their_join_rows(tmp_path: Path) -> None:
    path = _make(tmp_path, _seed(tmp_path)).placed.path
    ids = {r[0] for r in _rows(path, "SELECT Sm2TiItem_id FROM Sm2TiItem")}
    assert uid(0xB000) not in ids and uid(0xB001) not in ids and len(ids) == 4
    joined = {r[0] for r in _rows(path, "SELECT DbAssociate FROM Sm2TiItem_Sm2TiTrack")}
    assert joined == ids


def test_set_extents_is_the_last_end_over_the_rate_minus_the_start() -> None:
    start = 86400 / RATE
    assert sequence_length(RATE, start, [86448, 86460]) == 86460 / RATE - start
    assert sequence_length(RATE, start, []) == 0.0
    # Resolve's form and the frame count over the rate differ in the last bits here.
    assert 86460 / RATE - start != 60 / RATE


def test_the_written_extents_follow_the_items(tmp_path: Path) -> None:
    path = _make(tmp_path, _seed(tmp_path)).placed.path
    (extents,) = _rows(path, "SELECT MediaExtents FROM Sm2Sequence")[0]
    assert struct.unpack("<dd", extents) == (86400 / RATE, 86448 / RATE - 86400 / RATE)


def test_no_journal_is_left_and_the_file_stays_in_rollback_mode(tmp_path: Path) -> None:
    path = _make(tmp_path, _seed(tmp_path)).placed.path
    assert sorted(os.listdir(path.parent)) == ["Project.db"]
    assert _rows(path, "PRAGMA journal_mode") == [("delete",)]


def test_the_seed_is_never_changed(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    before = seed.read_bytes()
    _make(tmp_path, seed)
    assert seed.read_bytes() == before


# -- read-back -----------------------------------------------------------------------------------


def test_a_project_that_does_not_read_back_as_written_is_never_placed(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    picks = pick_item_templates([read_items(seed, options=OPTS)])

    def author(s: Any) -> Authored:
        items, specs = author_cut(s, [Cut(s.clips[0], 0, 24, 0)], picks)
        wrong = [ItemSpec(x.track, x.record_in, x.record_out + 1, x.src_in, x.path) for x in specs]
        return Authored([], items, wrong)

    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, author=author)
    assert info.value.code == "self_check_failed"
    projects = tmp_path / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(projects.iterdir()) == []


# -- placement -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "",
        "a:b",
        "CON",
        "trailing.",
        "trailing ",
        "a\\b",
        "a/b",
        "..",
        ".hidden",
        "x​y",
        "a\tb",
        "x" * 201,
    ],
)
def test_a_name_that_is_not_a_folder_name_on_the_share_is_refused(name: str) -> None:
    with pytest.raises(ProjectWriteError) as info:
        check_project_name(name)
    assert info.value.code == "bad_project_name"


def test_an_existing_project_of_that_name_in_any_case_is_refused(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    existing = library / "Resolve Projects" / "Users" / "guest" / "Projects" / "My Cut"
    existing.mkdir()
    (existing / "Project.db").write_bytes(b"theirs")
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, _seed(tmp_path), name="MY CUT")
    assert info.value.code == "project_exists"
    assert (existing / "Project.db").read_bytes() == b"theirs"


def test_a_folder_that_is_not_a_library_is_refused(tmp_path: Path) -> None:
    (tmp_path / "lib").mkdir()
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, _seed(tmp_path))
    assert info.value.code == "not_a_library"


def test_two_users_need_the_user_named(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    (library / "Resolve Projects" / "Users" / "other" / "Projects").mkdir(parents=True)
    seed = _seed(tmp_path)
    with pytest.raises(ProjectWriteError, match="2 users"):
        _make(tmp_path, seed)
    result = _make(tmp_path, seed, user="other")
    assert result.placed.user == "other"


@pytest.mark.skipif(not POSIX or ROOT, reason="needs POSIX permissions and a non-root user")
def test_a_read_only_library_is_refused_and_left_as_it_was(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    top = library / "Resolve Projects"
    top.chmod(0o555)
    try:
        with pytest.raises(ProjectWriteError) as info:
            _make(tmp_path, _seed(tmp_path))
        assert info.value.code == "library_unwritable"
        assert sorted(os.listdir(top)) == ["Users"]
    finally:
        top.chmod(0o755)


def test_a_library_that_disappears_mid_write_leaves_nothing_behind(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    inner = _cut(seed)
    projects = tmp_path / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects"

    def author(s: Any) -> Authored:
        out = inner(s)
        projects.rmdir()
        return out

    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, author=author)
    assert info.value.code == "library_unwritable"
    assert sorted(os.listdir(tmp_path / "lib" / "Resolve Projects")) == ["Users"]


def test_an_interrupted_build_places_nothing_and_removes_its_staging(tmp_path: Path) -> None:
    seed = _seed(tmp_path)

    def author(s: Any) -> Authored:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _make(tmp_path, seed, author=author)
    top = tmp_path / "lib" / "Resolve Projects"
    assert sorted(os.listdir(top)) == ["Users"]
    assert list((top / "Users" / "guest" / "Projects").iterdir()) == []


def test_a_killed_build_is_reported_by_the_next_placement(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    code = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "from aar.drp.libwrite.place import place_project\n"
        "def build(path):\n"
        "    path.write_bytes(b'half')\n"
        "    os._exit(9)\n"
        f"place_project(Path({str(library)!r}), 'Killed', build)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], check=False)
    assert done.returncode == 9
    projects = library / "Resolve Projects" / "Users" / "guest" / "Projects"
    assert list(projects.iterdir()) == []  # nothing half-placed
    placed, _ = place_project(library, "Next", lambda p: p.write_bytes(b"ok"))
    assert len(placed.leftovers) == 1 and placed.leftovers[0].startswith("build-")
    assert sorted(p.name for p in projects.iterdir()) == ["Next"]


@pytest.mark.skipif(not POSIX, reason="POSIX modes")
def test_a_new_project_takes_the_mode_of_the_library_projects(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    projects = library / "Resolve Projects" / "Users" / "guest" / "Projects"
    sibling = projects / "Theirs"
    sibling.mkdir()
    (sibling / "Project.db").write_bytes(b"x")
    sibling.chmod(0o750)
    (sibling / "Project.db").chmod(0o640)
    placed, _ = place_project(library, "Mine", lambda p: p.write_bytes(b"ok"))
    assert oct(os.stat(placed.folder).st_mode & 0o7777) == oct(0o750)
    assert oct(os.stat(placed.path).st_mode & 0o7777) == oct(0o640)
    assert placed.permissions[0] == "mode 750 on the folder and 640 on Project.db"


def test_placement_never_writes_into_an_existing_folder(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    projects = library / "Resolve Projects" / "Users" / "guest" / "Projects"

    def build(path: Path) -> None:
        path.write_bytes(b"ok")
        (projects / "Race").mkdir()
        (projects / "Race" / "Project.db").write_bytes(b"theirs")

    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Race", build)
    assert info.value.code == "project_exists"
    assert (projects / "Race" / "Project.db").read_bytes() == b"theirs"
    assert sorted(os.listdir(library / "Resolve Projects")) == ["Users"]


# -- fuzz ----------------------------------------------------------------------------------------


def _damage(tables: dict[str, list[dict[str, Any]]], rng: Any) -> None:
    table = rng.choice([t for t, rows in tables.items() if rows])
    row = rng.choice(tables[table])
    column = rng.choice(list(row))
    value = row[column]
    if isinstance(value, bytes) and value:
        data = bytearray(value)
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
        row[column] = bytes(data)
    else:
        row[column] = rng.choice([None, 0, -1, "", "x", b"\x00", 2**40, uid(rng.randrange(9))])


@pytest.mark.parametrize("seed_number", range(60))
def test_a_damaged_seed_gives_a_typed_refusal_or_a_checked_project(
    tmp_path: Path, seed_number: int
) -> None:
    import random

    rng = random.Random(seed_number)
    tables = seed_tables(
        clips=[CLIP], items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0)
    )
    for _ in range(rng.randint(1, 3)):
        _damage(tables, rng)
    seed = write_tables(tmp_path / "damaged.db", tables)
    template = _seed(tmp_path, "template.db")
    try:
        result = _make(tmp_path, seed, author=_cut(template))
    except ProjectWriteError as exc:
        assert exc.code in ("bad_seed", "bad_input", "self_check_failed")
        top = tmp_path / "lib" / "Resolve Projects"
        assert sorted(os.listdir(top)) == ["Users"]
        return
    except Exception as exc:  # an item or media refusal of the caller's authoring
        from aar.drp.mediapool.errors import MediaPoolError

        assert isinstance(exc, MediaPoolError), repr(exc)
        return
    assert result.placed.path.is_file()
    assert result.drp.issues == [] or all(i.code in READER_WARNING_CODES for i in result.drp.issues)


def test_a_project_name_with_a_reserved_name_after_a_dot_is_refused() -> None:
    with pytest.raises(ProjectWriteError):
        check_project_name("nul.txt")


def test_item_row_support_matches_the_reader(tmp_path: Path) -> None:
    """The support module's rows read back as the items they describe (a guard on the fixture
    itself, so the tests above test the writer and not a broken fixture)."""
    seed = _seed(tmp_path)
    items = read_items(seed, options=OPTS)
    assert sorted(r.kind for r in items.items) == ["picture", "sound"]
    row = item_row("picture", uid(1), VIDEO_TRACK, CLIP, start=1, duration=2, src_in=0)
    assert row["In"] is None
