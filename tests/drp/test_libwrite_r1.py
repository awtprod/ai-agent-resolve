"""S25c2 fix round 1: the validator's findings on the library writer, each with a test that
fails without its fix (items 1-5 and 7; item 10's surviving mutants are pinned here too).

Every value is made up (``libwrite_support``, ``mediapool_support``)."""

from __future__ import annotations

import dataclasses
import errno
import importlib
import inspect
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest
from libwrite_support import (
    PROJECT,
    make_library,
    mp4_template,
    seed_tables,
    write_seed,
    write_tables,
)
from mediapool_support import png_template, uid
from seed_fuzz import damaged_seed
from test_libwrite import CLIP, OPTS, POSIX, _cut, _make, _seed

import aar.drp.libwrite.cut as cut_module
import aar.drp.libwrite.place as place
import aar.drp.libwrite.write as write_module
from aar.drp.items import ItemError, ItemPlacement, author_item
from aar.drp.items.rows import read_items
from aar.drp.library import LibraryError, open_library, read_rows
from aar.drp.libwrite import (
    Authored,
    ProjectWriteError,
    check_project_name,
    check_written,
    make_project,
    place_project,
    read_seed,
)
from aar.drp.libwrite.cut import Cut, author_cut, pick_item_templates
from aar.drp.libwrite.place import STAGING, fold_name
from aar.drp.libwrite.write import build_project
from aar.drp.mediapool.errors import MediaPoolError
from aar.plan.timeline_import import READER_WARNING_CODES

pytest.importorskip("zstandard")


def _projects(tmp: Path) -> Path:
    return tmp / "lib" / "Resolve Projects" / "Users" / "guest" / "Projects"


def _top(tmp: Path) -> list[str]:
    return sorted(os.listdir(tmp / "lib" / "Resolve Projects"))


def _symlink(target: Path, link: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this system cannot make symbolic links")


# -- 1. a hostile seed gives a typed refusal, never a traceback ---------------------------------


def _schema_text_damaged(path: Path, table: str) -> None:
    """Give the third byte of ``table``'s stored CREATE statement its top bit: SQLite then
    reports a syntax error quoting a byte that is not UTF-8."""
    data = bytearray(path.read_bytes())
    at = data.index(f'CREATE TABLE "{table}" ('.encode()) + 2
    data[at] |= 0x80
    path.write_bytes(bytes(data))


def test_a_seed_whose_schema_text_is_not_utf8_is_a_library_refusal(tmp_path: Path) -> None:
    path = write_seed(tmp_path / "seed.db", clips=[CLIP])
    _schema_text_damaged(path, "Sm2Sequence")
    for read in (
        lambda: read_rows(path, ("SM_Project",), options=OPTS),
        lambda: open_library(path, options=OPTS),
    ):
        with pytest.raises(LibraryError) as info:
            read()
        assert info.value.code == "damaged" and "not UTF-8" in info.value.message
    with pytest.raises(ProjectWriteError) as refused:
        read_seed(path, options=OPTS)
    assert refused.value.code == "bad_seed"
    assert refused.value.details["library_code"] == "damaged"


def test_a_fixed_layout_value_of_another_type_is_bad_seed(tmp_path: Path) -> None:
    tables = seed_tables(clips=[CLIP])
    tables["Sm2Sequence"][0]["Resolution"] = 7
    with pytest.raises(ProjectWriteError) as info:
        read_seed(write_tables(tmp_path / "seed.db", tables), options=OPTS)
    assert info.value.code == "bad_seed" and "TypeError" in info.value.message


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("Sm2SequenceContainer_Sm2TiTrack", "DbAssociate"),
        ("BtLockableBlobMap", "BtLockableBlobMap_id"),
        ("SM_Project", "ProjectName"),
        ("Sm2TiItem", "MediaFilePath"),
        ("SM_Project", "SM_Project_id"),
    ],
)
def test_a_seed_with_a_column_renamed_is_bad_seed(tmp_path: Path, table: str, column: str) -> None:
    path = _seed(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute(f'ALTER TABLE "{table}" RENAME COLUMN "{column}" TO "{column}_old"')
    conn.commit()
    conn.close()
    with pytest.raises(ProjectWriteError) as info:
        read_seed(path, options=OPTS)
    assert info.value.code == "bad_seed" and column in info.value.message


@pytest.mark.parametrize("column", ["SM_Project_id", "ProjectVersion"])
def test_a_project_table_without_its_id_or_version_is_not_a_project(
    tmp_path: Path, column: str
) -> None:
    path = _seed(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute(f'ALTER TABLE "SM_Project" RENAME COLUMN "{column}" TO "{column}_old"')
    conn.commit()
    conn.close()
    with pytest.raises(LibraryError) as info:
        read_rows(path, ("SM_Project",), options=OPTS)
    assert info.value.code == "not_a_project" and column in info.value.message


def test_a_seed_with_a_trigger_resolve_does_not_make_is_refused(tmp_path: Path) -> None:
    path = _seed(tmp_path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TRIGGER planted AFTER UPDATE ON SM_Project BEGIN "
        "UPDATE Sm2Timeline SET Name = 'changed'; END"
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProjectWriteError, match="trigger Resolve does not make") as info:
        read_seed(path, options=OPTS)
    assert info.value.code == "bad_seed"


@pytest.mark.parametrize("chunk", range(6))
def test_a_damaged_seed_is_refused_typed_or_placed_verified(tmp_path: Path, chunk: int) -> None:
    """The aimed seed fuzz (``seed_fuzz``): rows, column types, fixed-layout blobs, other
    blobs, the schema and the file's bytes. On the code before this round, cases 0-599 found
    the TypeError (a fixed-layout value of another type), the KeyError (a renamed column) and
    the UnicodeDecodeError (schema text that is not UTF-8)."""
    template = _seed(tmp_path, "template.db")
    for number in range(chunk * 100, chunk * 100 + 100):
        case = tmp_path / f"case-{number}"
        case.mkdir()
        seed = case / "seed.db"
        what = damaged_seed(seed, number)
        try:
            read_seed(seed, options=OPTS)
        except ProjectWriteError as exc:
            assert exc.code == "bad_seed", (number, what, exc)
            continue
        library = make_library(case / "lib")
        try:
            result = make_project(
                library,
                seed,
                project_name="Fuzzed",
                timeline_name=None,
                author=_cut(template),
                allowed_issues=READER_WARNING_CODES,
                allow_unopened=True,
                options=OPTS,
            )
        except ProjectWriteError as exc:
            assert exc.code in ("bad_seed", "bad_input", "self_check_failed"), (number, what)
            assert sorted(os.listdir(library / "Resolve Projects")) == ["Users"], (number, what)
            continue
        except MediaPoolError:  # the caller's authoring refused the seed's clip: typed
            continue
        assert result.placed.path.is_file(), (number, what)


def test_every_public_entry_point_is_behind_the_typed_boundary() -> None:
    modules = [
        "aar.drp.items.author",
        "aar.drp.items.media",
        "aar.drp.items.rows",
        "aar.drp.items.fields",
        "aar.drp.items.gate",
        "aar.drp.libwrite",
        "aar.drp.libwrite.check",
        "aar.drp.libwrite.cut",
        "aar.drp.libwrite.ops",
        "aar.drp.libwrite.place",
        "aar.drp.libwrite.seed",
        "aar.drp.libwrite.verify",
        "aar.drp.libwrite.write",
        "aar.plan.export.library",
    ]
    # Pure functions of their own arguments that cannot raise on any input, and the
    # boundary's own helpers.
    pure = {"frames_text"}
    bare = []
    for name in modules:
        module = importlib.import_module(name)
        for public in getattr(module, "__all__", []):
            obj = getattr(module, public)
            if not inspect.isfunction(obj) or public in pure:
                continue
            if not hasattr(obj, "__wrapped__"):
                bare.append(f"{name}.{public}")
    assert bare == []


# -- 2. no symbolic link below the library ------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="symbolic links (Windows: the junction test below)")
@pytest.mark.parametrize(
    ("linked", "user"),
    [
        ("Resolve Projects/Users/guest/Projects", None),
        ("Resolve Projects/Users/guest/Projects", "guest"),
        ("Resolve Projects/Users/guest", "guest"),
        ("Resolve Projects/Users/guest", None),
        ("Resolve Projects", None),
        ("Resolve Projects/Users", None),
    ],
)
def test_a_linked_folder_below_the_library_is_refused(
    tmp_path: Path, linked: str, user: str | None
) -> None:
    seed = _seed(tmp_path)
    library = make_library(tmp_path / "lib")
    real = tmp_path / "elsewhere"
    shutil.move(str(library / linked), str(real))
    _symlink(real, library / linked)
    before = sorted(str(p.relative_to(real)) for p in real.rglob("*"))
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, user=user)
    assert info.value.code == "not_a_library"
    assert "symbolic link" in info.value.message
    assert sorted(str(p.relative_to(real)) for p in real.rglob("*")) == before


@pytest.mark.skipif(not POSIX, reason="symbolic links")
def test_a_linked_staging_folder_is_refused_and_its_target_left_alone(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    library = make_library(tmp_path / "lib")
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink(outside, library / "Resolve Projects" / STAGING)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "library_unwritable" and "symbolic link" in info.value.message
    assert list(outside.iterdir()) == []
    assert list(_projects(tmp_path).iterdir()) == []


@pytest.mark.skipif(not POSIX, reason="symbolic links")
def test_the_library_path_itself_may_be_a_link(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    make_library(tmp_path / "real")
    _symlink(tmp_path / "real", tmp_path / "lib")
    result = _make(tmp_path, seed, name="Through a link")
    assert (
        (tmp_path / "real" / "Resolve Projects" / "Users" / "guest" / "Projects")
        .joinpath("Through a link", "Project.db")
        .is_file()
    )
    assert result.placed.user == "guest"


@pytest.mark.skipif(not POSIX, reason="symbolic links")
def test_a_destination_that_resolves_outside_the_library_is_refused(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    (tmp_path / "out").mkdir()
    _symlink(tmp_path / "out", library / "elsewhere")
    with pytest.raises(ProjectWriteError) as info:
        place._inside(library / "elsewhere" / "x", library, "the test folder")
    assert info.value.code == "not_a_library" and "outside the library" in info.value.message
    place._inside(library / "Resolve Projects", library, "the library's own folder")


def test_a_windows_reparse_point_counts_as_a_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A junction is no symbolic link to ``S_ISLNK``: the reparse-point attribute marks it."""
    folder = tmp_path / "junction"
    folder.mkdir()
    real = os.lstat(folder)

    class Stat:
        st_mode = real.st_mode
        st_file_attributes = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT

    monkeypatch.setattr(place.os, "lstat", lambda path: Stat())
    assert place._is_link(folder)


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are Windows'")
def test_a_junction_below_the_library_is_refused_on_windows(tmp_path: Path) -> None:
    import _winapi  # type: ignore[import-not-found]

    seed = _seed(tmp_path)
    library = make_library(tmp_path / "lib")
    projects = library / "Resolve Projects" / "Users" / "guest" / "Projects"
    real = tmp_path / "elsewhere"
    shutil.move(str(projects), str(real))
    _winapi.CreateJunction(str(real), str(projects))
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "not_a_library"
    assert list(real.iterdir()) == []


# -- 3. what a failed build leaves is named ---------------------------------------------------


def _failing_rmtree(exc: BaseException) -> Any:
    def rmtree(path: Any, *args: Any, **kwargs: Any) -> None:
        raise exc

    return rmtree


def test_a_refusal_whose_staging_cannot_be_removed_names_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    monkeypatch.setattr(place.shutil, "rmtree", _failing_rmtree(OSError(errno.EBUSY, "busy")))
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, author=_cut(seed, lambda c: [Cut(c, 5, 24, 0)]), allow_unopened=False)
    assert info.value.code == "operation_not_opened"
    residue = info.value.details["residue"]
    assert STAGING in residue and "build-" in residue
    assert "left in the library" in info.value.message and residue in info.value.message
    assert "left in the library" in str(info.value)
    left = [p for p in (tmp_path / "lib" / "Resolve Projects" / STAGING).iterdir()]
    assert [p.name.split("-")[0] for p in left] == ["build"]  # the lock is removed
    assert list(_projects(tmp_path).iterdir()) == []


def test_an_interrupt_during_the_removal_names_what_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    monkeypatch.setattr(place.shutil, "rmtree", _failing_rmtree(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt) as info:
        _make(tmp_path, seed, author=_cut(seed, lambda c: [Cut(c, 5, 24, 0)]), allow_unopened=False)
    interrupt = info.value
    assert STAGING in interrupt.residue  # type: ignore[attr-defined]
    assert any("left in the library" in n for n in interrupt.__notes__)
    # The refusal it interrupted is kept as its context, with the residue named too.
    refusal = interrupt.__cause__
    assert isinstance(refusal, ProjectWriteError) and refusal.code == "operation_not_opened"
    assert refusal.details["residue"] == interrupt.residue  # type: ignore[attr-defined]
    assert list(_projects(tmp_path).iterdir()) == []


def test_an_error_reported_by_a_rename_that_happened_is_a_placed_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    real_move = place._move

    def late(folder: Path, target: Path) -> None:
        real_move(folder, target)
        raise OSError(errno.EIO, "the share answered late")

    monkeypatch.setattr(place, "_move", late)
    result = _make(tmp_path, seed, name="Placed anyway")
    assert result.placed.path.is_file()
    assert any("but the project is in place" in note for note in result.placed.cleanup)
    assert _top(tmp_path) == ["Users"]


def test_a_placed_project_whose_staging_cannot_be_removed_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    monkeypatch.setattr(place.shutil, "rmtree", _failing_rmtree(OSError(errno.EBUSY, "busy")))
    result = _make(tmp_path, seed, name="Placed")
    assert result.placed.path.is_file()
    assert len(result.placed.cleanup) == 1 and "build-" in result.placed.cleanup[0]
    assert "delete it by hand" in result.placed.cleanup[0]
    left = os.listdir(tmp_path / "lib" / "Resolve Projects" / STAGING)
    assert [x.split("-")[0] for x in left] == ["build"]


# -- 4. the independent verifier: every planted error is refused before placement -----------------


def _plant(monkeypatch: pytest.MonkeyPatch, change: Any) -> None:
    real = cut_module.author_item

    def planted(*args: Any, **kwargs: Any) -> Any:
        authored = real(*args, **kwargs)
        row = dict(authored.item)
        change(row, args[3])
        return dataclasses.replace(authored, item=row)

    monkeypatch.setattr(cut_module, "author_item", planted)


_RATE_25 = struct.pack("<d", 25.0) + bytes(8)


@pytest.mark.parametrize(
    ("field", "change"),
    [
        ("MediaFilePath", lambda row, p: row.update(MediaFilePath=row["MediaFilePath"] + "x")),
        ("MediaStartTime", lambda row, p: row.update(MediaStartTime=row["MediaStartTime"] + 1)),
        (
            "MediaStartTime",
            lambda row, p: row.update(
                MediaStartTime=struct.unpack(
                    "<d",
                    (struct.unpack("<q", struct.pack("<d", row["MediaStartTime"]))[0] + 1).to_bytes(
                        8, "little", signed=True
                    ),
                )[0]
            ),
        ),
        (
            "MediaTimemapBA",
            lambda row, p: row.update(MediaTimemapBA=b"\x02" + struct.pack(">d", 1.0)),
        ),
        ("MediaFrameRate", lambda row, p: row.update(MediaFrameRate=_RATE_25)),
        (
            "MediaTrackIdx",
            lambda row, p: row.update(MediaTrackIdx=0 if p.kind == "picture" else None),
        ),
        ("MediaTrackIdx", lambda row, p: p.kind == "sound" and row.update(MediaTrackIdx=1)),
    ],
)
def test_a_planted_item_value_is_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, change: Any
) -> None:
    seed = _seed(tmp_path)
    _plant(monkeypatch, change)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert info.value.details["field"] == field
    assert list(_projects(tmp_path).iterdir()) == [] and _top(tmp_path) == ["Users"]


def test_an_item_playing_another_clip_of_the_pool_is_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import aar.drp.libwrite as libwrite

    other = mp4_template(base=0x400, name="camera-d.MP4")
    seed = _seed(tmp_path, clips=[CLIP, other])
    _plant(monkeypatch, lambda row, p: row.update(MediaRef=other.media_id))
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"  # the reader's read-back sees it first
    # The verifier finds it on its own too (the read-back switched off).
    monkeypatch.setattr(libwrite, "check_written", lambda *a, **k: None)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert info.value.details["field"] == "MediaRef"
    assert list(_projects(tmp_path).iterdir()) == []


def test_an_item_naming_a_clip_that_is_not_there_is_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    _plant(monkeypatch, lambda row, p: row.update(MediaRef=uid(0xEEEE)))
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert list(_projects(tmp_path).iterdir()) == []


def test_pictures_linked_to_pictures_are_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)
    real = cut_module.author_item
    pictures: list[str] = []

    def planted(clip: Any, template: Any, clips: Any, placement: Any, **kwargs: Any) -> Any:
        if placement.kind == "picture" and pictures:
            placement = dataclasses.replace(placement, partners=(pictures[0],))
        authored = real(clip, template, clips, placement, **kwargs)
        if authored.kind == "picture":
            pictures.append(authored.item_id)
        return authored

    monkeypatch.setattr(cut_module, "author_item", planted)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert info.value.details["field"] == "FieldsBlob #1.#1.#1"
    assert list(_projects(tmp_path).iterdir()) == []


@pytest.mark.parametrize("seconds", [1.0, -1.0])
def test_a_wrong_sequence_length_is_refused_before_placement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seconds: float
) -> None:
    seed = _seed(tmp_path)
    real = write_module.sequence_length
    monkeypatch.setattr(write_module, "sequence_length", lambda *a: real(*a) + seconds)
    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert info.value.details["field"] == "MediaExtents"
    assert list(_projects(tmp_path).iterdir()) == []


def test_items_without_their_cut_are_refused(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    inner = _cut(seed)

    def author(s: Any) -> Authored:
        return dataclasses.replace(inner(s), cuts=())

    with pytest.raises(ProjectWriteError, match="without the cut") as info:
        _make(tmp_path, seed, author=author)
    assert info.value.code == "bad_input"


# -- 5. names that fold to one name, and names the share cannot hold ----------------------------


def test_a_name_that_folds_to_one_being_placed_is_refused_while_its_lock_is_held(
    tmp_path: Path,
) -> None:
    library = make_library(tmp_path / "lib")
    inner: list[ProjectWriteError] = []

    def build(path: Path) -> None:
        path.write_bytes(b"ok")
        try:
            place_project(library, "RACE", lambda p: p.write_bytes(b"other"))
        except ProjectWriteError as exc:
            inner.append(exc)

    place_project(library, "Race", build)
    assert [e.code for e in inner] == ["project_exists"]
    assert "lock" in inner[0].message
    assert sorted(os.listdir(_projects(tmp_path))) == ["Race"]
    assert _top(tmp_path) == ["Users"]  # the lock and the staging folder are gone


def test_concurrent_exports_of_names_differing_in_case_place_one(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "from aar.drp.libwrite import ProjectWriteError\n"
        "from aar.drp.libwrite.place import place_project\n"
        "def build(p):\n"
        "    p.write_bytes(b'x')\n"
        "    time.sleep(0.5)\n"
        "try:\n"
        "    place_project(Path(sys.argv[1]), sys.argv[2], build)\n"
        "    print('placed')\n"
        "except ProjectWriteError as exc:\n"
        "    print(exc.code)\n"
    )
    names = ["Race", "RACE", "race", "rAcE"]
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(library), name], stdout=subprocess.PIPE, text=True
        )
        for name in names
    ]
    outs = sorted(p.communicate(timeout=120)[0].strip() for p in procs)
    assert outs == ["placed", "project_exists", "project_exists", "project_exists"]
    assert len(os.listdir(_projects(tmp_path))) == 1
    assert _top(tmp_path) == ["Users"]


def test_a_lock_left_by_a_killed_placement_is_named_and_kept(tmp_path: Path) -> None:
    library = make_library(tmp_path / "lib")
    staging = library / "Resolve Projects" / STAGING
    staging.mkdir()
    lock = staging / place._lock_name("Kept")
    lock.write_bytes(b"1 kept\n")
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "KEPT", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "project_exists" and info.value.details["lock"] == str(lock)
    assert lock.read_bytes() == b"1 kept\n"


@pytest.mark.parametrize(
    "name",
    [
        " lead",
        "Project.db",
        "PROJECT.DB",
        "é",  # not NFC
        "\U0001f600" * 70,  # 70 characters, 280 bytes of UTF-8, 140 UTF-16 units
        "é" * 130,  # 130 characters, 260 bytes of UTF-8
    ],
)
def test_a_name_the_share_cannot_hold_as_one_folder_is_bad_project_name(name: str) -> None:
    with pytest.raises(ProjectWriteError) as info:
        check_project_name(name)
    assert info.value.code == "bad_project_name"


def test_a_name_too_long_once_encoded_is_refused_before_any_work(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    make_library(tmp_path / "lib")

    def author(s: Any) -> Authored:
        raise AssertionError("no work may be done for a refused name")

    with pytest.raises(ProjectWriteError) as info:
        _make(tmp_path, seed, name="\U0001f600" * 70, author=author)
    assert info.value.code == "bad_project_name"
    assert _top(tmp_path) == ["Users"]


def test_folded_names_join_case_unicode_forms_and_windows_upper_case() -> None:
    assert fold_name("Race") == fold_name("RACE")
    assert fold_name("straße") == fold_name("STRASSE")
    assert fold_name("é") == fold_name("É")
    assert fold_name(chr(0x131)) == fold_name("I")  # dotless i and I: one name to Windows
    assert fold_name("Race") != fold_name("Rice")


def test_an_existing_empty_folder_of_the_name_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    existing = _projects(tmp_path) / "Theirs"
    existing.mkdir()
    monkeypatch.setattr(place, "_exists", lambda projects, name: None)  # a check that missed
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Theirs", lambda p: p.write_bytes(b"ours"))
    assert info.value.code == "project_exists"
    assert list(existing.iterdir()) == []


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux's renameat2")
def test_the_no_replace_rename_refuses_a_folder_the_check_missed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The move's own guarantee on Linux: a folder of the name that appeared after every check
    (here: one the check does not see), even an empty one, which a plain rename replaces,
    stays as it is."""
    if place._NOREPLACE is None:
        pytest.skip("this C library has no renameat2")
    ours, theirs = tmp_path / "ours", tmp_path / "theirs"
    ours.mkdir()
    (ours / "Project.db").write_bytes(b"ours")
    theirs.mkdir()
    real_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: self != theirs and real_exists(self))
    with pytest.raises(OSError) as info:
        place._move(ours, theirs)
    assert info.value.errno == errno.EEXIST
    assert list(theirs.iterdir()) == [] and (ours / "Project.db").read_bytes() == b"ours"


def test_a_move_refused_as_not_empty_is_project_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A system without a no-replace rename: the rename meets a folder that is not empty."""
    library = make_library(tmp_path / "lib")
    monkeypatch.setattr(place, "_NOREPLACE", None)

    def rename(src: Any, dst: Any) -> None:
        raise OSError(errno.ENOTEMPTY, "Directory not empty", str(dst))

    monkeypatch.setattr(place.os, "rename", rename)
    with pytest.raises(ProjectWriteError) as info:
        place_project(library, "Busy", lambda p: p.write_bytes(b"ours"))
    assert info.value.code == "project_exists"
    assert list(_projects(tmp_path).iterdir()) == []
    assert _top(tmp_path) == ["Users"]


# -- 7. CurrentSelectorIdx -------------------------------------------------------------------------


def test_a_new_project_holds_no_selector_of_a_template(tmp_path: Path) -> None:
    seed = _seed(tmp_path)  # its items (the templates) hold 4242 on the picture
    path = _make(tmp_path, seed).placed.path
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        values = {r[0] for r in conn.execute("SELECT CurrentSelectorIdx FROM Sm2TiItem")}
    finally:
        conn.close()
    assert values == {0}


# -- 10. the validator's surviving mutants ------------------------------------------------------


def test_a_staging_folder_on_another_file_system_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = make_library(tmp_path / "lib")
    real_stat = os.stat

    def stat(path: Any, *args: Any, **kwargs: Any) -> Any:
        found = real_stat(path, *args, **kwargs)
        if Path(path).name.startswith("build-"):
            values = list(found)
            values[2] += 1  # st_dev
            return os.stat_result(values)
        return found

    monkeypatch.setattr(place.os, "stat", stat)
    with pytest.raises(ProjectWriteError, match="different file systems") as info:
        place_project(library, "Elsewhere", lambda p: p.write_bytes(b"ok"))
    assert info.value.code == "library_unwritable"
    assert list(_projects(tmp_path).iterdir()) == []


def test_a_seed_in_wal_mode_is_refused(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    conn = sqlite3.connect(seed)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.close()
    os.utime(seed, (1_000_000_000, 1_000_000_000))
    with pytest.raises(ProjectWriteError, match="journal mode") as info:
        _make(tmp_path, seed)
    assert info.value.code == "bad_input"


@pytest.mark.parametrize(("second", "overlaps"), [(23, True), (24, False)])
def test_items_overlapping_by_one_frame_are_refused_and_abutting_ones_are_not(
    tmp_path: Path, second: int, overlaps: bool
) -> None:
    seed = _seed(tmp_path)
    author = _cut(seed, lambda c: [Cut(c, 0, 24, 0), Cut(c, second, 24, 100)])
    if overlaps:
        with pytest.raises(ProjectWriteError, match="overlap") as info:
            _make(tmp_path, seed, author=author)
        assert info.value.code == "bad_input"
    else:
        assert _make(tmp_path, seed, author=author).placed.path.is_file()


def test_items_whose_db_index_is_not_their_record_order_are_refused(tmp_path: Path) -> None:
    seed = _seed(tmp_path)
    inner = _cut(seed)

    def author(s: Any) -> Authored:
        out = inner(s)
        items = list(out.items)
        pics = [i for i, x in enumerate(items) if x.kind == "picture"]
        a, b = items[pics[0]], items[pics[1]]
        items[pics[0]] = dataclasses.replace(a, link={**a.link, "DbIndex": b.link["DbIndex"]})
        items[pics[1]] = dataclasses.replace(b, link={**b.link, "DbIndex": a.link["DbIndex"]})
        return dataclasses.replace(out, items=items)

    with pytest.raises(ProjectWriteError, match="DbIndex is not their record order") as info:
        _make(tmp_path, seed, author=author)
    assert info.value.code == "bad_input"


@pytest.mark.parametrize("side", ["-journal", "-wal", "-shm"])
def test_a_file_left_beside_the_written_project_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, side: str
) -> None:
    """A file system that keeps SQLite's journal (or a WAL file) after the write: SQLite
    itself removes a stale one, so the leftover is made as the connection closes."""
    seed = _seed(tmp_path)

    class Conn(sqlite3.Connection):
        def __init__(self, database: Any, *args: Any, **kwargs: Any) -> None:
            super().__init__(database, *args, **kwargs)
            self.path = str(database)

        def close(self) -> None:
            super().close()
            Path(self.path + side).write_bytes(b"left")

    fake = types.SimpleNamespace(**{k: getattr(sqlite3, k) for k in dir(sqlite3)})
    fake.connect = lambda *a, **k: sqlite3.connect(*a, factory=Conn, **k)
    monkeypatch.setattr(write_module, "sqlite3", fake)
    with pytest.raises(ProjectWriteError, match="was left beside") as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert _top(tmp_path) == ["Users"]


def test_a_written_project_failing_the_integrity_check_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = _seed(tmp_path)

    class Conn(sqlite3.Connection):
        def execute(self, sql: str, *args: Any) -> Any:  # type: ignore[override]
            if sql.startswith("PRAGMA integrity_check"):
                return super().execute("SELECT 'row 1 missing from index'")
            return super().execute(sql, *args)

    fake = types.SimpleNamespace(**{k: getattr(sqlite3, k) for k in dir(sqlite3)})
    fake.connect = lambda *a, **k: sqlite3.connect(*a, factory=Conn, **k)
    monkeypatch.setattr(write_module, "sqlite3", fake)
    with pytest.raises(ProjectWriteError, match="integrity check") as info:
        _make(tmp_path, seed)
    assert info.value.code == "self_check_failed"
    assert list(_projects(tmp_path).iterdir()) == []


def _written(tmp_path: Path) -> tuple[Path, Any, Any]:
    seed_path = _seed(tmp_path)
    result = _make(tmp_path, seed_path, name="Read back")
    return result.placed.path, result.seed, result.authored.specs


def test_the_read_back_checks_the_project_name(tmp_path: Path) -> None:
    path, seed, specs = _written(tmp_path)
    kwargs = {"timeline_name": seed.timeline_name, "allowed_issues": READER_WARNING_CODES}
    check_written(path, seed, specs, project_name="Read back", **kwargs)
    with pytest.raises(ProjectWriteError, match="named 'Read back'") as info:
        check_written(path, seed, specs, project_name="Other", **kwargs)
    assert info.value.code == "self_check_failed"


@pytest.mark.parametrize(("change", "fragment"), [("start", "from frame"), ("rate", "fps")])
def test_the_read_back_checks_the_rate_and_the_start(
    tmp_path: Path, change: str, fragment: str
) -> None:
    from fractions import Fraction

    path, seed, specs = _written(tmp_path)
    other = (
        dataclasses.replace(seed, start_frame=seed.start_frame + 1)
        if change == "start"
        else dataclasses.replace(seed, rate=Fraction(25))
    )
    with pytest.raises(ProjectWriteError, match=fragment) as info:
        check_written(
            path,
            other,
            specs,
            timeline_name=seed.timeline_name,
            project_name="Read back",
            allowed_issues=READER_WARNING_CODES,
        )
    assert info.value.code == "self_check_failed"


def test_the_read_back_refuses_an_item_that_is_not_at_full_speed(tmp_path: Path) -> None:
    from mediapool_support import S, kd

    path, seed, specs = _written(tmp_path)
    copy = tmp_path / "retimed.db"
    shutil.copy(path, copy)
    conn = sqlite3.connect(copy)
    conn.execute(
        "UPDATE Sm2TiItem SET MediaTimemapBA = ? WHERE rowid = (SELECT min(rowid) FROM Sm2TiItem)",
        (kd(("DbType", S, "Sm2TimeMap")),),
    )
    conn.commit()
    conn.close()
    with pytest.raises(ProjectWriteError) as info:
        check_written(
            copy,
            seed,
            specs,
            timeline_name=seed.timeline_name,
            project_name="Read back",
            allowed_issues=READER_WARNING_CODES,
        )
    assert info.value.code == "self_check_failed"


def _placement(**change: Any) -> ItemPlacement:
    values: dict[str, Any] = {
        "kind": "picture",
        "track_id": uid(0xA005),
        "index": 0,
        "start": 86400,
        "duration": 24,
        "src_in": 0,
    }
    values.update(change)
    return ItemPlacement(**values)


def _template_item(tmp_path: Path, kind: str) -> Any:
    items = read_items(_seed(tmp_path), options=OPTS)
    return next(r for r in items.items if r.kind == kind), items.clips


def test_a_sound_item_of_a_clip_without_sound_is_a_placement_refusal(tmp_path: Path) -> None:
    template, clips = _template_item(tmp_path, "sound")
    with pytest.raises(ItemError, match="has no sound") as info:
        author_item(
            png_template(),
            template,
            clips,
            _placement(kind="sound", track_id=uid(0xA006)),
            item_id=uid(0xD100),
        )
    assert info.value.code == "bad_placement"


@pytest.mark.parametrize(("src_in", "past"), [(216, False), (217, True)])
def test_a_source_range_past_the_media_is_refused(tmp_path: Path, src_in: int, past: bool) -> None:
    template, clips = _template_item(tmp_path, "picture")  # the clip holds 240 frames
    place_ = _placement(src_in=src_in)
    if past:
        with pytest.raises(ItemError, match="reads past") as info:
            author_item(CLIP, template, clips, place_, item_id=uid(0xD100))
        assert info.value.code == "bad_placement"
    else:
        author_item(CLIP, template, clips, place_, item_id=uid(0xD100))


def test_an_item_named_as_its_own_partner_is_refused(tmp_path: Path) -> None:
    template, clips = _template_item(tmp_path, "picture")
    with pytest.raises(ItemError, match="its own partner") as info:
        author_item(CLIP, template, clips, _placement(partners=(uid(0xD100),)), item_id=uid(0xD100))
    assert info.value.code == "bad_placement"


def test_a_cut_asking_for_the_sound_of_a_clip_without_sound_is_refused(tmp_path: Path) -> None:
    seed_path = _seed(tmp_path, clips=[CLIP, png_template()])
    seed = read_seed(seed_path, options=OPTS)
    still = next(c for c in seed.clips if not c.audio)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    with pytest.raises(ProjectWriteError, match="has no sound to place") as info:
        author_cut(seed, [Cut(still, 0, 1, 0)], picks)
    assert info.value.code == "bad_input"
    items, _ = author_cut(seed, [Cut(still, 0, 1, 0, sound=False)], picks)
    assert [i.kind for i in items] == ["picture"]


def test_an_id_source_giving_a_taken_id_is_refused_by_the_cut(tmp_path: Path) -> None:
    seed_path = _seed(tmp_path)
    seed = read_seed(seed_path, options=OPTS)
    picks = pick_item_templates([read_items(seed_path, options=OPTS)])
    with pytest.raises(ProjectWriteError, match="not a fresh uuid") as info:
        author_cut(seed, [Cut(seed.clips[0], 0, 24, 0)], picks, new_id=lambda: PROJECT)
    assert info.value.code == "bad_input"


def test_a_new_project_id_already_in_the_seed_is_refused(tmp_path: Path) -> None:
    seed_path = _seed(tmp_path)
    staged = tmp_path / "staged.db"
    shutil.copy(seed_path, staged)
    seed = read_seed(staged, options=OPTS)
    with pytest.raises(ProjectWriteError, match="already used") as info:
        build_project(
            staged,
            seed,
            project_name="Taken",
            timeline_name=None,
            clips=[],
            items=[],
            new_id=lambda: PROJECT,
        )
    assert info.value.code == "bad_input"


def test_the_seed_library_reads_through_the_boundary(tmp_path: Path) -> None:
    """``pick_item_templates`` and ``author_cut`` keep the item writer's refusals typed."""
    seed = read_seed(_seed(tmp_path), options=OPTS)
    picks = pick_item_templates([read_items(seed.path, options=OPTS)])
    with pytest.raises(ItemError):
        author_cut(seed, [Cut(seed.clips[0], 0, 24, 230)], picks)  # past the 240 frames
    with pytest.raises(ProjectWriteError) as info:
        author_cut(seed, ["not a cut"], picks)  # type: ignore[list-item]
    assert info.value.code == "bad_input"
