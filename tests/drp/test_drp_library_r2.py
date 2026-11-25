"""S25a fix round 2: a library read's layout is linear and bounded, the whole read has one time
budget, the defaults read a real project ten times over and can be raised from every command,
and nothing a library holds can forge a field of the text output or be dropped without a word.
"""

from __future__ import annotations

import re
import sqlite3
import time
import tracemalloc
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from library_support import Tables, find, library_db, library_tree
from test_drp_library import FAST, edited, refusal, rich_members
from test_drp_synthetic import members

from aar.cli import main
from aar.drp import library, read_drp
from aar.drp.cli import _library_lines
from aar.drp.describe import _marker_line, describe
from aar.drp.library import (
    COLUMNS,
    DEFAULT_LISTING_BYTES,
    DEFAULT_LISTING_SECONDS,
    DEFAULT_MAX_DB_BYTES,
    DEFAULT_MAX_LOAD_BYTES,
    DEFAULT_MAX_READ_SECONDS,
    DEFAULT_MAX_ROWS,
    DEFAULT_MAX_VALUE_BYTES,
    LAYOUT_PER_LOAD_BYTE,
    LibraryError,
    LibraryOptions,
    list_library,
    open_library,
)
from aar.drp.model import Marker

MIB = 1024 * 1024


def _projects(root: Path) -> Path:
    return root / "Resolve Projects" / "Users" / "guest" / "Projects"


class _StepClock:
    """A fake clock that stands still until :meth:`start` and then moves ``step`` s a call."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 1000.0
        self.step = 0.0
        self.going = step

    def start(self) -> None:
        self.step = self.going

    def clock(self) -> float:
        self.t += self.step
        return self.t


def _start_clock_after_load(monkeypatch: pytest.MonkeyPatch, clock: _StepClock) -> None:
    """Let time pass only once every row is fetched, so only the layout can see it pass."""
    real = library._load

    def load(*args: Any) -> Any:
        tables = real(*args)
        clock.start()
        return tables

    monkeypatch.setattr(library, "_load", load)


def _outside_quotes(line: str) -> str:
    """``line`` with every quoted string (as :func:`aar.drp.describe.quote` writes it) removed."""
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', line)


# ---------------------------------------------------------------------------- N1. layout


def _siblings(count: int, name: str) -> Callable[[Tables], None]:
    def add(tables: Tables) -> None:
        for n in range(count):
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"sib{n}", "Name": name, "MpFolder": "bin1"}
            )

    return add


def test_many_folders_with_one_name_are_laid_out_in_linear_time(tmp_path: Path) -> None:
    # Before: every folder rebuilt the set of names taken and counted up from ~2 to a free one,
    # so 49,000 such folders took 362 s. Each one is now named in one step: 30,000 take well
    # under a second; the bound leaves a slow machine thirty times that.
    path = edited(tmp_path, rich_members(), _siblings(30_000, "Same"))
    started = time.monotonic()
    drp = read_drp(path, library_options=FAST)
    assert time.monotonic() - started < 30
    assert sum(1 for b in drp.bins if b.path == "Master/Same") == 30_000


def test_folders_with_one_path_are_each_read_as_a_folder(tmp_path: Path) -> None:
    # Two folders laid out under one path used to get a member "MpFolder~2.xml", which the
    # reader does not read as a folder: the second folder and its media were dropped silently.
    def twins(tables: Tables) -> None:
        _siblings(2, "Twin")(tables)
        clip = find(tables, "Sm2MpMedia", "clip1")
        for n in range(2):
            tables["Sm2MpMedia"].append(
                {**clip, "Sm2MpMedia_id": f"twin{n}", "Name": f"twin{n}.mov", "Video": None}
            )
            tables["Sm2MpFolder_Sm2MpMedia"].append(
                {
                    "DbOwner": f"sib{n}",
                    "DbAssociate": f"twin{n}",
                    "DbPropertyName": "MediaVec",
                    "DbIndex": 0,
                }
            )
            find(tables, "Sm2MpMedia", f"twin{n}")["Sm2MpFolder_id"] = f"sib{n}"

    path = edited(tmp_path, rich_members(), twins)
    with open_library(path, options=FAST) as view:
        assert "MediaPool/Master/Twin~2/MpFolder.xml" in view.names()
    drp = read_drp(path, library_options=FAST)
    twin_bins = sorted(b.id for b in drp.bins if b.path == "Master/Twin")
    assert twin_bins == ["sib0", "sib1"]
    assert {"twin0", "twin1"} <= {m.id for m in drp.media}


def test_folder_member_names_are_bounded_whatever_the_chain(tmp_path: Path) -> None:
    # A chain of folders with long names: every member name holds at most
    # _MAX_BIN_MEMBER_CHARS of path (before: the full path, 256 levels of 200 characters).
    def chain(tables: Tables) -> None:
        for n in range(60):
            parent = "bin1" if n == 0 else f"deep{n - 1}"
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"deep{n}", "Name": "n" * 300, "MpFolder": parent}
            )

    path = edited(tmp_path, rich_members(), chain)
    with open_library(path, options=FAST) as view:
        folders = [n for n in view.names() if n.startswith("MediaPool/")]
    longest = len("MediaPool/") + library._MAX_BIN_MEMBER_CHARS + len("/MpFolder.xml") + 8
    assert len(folders) == 62 and max(len(n) for n in folders) <= longest
    drp = read_drp(path, library_options=FAST)
    assert sum(1 for b in drp.bins if b.id.startswith("deep")) == 60  # every one still read


def test_a_bin_chain_deeper_than_the_guard_is_cut_to_it(tmp_path: Path) -> None:
    depth = library._MAX_BIN_DEPTH + 40

    def chain(tables: Tables) -> None:
        for n in range(depth):
            parent = "bin1" if n == 0 else f"deep{n - 1}"
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"deep{n}", "Name": "b", "MpFolder": parent}
            )

    with open_library(edited(tmp_path, rich_members(), chain), options=FAST) as view:
        parts = max(name.count("/") for name in view.names() if name.startswith("MediaPool/"))
    assert parts == library._MAX_BIN_DEPTH + 1  # MediaPool/<at most 256 bins>/MpFolder.xml


def _long_chain(count: int, name_bytes: int) -> Callable[[Tables], None]:
    def chain(tables: Tables) -> None:
        for n in range(count):
            parent = "bin1" if n == 0 else f"deep{n - 1}"
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"deep{n}", "Name": "n" * name_bytes, "MpFolder": parent}
            )

    return chain


def test_the_folder_paths_the_model_holds_are_charged_before_they_are_made(
    tmp_path: Path,
) -> None:
    # 400 folders in a chain with 1,000-byte names: 0.4 MB of rows, but the model's bin paths
    # come to ~50 MB (each holds its parents' names). Charged at layout, that is refused before
    # the reader makes them; without the charge it would be read (the rows lay out to 0.5 MB).
    path = edited(tmp_path, rich_members(), _long_chain(400, 1000))
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_layout_bytes=10 * MIB)
    tracemalloc.start()
    try:
        error = refusal(path, options)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert error.code == "too_large" and "10 MiB layout budget" in error.message
    assert "--max-layout-bytes" in error.hint
    assert peak < 8 * MIB


def test_a_cycle_of_folders_is_charged_its_whole_length_for_each_folder_below_it(
    tmp_path: Path,
) -> None:
    # 200 folders in a cycle with 1,000-byte names, and 300 folders under the one where a walk
    # from the first would cut the cycle: the reader's path for each of those holds the whole
    # cycle (~200 KB), so 300 of them are ~60 MB.
    def cycle(tables: Tables) -> None:
        for n in range(200):
            parent = f"cyc{(n + 1) % 200}"
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"cyc{n}", "Name": "c" * 1000, "MpFolder": parent}
            )
        for n in range(300):
            tables["Sm2MpFolder"].append(
                {"Sm2MpFolder_id": f"leaf{n}", "Name": "l", "MpFolder": "cyc199"}
            )

    path = edited(tmp_path, rich_members(), cycle)
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_layout_bytes=20 * MIB)
    assert refusal(path, options).code == "too_large"


def test_the_layout_looks_at_the_reads_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Time passes only after the last row is fetched: the SQL progress handler never sees it,
    # so the refusal can only come from the layout's own checks. The message gives the limit
    # as configured.
    path = edited(tmp_path, rich_members(), _siblings(5_000, "Same"))
    clock = _StepClock()
    _start_clock_after_load(monkeypatch, clock)
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=0, max_read_seconds=5, clock=clock.clock
    )
    error = refusal(path, options)
    assert error.code == "too_large" and error.timed_out == "read"
    assert error.message.endswith("reading it took longer than 5 s")
    assert "--max-read-seconds" in error.hint


def test_a_read_that_runs_past_its_time_while_the_model_is_built_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The layout finishes in time but the reader does not: the read is still refused rather
    # than returned late.
    path = library_db(tmp_path / "Project.db", rich_members())
    clock = _StepClock(step=0.0)
    real = library.open_library

    def opened(*args: Any, **kw: Any) -> Any:
        view = real(*args, **kw)
        clock.t += 100  # the reader's work starts with the read's time used up
        return view

    monkeypatch.setattr(library, "open_library", opened)
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=0, max_read_seconds=50, clock=clock.clock
    )
    error = refusal(path, options)
    assert error.timed_out == "read" and "longer than 50 s" in error.message


def test_a_listing_cuts_each_read_to_its_deadline_during_the_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each project's own read budget is far longer than the listing's deadline: only the
    # deadline handed down to the read (stop_at) can stop the first project's layout, and a
    # project stopped by it is NOT READ (the listing ran out of time), never REFUSED.
    def big(tables: Tables) -> None:
        _siblings(5_000, "Same")(tables)

    root = tmp_path / "lib"
    for i in range(2):
        folder = _projects(root) / f"P{i}"
        folder.mkdir(parents=True)
        library_db(folder / "Project.db", rich_members(), big)
    clock = _StepClock()
    _start_clock_after_load(monkeypatch, clock)
    options = LibraryOptions(
        settle_seconds=0, deadline_seconds=0, max_read_seconds=10**6, clock=clock.clock
    )
    started = clock.t
    listing = list_library(root, options=options, deadline_seconds=8)
    first, second = listing.projects
    assert first.refusal is None
    assert first.not_read == (
        "the listing's 8 s deadline passed while it was read (raise --deadline)"
    )
    assert second.not_read is not None and "before it was reached" in second.not_read
    assert clock.t - started < 8 + 5  # stopped within a few clock reads of the deadline
    assert not listing.complete


# ---------------------------------------------------------------------------- N2. charge first


def _shared_blob(blob_bytes: int, refs: int) -> Callable[[Tables], None]:
    """One BtVideoInfo row of ``blob_bytes`` that ``refs`` media rows all name."""

    def share(tables: Tables) -> None:
        find(tables, "BtVideoInfo", "vi1")["Clip"] = b"\x00" * blob_bytes
        clip = find(tables, "Sm2MpMedia", "clip1")
        for n in range(refs):
            tables["Sm2MpMedia"].append({**clip, "Sm2MpMedia_id": f"ref{n}", "Video": "vi1"})
            tables["Sm2MpFolder_Sm2MpMedia"].append(
                {
                    "DbOwner": "bin1",
                    "DbAssociate": f"ref{n}",
                    "DbPropertyName": "MediaVec",
                    "DbIndex": 100 + n,
                }
            )

    return share


def test_a_value_laid_out_many_times_is_refused_before_its_copies_are_made(
    tmp_path: Path,
) -> None:
    # A 6 MiB blob named by 20 media rows lays out to 20 x 12 MiB of hex. With an 8 MiB layout
    # budget not even the first copy fits: it is charged before it is made, so no hex is
    # ever made (charged after, the 12 MiB copy would exist when the refusal comes).
    path = edited(tmp_path, rich_members(), _shared_blob(6 * MIB, 20))
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_layout_bytes=8 * MIB)
    tracemalloc.start()
    try:
        error = refusal(path, options)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert error.code == "too_large" and "8 MiB layout budget" in error.message
    assert peak < 10 * MIB  # the 6 MiB row, never its 12 MiB of hex
    # The same holds per reference: 20 MiB fits one copy, and the second is never made.
    big = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_layout_bytes=20 * MIB)
    assert "20 MiB layout budget" in refusal(path, big).message


def test_the_layout_budget_follows_the_load_budget() -> None:
    assert LibraryOptions().layout_limit == 2 * DEFAULT_MAX_LOAD_BYTES
    assert LibraryOptions(max_load_bytes=1000).layout_limit == 2000
    assert LibraryOptions(max_load_bytes=1000, max_layout_bytes=1500).layout_limit == 1500


def test_the_layout_budget_is_its_own_not_the_readers(tmp_path: Path) -> None:
    # 30 MiB of hex: well inside the reader's 600 MiB, outside a 16 MiB layout budget.
    path = edited(tmp_path, rich_members(), _shared_blob(3 * MIB, 5))
    options = LibraryOptions(settle_seconds=0, deadline_seconds=0, max_load_bytes=8 * MIB)
    error = refusal(path, options)
    assert error.code == "too_large" and "16 MiB layout budget" in error.message
    assert read_drp(path, library_options=FAST).project is not None


def _value_charge(path: Path) -> int:
    """What ``_load`` charges for ``path``: every value's bytes (text and blobs) plus 16."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        total = 0
        for table, columns in COLUMNS.items():
            (rows,) = conn.execute(f'SELECT count(*) FROM "{table}"').fetchone()
            total += rows * len(columns) * 16
            for column in columns:
                (size,) = conn.execute(
                    f"SELECT coalesce(sum(CASE WHEN typeof(\"{column}\") IN ('text', 'blob') "
                    f'THEN length(CAST("{column}" AS BLOB)) ELSE 0 END), 0) FROM "{table}"'
                ).fetchone()
                total += size
        return int(total)
    finally:
        conn.close()


def test_every_value_is_charged_sixteen_bytes_on_top_of_its_size(tmp_path: Path) -> None:
    # Many values with no bytes (NULLs and numbers) cost memory too: the load budget charges
    # each one 16 bytes, so a budget one byte short of the exact charge is refused.
    path = library_db(tmp_path / "Project.db", rich_members())
    charge = _value_charge(path)

    def opts(budget: int) -> LibraryOptions:
        return LibraryOptions(settle_seconds=0, deadline_seconds=0, max_load_bytes=budget)

    assert read_drp(path, library_options=opts(charge)).project is not None
    assert "load budget" in refusal(path, opts(charge - 1)).message


# ---------------------------------------------------------------------------- N3. defaults

#: The largest real save measured (a show project, 9,878 timeline items in 39 timelines):
#: rows in the tables read, load charge, laid-out bytes, file size and largest value.
REAL_ROWS = 21_637
REAL_LOAD_CHARGE = 7_265_740
REAL_LAYOUT = 7_085_851
REAL_FILE_BYTES = 29_470_720
REAL_LARGEST_VALUE = 28_984


def test_the_defaults_are_pinned() -> None:
    assert DEFAULT_MAX_ROWS == 220_000
    assert DEFAULT_MAX_LOAD_BYTES == 80 * MIB
    assert LAYOUT_PER_LOAD_BYTE == 2
    assert DEFAULT_MAX_VALUE_BYTES == 16 * MIB
    assert DEFAULT_MAX_DB_BYTES == 256 * MIB
    assert DEFAULT_MAX_READ_SECONDS == 120.0
    assert DEFAULT_LISTING_SECONDS == 120.0
    assert DEFAULT_LISTING_BYTES == 2 * 1024 * MIB
    defaults = LibraryOptions()
    assert (defaults.max_rows, defaults.max_load_bytes, defaults.max_layout_bytes) == (
        220_000,
        80 * MIB,
        None,
    )


def test_the_defaults_read_the_largest_real_project_ten_times_over() -> None:
    assert DEFAULT_MAX_ROWS >= 10 * REAL_ROWS
    assert DEFAULT_MAX_LOAD_BYTES >= 10 * REAL_LOAD_CHARGE
    assert LibraryOptions().layout_limit >= 10 * REAL_LAYOUT
    assert DEFAULT_MAX_DB_BYTES >= 8 * REAL_FILE_BYTES
    assert DEFAULT_MAX_VALUE_BYTES >= 100 * REAL_LARGEST_VALUE


def _many_items(count: int) -> Callable[[Tables], None]:
    def many(tables: Tables) -> None:
        base = find(tables, "Sm2TiItem", "v1i1")
        tables["Sm2TiItem"].extend({**base, "Sm2TiItem_id": f"x{n}"} for n in range(count))

    return many


#: Each budget option, a value that refuses the rich project, and what its refusal says.
BUDGETS = [
    ("--max-rows", "50", "50-row limit"),
    ("--max-load-bytes", "2000", "load budget"),
    ("--max-layout-bytes", "2000", "layout budget"),
    ("--max-value-bytes", "100", "per-value limit"),
    ("--max-db-bytes", "1000", "limit"),
]


@pytest.mark.parametrize(("option", "value", "said"), BUDGETS)
@pytest.mark.parametrize("command", ["describe", "inspect", "diff", "library"])
def test_every_drp_command_takes_every_budget_option_and_names_it_when_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    option: str,
    value: str,
    said: str,
) -> None:
    root = library_tree(tmp_path / "lib", [("P", rich_members())])
    db = str(_projects(root) / "P" / "Project.db")
    argv = {
        "describe": ["drp", "describe", db],
        "inspect": ["drp", "inspect", db],
        "diff": ["drp", "diff", db, db],
        "library": ["drp", "library", str(root), "--wait", "0"],
    }[command]
    code = main([*argv, option, value])
    out, err = capsys.readouterr()
    assert code == (2 if command == "diff" else 1)
    assert said in out + err and f"raise the limit with {option}" in out + err
    assert main([*argv, option, str(10**8)]) == 0  # raised, it reads
    capsys.readouterr()


def test_the_time_budget_option_is_taken_and_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db = str(library_db(tmp_path / "Project.db", rich_members()))
    assert main(["drp", "describe", db, "--max-read-seconds", "60"]) == 0
    capsys.readouterr()
    assert main(["drp", "describe", db, "--max-read-seconds", "0"]) == 2
    assert main(["drp", "describe", db, "--max-layout-bytes", str(601 * MIB)]) == 2
    assert "at most" in capsys.readouterr().err


def test_the_row_limit_is_raised_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 300 more items than a 200-row limit allows: refused at the limit, read once it is raised.
    path = str(edited(tmp_path, rich_members(), _many_items(300)))
    assert main(["drp", "describe", path, "--max-rows", "200"]) == 1
    assert "raise the limit with --max-rows" in capsys.readouterr().err
    assert main(["drp", "describe", path, "--max-rows", "1000"]) == 0


def test_the_listing_byte_budget_is_raised_from_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = library_tree(tmp_path / "lib", [("P0", members()), ("P1", members())])
    size = (_projects(root) / "P0" / "Project.db").stat().st_size
    argv = ["drp", "library", str(root), "--wait", "0"]
    assert main([*argv, "--max-listing-bytes", str(size)]) == 3
    assert "raise --max-listing-bytes" in capsys.readouterr().out
    assert main([*argv, "--max-listing-bytes", str(2 * size)]) == 0


def test_every_refusal_hint_names_an_option() -> None:
    # Each LibraryError raised for a budget names the option that raises it.
    source = Path(library.__file__).read_text()
    for option in (
        "--max-rows",
        "--max-load-bytes",
        "--max-layout-bytes",
        "--max-value-bytes",
        "--max-db-bytes",
        "--max-read-seconds",
    ):
        assert f"raise the limit with {option}" in source
    assert "raise max_" not in source


# ---------------------------------------------------------------------------- N4. quoting


FORGED = 'x" src 00:00:00:00-99:00:00:00 speed 9x media "evil'


def test_an_item_name_cannot_forge_fields_of_its_describe_line(tmp_path: Path) -> None:
    def forge(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["Name"] = FORGED
        find(tables, "Sm2MpMedia", "clip1")["Name"] = FORGED

    drp = read_drp(edited(tmp_path, rich_members(), forge), library_options=FAST)
    text = describe(drp, max_items=0)
    lines = [line for line in text.splitlines() if "evil" in line]
    assert len(lines) >= 2  # the media clip and every item that uses it
    for line in lines:
        bare = _outside_quotes(line)
        assert "evil" not in bare and "99:00:00:00" not in bare and "speed 9x" not in bare


def test_marker_names_notes_and_colours_are_quoted() -> None:
    marker = Marker("i", "item", 5, "item", 'Red" frame 9', FORGED, 'n" dur 99', 1, "")
    line = _marker_line(marker, (0, 25.0))
    bare = _outside_quotes(line)
    assert "dur 99" not in bare and "frame 9" not in bare and "evil" not in bare


def test_media_paths_codecs_and_warnings_are_quoted(tmp_path: Path) -> None:
    def forge(tables: Tables) -> None:
        find(tables, "Sm2TiItem", "v1i1")["MediaRef"] = 'gone" media "evil'

    drp = read_drp(edited(tmp_path, rich_members(), forge), library_options=FAST)
    text = describe(drp, max_items=0)
    for line in text.splitlines():
        if line.lstrip().startswith("["):  # a media clip or a warning
            assert line.count('"') >= 2
        if "gone" in line:
            assert "evil" not in _outside_quotes(line)


def test_refused_and_not_read_lines_quote_their_messages(tmp_path: Path) -> None:
    root = library_tree(tmp_path / 'li"b', [('P" REFUSED ok', members()), ("Q", members())])
    (_projects(root) / 'P" REFUSED ok' / "Project.db-journal").write_bytes(b"")
    size = (_projects(root) / "Q" / "Project.db").stat().st_size
    listing = list_library(root, options=FAST, max_bytes=size + 1)  # room for the first only
    text = _library_lines(listing, 1)
    refused = next(line for line in text if "REFUSED" in line and "busy" in line)
    not_read = next(line for line in text if "NOT READ" in line)
    assert _outside_quotes(refused).count("REFUSED") == 1  # the folder's own is quoted
    assert 'REFUSED busy: "' in refused and 'NOT READ: "' in not_read
    assert text[0] == f"library: {_quote_path(root)}"


def _quote_path(path: Path) -> str:
    return '"' + str(path).replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------- N5. nested


def test_a_project_nested_in_a_project_folder_is_listed_as_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = library_tree(tmp_path / "lib", [("Outer", members()), ("Outer/Inner", members())])
    listing = list_library(root, options=FAST)
    assert [p.folder for p in listing.projects] == ["Outer"]
    assert listing.skipped["nested"].count == 1
    assert listing.skipped["nested"].first == [
        "guest/Projects/Outer/Inner/Project.db (inside Outer)"
    ]
    assert not listing.complete
    assert main(["drp", "library", str(root), "--wait", "0"]) == 3
    out = capsys.readouterr().out
    assert "skipped 1 project(s) inside another project's folder, not read" in out


def test_a_folder_of_projects_is_still_searched(tmp_path: Path) -> None:
    root = library_tree(tmp_path / "lib", [("Shows/A", members()), ("Shows/B", members())])
    listing = list_library(root, options=FAST)
    assert [p.folder for p in listing.projects] == ["Shows/A", "Shows/B"] and listing.complete


def test_library_error_timed_out_is_none_for_other_refusals(tmp_path: Path) -> None:
    error = refusal(library_db(tmp_path / "Project.db", rich_members()), FAST, max_member_bytes=1)
    assert isinstance(error, LibraryError) and error.timed_out is None
