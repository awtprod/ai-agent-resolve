"""Tests for the Windows <-> Mac path mapper in ``aar.drp.modify.paths``."""

from __future__ import annotations

import pytest

from aar.drp.modify.paths import Direction, MapResult, PathMapper, TableError

W2M: Direction = "windows_to_mac"
M2W: Direction = "mac_to_windows"


def _map(direction: Direction, table: list[tuple[str, str]], path: str) -> MapResult:
    return PathMapper(direction, table).map(path)


def _ok(new: str) -> MapResult:
    return MapResult(new=new, reason=None)


def _no(reason: str) -> MapResult:
    return MapResult(new=None, reason=reason)


# --------------------------------------------------------------------------- windows_to_mac


@pytest.mark.parametrize(
    ("row", "path"),
    [("E:", "e:\\x"), ("E:", "E:\\x"), ("e:", "E:\\x"), ("e:", "e:\\x")],
)
def test_w2m_drive_letter_matches_case_insensitively(row: str, path: str) -> None:
    assert _map(W2M, [(row, "/Volumes/M")], path) == _ok("/Volumes/M/x")


def test_w2m_accepts_both_separators_in_the_path() -> None:
    table = [("E:", "/Volumes/M")]
    assert _map(W2M, table, "E:/a\\b.mov") == _ok("/Volumes/M/a/b.mov")
    assert _map(W2M, table, "E:\\a/b\\c.mov") == _ok("/Volumes/M/a/b/c.mov")


def test_w2m_accepts_both_separators_in_the_table_row() -> None:
    assert _map(W2M, [("E:/Media/Raw", "/Volumes/R")], "E:\\Media\\Raw\\x") == _ok("/Volumes/R/x")
    assert _map(W2M, [("E:\\Media\\Raw", "/Volumes/R")], "E:/Media/Raw/x") == _ok("/Volumes/R/x")


def test_w2m_prefix_components_match_case_insensitively_but_the_rest_keeps_its_spelling() -> None:
    table = [("E:\\Media", "/Volumes/M")]
    assert _map(W2M, table, "e:\\MEDIA\\Clip.MOV") == _ok("/Volumes/M/Clip.MOV")
    assert _map(W2M, table, "E:\\media\\Sub Dir\\Ünï.mov") == _ok("/Volumes/M/Sub Dir/Ünï.mov")


def test_w2m_uses_the_mac_prefix_spelling_from_the_table() -> None:
    table = [("E:", "/Volumes/My Media Drive")]
    assert _map(W2M, table, "E:\\x") == _ok("/Volumes/My Media Drive/x")


def test_w2m_a_trailing_separator_in_a_table_row_changes_nothing() -> None:
    assert _map(W2M, [("E:\\Media\\", "/Volumes/M/")], "E:\\Media\\x") == _ok("/Volumes/M/x")


@pytest.mark.parametrize("order", [0, 1])
def test_w2m_longest_prefix_wins_whatever_the_row_order(order: int) -> None:
    rows = [("E:", "/Volumes/M"), ("E:\\Deep", "/Volumes/Deep")]
    table = rows if order == 0 else rows[::-1]
    mapper = PathMapper(W2M, table)
    assert mapper.map("E:\\deep\\x.mov") == _ok("/Volumes/Deep/x.mov")
    assert mapper.map("E:\\Deep") == _ok("/Volumes/Deep")
    assert mapper.map("E:\\elsewhere\\x.mov") == _ok("/Volumes/M/elsewhere/x.mov")


def test_w2m_prefix_matches_whole_components_only() -> None:
    mapper = PathMapper(W2M, [("E:\\Foo", "/Volumes/F")])
    assert mapper.map("E:\\Foobar\\x") == _no("no-mapping")
    assert mapper.map("E:\\Foo.bak\\x") == _no("no-mapping")
    assert mapper.map("E:\\Foo\\x") == _ok("/Volumes/F/x")
    assert mapper.map("E:\\Foo") == _ok("/Volumes/F")


def test_w2m_a_deeper_row_does_not_cover_its_parent() -> None:
    mapper = PathMapper(W2M, [("E:\\Foo\\Bar", "/Volumes/FB")])
    assert mapper.map("E:\\Foo\\x") == _no("no-mapping")
    assert mapper.map("E:\\Foo") == _no("no-mapping")
    assert mapper.map("E:\\") == _no("no-mapping")


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("E:\\", "/Volumes/M/"),
        ("E:/", "/Volumes/M/"),
        ("E:\\a\\", "/Volumes/M/a/"),
        ("E:/a/", "/Volumes/M/a/"),
        ("E:\\a/", "/Volumes/M/a/"),
        ("E:\\a", "/Volumes/M/a"),
        ("E:", "/Volumes/M"),
    ],
)
def test_w2m_trailing_separator_is_preserved(path: str, expected: str) -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], path) == _ok(expected)


def test_w2m_unmatched_drive_is_no_mapping() -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], "F:\\x") == _no("no-mapping")


def test_w2m_unc_path_is_mapped_by_a_unc_row() -> None:
    table = [("\\\\nas\\share", "/Volumes/share")]
    assert _map(W2M, table, "\\\\nas\\share\\q.wav") == _ok("/Volumes/share/q.wav")
    assert _map(W2M, table, "\\\\nas\\share\\a\\b\\q.wav") == _ok("/Volumes/share/a/b/q.wav")
    assert _map(W2M, table, "\\\\nas\\share") == _ok("/Volumes/share")


def test_w2m_unc_server_and_share_match_case_insensitively() -> None:
    table = [("\\\\NAS\\Share", "/Volumes/share")]
    assert _map(W2M, table, "\\\\nas\\SHARE\\Q.wav") == _ok("/Volumes/share/Q.wav")


def test_w2m_unc_accepts_forward_slashes_in_path_and_row() -> None:
    assert _map(W2M, [("//nas/share", "/Volumes/s")], "//NAS/share/a/b") == _ok("/Volumes/s/a/b")
    assert _map(W2M, [("\\\\nas\\share", "/Volumes/s")], "//nas/share\\a") == _ok("/Volumes/s/a")


def test_w2m_unc_row_can_cover_a_subfolder_of_a_share() -> None:
    mapper = PathMapper(W2M, [("\\\\nas\\share", "/Volumes/s"), ("\\\\nas\\share\\Proj", "/V/P")])
    assert mapper.map("\\\\nas\\share\\proj\\x") == _ok("/V/P/x")
    assert mapper.map("\\\\nas\\share\\other\\x") == _ok("/Volumes/s/other/x")


def test_w2m_unc_path_is_not_mapped_by_a_drive_row() -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], "\\\\nas\\share\\q") == _no("unc-not-mapped")


def test_w2m_unc_path_with_an_unknown_share_is_unc_not_mapped() -> None:
    table = [("\\\\nas\\share", "/Volumes/s"), ("E:", "/Volumes/M")]
    mapper = PathMapper(W2M, table)
    assert mapper.map("\\\\nas\\other\\q") == _no("unc-not-mapped")
    assert mapper.map("\\\\other\\share\\q") == _no("unc-not-mapped")


def test_w2m_drive_path_is_not_mapped_by_a_unc_row() -> None:
    assert _map(W2M, [("\\\\nas\\share", "/Volumes/s")], "E:\\x") == _no("no-mapping")


@pytest.mark.parametrize(
    "path",
    ["/Volumes/M/x", "/", "relative\\dir\\x", "x", "E:relative", "", "\\rooted\\x", "\\\\server"],
)
def test_w2m_other_forms_are_reported(path: str) -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], path) == _no("other-form")


@pytest.mark.parametrize(
    "path",
    ["E:\\a\\\\b", "E:\\\\b", "E:/a//b", "E:\\a\\..\\b", "E:\\..", "E:\\a\\.\\b", "E:\\."],
)
def test_w2m_empty_dot_and_dotdot_components_are_irregular(path: str) -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], path) == _no("irregular-path")


def test_w2m_nul_in_a_component_is_unrepresentable() -> None:
    assert _map(W2M, [("E:", "/Volumes/M")], "E:\\a\x00b") == _no("unrepresentable")


def test_w2m_a_mapping_does_not_alter_characters_that_are_fine_on_a_mac() -> None:
    table = [("E:", "/Volumes/M")]
    assert _map(W2M, table, "E:\\a b\\c.d.e\\.hidden\\x ") == _ok("/Volumes/M/a b/c.d.e/.hidden/x ")


# --------------------------------------------------------------------------- mac_to_windows


def test_m2w_prefix_match_is_case_sensitive() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map("/Volumes/Media/x") == _ok("E:\\x")
    assert mapper.map("/Volumes/media/x") == _no("no-mapping")
    assert mapper.map("/volumes/Media/x") == _no("no-mapping")


def test_m2w_rows_differing_only_in_case_are_distinct() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media"), ("F:", "/Volumes/media")])
    assert mapper.map("/Volumes/Media/x") == _ok("E:\\x")
    assert mapper.map("/Volumes/media/x") == _ok("F:\\x")


def test_m2w_prefix_matches_whole_components_only() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map("/Volumes/MediaX/a") == _no("no-mapping")
    assert mapper.map("/Volumes/Media2") == _no("no-mapping")
    assert mapper.map("/Volumes/Media/a") == _ok("E:\\a")


def test_m2w_a_deeper_row_does_not_cover_its_parent() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media/Deep")])
    assert mapper.map("/Volumes/Media/x") == _no("no-mapping")
    assert mapper.map("/Volumes/Media") == _no("no-mapping")


@pytest.mark.parametrize("order", [0, 1])
def test_m2w_longest_prefix_wins_whatever_the_row_order(order: int) -> None:
    rows = [("E:", "/Volumes/Media"), ("F:\\Deep", "/Volumes/Media/Deep")]
    table = rows if order == 0 else rows[::-1]
    mapper = PathMapper(M2W, table)
    assert mapper.map("/Volumes/Media/Deep/x.mov") == _ok("F:\\Deep\\x.mov")
    assert mapper.map("/Volumes/Media/Deep") == _ok("F:\\Deep")
    assert mapper.map("/Volumes/Media/Deeper/x.mov") == _ok("E:\\Deeper\\x.mov")


def test_m2w_output_uses_backslashes_and_the_table_spelling_of_the_prefix() -> None:
    mapper = PathMapper(M2W, [("E:/Media/Raw", "/Volumes/R")])
    assert mapper.map("/Volumes/R/a/b/C.mov") == _ok("E:\\Media\\Raw\\a\\b\\C.mov")
    # The row's own component spelling is kept, not folded.
    assert _map(M2W, [("E:\\MeDiA", "/Volumes/R")], "/Volumes/R/x") == _ok("E:\\MeDiA\\x")


def test_m2w_drive_letter_is_written_in_upper_case() -> None:
    # The table is parsed through the same code as windows_to_mac, which normalises the letter.
    assert _map(M2W, [("e:", "/Volumes/M")], "/Volumes/M/x") == _ok("E:\\x")
    assert _map(M2W, [("e:\\media", "/Volumes/M")], "/Volumes/M/x") == _ok("E:\\media\\x")


def test_m2w_a_path_equal_to_the_prefix_maps_to_the_drive_root() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map("/Volumes/Media") == _ok("E:\\")
    assert mapper.map("/Volumes/Media/") == _ok("E:\\")


def test_m2w_a_path_equal_to_a_deeper_prefix_maps_to_that_folder() -> None:
    mapper = PathMapper(M2W, [("E:\\Deep", "/Volumes/Media/Deep")])
    assert mapper.map("/Volumes/Media/Deep") == _ok("E:\\Deep")
    assert mapper.map("/Volumes/Media/Deep/") == _ok("E:\\Deep\\")


def test_m2w_trailing_separator_is_preserved() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map("/Volumes/Media/a/b/") == _ok("E:\\a\\b\\")
    assert mapper.map("/Volumes/Media/a/b") == _ok("E:\\a\\b")


def test_m2w_a_trailing_separator_in_the_mac_row_changes_nothing() -> None:
    assert _map(M2W, [("E:", "/Volumes/Media/")], "/Volumes/Media/x") == _ok("E:\\x")


def test_m2w_unc_target_row_works() -> None:
    mapper = PathMapper(M2W, [("\\\\nas\\share", "/Volumes/share")])
    assert mapper.map("/Volumes/share/q.wav") == _ok("\\\\nas\\share\\q.wav")
    assert mapper.map("/Volumes/share/a/b/q.wav") == _ok("\\\\nas\\share\\a\\b\\q.wav")
    assert mapper.map("/Volumes/share") == _ok("\\\\nas\\share\\")


def test_m2w_unc_target_row_keeps_the_table_spelling_and_subfolder() -> None:
    mapper = PathMapper(M2W, [("//NAS/Share/Proj", "/Volumes/share")])
    assert mapper.map("/Volumes/share/q.wav") == _ok("\\\\NAS\\Share\\Proj\\q.wav")


@pytest.mark.parametrize(
    "component",
    [
        "a:b",
        "a?b",
        "a*b",
        "a<b",
        "a>b",
        'a"b',
        "a|b",
        "a\\b",
        "a\x01b",
        "a\x1fb",
        "a ",
        "a.",
        "a..",
        "ends with space ",
        "trailing.dot.",
    ],
)
def test_m2w_names_windows_cannot_hold_are_unrepresentable(component: str) -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map(f"/Volumes/Media/ok/{component}/x") == _no("unrepresentable")
    assert mapper.map(f"/Volumes/Media/{component}") == _no("unrepresentable")


@pytest.mark.parametrize("component", ["a b", "a.b", ".hidden", "ünï", "a-b_c", "a~1", "a,b;c"])
def test_m2w_names_windows_can_hold_are_kept_verbatim(component: str) -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/Media")])
    assert mapper.map(f"/Volumes/Media/{component}/x") == _ok(f"E:\\{component}\\x")


@pytest.mark.parametrize(
    "path",
    ["E:\\x", "E:/x", "\\\\nas\\share\\x", "relative/x", "x", "", "//Volumes/Media/x"],
)
def test_m2w_other_forms_are_reported(path: str) -> None:
    assert _map(M2W, [("E:", "/Volumes/Media")], path) == _no("other-form")


@pytest.mark.parametrize(
    "path",
    ["/Volumes/Media//x", "/Volumes/Media/../x", "/Volumes/Media/./x", "/Volumes//Media/x"],
)
def test_m2w_empty_dot_and_dotdot_components_are_irregular(path: str) -> None:
    assert _map(M2W, [("E:", "/Volumes/Media")], path) == _no("irregular-path")


@pytest.mark.parametrize("path", ["/other/x", "/", "/Volumes", "/Volumes/Other/x"])
def test_m2w_unmatched_path_is_no_mapping(path: str) -> None:
    assert _map(M2W, [("E:", "/Volumes/Media")], path) == _no("no-mapping")


# --------------------------------------------------------------------------- TableError


def test_table_error_is_a_value_error() -> None:
    assert issubclass(TableError, ValueError)


def test_table_must_not_be_empty() -> None:
    with pytest.raises(TableError, match="empty"):
        PathMapper(W2M, [])


def test_direction_must_be_known() -> None:
    with pytest.raises(TableError, match="unknown direction 'sideways'"):
        PathMapper("sideways", [("E:", "/Volumes/M")])  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "windows",
    [
        "Media",
        "",
        "E",
        "E:foo",
        "/Volumes/M",
        "\\\\nas",
        "\\\\nas\\",
        "\\rooted",
        "E:\\a\\\\b",
        "E:\\..",
        "E:\\a\\.\\b",
    ],
)
@pytest.mark.parametrize("direction", [W2M, M2W])
def test_windows_side_must_be_a_drive_or_unc_prefix(direction: Direction, windows: str) -> None:
    with pytest.raises(TableError, match="not a Windows drive or UNC prefix"):
        PathMapper(direction, [(windows, "/Volumes/M")])


@pytest.mark.parametrize(
    "mac",
    [
        "Volumes/M",
        "",
        "/",
        "//",
        "//Volumes/M",
        "/Volumes\\M",
        "\\Volumes\\M",
        "E:\\x",
        "/Volumes//M",
        "/Volumes/../M",
        "/Volumes/./M",
    ],
)
@pytest.mark.parametrize("direction", [W2M, M2W])
def test_mac_side_must_be_an_absolute_posix_prefix(direction: Direction, mac: str) -> None:
    with pytest.raises(TableError, match="not an absolute Mac path prefix"):
        PathMapper(direction, [("E:", mac)])


def test_table_error_names_the_offending_entry() -> None:
    with pytest.raises(TableError, match="entry 2: 'Media'"):
        PathMapper(W2M, [("E:", "/V/A"), ("F:", "/V/B"), ("Media", "/V/C")])
    with pytest.raises(TableError, match="entry 1: 'V/B'"):
        PathMapper(M2W, [("E:", "/V/A"), ("F:", "V/B")])


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("E:", "E:"),
        ("E:", "e:\\"),
        ("E:", "E:/"),
        ("E:\\Media", "e:/MEDIA/"),
        ("\\\\nas\\share", "//NAS/Share/"),
        ("\\\\nas\\share\\Proj", "\\\\NAS\\SHARE\\proj"),
    ],
)
def test_w2m_two_rows_for_the_same_windows_prefix_are_refused(first: str, second: str) -> None:
    with pytest.raises(TableError, match="entries 0 and 1 map the same prefix"):
        PathMapper(W2M, [(first, "/Volumes/A"), (second, "/Volumes/B")])


def test_w2m_the_same_mac_prefix_under_two_windows_prefixes_is_allowed() -> None:
    mapper = PathMapper(W2M, [("E:", "/Volumes/M"), ("F:", "/Volumes/M")])
    assert mapper.map("E:\\x") == _ok("/Volumes/M/x")
    assert mapper.map("F:\\x") == _ok("/Volumes/M/x")


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("/Volumes/M", "/Volumes/M"),
        ("/Volumes/M", "/Volumes/M/"),
        ("/Volumes/M/", "/Volumes/M/"),
    ],
)
def test_m2w_two_rows_for_the_same_mac_prefix_are_refused(first: str, second: str) -> None:
    with pytest.raises(TableError, match="entries 0 and 1 map the same prefix"):
        PathMapper(M2W, [("E:", first), ("F:", second)])


def test_m2w_mac_prefixes_differing_only_in_case_are_not_the_same_prefix() -> None:
    PathMapper(M2W, [("E:", "/Volumes/M"), ("F:", "/Volumes/m")])


def test_m2w_two_mac_prefixes_for_one_windows_prefix_are_allowed() -> None:
    mapper = PathMapper(M2W, [("E:", "/Volumes/A"), ("E:", "/Volumes/B")])
    assert mapper.map("/Volumes/A/x") == _ok("E:\\x")
    assert mapper.map("/Volumes/B/x") == _ok("E:\\x")


def test_a_table_may_hold_exactly_256_rows() -> None:
    table = [(f"E:\\d{i}", f"/Volumes/v{i}") for i in range(256)]
    mapper = PathMapper(W2M, table)
    assert len(mapper.entries) == 256
    assert mapper.map("E:\\d255\\x") == _ok("/Volumes/v255/x")


@pytest.mark.parametrize("direction", [W2M, M2W])
def test_a_table_of_257_rows_is_refused(direction: Direction) -> None:
    table = [(f"E:\\d{i}", f"/Volumes/v{i}") for i in range(257)]
    with pytest.raises(TableError, match="more than 256 entries"):
        PathMapper(direction, table)


def test_a_prefix_of_exactly_4096_characters_is_accepted() -> None:
    windows = "E:\\" + "a" * 4093
    mac = "/" + "a" * 4095
    assert len(windows) == len(mac) == 4096
    PathMapper(W2M, [(windows, "/Volumes/M")])
    PathMapper(W2M, [("E:", mac)])


@pytest.mark.parametrize("direction", [W2M, M2W])
def test_a_windows_prefix_over_4096_characters_is_refused(direction: Direction) -> None:
    windows = "E:\\" + "a" * 4094
    assert len(windows) == 4097
    with pytest.raises(TableError, match="entry 0: prefix longer than 4096 characters"):
        PathMapper(direction, [(windows, "/Volumes/M")])


@pytest.mark.parametrize("direction", [W2M, M2W])
def test_a_mac_prefix_over_4096_characters_is_refused(direction: Direction) -> None:
    mac = "/" + "a" * 4096
    assert len(mac) == 4097
    with pytest.raises(TableError, match="entry 0: prefix longer than 4096 characters"):
        PathMapper(direction, [("E:", mac)])


# --------------------------------------------------------------------------- round trips

ROUND_TRIP_TABLE = [
    ("E:", "/Volumes/Media"),
    ("E:\\Proj", "/Volumes/Proj"),
    ("\\\\nas\\share", "/Volumes/share"),
]


@pytest.mark.parametrize(
    ("windows", "back"),
    [
        ("E:\\a\\b.mov", "E:\\a\\b.mov"),
        ("e:\\a\\b.mov", "E:\\a\\b.mov"),
        ("E:/a/b.mov", "E:\\a\\b.mov"),
        ("e:/a\\b/Mixed Case.MOV", "E:\\a\\b\\Mixed Case.MOV"),
        ("E:\\dir\\", "E:\\dir\\"),
        ("E:/dir/", "E:\\dir\\"),
        ("E:\\", "E:\\"),
        ("E:\\Proj\\Clip.mov", "E:\\Proj\\Clip.mov"),
        ("e:\\PROJ\\clip.mov", "E:\\Proj\\clip.mov"),
        ("E:\\Project\\clip.mov", "E:\\Project\\clip.mov"),
        ("\\\\nas\\share\\q.wav", "\\\\nas\\share\\q.wav"),
        ("//NAS/Share/Dir/Q.wav", "\\\\nas\\share\\Dir\\Q.wav"),
        ("\\\\nas\\share\\Dir\\", "\\\\nas\\share\\Dir\\"),
    ],
)
def test_round_trip_windows_to_mac_and_back(windows: str, back: str) -> None:
    to_mac = PathMapper(W2M, ROUND_TRIP_TABLE).map(windows)
    assert to_mac.new is not None and to_mac.new.startswith("/Volumes/")
    to_windows = PathMapper(M2W, ROUND_TRIP_TABLE).map(to_mac.new)
    assert to_windows == _ok(back)


@pytest.mark.parametrize(
    "mac",
    [
        "/Volumes/Media/a/b.mov",
        "/Volumes/Media/Mixed Case/Ünï.MOV",
        "/Volumes/Media/dir/",
        "/Volumes/Proj/Clip.mov",
        "/Volumes/share/q.wav",
    ],
)
def test_round_trip_mac_to_windows_and_back(mac: str) -> None:
    to_windows = PathMapper(M2W, ROUND_TRIP_TABLE).map(mac)
    assert to_windows.new is not None and "/" not in to_windows.new
    assert PathMapper(W2M, ROUND_TRIP_TABLE).map(to_windows.new) == _ok(mac)


def test_a_bare_drive_comes_back_as_the_drive_root() -> None:
    # "E:" has no trailing separator, but the Windows form of the volume root always has one.
    to_mac = PathMapper(W2M, ROUND_TRIP_TABLE).map("E:")
    assert to_mac == _ok("/Volumes/Media")
    assert PathMapper(M2W, ROUND_TRIP_TABLE).map("/Volumes/Media") == _ok("E:\\")


def test_unc_share_root_keeps_its_trailing_separator() -> None:
    m = PathMapper("windows_to_mac", [("\\\\nas\\share", "/Volumes/share")])
    assert m.map("\\\\nas\\share\\").new == "/Volumes/share/"
    assert m.map("\\\\nas\\share").new == "/Volumes/share"


@pytest.mark.parametrize("name", ["CON", "con.txt", "NUL", "COM1", "lpt9.log", "AUX.mov"])
def test_windows_device_names_are_unrepresentable(name: str) -> None:
    m = PathMapper("mac_to_windows", [("E:", "/Volumes/M")])
    assert m.map(f"/Volumes/M/{name}") == MapResult(None, "unrepresentable")


def test_names_that_only_start_like_device_names_map() -> None:
    m = PathMapper("mac_to_windows", [("E:", "/Volumes/M")])
    assert m.map("/Volumes/M/CONsole/COM10.wav").new == "E:\\CONsole\\COM10.wav"
