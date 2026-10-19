"""Windows-path validation, scp remote specs (both forms), and the edit-PC <-> hub map."""

from __future__ import annotations

import pytest

from aar.host.errors import PathRefused
from aar.host.pathmap import HostPathMap
from aar.host.winpath import (
    normalize_root,
    remote_spec,
    scp_target,
    validate_windows_path,
    windows_literal,
)

ROOTS = (normalize_root("E:\\Show"),)


def _validate(path: str, roots=ROOTS):
    return validate_windows_path(path, roots)


# -- acceptance ------------------------------------------------------------------------------


def test_accepts_a_path_inside_the_root() -> None:
    win = _validate("E:\\Show\\Review\\cut v2.mp4")
    assert win.parts == ("Show", "Review", "cut v2.mp4")


def test_drive_form_remote_spec() -> None:
    win = _validate("E:\\Show\\Review\\cut v2.mp4")
    assert remote_spec(win, "drive") == "E:/Show/Review/cut v2.mp4"
    assert scp_target("edit-pc", win, "drive") == "edit-pc:E:/Show/Review/cut v2.mp4"


def test_slash_form_remote_spec() -> None:
    win = _validate("E:\\Show\\Review\\cut v2.mp4")
    assert remote_spec(win, "slash") == "/E:/Show/Review/cut v2.mp4"
    assert scp_target("edit-pc", win, "slash") == "edit-pc:/E:/Show/Review/cut v2.mp4"


def test_mixed_separators_are_accepted_and_normalised() -> None:
    win = _validate("E:\\Show/Review\\cut v2.mp4")
    assert remote_spec(win, "drive") == "E:/Show/Review/cut v2.mp4"
    assert windows_literal(win) == "E:\\Show\\Review\\cut v2.mp4"


def test_forward_slash_input_is_accepted() -> None:
    win = _validate("E:/Show/Review/x.mp4")
    assert remote_spec(win, "drive") == "E:/Show/Review/x.mp4"


def test_lowercase_drive_and_case_insensitive_root_match() -> None:
    # case differs from the root ("E:\\Show") but NTFS compares case-insensitively
    win = _validate("e:\\show\\Review\\x.mp4")
    assert win.root == "E:"  # drive letter is upper-cased
    assert remote_spec(win, "drive") == "E:/show/Review/x.mp4"  # non-drive spelling preserved


def test_leading_dash_component_stays_behind_the_alias() -> None:
    # A path whose component starts with '-' must never become a bare scp option.
    win = _validate("E:\\Show\\-rf")
    target = scp_target("edit-pc", win, "drive")
    assert target == "edit-pc:E:/Show/-rf"
    assert not target.startswith("-")


def test_windows_literal_uses_backslashes_and_upper_drive() -> None:
    win = _validate("e:/Show/Review/x.mp4")
    assert windows_literal(win) == "E:\\Show\\Review\\x.mp4"


# -- refusals (adversarial) ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("E:\\Show\\..\\..\\Windows", "dot-or-empty-component"),
        ("E:/Show/../x", "dot-or-empty-component"),
        ("\\\\?\\E:\\Show\\x", "device-namespace-path"),
        ("E:Show", "not-a-drive-or-unc-path"),  # drive-relative
        ("Show\\x.mp4", "not-a-drive-or-unc-path"),  # relative
        ("E:\\Show\\CON", "reserved-device-name"),
        ("E:\\Show\\CON.txt", "reserved-device-name"),
        ("E:\\Show\\name ", "trailing-dot-or-space"),
        ("E:\\Show\\name.", "trailing-dot-or-space"),
        ("E:\\Show\\a\x01b", "control-character"),
        ("E:\\Show\\a:b", "illegal-character"),  # alternate data stream
        ("E:\\Other\\x", "outside-allowed-roots"),
        ("E:\\", "no-file-component"),
        ("", "empty"),
    ],
)
def test_refuses_adversarial_paths(path: str, reason: str) -> None:
    with pytest.raises(PathRefused) as info:
        _validate(path)
    assert info.value.reason == reason


def test_case_differing_but_outside_root_is_refused() -> None:
    with pytest.raises(PathRefused) as info:
        _validate("E:\\Shows\\x.mp4")  # "Shows" != "Show"
    assert info.value.reason == "outside-allowed-roots"


def test_unc_refused_when_no_unc_root() -> None:
    with pytest.raises(PathRefused) as info:
        _validate("\\\\server\\share\\x.mp4")
    assert info.value.reason == "unc-not-allowed"


def test_unc_accepted_only_with_a_unc_root() -> None:
    roots = (normalize_root("\\\\server\\share"),)
    win = validate_windows_path("\\\\server\\share\\dir\\x.mp4", roots)
    assert win.unc
    assert remote_spec(win, "drive") == "//server/share/dir/x.mp4"


def test_unc_outside_the_unc_root_is_refused() -> None:
    roots = (normalize_root("\\\\server\\share"),)
    with pytest.raises(PathRefused) as info:
        validate_windows_path("\\\\server\\other\\x.mp4", roots)
    assert info.value.reason == "outside-allowed-roots"


def test_normalize_root_rejects_a_relative_prefix() -> None:
    with pytest.raises(PathRefused) as info:
        normalize_root("Show")
    assert info.value.reason == "bad-root"


# -- path map (reusing PathMapper) -----------------------------------------------------------


def test_path_map_round_trips_edit_and_hub() -> None:
    mapper = HostPathMap((("E:\\Show", "/srv/show"),))
    assert mapper.edit_to_hub("E:\\Show\\Review\\cut.mp4").new == "/srv/show/Review/cut.mp4"
    assert mapper.hub_to_edit("/srv/show/Review/cut.mp4").new == "E:\\Show\\Review\\cut.mp4"


def test_path_map_posix_side_takes_any_absolute_prefix() -> None:
    # confirm PathMapper's POSIX side is not restricted to /Volumes
    for hub in ("/srv/x", "/data/cuts", "/a/b/c/d"):
        mapper = HostPathMap((("E:\\Show", hub),))
        assert mapper.edit_to_hub("E:\\Show\\f.mp4").new == f"{hub}/f.mp4"


def test_path_map_is_case_insensitive_on_the_windows_side() -> None:
    mapper = HostPathMap((("E:\\Show", "/srv/show"),))
    assert mapper.edit_to_hub("e:\\SHOW\\x").new == "/srv/show/x"


def test_path_map_requires_a_table() -> None:
    from aar.host.errors import ConfigError

    with pytest.raises(ConfigError):
        HostPathMap(())
