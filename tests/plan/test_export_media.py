"""Media path rules of the FCPXML exporter: file URLs, media maps and refusals."""

from __future__ import annotations

from pathlib import Path

import pytest
from apply_builders import plan_data, source
from export_support import permissive

from aar.plan.apply.errors import CompileError
from aar.plan.export import export_fcpxml
from aar.plan.export.errors import EXPORT_CODES, ExportError
from aar.plan.export.media import (
    MEDIA_MAP_MAX_BYTES,
    load_media_map,
    media_map_from_mapping,
    media_ref,
)
from aar.plan.model import EditPlan, Source

MAP = {"/srv/media": "E:\\Media"}


def ref(media: str, mapping: object = None):
    m = media_map_from_mapping(mapping) if mapping is not None else None
    return media_ref(Source(id="s1", media=media), m)


def refused(media: str, code: str, mapping: object = None) -> ExportError:
    with pytest.raises(ExportError) as info:
        ref(media, mapping)
    assert info.value.code == code
    assert code in EXPORT_CODES
    return info.value


def test_windows_path() -> None:
    r = ref("E:\\CCN\\Ep 1\\A001.braw")
    assert r.rule == "windows-path"
    assert r.src == "file:///E:/CCN/Ep%201/A001.braw"
    assert r.windows_path == "E:\\CCN\\Ep 1\\A001.braw"
    assert r.name == "A001.braw"


def test_forward_slashes() -> None:
    assert ref("E:/CCN/A.mov").src == "file:///E:/CCN/A.mov"


@pytest.mark.parametrize(
    ("name", "encoded"),
    [
        ("a#b.mov", "a%23b.mov"),
        ("100%.mov", "100%25.mov"),
        ("a&b.mov", "a%26b.mov"),
        ("a[1].mov", "a%5B1%5D.mov"),
        ("it's.mov", "it%27s.mov"),
        ("Café.mov", "Caf%C3%A9.mov"),
        ("it\u2019s.mov", "it%E2%80%99s.mov"),
    ],
)
def test_percent_encoding(name: str, encoded: str) -> None:
    assert ref(f"E:\\CCN\\{name}").src == f"file:///E:/CCN/{encoded}"


def test_unc() -> None:
    r = ref("\\\\nas\\share\\show\\a.mov")
    assert r.src == "file://nas/share/show/a.mov"
    assert r.rule == "windows-path"


def test_file_url_as_is() -> None:
    r = ref("file:///E:/CCN/A001.braw")
    assert r.rule == "file-url"
    assert r.src == "file:///E:/CCN/A001.braw"


def test_file_url_unc() -> None:
    r = ref("file://nas/share/a.mov")
    assert r.rule == "file-url"
    assert r.windows_path == "\\\\nas\\share\\a.mov"


def test_file_url_uppercase_scheme() -> None:
    assert ref("FILE:///E:/x.mov").rule == "file-url"


@pytest.mark.parametrize(
    "media",
    [
        "file:///E:/a b.mov",
        "file:///E:/a.mov?x=1",
        "file:///E:/a.mov#frag",
        "file:/E:/x.mov",
        "file:///relative",
    ],
)
def test_file_url_refused(media: str) -> None:
    refused(media, "media_not_on_edit_pc")


def test_hub_file_url_needs_map() -> None:
    refused("file:///srv/media/a.mov", "media_not_on_edit_pc")
    r = ref("file:///srv/media/a.mov", MAP)
    assert r.rule == "media-map"
    assert r.src == "file:///E:/Media/a.mov"


def test_linux_path_without_map() -> None:
    refused("/srv/media/ep1/A.mov", "media_not_on_edit_pc")


def test_linux_path_with_map() -> None:
    r = ref("/srv/media/ep1/A.mov", MAP)
    assert r.rule == "media-map"
    assert r.windows_path == "E:\\Media\\ep1\\A.mov"
    assert r.src == "file:///E:/Media/ep1/A.mov"


def test_linux_outside_map() -> None:
    refused("/other/x.mov", "media_not_on_edit_pc", MAP)


def test_linux_dotdot() -> None:
    refused("/srv/media/../etc/x.mov", "media_not_on_edit_pc", MAP)


def test_longest_prefix_wins() -> None:
    mapping = {"/srv/media": "E:\\Media", "/srv/media/ccn": "F:\\CCN"}
    assert ref("/srv/media/ccn/a.mov", mapping).windows_path == "F:\\CCN\\a.mov"
    assert ref("/srv/media/other/a.mov", mapping).windows_path == "E:\\Media\\other\\a.mov"


def test_pool_name() -> None:
    err = refused("Ep1 Multicam", "pool_source_not_exportable")
    assert "camera file" in err.fix_hint


@pytest.mark.parametrize(
    "media",
    [
        "clip.mov",
        "media/clip.mov",
        "E:clip.mov",
        "\\\\?\\E:\\x.mov",
        "//server/share/x.mov",
        "E:\\a\\..\\b.mov",
    ],
)
def test_malformed_paths(media: str) -> None:
    refused(media, "media_not_on_edit_pc")


@pytest.mark.parametrize(
    "data",
    [{}, ["a"], {"/srv/media": 1}, {"srv/media": "E:\\Media"}, {"/srv/media": "/mnt/x"}],
)
def test_bad_media_map(data: object) -> None:
    with pytest.raises(ExportError) as info:
        media_map_from_mapping(data)
    assert info.value.code == "bad_media_map"
    assert "bad_media_map" in EXPORT_CODES


def test_load_media_map(tmp_path: Path) -> None:
    f = tmp_path / "map.json"
    f.write_text('{"/srv/media": "E:\\\\Media"}', encoding="utf-8")
    assert load_media_map(f).hub_to_edit("/srv/media/a.mov").new == "E:\\Media\\a.mov"


def test_load_media_map_invalid_json(tmp_path: Path) -> None:
    f = tmp_path / "map.json"
    f.write_text("{nope", encoding="utf-8")
    with pytest.raises(ExportError) as info:
        load_media_map(f)
    assert info.value.code == "bad_media_map"


def test_load_media_map_too_big(tmp_path: Path) -> None:
    f = tmp_path / "map.json"
    f.write_bytes(b" " * (MEDIA_MAP_MAX_BYTES + 1))
    with pytest.raises(ExportError) as info:
        load_media_map(f)
    assert info.value.code == "bad_media_map"
    assert "limit" in str(info.value)


# -- malformed URLs: only a typed refusal escapes (Prism, PR #13) ------------------------------

MEDIA_FUZZ = [
    # urlsplit raises ValueError on an unmatched bracket: refused as malformed-url
    ("file://[/share/clip.mov", "malformed-url"),
    ("file://[::1/x.mov", "malformed-url"),
    ("file://]/x.mov", "malformed-url"),
    # a bad port parses (only .port would raise, which is never read): a host that is no UNC name
    ("file://host:notaport/x.mov", "host-not-unc"),
    ("file://host:99999/x.mov", "host-not-unc"),
    ("file://[::1]:notaport/x.mov", "host-not-unc"),
    ("file://[v1.x]/x.mov", "host-not-unc"),
    ("file://", None),
    ("file:", None),
    ("file:///", None),
    ("file:////", None),
    ("file://%/x.mov", None),
    ("file://h/%zz.mov", None),
]


@pytest.mark.parametrize(("media", "reason"), MEDIA_FUZZ, ids=[m for m, _ in MEDIA_FUZZ])
def test_a_malformed_file_url_is_a_typed_refusal(media: str, reason: str | None) -> None:
    """Only a CompileError (here media_not_on_edit_pc) escapes; never a raw ValueError."""
    try:
        media_ref(Source(id="s", media=media))
    except CompileError as exc:
        assert exc.code == "media_not_on_edit_pc"
        if reason is not None:
            assert exc.details["reason"] == reason
    else:
        pytest.fail(f"{media!r} was accepted")


@pytest.mark.parametrize("media", ["file://[/share/clip.mov", "file://[::1/x.mov"])
def test_the_export_refuses_a_malformed_url_with_a_typed_error(media: str) -> None:
    data = plan_data(sources=[source(media=media, frame_size=[1920, 1080])])
    with pytest.raises(CompileError) as caught:
        export_fcpxml(EditPlan.model_validate(data), permissive())
    assert caught.value.code == "media_not_on_edit_pc"
    assert caught.value.details["reason"] == "malformed-url"
