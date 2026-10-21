"""Regression tests for the S1 review follow-ups (hostile input, error paths, CLI output)."""

from __future__ import annotations

import re
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from drp_helpers import ZSTD_REASON, envelope, kd, pb_len, pb_var, xml

from aar.cli import main
from aar.drp import blobs
from aar.drp import scan as scan_module
from aar.drp.container import parse_version_header
from aar.drp.reader import read_drp
from aar.drp.scan import scan

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"

# ----------------------------------------------------------------------------- crafted zips


def small_drp(path: Path, project_body: str = "<ProjectName>P</ProjectName>") -> Path:
    """A minimal project: small deflated members (each under 1 KB)."""
    members = {
        "project.xml": xml(f'<SM_Project DbId="p1">{project_body}</SM_Project>\n'),
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>\n'),
        "MediaPool/Master/MpFolder.xml": xml(
            '<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def _central_entry(data: bytes, member: str) -> int:
    """Offset of the central-directory record for ``member``."""
    pos = 0
    while True:
        pos = data.index(b"PK\x01\x02", pos)
        (name_len,) = struct.unpack_from("<H", data, pos + 28)
        if data[pos + 46 : pos + 46 + name_len] == member.encode():
            return pos
        pos += 4


def patch_member(path: Path, member: str, how: str) -> None:
    data = bytearray(path.read_bytes())
    cd = _central_entry(bytes(data), member)
    if how == "encrypted":
        (flags,) = struct.unpack_from("<H", data, cd + 8)
        struct.pack_into("<H", data, cd + 8, flags | 0x1)
    elif how == "unknown-method":
        struct.pack_into("<H", data, cd + 10, 99)
    elif how == "bad-crc":
        (crc,) = struct.unpack_from("<I", data, cd + 16)
        struct.pack_into("<I", data, cd + 16, crc ^ 0xFFFFFFFF)
    elif how == "bad-deflate":
        (local,) = struct.unpack_from("<I", data, cd + 42)
        name_len, extra_len = struct.unpack_from("<HH", data, local + 26)
        start = local + 30 + name_len + extra_len
        data[start : start + 4] = b"\xff\xff\xff\xff"  # deflate block type 3 is invalid
    else:
        raise AssertionError(how)
    path.write_bytes(bytes(data))


def issues(drp, code: str) -> list:  # type: ignore[no-untyped-def]
    return [i for i in drp.issues if i.code == code]


# ----------------------------------------------------------------------------- item 1


@pytest.mark.parametrize(
    ("how", "exc_name"),
    [
        ("bad-deflate", "zlib.error"),
        ("encrypted", "RuntimeError"),
        ("unknown-method", "NotImplementedError"),
    ],
)
@pytest.mark.parametrize("member", ["project.xml", "Gallery.xml"])
def test_damaged_member_becomes_an_issue_not_a_traceback(
    tmp_path: Path, how: str, exc_name: str, member: str
) -> None:
    # project.xml goes through Container.read(); Gallery.xml (not interpreted) only through
    # Container.head() for its version header.
    path = small_drp(tmp_path / "x.drp")
    patch_member(path, member, how)
    drp = read_drp(path, check_blobs=True)
    found = issues(drp, "unreadable-member")
    assert len(found) == 1
    assert member in found[0].message and exc_name in found[0].message
    if member == "Gallery.xml":
        assert drp.project is not None and drp.project.name == "P"
    else:
        assert drp.project is None


def test_damaged_member_cli_exits_cleanly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = small_drp(tmp_path / "x.drp")
    patch_member(path, "Gallery.xml", "bad-deflate")
    assert main(["drp", "describe", str(path)]) == 0
    assert "unreadable-member" in capsys.readouterr().out


# ----------------------------------------------------------------------------- item 2

HOSTILE = {
    "comment": b"<!--" * 262_144,
    "cdata": b"<![CDATA[" * 116_509,
    "pi": b"<?x" * 349_525,
    "decl": b"<!x" * 349_525,
    "nested": b"<a>" + b"<b><!--" * 149_796,
    "attribute": b'<a x="' * 174_763,
    "end-tag": b"</a" * 349_525,
    "open-tag": b"<a " * 349_525,
    "quotes": b"<a x=\"<b y='" * 90_000,
}


@pytest.mark.parametrize("kind", sorted(HOSTILE))
def test_hostile_megabyte_member_scans_in_linear_time(kind: str) -> None:
    # Counted, not timed: Document.work is the bytes the scanner examined plus its tag-match
    # attempts. Linear scanning stays within a small multiple of the input size; the old
    # regex scanner re-read the rest of the member at every "<!--" (about n^2 / 8 bytes).
    data = HOSTILE[kind]
    assert len(data) >= 1_000_000
    doc = scan(data, "hostile.xml")
    assert doc.work <= 3 * len(data), f"{kind}: work {doc.work} for {len(data)} bytes"
    assert doc.issues  # the unterminated construct is reported
    assert len(doc.issues) < 1000  # repeated problems are counted, not listed one by one


def _best_of_three(data: bytes) -> float:
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        scan(data)
        times.append(time.perf_counter() - t0)
    return min(times)


def test_deep_nesting_with_stray_end_tags_scales_linearly() -> None:
    # Each "</z>" used to search the whole open-element stack: quadratic for deep nesting.
    # Linear: 4x the input takes about 4x the time. Quadratic would be about 16x.
    small = _best_of_three(b"<a>" * 5_000 + b"</z>" * 5_000)
    large = _best_of_three(b"<a>" * 20_000 + b"</z>" * 20_000)
    assert large < 8 * small, f"4x input took {large / small:.1f}x time"


def test_repeated_issues_are_capped_and_counted() -> None:
    doc = scan(b"<r>" + b"</x>" * 150 + b"</r>")
    stray = [i for i in doc.issues if i.code == "stray-end-tag"]
    assert len(stray) == 100
    [summary] = [i for i in doc.issues if i.code == "issues-truncated"]
    assert summary.message == "50 more stray-end-tag issue(s) not listed"


def test_unterminated_comment_consumes_to_end_of_member() -> None:
    data = b"<a><b>x</b><!-- never closed <c>y</c></a>"
    doc = scan(data)
    codes = [i.code for i in doc.issues]
    assert "unterminated-comment" in codes
    assert doc.root is not None
    assert [c.tag for c in doc.root.children] == ["b"]  # <c> sits inside the comment
    assert doc.root.end == len(data)
    assert doc.comments == []


def test_terminated_constructs_still_parse() -> None:
    data = b"<?xml version='1.0'?><!--c--><!DOCTYPE a><a><b><![CDATA[<x>]]><!--n--></b></a>"
    doc = scan(data)
    assert doc.issues == []
    assert [c[2] for c in doc.comments] == ["c", "n"]
    assert doc.comments[0] == (21, 29, "c")
    assert doc.root is not None and doc.root.child_text("b") == "<x>"


# ----------------------------------------------------------------------------- item 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("17", 17),
        (" -42 ", -42),
        ("²", None),  # str.isdigit() is true, int() raises
        ("١٢", None),  # int() accepts Arabic-Indic digits; Resolve never writes them
        ("1_000", None),  # int() accepts underscores
        ("9" * 4301, None),  # int() raises past 4300 digits
        ("", None),
        ("-", None),
    ],
)
def test_parse_int_is_strict_and_never_raises(text: str, expected: int | None) -> None:
    assert scan_module.parse_int(text) == expected


@pytest.mark.parametrize("value", ["²", "\u0661\u0667", "9" * 30])
def test_version_header_with_odd_digits(value: str) -> None:
    header = f'<!--DbAppVer="21.0" DbPrjVer="{value}"-->'.encode()
    v = parse_version_header(header)
    assert v is not None and v.app_version == "21.0" and v.project_version is None


def _marker_blob(duration: str) -> str:
    detail = pb_var(1, 2) + pb_len(3, b"") + pb_len(3, duration.encode()) + pb_len(3, b"m")
    inner = pb_len(1, detail)
    body = struct.pack(">II", 2, len(inner)) + inner
    entry = pb_len(1, pb_var(1, 7) + pb_len(2, body))
    data = envelope(10001, 0x80, pb_len(2, entry))
    return kd([("BlobData", 0x0C, struct.pack(">I", len(data)) + data)]).hex()


@pytest.mark.parametrize(
    ("duration", "expected"),
    [("²", None), ("9" * 4301, None), ("\u0661", None), ("-3", None), ("12", 12), ("", None)],
)
def test_marker_duration_with_odd_digits(
    tmp_path: Path, duration: str, expected: int | None
) -> None:
    body = (
        "<ProjectName>P</ProjectName><LocableBlobSet><Element>"
        f'<Sm2SequenceLockableBlob DbId="l"><FieldsBlob>{_marker_blob(duration)}</FieldsBlob>'
        "<BlobOwner>seq</BlobOwner></Sm2SequenceLockableBlob></Element></LocableBlobSet>"
    )
    path = small_drp(tmp_path / "m.drp", body)
    timeline = (
        '<Sm2MpFolder DbId="b2"><Name>T</Name><MediaVec><Element>'
        '<Sm2MpTimelineClip DbId="tc"><Name>T</Name><TimelineSharedHandle>'
        '<Sm2Timeline DbId="tl"><Name>T</Name><Sequence><Sm2Sequence DbId="seq"/></Sequence>'
        "</Sm2Timeline></TimelineSharedHandle></Sm2MpTimelineClip></Element></MediaVec>"
        "</Sm2MpFolder>"
    )
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("MediaPool/Master/T/MpFolder.xml", xml(timeline))
    drp = read_drp(path)
    [tl] = drp.timelines
    [m] = tl.markers
    assert (m.frame, m.color, m.name, m.note, m.duration) == (7, "Blue", "m", "", expected)
    assert not issues(drp, "marker-decode") and not issues(drp, "orphan-markers")


def test_non_ascii_digits_in_integer_fields_are_rejected(tmp_path: Path) -> None:
    seq = xml(
        '<Sm2SequenceContainer DbId="c"><VideoTrackVec><Element><Sm2TiTrack DbId="t">'
        "<Type>0</Type><SubType>١٢</SubType><Flags>0</Flags><Sequence>s</Sequence><Items>"
        '<Element><Sm2TiVideoClip DbId="i"><Name>n</Name><Start>1_000</Start>'
        "<Duration>5</Duration></Sm2TiVideoClip></Element></Items></Sm2TiTrack></Element>"
        "</VideoTrackVec></Sm2SequenceContainer>"
    )
    path = small_drp(tmp_path / "d.drp")
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("SeqContainer/c.xml", seq)
    drp = read_drp(path)
    [tl] = drp.timelines
    track = tl.tracks[0]
    assert track.subtype is None and track.items[0].record_start is None
    assert issues(drp, "bad-integer") and issues(drp, "bad-frame-count")


# ----------------------------------------------------------------------------- item 4


def test_lone_surrogate_does_not_crash_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = small_drp(tmp_path / "s.drp", "<ProjectName>a&#xD800;b</ProjectName>")
    assert main(["drp", "describe", str(path)]) == 0
    assert 'project "a\\ud800b"' in capsys.readouterr().out
    assert main(["drp", "inspect", "--json", str(path)]) == 0
    assert '"name": "a\\ud800b"' in capsys.readouterr().out
    assert main(["drp", "inspect", str(path)]) == 0
    assert "project: 'a\\ud800b'" in capsys.readouterr().out


@pytest.mark.parametrize("command", [["inspect", "--json"], ["describe", "--max-items", "0"]])
def test_closed_pipe_exits_without_traceback(tmp_path: Path, command: list[str]) -> None:
    # Enough output to overflow a pipe buffer, so the writer must block and then hit EPIPE.
    item = (
        '<Element><Sm2TiVideoClip DbId="i{n}"><Name>clip {n}</Name><Start>{n}</Start>'
        "<Duration>1</Duration><In>0</In></Sm2TiVideoClip></Element>"
    )
    items = "".join(item.format(n=n) for n in range(3000))
    seq = xml(
        '<Sm2SequenceContainer DbId="c"><VideoTrackVec><Element><Sm2TiTrack DbId="t">'
        f"<Type>0</Type><Sequence>s</Sequence><Items>{items}</Items></Sm2TiTrack></Element>"
        "</VideoTrackVec></Sm2SequenceContainer>"
    )
    path = small_drp(tmp_path / "big.drp")
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("SeqContainer/c.xml", seq)
    code = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
    proc = subprocess.Popen(
        [sys.executable, "-c", code, "drp", *command, str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None
    proc.stdout.read(10)
    proc.stdout.close()
    stderr = proc.stderr.read().decode()
    assert proc.wait(timeout=60) == 141
    assert "Traceback" not in stderr and "BrokenPipeError" not in stderr
    assert "Exception ignored" not in stderr


# ----------------------------------------------------------------------------- items 5 and 6


@pytest.fixture
def no_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))


def test_video_codec_is_never_borrowed_from_audio(
    no_zstd: None, capsys: pytest.CaptureFixture[str]
) -> None:
    drp = read_drp(FIXTURES / "media-clip-h264.drp")
    [clip] = [m for m in drp.media if m.type == "video"]
    assert clip.codec is None
    assert any("1 media clip(s)" in i.message for i in issues(drp, "codec-unknown"))
    assert main(["drp", "describe", str(FIXTURES / "media-clip-h264.drp")]) == 0
    out = capsys.readouterr().out
    assert "codec unknown" in out and "codec AAC" not in out


def test_audio_only_clip_takes_its_codec_from_audio(tmp_path: Path) -> None:
    message = pb_len(1, b"/m") + pb_len(2, b"a.wav") + pb_len(5, b"pcm")
    pool = xml(
        '<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec><Element>'
        '<Sm2MpAudioClip DbId="a1"><Name>a.wav</Name><EmbeddedAudioVec><Element>'
        f'<BtAudioInfo DbId="x"><Clip>{envelope(2, 0x80, message).hex()}</Clip></BtAudioInfo>'
        "</Element></EmbeddedAudioVec></Sm2MpAudioClip></Element></MediaVec></Sm2MpFolder>"
    )
    path = small_drp(tmp_path / "a.drp")
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("MediaPool/Master/Sub/MpFolder.xml", pool.replace('"b1"', '"b2"'))
    clip = read_drp(path).media_by_id()["a1"]
    assert (clip.codec, clip.path) == ("pcm", "/m/a.wav")


def test_zstd_unavailable_is_reported_once_per_file_with_a_count(no_zstd: None) -> None:
    drp = read_drp(FIXTURES / "media-clip-h264.drp")
    [issue] = issues(drp, "zstd-unavailable")
    assert issue.message.startswith("2 blob(s)") and "zstandard" in issue.message


# ----------------------------------------------------------------------------- item 7


def test_bad_crc_is_reported_once(tmp_path: Path) -> None:
    path = small_drp(tmp_path / "c.drp")
    patch_member(path, "project.xml", "bad-crc")
    patch_member(path, "Gallery.xml", "bad-crc")
    drp = read_drp(path, check_blobs=True)
    found = issues(drp, "unreadable-member")
    assert sorted((i.severity, "project.xml" in i.message) for i in found) == [
        ("error", True),
        ("warning", False),
    ]
    assert all("CRC" in i.message for i in found)


# ----------------------------------------------------------------------------- item 10


def test_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    not_zip = tmp_path / "plain.drp"
    not_zip.write_bytes(b"not a zip at all")
    for command in (["describe"], ["inspect"], ["inspect", "--json"]):
        assert main(["drp", *command, str(not_zip)]) == 1
        out = capsys.readouterr()
        assert "not a zip archive" in out.out + out.err
    drb = tmp_path / "bin.drb"
    with zipfile.ZipFile(drb, "w") as zf:
        zf.writestr(
            "MediaPool/Master/MpFolder.xml",
            xml('<Sm2MpFolder DbId="b"><Name>Master</Name><MediaVec/></Sm2MpFolder>'),
        )
    assert main(["drp", "describe", str(drb)]) == 0  # a bin export has no project.xml
    assert main(["drp", "describe", str(FIXTURES / "empty-project.drp")]) == 0
    capsys.readouterr()
    for command in ("describe", "inspect"):
        assert main(["drp", command, "--help"]) == 0
        out = capsys.readouterr().out
        assert "exit status:" in out and "141  the output pipe was closed early" in out


# ----------------------------------------------------------------------------- item 8


def test_fixture_readme_names_existing_test_files() -> None:
    readme = (FIXTURES / "README.md").read_text(encoding="utf-8")
    named = re.findall(r"`(tests/[\w/]+\.py)`", readme)
    assert named, "README should name the corpus test"
    root = FIXTURES.parent.parent.parent
    assert [p for p in named if not (root / p).is_file()] == []


# ----------------------------------------------------------------------------- clip path


def _synced_clip_drp(path: Path, *, items: str) -> Path:
    """A video clip with synced external audio: the two Clip blobs name different files.

    The video Clip blob is a compressed zstd frame (needs a real decompressor); the audio one
    is plain protobuf, so without zstandard only the audio stream's blob can be read.
    """
    zstandard = pytest.importorskip("zstandard", reason=ZSTD_REASON)
    video_msg = pb_len(1, b"/cam") + pb_len(2, b"a.mov") + pb_len(5, b"apcn") + pb_len(9, b"x" * 64)
    audio_msg = pb_len(1, b"/rec") + pb_len(2, b"a.wav") + pb_len(5, b"pcm")
    video_blob = envelope(2, 0x81, zstandard.ZstdCompressor().compress(video_msg))
    audio_blob = envelope(2, 0x80, audio_msg)
    pool = xml(
        '<Sm2MpFolder DbId="b2"><Name>Synced</Name><MediaVec><Element>'
        '<Sm2MpVideoClip DbId="v1"><Name>a.mov</Name><MpFolder>b2</MpFolder>'
        f'<Video><BtVideoInfo DbId="vi"><Clip>{video_blob.hex()}</Clip></BtVideoInfo></Video>'
        '<EmbeddedAudioVec><Element><BtAudioInfo DbId="ai">'
        f"<Clip>{audio_blob.hex()}</Clip></BtAudioInfo></Element></EmbeddedAudioVec>"
        "</Sm2MpVideoClip></Element></MediaVec></Sm2MpFolder>"
    )
    seq = xml(f'<Sm2SequenceContainer DbId="c">{items}</Sm2SequenceContainer>')
    small_drp(path)
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("MediaPool/Master/Sub/MpFolder.xml", pool)
        zf.writestr("SeqContainer/c.xml", seq)
    return path


def _track(vec: str, item_tag: str, media_file_path: str) -> str:
    return (
        f'<{vec}><Element><Sm2TiTrack DbId="t-{item_tag}"><Type>0</Type><Sequence>s</Sequence>'
        f'<Items><Element><{item_tag} DbId="i-{item_tag}"><Name>a</Name><Start>0</Start>'
        f"<Duration>10</Duration><MediaRef>v1</MediaRef>"
        f"<MediaFilePath>{media_file_path}</MediaFilePath></{item_tag}></Element></Items>"
        f"</Sm2TiTrack></Element></{vec}>"
    )


VIDEO_ITEM = _track("VideoTrackVec", "Sm2TiVideoClip", "/old/a.mov")
AUDIO_ITEM = _track("AudioTrackVec", "Sm2TiAudioClip", "/rec/a.wav")


def test_synced_clip_path_with_zstandard(tmp_path: Path) -> None:
    drp = read_drp(_synced_clip_drp(tmp_path / "s.drp", items=AUDIO_ITEM + VIDEO_ITEM))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source, clip.codec) == ("/cam/a.mov", "media_pool_blob", "apcn")
    assert clip.linked_paths == ["/rec/a.wav"]


def test_synced_clip_path_never_comes_from_the_audio_stream(tmp_path: Path, no_zstd: None) -> None:
    # Audio track listed first, so its <MediaFilePath> is the first timeline path seen.
    drp = read_drp(_synced_clip_drp(tmp_path / "s.drp", items=AUDIO_ITEM + VIDEO_ITEM))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source) == ("/old/a.mov", "timeline_item")
    assert clip.blob_path is None and clip.codec is None
    assert clip.linked_paths == []  # unknown whether the audio file differs from the video
    assert "/rec/a.wav" in clip.timeline_paths
    assert issues(drp, "zstd-unavailable") and not issues(drp, "path-unknown")


def test_synced_clip_without_any_video_path_has_no_path(tmp_path: Path, no_zstd: None) -> None:
    drp = read_drp(_synced_clip_drp(tmp_path / "s.drp", items=AUDIO_ITEM))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source) == (None, "none")
    [issue] = issues(drp, "path-unknown")
    assert issue.message.startswith("1 media clip(s)")
