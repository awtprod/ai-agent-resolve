"""Regression tests for the second S1h review (clip paths, budgets, CLI output, caps)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from drp_helpers import ZSTD_REASON, envelope, pb_len, xml
from test_drp_hardening import issues, small_drp

from aar.cli import main
from aar.drp import blobs
from aar.drp.describe import describe, printable
from aar.drp.model import MemberSummary
from aar.drp.reader import MARKER_COLORS, join_clip_path, read_drp
from aar.drp.scan import scan

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "drp"
RUN_CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"


@pytest.fixture
def no_zstd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))


def clip_blob(directory: str, filename: str, codec: str = "c") -> str:
    message = (
        pb_len(1, directory.encode()) + pb_len(2, filename.encode()) + pb_len(5, codec.encode())
    )
    return envelope(2, 0x80, message).hex()


UNREADABLE_BLOB = envelope(2, 0x80, b"\x08").hex()  # truncated varint: never decodes
VIDEO_INFO = '<Video><BtVideoInfo DbId="v{n}"><Clip>{blob}</Clip></BtVideoInfo></Video>'
AUDIO_INFO = (
    '<EmbeddedAudioVec><Element><BtAudioInfo DbId="a{n}"><Clip>{blob}</Clip></BtAudioInfo>'
    "</Element></EmbeddedAudioVec>"
)


def drp_with(tmp_path: Path, pool_items: str, seq_body: str = "") -> Path:
    """small_drp plus a bin holding ``pool_items`` and a SeqContainer holding ``seq_body``."""
    path = small_drp(tmp_path / "x.drp")
    pool = f'<Sm2MpFolder DbId="b2"><Name>Bin</Name><MediaVec>{pool_items}</MediaVec></Sm2MpFolder>'
    with zipfile.ZipFile(path, "a") as zf:
        zf.writestr("MediaPool/Master/Bin/MpFolder.xml", xml(pool))
        if seq_body:
            zf.writestr(
                "SeqContainer/c.xml",
                xml(f'<Sm2SequenceContainer DbId="c">{seq_body}</Sm2SequenceContainer>'),
            )
    return path


def items(vec: str, item_tag: str, *paths: str, ref: str = "v1") -> str:
    elements = "".join(
        f'<Element><{item_tag} DbId="{item_tag}{i}"><Name>n</Name><Start>{i * 10}</Start>'
        f"<Duration>10</Duration><MediaRef>{ref}</MediaRef>"
        f"<MediaFilePath>{p}</MediaFilePath></{item_tag}></Element>"
        for i, p in enumerate(paths)
    )
    return (
        f'<{vec}><Element><Sm2TiTrack DbId="t{vec}"><Type>0</Type><Sequence>s</Sequence>'
        f"<Items>{elements}</Items></Sm2TiTrack></Element></{vec}>"
    )


def video_items(*paths: str, ref: str = "v1") -> str:
    return items("VideoTrackVec", "Sm2TiVideoClip", *paths, ref=ref)


def audio_items(*paths: str, ref: str = "v1") -> str:
    return items("AudioTrackVec", "Sm2TiAudioClip", *paths, ref=ref)


# ----------------------------------------------------------------------------- M1


def test_duplicate_audio_clip_cannot_change_the_first_clips_stream(tmp_path: Path) -> None:
    pool = (
        '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name></Sm2MpVideoClip></Element>'
        '<Element><Sm2MpAudioClip DbId="v1"><Name>a</Name>'
        + AUDIO_INFO.format(n=1, blob=clip_blob("/rec", "a.wav"))
        + "</Sm2MpAudioClip></Element>"
    )
    drp = read_drp(drp_with(tmp_path, pool, audio_items("/rec/a.wav")))
    clip = drp.media_by_id()["v1"]
    assert (clip.element, clip.name) == ("Sm2MpVideoClip", "v")
    assert (clip.path, clip.path_source) == (None, "none")
    assert issues(drp, "duplicate-id") and issues(drp, "path-unknown")


@pytest.mark.parametrize("element", ["Sm2MpVideoClip", "Sm2MpStillClip"])
def test_stream_less_file_clip_never_takes_an_audio_path(tmp_path: Path, element: str) -> None:
    pool = f'<Element><{element} DbId="v1"><Name>v</Name></{element}></Element>'
    drp = read_drp(drp_with(tmp_path, pool, audio_items("/rec/a.wav")))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source) == (None, "none")
    assert issues(drp, "path-unknown") and issues(drp, "no-main-stream")


def test_stream_less_compound_clip_has_no_file_path(tmp_path: Path) -> None:
    pool = '<Element><Sm2MpCompoundClip DbId="v1"><Name>c</Name></Sm2MpCompoundClip></Element>'
    drp = read_drp(drp_with(tmp_path, pool, audio_items("/rec/a.wav") + video_items("/x.mov")))
    clip = drp.media_by_id()["v1"]
    # A compound/multicam/timeline clip is a container: it never gets a file path.
    assert (clip.path, clip.path_source) == (None, "none")


def test_video_clip_with_only_an_audio_node_never_takes_the_audio_file(tmp_path: Path) -> None:
    pool = (
        '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name>'
        + AUDIO_INFO.format(n=1, blob=clip_blob("/rec", "a.wav", "pcm"))
        + "</Sm2MpVideoClip></Element>"
    )
    drp = read_drp(drp_with(tmp_path, pool, audio_items("/rec/a.wav") + video_items("/cam/a.mov")))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source, clip.blob_path) == ("/cam/a.mov", "timeline_item", None)
    assert clip.codec is None
    assert issues(drp, "no-main-stream")


def test_fallback_path_is_the_first_main_stream_item_and_conflicts_are_reported(
    tmp_path: Path,
) -> None:
    pool = (
        '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name>'
        + VIDEO_INFO.format(n=1, blob=UNREADABLE_BLOB)
        + "</Sm2MpVideoClip></Element>"
    )
    seq = audio_items("/rec/a.wav") + video_items("/first.mov", "/second.mov", "/third.mov")
    drp = read_drp(drp_with(tmp_path, pool, seq))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source) == ("/first.mov", "timeline_item")
    assert clip.timeline_paths == ["/rec/a.wav", "/first.mov", "/second.mov", "/third.mov"]
    [conflict] = issues(drp, "timeline-path-conflict")
    assert conflict.message.startswith("1 media clip(s)")


def test_agreeing_main_stream_items_raise_no_conflict(tmp_path: Path) -> None:
    pool = '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name></Sm2MpVideoClip></Element>'
    drp = read_drp(drp_with(tmp_path, pool, video_items("/a.mov", "/a.mov")))
    assert drp.media_by_id()["v1"].path == "/a.mov"
    assert not issues(drp, "timeline-path-conflict")


def test_path_from_a_later_video_stream_is_flagged(tmp_path: Path) -> None:
    pool = (
        '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name>'
        + VIDEO_INFO.format(n=1, blob=UNREADABLE_BLOB)
        + VIDEO_INFO.format(n=2, blob=clip_blob("/cam", "right.mov", "h264"))
        + "</Sm2MpVideoClip></Element>"
    )
    drp = read_drp(drp_with(tmp_path, pool))
    clip = drp.media_by_id()["v1"]
    assert (clip.path, clip.path_source) == ("/cam/right.mov", "media_pool_blob")
    [flag] = issues(drp, "path-from-later-stream")
    assert flag.message.startswith("1 media clip(s)")


def test_readable_first_video_stream_is_not_flagged(tmp_path: Path) -> None:
    pool = (
        '<Element><Sm2MpVideoClip DbId="v1"><Name>v</Name>'
        + VIDEO_INFO.format(n=1, blob=clip_blob("/cam", "left.mov"))
        + VIDEO_INFO.format(n=2, blob=clip_blob("/cam", "right.mov"))
        + "</Sm2MpVideoClip></Element>"
    )
    drp = read_drp(drp_with(tmp_path, pool))
    clip = drp.media_by_id()["v1"]
    assert clip.path == "/cam/left.mov" and clip.linked_paths == ["/cam/right.mov"]
    assert not issues(drp, "path-from-later-stream")


def test_join_clip_path_is_public_and_unchanged() -> None:
    assert join_clip_path("D:\\Footage", "a.mov") == "D:\\Footage\\a.mov"
    assert join_clip_path("/Volumes/x", "a.mov") == "/Volumes/x/a.mov"
    assert join_clip_path("/Volumes/x/", "a.mov") == "/Volumes/x/a.mov"
    assert join_clip_path(None, "a.mov") == "a.mov"
    assert join_clip_path("/dir", None) == "/dir"
    assert join_clip_path(None, None) is None
    assert MARKER_COLORS[2] == "Blue"


# ----------------------------------------------------------------------------- M2


class CountingList(list[str]):
    iterations = 0

    def __iter__(self):  # type: ignore[no-untyped-def]
        CountingList.iterations += 1
        return super().__iter__()


def test_describe_reads_the_raw_member_list_a_constant_number_of_times() -> None:
    drp = read_drp(FIXTURES / "empty-project.drp")
    names = [f"stills/{i}.xml" for i in range(500)]
    drp.members = [MemberSummary(n, 1, 1, "gallery-still", False) for n in names]
    drp.raw_members = CountingList(names)
    CountingList.iterations = 0
    text = describe(drp)
    assert "not interpreted (kept raw): 500 member(s): 500 gallery-still" in text
    assert CountingList.iterations <= 2  # was once per member: quadratic in member count


# ----------------------------------------------------------------------------- M3


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
def test_inspect_json_is_valid_in_any_output_encoding(tmp_path: Path, encoding: str) -> None:
    name = "Caf\u00e9 \U0001f600 \ud800 \x1b"
    path = small_drp(
        tmp_path / "e.drp",
        f"<ProjectName>{name.replace(chr(0xD800), '&#xD800;')}</ProjectName>".replace(
            "\x1b", "&#x1B;"
        ),
    )
    env = {**os.environ, "PYTHONIOENCODING": encoding}
    proc = subprocess.run(
        [sys.executable, "-c", RUN_CLI, "drp", "inspect", "--json", str(path)],
        capture_output=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.isascii()
    data = json.loads(proc.stdout.decode("ascii"))
    assert data["project"]["name"] == name


# ----------------------------------------------------------------------------- M4


def test_element_budget_stops_the_scan() -> None:
    doc = scan(b"<r>" + b"<a/>" * 50 + b"</r>", max_elements=10)
    assert doc.element_count == 10 and doc.stopped_at is not None
    assert [i.code for i in doc.issues] == ["budget-exceeded"]
    assert doc.root is not None and len(doc.root.children) == 9


def test_element_budget_is_per_file(tmp_path: Path) -> None:
    # 2 MB of "<a/>" deflates to a few KB: without a budget this is ~500k elements (~230 MB).
    path = small_drp(tmp_path / "bomb.drp")
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("Gallery2.xml", xml("<G>" + "<a/>" * 500_000 + "</G>"))
        zf.writestr("Gallery3.xml", xml("<G>" + "<a/>" * 500_000 + "</G>"))
    assert path.stat().st_size < 50_000
    drp = read_drp(path, check_blobs=True, max_elements=1_000)
    stops = issues(drp, "scan-budget-exceeded")
    assert [i.where.member for i in stops if i.where] == ["Gallery2.xml", "Gallery3.xml"]
    assert drp.project is not None and drp.project.name == "P"


def test_xml_budget_is_per_file(tmp_path: Path) -> None:
    path = small_drp(tmp_path / "x.drp")
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "SeqContainer/big.xml",
            xml("<Sm2SequenceContainer>" + " " * 200_000 + "</Sm2SequenceContainer>"),
        )
    drp = read_drp(path, max_total_xml_bytes=10_000)
    [refused] = issues(drp, "xml-budget-exceeded")
    assert refused.message.startswith("1 member(s)")
    assert drp.project is not None and drp.project.name == "P"


def _blob_bomb(tmp_path: Path, count: int, size: int, *, declare: bool) -> Path:
    zstandard = pytest.importorskip("zstandard", reason=ZSTD_REASON)
    frame = zstandard.ZstdCompressor(write_content_size=declare).compress(b"\x00" * size)
    blob = envelope(2, 0x81, frame).hex()
    body = "<ProjectName>P</ProjectName>" + f"<X><FieldsBlob>{blob}</FieldsBlob></X>" * count
    return small_drp(tmp_path / "z.drp", body)


@pytest.mark.parametrize("declare", [True, False])
def test_decompression_budget_is_per_file(tmp_path: Path, declare: bool) -> None:
    path = _blob_bomb(tmp_path, count=20, size=4 * 1024 * 1024, declare=declare)
    assert path.stat().st_size < 50_000
    drp = read_drp(path, check_blobs=True, max_total_blob_output=10 * 1024 * 1024)
    [refused] = issues(drp, "blob-budget-exceeded")
    # Two 4 MB blobs fit; the third is refused up front (declared size) or stopped at the
    # budget (undeclared), and the rest are refused without decompressing anything.
    assert refused.message.startswith("18 blob(s)")
    failures = drp.blob_stats.failures
    assert sum(n for f, n in failures.items() if "budget-exceeded" in f) == 18


def test_decode_budget_charges_failed_decodes() -> None:
    zstandard = pytest.importorskip("zstandard", reason=ZSTD_REASON)
    frame = zstandard.ZstdCompressor(write_content_size=False).compress(b"\x00" * 3000)
    budget = blobs.DecodeBudget(limit=10_000)
    with pytest.raises(blobs.BlobError, match="exceeds 1000"):
        blobs.zstd_decompress(frame, max_output=1000, budget=budget)
    assert budget.used == 1000  # the output produced before failing still counts


# ----------------------------------------------------------------------------- minors: output


def test_closed_stdout_exits_74_without_traceback() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", RUN_CLI, "drp", "describe", str(FIXTURES / "empty-project.drp")],
        stderr=subprocess.PIPE,
        preexec_fn=lambda: os.close(1),
        timeout=60,
    )
    assert proc.returncode == 74
    assert b"stdout is closed" in proc.stderr and b"Traceback" not in proc.stderr


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
@pytest.mark.parametrize("command", [["describe"], ["inspect"], ["inspect", "--json"]])
def test_full_disk_exits_74_without_traceback(command: list[str]) -> None:
    with open("/dev/full", "wb") as full:
        proc = subprocess.run(
            [sys.executable, "-c", RUN_CLI, "drp", *command, str(FIXTURES / "empty-project.drp")],
            stdout=full,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    assert proc.returncode == 74
    assert b"No space left" in proc.stderr
    assert b"Traceback" not in proc.stderr and b"Exception ignored" not in proc.stderr


def test_small_output_into_a_closed_pipe_exits_141_quietly() -> None:
    # The read end is closed before the child starts, so even a tiny output hits EPIPE at the
    # final flush. Without stdout being redirected to /dev/null the interpreter's own flush at
    # exit fails again and Python exits 120 with "Exception ignored".
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", RUN_CLI, "drp", "describe", str(FIXTURES / "empty-project.drp")],
            stdout=write_end,
            stderr=subprocess.PIPE,
            timeout=60,
        )
    finally:
        os.close(write_end)
    assert proc.returncode == 141
    assert proc.stderr == b""


def test_control_and_bidi_characters_are_escaped_in_text_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hostile = "&#x0;a&#x1B;[2Jb&#x202E;c&#x2028;d\U0001f600"
    path = small_drp(tmp_path / "h.drp", f"<ProjectName>{hostile}</ProjectName>")
    assert main(["drp", "describe", str(path)]) == 0
    out = capsys.readouterr().out
    assert 'project "\\x00a\\x1b[2Jb\\u202ec\\u2028d\U0001f600"' in out
    assert main(["drp", "inspect", str(path)]) == 0
    out += capsys.readouterr().out
    for ch in ("\x00", "\x1b", "\u202e", "\u2028"):
        assert ch not in out
    assert printable("tab\there") == "tab\\x09here"


# ----------------------------------------------------------------------------- minors: issues


def test_reader_issues_are_capped_per_code(tmp_path: Path) -> None:
    clips = "".join(
        '<Element><Sm2MpVideoClip DbId="dup"><Name>n</Name></Sm2MpVideoClip></Element>'
        for _ in range(301)
    )
    drp = read_drp(drp_with(tmp_path, clips))
    assert len(issues(drp, "duplicate-id")) == 100
    [summary] = issues(drp, "issues-truncated")
    assert summary.message == "200 more duplicate-id issue(s) not listed"
