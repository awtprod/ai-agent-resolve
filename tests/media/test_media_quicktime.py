"""S18a: the start timecode from a QuickTime/MP4/BRAW file's tmcd track, without ffprobe.

ffmpeg-made ``.mov``/``.mp4`` files must give what ffprobe gives; a BRAW-like file built from
scratch (``qt_support``) must give its sample; malformed files must give a typed
``TimecodeUnreadable`` within a bounded number of reads; and, when ``AAR_BRAW_SAMPLE_DIR`` points
at the head and tail of the real CCN BRAW (never committed), it must read 521420 at 30000/1001.
"""

from __future__ import annotations

import io
import json
import os
import random
import struct
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import pytest
from media_support import FFPROBE, ffmpeg, needs_ffmpeg

from aar.cli import main
from aar.media import MediaError, MediaInputError, quicktime
from aar.media.probe import (
    STANDARD_RATES,
    frames_timecode,
    parse_probe_json,
    snap_rate,
    timecode_frames,
)
from aar.media.quicktime import (
    MAX_ATOMS,
    MAX_MOOV_BYTES,
    MAX_TOP_LEVEL_ATOMS,
    QuickTimeTimecode,
    TimecodeUnreadable,
    frames_per_day,
    read_timecode,
    read_timecode_from,
)
from qt_support import TMCD_AT, Movie, Sparse, atom, co64, hdlr, stco, stsd, tmcd_entry, trak

NTSC = Fraction(30000, 1001)
NTSC60 = Fraction(60000, 1001)


def read_bytes(data: bytes) -> QuickTimeTimecode:
    return read_timecode_from(io.BytesIO(data), len(data))


def refused(data: bytes, code: str, match: str = "") -> TimecodeUnreadable:
    with pytest.raises(TimecodeUnreadable, match=match) as caught:
        read_bytes(data)
    assert caught.value.code == code, caught.value
    return caught.value


# -- frames_timecode, the label ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("frames", "fps", "drop", "text"),
    [
        (0, Fraction(24), False, "00:00:00:00"),
        (86400, Fraction(24), False, "01:00:00:00"),
        (86400, Fraction(24000, 1001), False, "01:00:00:00"),
        (90000, Fraction(25), False, "01:00:00:00"),
        (108000, NTSC, False, "01:00:00:00"),
        (521420, NTSC, False, "04:49:40:20"),  # the CCN BRAW
        (1799, NTSC, True, "00:00:59;29"),
        (1800, NTSC, True, "00:01:00;02"),
        (17982, NTSC, True, "00:10:00;00"),
        (107892, NTSC, True, "01:00:00;00"),
        (3600, NTSC60, True, "00:01:00;04"),
        (215784, NTSC60, True, "01:00:00;00"),
        (24 * 86400, Fraction(24), False, "24:00:00:00"),  # hours are not wrapped
    ],
)
def test_frames_timecode_labels_a_frame(frames: int, fps: Fraction, drop: bool, text: str) -> None:
    assert frames_timecode(frames, fps, drop_frame=drop) == text


@pytest.mark.parametrize(("fps", "drop"), [(Fraction(24), False), (NTSC, False), (NTSC, True),
                                           (NTSC60, True), (Fraction(25), False)])  # fmt: skip
def test_frames_timecode_inverts_timecode_frames(fps: Fraction, drop: bool) -> None:
    rng = random.Random(18)
    day = frames_per_day(fps, drop_frame=drop)
    samples = [*range(0, 40000, 7), *(rng.randrange(day) for _ in range(3000)), day - 1]
    for frames in samples:
        assert timecode_frames(frames_timecode(frames, fps, drop_frame=drop), fps) == frames


@pytest.mark.parametrize(
    ("frames", "fps", "drop", "why"),
    [
        (-1, Fraction(24), False, "before 00:00:00:00"),
        (0, Fraction(1, 3), False, "no frames a second"),
        (0, Fraction(24), True, "drop-frame exists only"),
        (0, Fraction(25), True, "drop-frame exists only"),
    ],
)
def test_frames_timecode_refuses(frames: int, fps: Fraction, drop: bool, why: str) -> None:
    with pytest.raises(ValueError, match=why):
        frames_timecode(frames, fps, drop_frame=drop)


@pytest.mark.parametrize(
    ("rate", "snapped"),
    [
        (Fraction(2953125, 98536), NTSC),  # the CCN BRAW's video average
        (Fraction(2997, 100), NTSC),
        (NTSC, NTSC),
        (Fraction(30), Fraction(30)),
        (Fraction(24), Fraction(24)),
        (Fraction(2997, 125), Fraction(24000, 1001)),  # 23.976
        (Fraction(5994, 100), NTSC60),
        (Fraction(47952, 1000), Fraction(48000, 1001)),  # 47.952
        (Fraction(48000, 1001), Fraction(48000, 1001)),
        (Fraction(4801, 100), Fraction(48)),
        (Fraction(48), Fraction(48)),
        (Fraction(50), Fraction(50)),
        (Fraction(60), Fraction(60)),
        (Fraction(2401, 100), Fraction(24)),  # 0.04 % off
        (Fraction(241, 10), None),  # 0.42 % off: no standard rate
        (Fraction(25, 2), None),
        (Fraction(120), None),
        (Fraction(0), None),
        (None, None),
    ],
)
def test_snap_rate(rate: Fraction | None, snapped: Fraction | None) -> None:
    assert snap_rate(rate) == snapped


def test_the_standard_rates_are_pinned() -> None:
    """Validator R2-5 (N2): the list, 47.952 and 48 included."""
    assert tuple(
        Fraction(n, d)
        for n, d in [(24000, 1001), (24, 1), (25, 1), (30000, 1001), (30, 1), (48000, 1001),
                     (48, 1), (50, 1), (60000, 1001), (60, 1)]
    ) == STANDARD_RATES  # fmt: skip
    assert all(snap_rate(rate) == rate for rate in STANDARD_RATES)


def test_the_probe_keeps_r_frame_rate_beside_the_average() -> None:
    raw = {"index": 0, "codec_type": "video", "avg_frame_rate": "2953125/98536",
           "r_frame_rate": "30000/1001"}  # fmt: skip
    video = parse_probe_json({"streams": [raw], "format": {}}, "x.braw", 1).video
    assert video is not None and (video.fps, video.base_fps) == (Fraction(2953125, 98536), NTSC)
    assert video.to_dict()["base_fps"] == "30000/1001"
    only_r = parse_probe_json(
        {"streams": [{**raw, "avg_frame_rate": "0/0"}], "format": {}}, "x.mov", 1
    ).video
    assert only_r is not None and only_r.fps == only_r.base_fps == NTSC


def test_frames_per_day() -> None:
    assert frames_per_day(Fraction(24), drop_frame=False) == 2073600
    assert frames_per_day(NTSC, drop_frame=False) == 2592000
    assert frames_per_day(NTSC, drop_frame=True) == 2589408
    assert frames_per_day(NTSC60, drop_frame=True) == 5178816


# -- the BRAW-like file ------------------------------------------------------------------------


def test_a_braw_like_file_gives_its_tmcd_sample(tmp_path: Path) -> None:
    path = tmp_path / "B006_10010449_C074.braw"
    path.write_bytes(Movie().build())
    got = read_timecode(path)
    assert got == QuickTimeTimecode(
        frames=521420,
        rate=NTSC,
        drop_frame=False,
        wraps_24h=True,
        flags=2,
        timescale=30000,
        frame_duration=1001,
        frames_per_second=30,
        sample_offset=TMCD_AT,
    )
    assert got.timecode == "04:49:40:20"
    data = got.to_dict()
    assert (data["frames"], data["rate"], data["drop_frame"]) == (521420, "30000/1001", False)
    assert data["timecode"] == "04:49:40:20" and data["fps"] == 29.97003
    assert data["stub"] == {
        "start_timecode": 521420,
        "start_timecode_rate": {"num": 30000, "den": 1001},
    }


def test_the_stub_fields_take_the_snapped_rate() -> None:
    movie = Movie(frame=108000)
    movie.tmcd_stbl = [stsd(tmcd_entry(0, 2997, 100, 30)), co64(TMCD_AT)]
    got = read_bytes(movie.build())
    data = got.to_dict()
    assert got.rate == Fraction(2997, 100) and data["rate"] == "2997/100"  # as written
    assert data["stub"]["start_timecode_rate"] == {"num": 30000, "den": 1001}
    movie.tmcd_stbl = [stsd(tmcd_entry(0, 25, 2, 12)), co64(TMCD_AT)]  # 12.5: no standard rate
    assert read_bytes(movie.build()).to_dict()["stub"]["start_timecode_rate"] == {
        "num": 25,
        "den": 2,
    }


def test_a_64_bit_mdat_and_a_co64_offset_past_4_gib_are_read_by_seeking() -> None:
    view, size = Movie(frame=1234).big(5 * 2**30, 4 * 2**30 + 12345)
    got = read_timecode_from(view, size)
    assert (got.frames, got.sample_offset) == (1234, 4 * 2**30 + 12345)
    # Only headers and a few small fields: never the mdat, never the whole moov.
    assert view.bytes_read < 1024 and len(view.reads) < 64


def test_moov_at_the_start_and_32_bit_chunk_offsets() -> None:
    movie = Movie()
    ftyp = atom(b"ftyp", b"qt  \0\0\0\0")
    movie.tmcd_stbl = [stsd(tmcd_entry(0, 24, 1, 24)), stco(0)]
    sample_at = len(ftyp) + len(movie.moov()) + 8  # the moov's length does not depend on it
    movie.tmcd_stbl = [stsd(tmcd_entry(0, 24, 1, 24)), stco(sample_at)]
    data = ftyp + movie.moov() + atom(b"mdat", struct.pack(">I", 86400))
    got = read_bytes(data)
    assert got.sample_offset == sample_at == len(data) - 4
    assert (got.frames, got.rate, got.timecode) == (86400, Fraction(24), "01:00:00:00")


def test_the_media_handler_is_the_one_in_mdia_not_minf() -> None:
    """Every track's minf carries an 'alis' data handler; a track whose minf handler says
    'tmcd' but whose mdia handler says 'vide' is not the timecode track."""
    decoy = trak(b"vide", stsd(tmcd_entry()), co64(10**9), minf_extra=[hdlr(b"tmcd")])
    got = read_bytes(Movie(tracks_before=[decoy]).build())
    assert got.frames == 521420


def test_only_a_trak_is_a_track() -> None:
    """An atom shaped like a tmcd track but not called 'trak' (here 'udta', pointing at other
    bytes of the mdat) is not the timecode track."""
    movie = Movie()
    decoy = bytearray(movie.tmcd_track(TMCD_AT + 4))
    decoy[4:8] = b"udta"
    movie.tracks_before = [bytes(decoy)]
    assert read_bytes(movie.build()).frames == 521420


def test_the_first_tmcd_track_wins() -> None:
    movie = Movie(frame=7)
    movie.tracks_after = [movie.tmcd_track(TMCD_AT + 4)]  # a second one, pointing at zeros
    assert read_bytes(movie.build()).frames == 7


@pytest.mark.parametrize(
    ("flags", "scale", "duration", "per_second", "frame", "want"),
    [
        (0x1, 30000, 1001, 30, 107892, "01:00:00;00"),  # drop-frame: real frames, used as given
        (0x1, 60000, 1001, 60, 215784, "01:00:00;00"),
        (0x0, 24000, 1001, 24, 86400, "01:00:00:00"),
        (0x0, 25, 1, 25, 90000, "01:00:00:00"),
        (0x2, 2400, 100, 24, 2073600 + 5, "00:00:00:05"),  # past 24 h with the wrap flag
        (0x3, 30000, 1001, 30, 2589408 + 1800, "00:01:00;02"),
        (0x8, 24, 1, 24, 86400, "01:00:00:00"),  # the counter flag: still a frame number
    ],
)
def test_tmcd_flags_and_rates(
    flags: int, scale: int, duration: int, per_second: int, frame: int, want: str
) -> None:
    movie = Movie(frame=frame)
    movie.tmcd_stbl = [stsd(tmcd_entry(flags, scale, duration, per_second)), co64(TMCD_AT)]
    got = read_bytes(movie.build())
    assert got.timecode == want
    assert got.rate == Fraction(scale, duration)
    assert got.drop_frame is bool(flags & 1) and got.wraps_24h is bool(flags & 2)
    assert got.frames == frame % frames_per_day(got.rate, drop_frame=got.drop_frame)


# -- against ffprobe ---------------------------------------------------------------------------


def _ffprobe_timecode(path: Path) -> str:
    assert FFPROBE is not None
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", f"file:{path}"],
        check=True,
        capture_output=True,
        timeout=60,
    ).stdout
    tags = {s.get("tags", {}).get("timecode") for s in json.loads(out)["streams"]} - {None}
    assert len(tags) == 1, tags
    return tags.pop()


@needs_ffmpeg
@pytest.mark.parametrize(
    ("rate", "timecode", "suffix", "extra"),
    [
        ("24", "01:00:00:00", ".mov", ()),
        ("25", "10:11:12:13", ".mov", ()),
        ("30000/1001", "01:00:00:00", ".mov", ()),  # 29.97 non-drop
        ("30000/1001", "01:00:00;00", ".mov", ()),  # 29.97 drop-frame
        ("60000/1001", "00:10:00;00", ".mov", ()),
        ("24000/1001", "23:59:59:23", ".mp4", ()),
        ("24", "01:00:00:00", ".mov", ("-movflags", "+faststart")),  # moov first
    ],
)
def test_the_reader_matches_ffprobe(
    tmp_path: Path, rate: str, timecode: str, suffix: str, extra: tuple[str, ...]
) -> None:
    out = tmp_path / f"tc{suffix}"
    ffmpeg(
        "-f", "lavfi", "-i", f"testsrc=size=64x36:rate={rate}", "-t", "0.5",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-timecode", timecode, *extra, str(out),
    )  # fmt: skip
    fps = Fraction(rate)
    tag = _ffprobe_timecode(out)
    got = read_timecode(out)
    assert got.timecode == tag == timecode
    assert got.frames == timecode_frames(tag, fps)
    assert got.rate == fps and got.drop_frame is (";" in timecode)


@needs_ffmpeg
def test_a_movie_without_a_timecode_track_says_so(tmp_path: Path) -> None:
    out = tmp_path / "plain.mov"
    ffmpeg(
        "-f", "lavfi", "-i", "testsrc=size=64x36:rate=24", "-t", "0.5",
        "-c:v", "libx264", "-preset", "ultrafast", str(out),
    )  # fmt: skip
    with pytest.raises(TimecodeUnreadable, match="no timecode") as caught:
        read_timecode(out)
    assert caught.value.code == "no_timecode"


# -- malformed input ---------------------------------------------------------------------------


def _tmcd(*children: bytes, frame: int = 521420) -> bytes:
    movie = Movie(frame=frame)
    movie.tmcd_stbl = list(children)
    return movie.build()


GOOD_STSD = stsd(tmcd_entry())


@pytest.mark.parametrize(
    ("data", "code", "match"),
    [
        pytest.param(b"", "not_quicktime", "too short", id="empty"),
        pytest.param(b"\0\0\0\x08wid", "not_quicktime", "too short", id="7-bytes"),
        pytest.param(atom(b"wide") + atom(b"free", bytes(64)), "not_quicktime", "no moov",
                     id="no-moov"),
        pytest.param(Movie().build()[:-100], "malformed", "past the end of the file",
                     id="truncated-moov"),
        pytest.param(Movie().build()[:5000], "malformed", "'mdat'.*past the end of the file",
                     id="truncated-mdat"),
        pytest.param(atom(b"wide") + struct.pack(">I4s", 10**9, b"mdat"), "malformed",
                     "past the end", id="size-past-eof"),
        pytest.param(atom(b"wide") + struct.pack(">I4s", 4, b"free") + bytes(8), "malformed",
                     "less than its 8-byte header", id="size-under-header"),
        pytest.param(atom(b"wide") + struct.pack(">I4sQ", 1, b"free", 0) + bytes(8),
                     "malformed", "less than its 16-byte header", id="loop-64-bit-zero"),
        pytest.param(atom(b"wide") + struct.pack(">I4sQ", 1, b"free", 8) + bytes(8),
                     "malformed", "less than its 16-byte header", id="loop-64-bit-eight"),
        pytest.param(atom(b"wide") + struct.pack(">I4s", 1, b"mdat") + b"\0\0", "malformed",
                     "64-bit size is cut off", id="cut-64-bit-size"),
        pytest.param(Movie(with_tmcd=False).build(), "no_timecode", "no timecode", id="no-tmcd"),
        pytest.param(Movie().build()[:-4], "malformed", "'moov'.*past the end of the file",
                     id="moov-short-by-4"),
        pytest.param(Movie().build()[:-1], "malformed", "'moov'.*past the end of the file",
                     id="moov-short-by-1"),
        pytest.param(_tmcd(GOOD_STSD, co64(10**12)), "malformed", "the tmcd sample at",
                     id="offset-past-eof"),
        pytest.param(_tmcd(GOOD_STSD, stco(2**32 - 2)), "malformed", "the tmcd sample at",
                     id="stco-offset-past-eof"),
        pytest.param(_tmcd(GOOD_STSD, co64(count=0)), "no_timecode", "no samples",
                     id="no-samples"),
        pytest.param(_tmcd(GOOD_STSD, co64(TMCD_AT, count=2)), "malformed", "lists 2 chunks",
                     id="co64-count-overflow"),
        pytest.param(_tmcd(GOOD_STSD, stco(TMCD_AT, count=2)), "malformed", "lists 2 chunks",
                     id="stco-count-overflow"),
        pytest.param(_tmcd(GOOD_STSD, atom(b"co64", bytes(4))), "malformed", "too short",
                     id="co64-cut"),
        pytest.param(_tmcd(GOOD_STSD), "malformed", "no chunk offsets", id="no-stco"),
        pytest.param(_tmcd(co64(TMCD_AT)), "malformed", "no 'stsd'", id="no-stsd"),
        pytest.param(_tmcd(stsd(count=0), co64(TMCD_AT)), "malformed", "no entries",
                     id="stsd-empty"),
        pytest.param(_tmcd(stsd(tmcd_entry(kind=b"avc1")), co64(TMCD_AT)), "malformed",
                     "'avc1', not 'tmcd'", id="stsd-not-tmcd"),
        pytest.param(_tmcd(stsd(tmcd_entry(size=33)), co64(TMCD_AT)), "malformed",
                     "claims 33 bytes", id="stsd-entry-short"),
        pytest.param(_tmcd(stsd(tmcd_entry(size=99)), co64(TMCD_AT)), "malformed",
                     "claims 99 bytes", id="stsd-entry-long"),
        pytest.param(_tmcd(stsd(tmcd_entry(timescale=0)), co64(TMCD_AT)), "malformed",
                     "must all be positive", id="timescale-0"),
        pytest.param(_tmcd(stsd(tmcd_entry(duration=0)), co64(TMCD_AT)), "malformed",
                     "must all be positive", id="duration-0"),
        pytest.param(_tmcd(stsd(tmcd_entry(per_second=0)), co64(TMCD_AT)), "malformed",
                     "must all be positive", id="frames-0"),
        pytest.param(_tmcd(stsd(tmcd_entry(per_second=60)), co64(TMCD_AT)), "unsupported",
                     "counts 60 frames a second at 29.97", id="frames-per-second-mismatch"),
        pytest.param(_tmcd(stsd(tmcd_entry(0x1, 25, 1, 25)), co64(TMCD_AT)), "unsupported",
                     "drop-frame at 25", id="drop-frame-at-25"),
        pytest.param(_tmcd(stsd(tmcd_entry(0x0)), co64(TMCD_AT), frame=2592000), "unsupported",
                     "past 24 hours", id="past-24h-no-wrap"),
        pytest.param(_tmcd(stsd(tmcd_entry(0x0)), co64(TMCD_AT), frame=2**32 - 1),
                     "unsupported", "past 24 hours", id="garbage-sample"),
    ],
)  # fmt: skip
def test_malformed_input_is_a_typed_refusal(data: bytes, code: str, match: str) -> None:
    started = time.perf_counter()
    error = refused(data, code, match)
    assert time.perf_counter() - started < 2.0
    assert isinstance(error, MediaError) and error.tool == "quicktime"


def test_the_last_byte_of_the_file_holds_a_sample() -> None:
    """The tmcd sample may end exactly at the end of the file, one byte later it may not."""
    movie = Movie()
    movie.tmcd_stbl = [GOOD_STSD, co64(0)]
    moov = movie.moov()
    data = atom(b"wide") + moov + atom(b"mdat", struct.pack(">I", 1234))
    for at, ok in ((len(data) - 4, True), (len(data) - 3, False)):
        movie.tmcd_stbl = [GOOD_STSD, co64(at)]
        built = atom(b"wide") + movie.moov() + atom(b"mdat", struct.pack(">I", 1234))
        if ok:
            assert read_bytes(built).frames == 1234
        else:
            refused(built, "malformed", f"the tmcd sample at {at} is past the end")


def test_a_child_that_overruns_its_parent_is_malformed() -> None:
    """A trak whose mdia claims 4 bytes more than the trak holds (the file is long enough)."""
    good = trak(b"vide", stsd(atom(b"brvl", bytes(8))), co64(0))
    tkhd_end = 8 + 8 + 84  # trak header, tkhd
    mdia_size = struct.unpack(">I", good[tkhd_end : tkhd_end + 4])[0]
    for extra in (1, 4):
        bad = good[:tkhd_end] + struct.pack(">I", mdia_size + extra) + good[tkhd_end + 4 :]
        refused(Movie(tracks_before=[bad]).build(), "malformed", "'mdia'.*past the end of 'trak'")


def test_padding_under_a_header_inside_a_container_is_ignored() -> None:
    """QuickTime allows a 32-bit zero terminator at the end of a container: a moov with no
    tmcd track and 4 zero bytes at its end has no timecode; it is not malformed."""
    moov = Movie(with_tmcd=False).moov()
    padded = struct.pack(">I", len(moov) + 4) + moov[4:] + bytes(4)
    data = atom(b"wide") + atom(b"mdat", bytes(64)) + padded
    refused(data, "no_timecode", "no timecode")
    lonely = atom(b"trak", atom(b"tkhd", bytes(84)), bytes(4))  # no mdia, then 4 bytes
    assert read_bytes(Movie(tracks_before=[lonely]).build()).frames == 521420


def test_a_short_media_handler_is_malformed() -> None:
    cut = atom(b"trak", atom(b"mdia", atom(b"hdlr", bytes(8))))
    refused(Movie(tracks_before=[cut]).build(), "malformed", "too short")


def test_a_tmcd_track_without_a_sample_table_is_malformed() -> None:
    bare = atom(b"trak", atom(b"mdia", hdlr(b"tmcd"), atom(b"minf", hdlr(b"alis", b"dhlr"))))
    refused(Movie(with_tmcd=False, tracks_before=[bare]).build(), "malformed", "no 'stbl'")
    refused(
        Movie(with_tmcd=False, tracks_before=[atom(b"trak", atom(b"mdia", hdlr(b"tmcd")))]).build(),
        "malformed",
        "no 'minf'",
    )


def test_a_huge_moov_is_refused_before_it_is_read() -> None:
    size = MAX_MOOV_BYTES + 1
    head = atom(b"wide") + struct.pack(">I4s", 8 + size, b"moov")
    view = Sparse(len(head) + size, {0: head})  # any read inside the moov raises LookupError
    with pytest.raises(TimecodeUnreadable, match="over the") as caught:
        read_timecode_from(view, view.size)
    assert caught.value.code == "too_large"
    # Exactly at the limit it is walked (and here has no tracks).
    head = atom(b"wide") + struct.pack(">I4s", 8 + MAX_MOOV_BYTES, b"moov")
    view = Sparse(len(head) + MAX_MOOV_BYTES, {0: head}, fill=True)
    with pytest.raises(TimecodeUnreadable) as caught:
        read_timecode_from(view, view.size)
    assert caught.value.code == "no_timecode"  # one zero-size child runs to its end: no trak


def test_the_top_level_walk_stops_at_1000_atoms() -> None:
    """Validator R2-3: each top-level atom costs a seek (a page read on a cold disk), so a walk
    visits at most MAX_TOP_LEVEL_ATOMS of them. wide + mdat + N 'free' + moov: 1000 in all is
    read, 1001 is refused before the 1001st header is read."""
    assert MAX_TOP_LEVEL_ATOMS == 1000
    good = Movie().build()
    moov_at = len(good) - len(Movie().moov())
    for free, ok in ((MAX_TOP_LEVEL_ATOMS - 3, True), (MAX_TOP_LEVEL_ATOMS - 2, False)):
        data = good[:moov_at] + atom(b"free") * free + good[moov_at:]
        view = Sparse(len(data), {0: data})
        if ok:
            assert read_timecode_from(view, len(data)).frames == 521420
            continue
        with pytest.raises(TimecodeUnreadable, match="more than 1000 atoms in the file") as caught:
            read_timecode_from(view, len(data))
        assert caught.value.code == "malformed"
        assert len(view.reads) == MAX_TOP_LEVEL_ATOMS and view.bytes_read == 8 * 1000


def test_the_walk_to_the_mdat_is_capped_too() -> None:
    """moov first, then 'free' atoms, then the mdat holding the sample: the second top-level
    walk (to check the sample is media) has the same 1000-atom cap."""
    movie = Movie()
    head = atom(b"ftyp", b"qt  \0\0\0\0")
    movie.tmcd_stbl = [GOOD_STSD, co64(0)]
    moov_len = len(movie.moov())
    for free, ok in ((MAX_TOP_LEVEL_ATOMS - 3, True), (MAX_TOP_LEVEL_ATOMS - 2, False)):
        sample_at = len(head) + moov_len + 8 * free + 8
        movie.tmcd_stbl = [GOOD_STSD, co64(sample_at)]
        data = head + movie.moov() + atom(b"free") * free + atom(b"mdat", struct.pack(">I", 99))
        if ok:
            assert read_bytes(data).frames == 99
        else:
            refused(data, "malformed", "more than 1000 atoms in the file")


def test_an_atom_flood_inside_moov_stops_at_the_atom_cap() -> None:
    """Size-8 'free' atoms inside moov: the walk is cut off at MAX_ATOMS headers in all."""
    count = MAX_ATOMS + 50
    moov = atom(b"moov", atom(b"free") * count)
    data = atom(b"wide") + moov
    view = Sparse(len(data), {0: data})
    started = time.perf_counter()
    with pytest.raises(TimecodeUnreadable, match=f"more than {MAX_ATOMS} atoms before") as caught:
        read_timecode_from(view, len(data))
    assert caught.value.code == "malformed"
    assert len(view.reads) == MAX_ATOMS and time.perf_counter() - started < 10.0


def test_100k_atoms_64_kib_apart_stop_at_the_top_level_cap() -> None:
    """The validator's case: 100k top-level 'free' atoms 64 KiB apart (a 6.5 GB file). Every
    header costs exactly 8 bytes and the walk stops typed after 1000 of them: 8 KB read, 1000
    seeks (a few MB from a cold disk), where round 1 read 409.6 MB."""
    count, step = 100_000, 65536
    segments = {i * step: struct.pack(">I4s", step, b"free") for i in range(count)}
    view = Sparse(count * step + 8, segments, fill=True)
    with pytest.raises(TimecodeUnreadable, match="more than 1000 atoms in the file") as caught:
        read_timecode_from(view, view.size)
    assert caught.value.code == "malformed"
    assert len(view.reads) == MAX_TOP_LEVEL_ATOMS and view.bytes_read == 8 * 1000
    assert all(n == 8 for _, n in view.reads)


def test_the_read_budget_is_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quicktime, "MAX_READ_BYTES", 100)
    with pytest.raises(TimecodeUnreadable, match="more than 100 bytes read") as caught:
        read_bytes(Movie().build())
    assert caught.value.code == "too_large"
    monkeypatch.setattr(quicktime, "MAX_READ_BYTES", 400)  # the whole read takes under 400
    assert read_bytes(Movie().build()).frames == 521420


@pytest.mark.skipif(not Path("/proc/self/io").exists(), reason="needs /proc/self/io (Linux)")
def test_reads_from_disk_are_unbuffered(tmp_path: Path) -> None:
    """A buffered file would fill 4-8 KiB on every seek; the reader's 8-byte header reads must
    stay 8 bytes at the system call (rchar), so a spaced-out file costs bytes, not megabytes."""
    count, step = 1500, 65536
    path = tmp_path / "spaced.mov"
    with path.open("wb") as out:
        out.truncate(count * step + 8)
        for i in range(count):
            out.seek(i * step)
            out.write(struct.pack(">I4s", step, b"free"))

    def rchar() -> int:
        text = Path("/proc/self/io").read_text()
        return int(next(line for line in text.splitlines() if line.startswith("rchar")).split()[1])

    before = rchar()
    with pytest.raises(TimecodeUnreadable):
        read_timecode(path)
    assert rchar() - before < 64 * 1024  # 1000 headers of 8 bytes, plus /proc itself


@pytest.mark.parametrize(
    ("where", "ok"),
    [("mdat-first", True), ("mdat-last", True), ("zero", False), ("mdat-header", False),
     ("moov", False), ("straddles-mdat-end", False), ("free", False)],
)  # fmt: skip
def test_the_sample_must_lie_in_an_mdat(where: str, ok: bool) -> None:
    """The tmcd sample's 4 bytes must be inside a top-level mdat's payload; an offset into the
    moov, a header or another atom is refused rather than read as a frame number."""
    payload = struct.pack(">I", 777) + bytes(8) + struct.pack(">I", 888)
    free = atom(b"free", bytes(16))
    movie = Movie()
    movie.tmcd_stbl = [GOOD_STSD, co64(0)]
    head = atom(b"wide") + free
    mdat_at = len(head)
    mdat = atom(b"mdat", payload)
    moov_at = mdat_at + len(mdat)
    offsets = {
        "mdat-first": mdat_at + 8,
        "mdat-last": mdat_at + len(mdat) - 4,
        "zero": 0,
        "mdat-header": mdat_at + 2,
        "moov": moov_at + 8,
        "straddles-mdat-end": mdat_at + len(mdat) - 2,
        "free": 8 + 8,
    }
    movie.tmcd_stbl = [GOOD_STSD, co64(offsets[where])]
    data = head + mdat + movie.moov()
    if ok:
        assert read_bytes(data).frames == {"mdat-first": 777, "mdat-last": 888}[where]
    else:
        refused(data, "malformed", "not inside an mdat")


def test_a_zero_size_atom_runs_to_the_end_of_its_parent() -> None:
    movie = Movie()
    data = movie.build()
    # The last top-level atom (moov) with size 0: to the end of the file.
    moov_at = len(data) - len(movie.moov())
    rewritten = data[:moov_at] + struct.pack(">I", 0) + data[moov_at + 4 :]
    assert read_bytes(rewritten).frames == 521420


def test_trailing_bytes_under_a_header_are_ignored() -> None:
    assert read_bytes(Movie().build() + b"\0\0\0").frames == 521420


def test_random_corruption_never_leaks_another_exception() -> None:
    """Seeded fuzz: flip 1-4 bytes of the movie's moov or headers, or cut it anywhere. Each
    read gives a timecode or a TimecodeUnreadable, never another exception."""
    rng = random.Random(1808)
    good = Movie().build()
    moov_at = len(good) - len(Movie().moov())
    outcomes = {"read": 0, "refused": 0}
    started = time.perf_counter()
    for _ in range(3000):
        data = bytearray(good)
        for _ in range(rng.randint(1, 4)):
            at = rng.choice([rng.randrange(0, 32), rng.randrange(moov_at, len(good))])
            data[at] = rng.randrange(256)
        if rng.random() < 0.2:
            data = data[: rng.randrange(len(data))]
        try:
            read_bytes(bytes(data))
            outcomes["read"] += 1
        except TimecodeUnreadable:
            outcomes["refused"] += 1
    assert outcomes["read"] > 100 and outcomes["refused"] > 100
    assert time.perf_counter() - started < 30.0


def test_a_file_that_shrinks_mid_read_is_an_io_refusal() -> None:
    data = Movie().build()
    with pytest.raises(TimecodeUnreadable, match="was it truncated") as caught:
        cut = len(data) - len(Movie().moov()) + 200  # into the moov, before the tmcd track
        read_timecode_from(io.BytesIO(data[:cut]), len(data))  # the size claims more
    assert caught.value.code == "io"


def test_a_failing_stream_is_an_io_refusal() -> None:
    class Broken(io.RawIOBase):
        def seek(self, offset: int, whence: int = 0) -> int:
            raise OSError(5, "Input/output error")

    with pytest.raises(TimecodeUnreadable, match="Input/output error") as caught:
        read_timecode_from(Broken(), 1000)
    assert caught.value.code == "io"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs FIFOs")
def test_a_fifo_swapped_in_after_the_check_is_refused_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The path check and the open race: the descriptor is opened non-blocking and checked
    again with fstat, so a FIFO put there in between is refused at once."""
    fifo = tmp_path / "swapped.mov"
    os.mkfifo(fifo)
    monkeypatch.setattr(quicktime, "require_regular_file", lambda path, tool="": Path(path))
    started = time.perf_counter()
    with pytest.raises(TimecodeUnreadable, match="not a regular file") as caught:
        read_timecode(fifo)
    assert caught.value.code == "io" and time.perf_counter() - started < 5.0


def test_missing_and_irregular_paths_are_input_errors(tmp_path: Path) -> None:
    with pytest.raises(MediaInputError, match="file not found"):
        read_timecode(tmp_path / "gone.braw")
    with pytest.raises(MediaInputError, match="not a regular file"):
        read_timecode(tmp_path)
    if hasattr(os, "mkfifo"):
        fifo = tmp_path / "pipe.mov"
        os.mkfifo(fifo)
        with pytest.raises(MediaInputError, match="not a regular file"):
            read_timecode(fifo)  # refused at once, never blocks on open


# -- the CLI -----------------------------------------------------------------------------------


def test_the_cli_prints_the_timecode_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "A001.braw"
    path.write_bytes(Movie().build())
    assert main(["media", "timecode", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True and out["path"] == str(path)
    assert (out["frames"], out["rate"], out["drop_frame"], out["timecode"]) == (
        521420,
        "30000/1001",
        False,
        "04:49:40:20",
    )
    assert out["stub"]["start_timecode_rate"] == {"num": 30000, "den": 1001}


def test_the_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a movie at all")
    assert main(["media", "timecode", str(junk)]) == 1
    error = json.loads(capsys.readouterr().err)["error"]
    assert error["code"] == "timecode_malformed"
    plain = tmp_path / "plain.mov"
    plain.write_bytes(Movie(with_tmcd=False).build())
    assert main(["media", "timecode", str(plain)]) == 1
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "timecode_no_timecode"
    assert main(["media", "timecode", str(tmp_path / "gone.braw")]) == 2
    assert json.loads(capsys.readouterr().err)["error"]["code"] == "invalid_input"


# -- the real BRAW (evidence only, never committed) --------------------------------------------

SAMPLE_DIR = os.environ.get("AAR_BRAW_SAMPLE_DIR", "").strip()
#: B006_10010449_C074.braw, of which the sample dir holds the first and last 4 MiB.
SAMPLE_SIZE = int(os.environ.get("AAR_BRAW_SAMPLE_SIZE", "3628178508"))


@pytest.mark.skipif(not SAMPLE_DIR, reason="set AAR_BRAW_SAMPLE_DIR to the BRAW head/tail dir")
def test_the_real_ccn_braw_reads_521420_at_29_97() -> None:
    """Reassemble the 3.6 GB file as a sparse view of its head and tail: any read outside them
    raises, so this also proves the reader touches nothing in between."""
    head = (Path(SAMPLE_DIR) / "braw_head.bin").read_bytes()
    tail = (Path(SAMPLE_DIR) / "braw_tail.bin").read_bytes()
    view = Sparse(SAMPLE_SIZE, {0: head, SAMPLE_SIZE - len(tail): tail})
    got = read_timecode_from(view, SAMPLE_SIZE)
    print(json.dumps(got.to_dict()), f"reads={len(view.reads)} bytes={view.bytes_read}")
    assert (got.frames, got.rate, got.drop_frame) == (521420, NTSC, False)
    assert got.timecode == "04:49:40:20"
    assert (got.flags, got.frames_per_second, got.sample_offset) == (0x2, 30, 4096)
    assert view.bytes_read < 4096
