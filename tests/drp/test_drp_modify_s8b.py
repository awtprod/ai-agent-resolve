"""S8b: follow-ups from the S8 validation (findings F-A to F-J and the listed test gaps).

Each test names the finding it pins. They run on the bundled templates only.
"""

from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import pytest
from diff_testing import CLIP, H264, POOL, SEQUENCE, TIMELINE
from drp_edit import add_carrier, carrier, edit_zip, marker_entry
from drp_helpers import kd, needs_zstandard, pb_len, zstd_raw_frame

from aar.drp import blobs
from aar.drp.modify import encode, parse_plan
from aar.drp.modify.encode import EncodeError
from aar.drp.modify.write import edit_file
from aar.drp.reader import read_drp

MAC_DIR = "/Users/samuelgursky/Desktop"
BLUE = 2


def plan(*edits: dict[str, Any], **top: Any) -> Any:
    return parse_plan(json.dumps({"edits": list(edits), **top}))


def run(src: Path, tmp_path: Path, *edits: dict[str, Any], **kw: Any) -> Any:
    out = tmp_path / "out.drp"
    return edit_file(src, plan(*edits), out, **kw), out


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name.startswith("."))


# --------------------------------------------------------------------------- F-A


def _split_frames(payload: bytes) -> bytes:
    """``payload`` as two concatenated zstd frames. With zstandard the reader decodes both (the
    content is the same); without a library only the first frame is read."""
    half = len(payload) // 2
    return encode.zstd_raw_frame(payload[:half]) + encode.zstd_raw_frame(payload[half:])


def _clip_in_two_frames(data: bytes) -> bytes:
    def repl(m: re.Match[bytes]) -> bytes:
        raw = bytes.fromhex(m.group(1).decode())
        env = blobs.parse_envelope(raw)
        assert env is not None and env.codec == "zstd" and env.version is not None
        body = b"\x81" + _split_frames(blobs.unwrap_message(raw))
        new = struct.pack(">II", env.version, len(body)) + body
        return b"<Clip>" + new.hex().encode() + b"</Clip>"

    new, n = re.subn(rb"<Clip>([0-9a-f]+)</Clip>", repl, data)
    assert n >= 1
    return new


@needs_zstandard
def test_fa_path_map_refuses_a_clip_blob_in_two_zstd_frames(tmp_path: Path) -> None:
    src = edit_zip(H264, tmp_path / "two-frames.drp", {POOL: _clip_in_two_frames})
    drp = read_drp(src)
    assert not [i for i in drp.issues if i.severity != "info"]
    assert drp.media_by_id()[CLIP].path == read_drp(H264).media_by_id()[CLIP].path
    table = [{"windows": "E:", "mac": MAC_DIR}]
    outcome, out = run(
        src, tmp_path, {"op": "path_map", "direction": "mac_to_windows", "table": table}
    )
    assert outcome.status == "refused"
    [refusal] = outcome.refusals
    assert refusal.code == "cannot-encode" and "one zstd frame" in refusal.message
    assert not out.exists()


def _two_frame_markers(*entries: bytes) -> str:
    body = b"\x81" + _split_frames(pb_len(2, b"".join(entries)))
    blob_data = struct.pack(">II", 10001, len(body)) + body
    return kd([("BlobData", 0x0C, struct.pack(">I", len(blob_data)) + blob_data)]).hex()


@needs_zstandard
def test_fa_marker_edit_refuses_blob_data_in_two_zstd_frames(tmp_path: Path) -> None:
    element = carrier(
        "0b0b0b0b-1111-2222-3333-444444444444",
        SEQUENCE,
        _two_frame_markers(marker_entry(10, BLUE, "Intro"), marker_entry(20, BLUE, "Outro")),
    )
    src = edit_zip(H264, tmp_path / "markers.drp", {"project.xml": add_carrier(element)})
    assert [m.name for m in read_drp(src).timelines[0].markers] == ["Intro", "Outro"]
    edit = {
        "op": "marker_update",
        "owner": "timeline",
        "owner_id": TIMELINE,
        "frame": 10,
        "set": {"name": "X"},
    }
    outcome, out = run(src, tmp_path, edit)
    assert [r.code for r in outcome.refusals] == ["no-marker-list"]
    assert "one zstd frame" in outcome.refusals[0].message
    assert not out.exists()


@pytest.mark.parametrize(
    ("frame", "why"),
    [
        (zstd_raw_frame(b"abc") + b"\x00", "trailing"),
        (zstd_raw_frame(b"abc") + zstd_raw_frame(b"d"), "trailing"),
        (zstd_raw_frame(b"abc")[:-1], "truncated"),
        (b"\x50\x2a\x4d\x18" + b"\x00" * 8, "not a zstd frame"),  # a skippable frame
        (zstd_raw_frame(b"abc")[:4] + bytes([0x21]) + zstd_raw_frame(b"abc")[5:], "dictionary"),
        (zstd_raw_frame(b"abc")[:6] + (7 << 1 | 1).to_bytes(3, "little"), "reserved"),
    ],
    ids=["byte-after", "second-frame", "truncated", "skippable", "dictionary", "reserved-block"],
)
def test_fa_check_single_frame_refuses(frame: bytes, why: str) -> None:
    with pytest.raises(EncodeError, match=why):
        encode.check_single_frame(frame)


def test_fa_check_single_frame_sizes() -> None:
    for payload in (b"", b"x" * 300, b"y" * 140_000):
        frame = encode.zstd_raw_frame(payload)
        assert encode.check_single_frame(frame) == len(frame)
    rle = b"\x28\xb5\x2f\xfd\x20\x10" + ((16 << 3) | (1 << 1) | 1).to_bytes(3, "little") + b"z"
    assert encode.check_single_frame(rle) == len(rle)
    with_checksum = (
        b"\x28\xb5\x2f\xfd\x24\x01" + ((1 << 3) | 1).to_bytes(3, "little") + b"q" + b"\0" * 4
    )
    assert encode.check_single_frame(with_checksum) == len(with_checksum)
    with pytest.raises(EncodeError, match="truncated"):
        encode.check_single_frame(with_checksum[:-1])


@needs_zstandard
def test_fa_compressed_frames_are_walked_by_block_size() -> None:
    import zstandard

    payload = b"compressible " * 5000
    frame = zstandard.ZstdCompressor(level=3).compress(payload)
    assert encode.check_single_frame(frame) == len(frame)
    with pytest.raises(EncodeError, match="trailing"):
        encode.check_single_frame(frame + frame)
