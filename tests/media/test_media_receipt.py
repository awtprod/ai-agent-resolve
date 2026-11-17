from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from aar.media.receipt import ReceiptError, check_receipt, receipt_path, write_receipt


@pytest.mark.parametrize("changed_field", [None, "st_ino", "st_size", "st_mtime_ns"])
def test_open_regular_accepts_different_path_and_descriptor_ctime(
    tmp_path, monkeypatch, changed_field
):
    from types import SimpleNamespace

    from aar.media import receipt

    path = tmp_path / "proxy.mp4"
    path.write_bytes(b"unchanged")
    original = receipt.os.fstat

    def windows_fstat(fd):
        value = original(fd)
        fields = {
            name: getattr(value, name)
            for name in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        }
        # CPython on Windows can report birth time by path and change time by fd.
        fields["st_ctime_ns"] += 1_000_000
        if changed_field is not None:
            fields[changed_field] += 1
        return SimpleNamespace(**fields)

    monkeypatch.setattr(receipt.os, "fstat", windows_fstat)
    if changed_field is None:
        with receipt.open_regular(path) as stream:
            assert stream.read() == b"unchanged"
    else:
        with (
            pytest.raises(ReceiptError, match="changed while opening"),
            receipt.open_regular(path),
        ):
            pass


def test_open_regular_rechecks_path_metadata_after_open(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from aar.media import receipt

    path = tmp_path / "proxy.mp4"
    path.write_bytes(b"media")
    original = Path.lstat
    calls = 0

    def changed_lstat(self, *args, **kwargs):
        nonlocal calls
        value = original(self, *args, **kwargs)
        if self != path:
            return value
        calls += 1
        fields = {
            name: getattr(value, name)
            for name in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        }
        if calls > 1:
            fields["st_ctime_ns"] += 1_000_000
        return SimpleNamespace(**fields)

    monkeypatch.setattr(Path, "lstat", changed_lstat)
    with pytest.raises(ReceiptError, match="changed while opening"), receipt.open_regular(path):
        pass


def test_receipt_binds_both_files_and_rejects_changed_copy(tmp_path: Path) -> None:
    source, copy = tmp_path / "source.mp4", tmp_path / "copy.mp4"
    source.write_bytes(b"original audio and video\r\n\x1a\x00")
    copy.write_bytes(source.read_bytes())
    write_receipt(source, copy)
    receipt = check_receipt(copy, str(source.resolve()))
    assert receipt["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["verification"] == "byte_identical"
    copy.write_bytes(b"different")
    with pytest.raises(ReceiptError, match="changed"):
        check_receipt(copy, str(source.resolve()))


def test_receipt_rejects_wrong_pc_path(tmp_path: Path) -> None:
    source, copy = tmp_path / "source.mp4", tmp_path / "copy.mp4"
    source.write_bytes(b"media")
    copy.write_bytes(b"media")
    write_receipt(source, copy)
    with pytest.raises(ReceiptError, match="path"):
        check_receipt(copy, "E:\\Wrong.mp4")


@pytest.mark.parametrize("data", [b"[]", b"{}", b"{", b"x" * 65537])
def test_malformed_receipt_is_refused(tmp_path: Path, data: bytes) -> None:
    media = tmp_path / "copy.mp4"
    media.write_bytes(b"media")
    receipt_path(media).write_bytes(data)
    with pytest.raises(ReceiptError):
        check_receipt(media, "E:\\Show.mp4")


def test_missing_receipt_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ReceiptError, match="missing"):
        check_receipt(tmp_path / "copy.mp4", "E:\\Show.mp4")


def test_receipt_never_replaces_foreign_sidecar(tmp_path: Path) -> None:
    source, copy = tmp_path / "source.mp4", tmp_path / "copy.mp4"
    source.write_bytes(b"media")
    copy.write_bytes(b"media")
    receipt_path(copy).write_text("human notes")
    with pytest.raises(ReceiptError):
        write_receipt(source, copy)
    assert receipt_path(copy).read_text() == "human notes"


def test_false_verification_and_nonfinite_time_refused(tmp_path: Path) -> None:
    source, copy = tmp_path / "source.mp4", tmp_path / "copy.mp4"
    source.write_bytes(b"media")
    copy.write_bytes(b"media")
    write_receipt(source, copy)
    body = json.loads(receipt_path(copy).read_text())
    body["verification"] = "trust_me"
    receipt_path(copy).write_text(json.dumps(body))
    with pytest.raises(ReceiptError):
        check_receipt(copy, str(source.resolve()))


def test_export_requires_receipts_for_every_mapped_source(tmp_path: Path) -> None:
    from aar.recipes.ccn.media_check import media_evidence
    from aar.recipes.context import StepFailure

    media = tmp_path / "proxy.mp4"
    media.write_bytes(b"frames and audio")
    rows = [{"plan_media": str(media), "windows_path": "E:\\Show\\proxy.mp4"}]
    with pytest.raises(StepFailure, match="media_receipt_missing"):
        media_evidence(rows)
    write_receipt(media, media)
    body = json.loads(receipt_path(media).read_text())
    body["source"] = "E:\\Show\\proxy.mp4"
    receipt_path(media).write_text(json.dumps(body))
    checked = media_evidence(rows)
    assert checked["status"] == "verified_snapshot"
    assert checked["files"][0]["path"] == body["source"]
    media.write_bytes(b"changed")
    with pytest.raises(StepFailure, match="media_receipt_invalid"):
        media_evidence(rows)


def test_shrink_command_writes_verified_receipt_with_audio(tmp_path: Path) -> None:
    import subprocess
    import sys

    from media_support import FFMPEG

    if not FFMPEG:
        pytest.skip("ffmpeg missing")
    source, copy = tmp_path / "source.mov", tmp_path / "share" / "copy.mov"
    copy.parent.mkdir()
    subprocess.run(
        [
            FFMPEG,
            "-hide_banner",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "pcm_s16le",
            "-timecode",
            "01:00:00:00",
            str(source),
        ],
        check=True,
        timeout=30,
        capture_output=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aar.media.shrink",
            str(source),
            "--out",
            str(copy),
            "--receipt",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checked = check_receipt(copy, str(source.resolve()))
    assert checked["verification"] == "verified_shrink"
    assert checked["facts"]["frames"] == 25
    assert checked["facts"]["audio"][0]["sample_rate"] == 48000


@pytest.mark.skipif(not hasattr(__import__("os"), "mkfifo"), reason="POSIX FIFO")
def test_receipt_fifo_is_refused_without_blocking(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    source = tmp_path / "proxy.mp4"
    source.write_bytes(b"valid")
    os.mkfifo(receipt_path(source))
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from aar.media.receipt import check_receipt; "
            'import sys; check_receipt(Path(sys.argv[1]), "E:\\\\proxy.mp4")',
            str(source),
        ],
        capture_output=True,
        timeout=2,
    )
    assert result.returncode != 0
    assert b"regular file" in result.stderr


def test_receipt_rejects_source_changed_with_restored_mtime(tmp_path: Path, monkeypatch) -> None:
    import os

    from aar.media import receipt

    source, copy = tmp_path / "source.mp4", tmp_path / "copy.mp4"
    source.write_bytes(b"first")
    copy.write_bytes(b"first")
    before = source.stat()
    original = receipt._hash

    def changed(path):
        result = original(path)
        if path == copy:
            source.write_bytes(b"other")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return result

    monkeypatch.setattr(receipt, "_hash", changed)
    with pytest.raises(ReceiptError, match="changed"):
        write_receipt(source, copy)
    assert not receipt_path(copy).exists()
