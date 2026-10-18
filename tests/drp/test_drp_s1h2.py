"""Follow-ups from the S1h and S2 reviews (dispatched as slice S1h2).

Reader/blob hardening: cumulative and per-kind budget charging, the default budget values, a
huge FrameRate x MediaExtents, a non-finite retime speed, one issue per budget trip, and a
timing guard. Diff/gate: the prolog X12/X15 probes, DbPrjVer spelling, exit codes when stderr
is broken, the short id of a one-character project-id change, and the zstd-unavailable gate.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

import pytest
from diff_testing import H264, SEQ_MEMBER, assert_mirrors, diff, edited, only_change
from drp_edit import once
from drp_helpers import kd, qstr, xml
from test_drp_hardening import issues, small_drp
from test_drp_synthetic import EXTENTS, members

from aar.drp import blobs, reader
from aar.drp.reader import read_drp

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"


# --------------------------------------------------------------------------- task 1: budgets


def test_element_budget_charges_cumulatively_across_members(tmp_path: Path) -> None:
    # One member spends most of the shared budget; the next then trips it. A per-member budget
    # (the bug this guards against) would read both whole and trip nothing.
    path = small_drp(tmp_path / "c.drp")
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("GalA.xml", xml("<G>" + "<a/>" * 40 + "</G>"))  # 44 tokens (incl. prolog)
        zf.writestr("GalB.xml", xml("<G>" + "<a/>" * 40 + "</G>"))  # 44 tokens
    # 80 holds the small members plus GalA but not GalB; a per-member budget would read both.
    drp = read_drp(path, check_blobs=True, max_elements=80)
    [stop] = issues(drp, "scan-budget-exceeded")
    assert stop.message.startswith("1 member(s)")


def test_xml_byte_budget_charges_cumulatively_across_members(tmp_path: Path) -> None:
    big = xml("<Sm2SequenceContainer>" + " " * 5_000 + "</Sm2SequenceContainer>")
    path = small_drp(tmp_path / "x.drp")
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SeqContainer/a.xml", big)
        zf.writestr("SeqContainer/b.xml", big)
    # ~5 KB each; 8 KB holds the small members plus one big one, not both.
    drp = read_drp(path, max_total_xml_bytes=8_000)
    [refused] = issues(drp, "xml-budget-exceeded")
    assert refused.message.startswith("1 member(s)")


def _qcompress(declared: int, actual: int) -> bytes:
    """A Qt qCompress blob declaring ``declared`` bytes but holding only ``actual`` zeros."""
    return struct.pack(">I", declared) + zlib.compress(b"\x00" * actual)


def test_qcompress_budget_is_charged(tmp_path: Path) -> None:
    blob = _qcompress(4 * 1024 * 1024, 4 * 1024 * 1024).hex()
    body = "<ProjectName>P</ProjectName>" + f"<X><RenderCacheBA>{blob}</RenderCacheBA></X>" * 3
    path = small_drp(tmp_path / "q.drp", body)
    drp = read_drp(path, check_blobs=True, max_total_blob_output=5 * 1024 * 1024)
    [refused] = issues(drp, "blob-budget-exceeded")
    # One 4 MB blob fits the 5 MB budget; the other two are refused.
    assert refused.message.startswith("2 blob(s)")
    assert any("budget-exceeded" in f for f in drp.blob_stats.failures)


def test_nested_keyed_dict_decode_charges_the_budget(tmp_path: Path) -> None:
    # A keyed dict whose byte-array value is itself a qCompress blob: the shared budget must be
    # charged one level down, not only for top-level blobs.
    inner = _qcompress(4 * 1024 * 1024, 4 * 1024 * 1024)
    blob = kd([("Cache", 0x0C, struct.pack(">I", len(inner)) + inner)]).hex()
    body = "<ProjectName>P</ProjectName>" + f"<X><FieldsBlob>{blob}</FieldsBlob></X>" * 3
    path = small_drp(tmp_path / "n.drp", body)
    drp = read_drp(path, check_blobs=True, max_total_blob_output=5 * 1024 * 1024)
    [refused] = issues(drp, "blob-budget-exceeded")
    assert refused.message.startswith("2 blob(s)")


def test_default_budget_values() -> None:
    # Pin the documented defaults; the element default was lowered so a 29 KB bomb stays under
    # ~1 GB of RAM (see reader.py and the S1h2 report).
    assert reader.DEFAULT_MAX_ELEMENTS == 3_000_000
    assert reader.DEFAULT_MAX_TOTAL_XML_BYTES == 600 * 1024 * 1024
    assert blobs.DEFAULT_MAX_TOTAL_OUTPUT == 400 * 1024 * 1024
    assert blobs.DEFAULT_MAX_OUTPUT == 64 * 1024 * 1024


# --------------------------------------------------------------------------- task 2: overflow


def _timeline_members(rate_hex: str, extents_hex: str) -> dict[str, str | bytes]:
    m = members(frame_rate_hex=rate_hex)
    pool = m["MediaPool/Master/MpFolder.xml"]
    assert isinstance(pool, str) and EXTENTS.hex() in pool
    m["MediaPool/Master/MpFolder.xml"] = pool.replace(EXTENTS.hex(), extents_hex, 1)
    return m


def test_huge_frame_rate_times_extents_is_an_issue_not_a_crash(make_zip) -> None:  # type: ignore[no-untyped-def]
    rate_hex = (struct.pack("<d", 1e308) + b"\x00" * 8).hex()
    extents_hex = struct.pack("<dd", 1e308, 1e308).hex()  # start * rate overflows to inf
    drp = read_drp(make_zip("huge.drp", _timeline_members(rate_hex, extents_hex)))
    [issue] = issues(drp, "implausible-extents")
    assert issue.severity == "warning"
    tl = drp.timelines[0]
    assert tl.frame_rate == 1e308  # the rate decoded fine; only the frame count overflowed
    assert tl.start_frame is None and tl.end_frame is None


# --------------------------------------------------------------------------- task 3: retime


def _retimed_timemap(x_max: float, last_valid_y: float) -> str:
    return kd(
        [
            ("DbType", 0x0A, qstr("Sm2TimeMap")),
            ("XMax", 0x06, struct.pack(">d", x_max)),
            ("LastValidYOffset", 0x06, struct.pack(">d", last_valid_y)),
        ]
    ).hex()


def test_non_finite_retime_speed_is_rejected_by_the_decoder() -> None:
    data = bytes.fromhex(_retimed_timemap(1.0, float("inf")))
    with pytest.raises(blobs.BlobError, match="finite"):
        blobs.decode_timemap(data)


def test_non_finite_retime_speed_stays_null_and_valid_json(make_zip) -> None:  # type: ignore[no-untyped-def]
    extra = (
        '<Element><Sm2TiVideoClip DbId="i9"><Name>retimed</Name><Start>0</Start>'
        "<Duration>10</Duration><In>0</In><MediaRef>clip1</MediaRef>"
        f"<MediaTimemapBA>{_retimed_timemap(1.0, float('inf'))}</MediaTimemapBA>"
        "</Sm2TiVideoClip></Element>"
    )
    drp = read_drp(make_zip("tm.drp", members(extra_item=extra)))
    item = next(i for t in drp.timelines for tr in t.tracks for i in tr.items if i.id == "i9")
    assert item.speed is None  # null, not inf
    assert any(i.code == "blob-decode" and "MediaTimemapBA" in i.message for i in drp.issues)
    text = json.dumps(drp.to_dict(), ensure_ascii=True)
    assert "Infinity" not in text and "NaN" not in text

    def reject(constant: str) -> float:
        raise AssertionError(f"invalid JSON constant {constant!r}")

    json.loads(text, parse_constant=reject)  # a strict parser must accept it


# --------------------------------------------------------------------------- task 4: one issue


def test_one_scan_budget_issue_per_file_not_per_member(tmp_path: Path) -> None:
    # One bomb drains the budget; the innocent members scanned after it (under --blobs) must
    # not each get their own scan-budget-exceeded issue.
    body = "<ProjectName>P</ProjectName><G>" + "<a/>" * 5_000 + "</G>"
    path = small_drp(tmp_path / "one.drp", body)
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
        for i in range(6):
            zf.writestr(f"Extra{i}.xml", xml("<Gallery::GyGallery DbId='x'/>"))
    drp = read_drp(path, check_blobs=True, max_elements=100)
    assert len(issues(drp, "scan-budget-exceeded")) == 1


# --------------------------------------------------------------------------- task 6: timing


@pytest.mark.parametrize("kind", ["elements", "xml", "blobs"])
def test_hostile_input_in_each_budgeted_class_stays_bounded(tmp_path: Path, kind: str) -> None:
    # Each budget caps the work it counts; this guards the uncounted work (reading/scanning the
    # bytes a stopped member leaves behind) from running unbounded. The bound is deliberately
    # generous so a loaded machine still passes; the point is that it is finite, not n^2/OOM.
    path = small_drp(tmp_path / f"{kind}.drp")
    if kind == "elements":
        with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Bomb.xml", xml("<G>" + "<a/>" * 4_000_000 + "</G>"))
        kwargs = {"check_blobs": True, "max_elements": 500}
        code = "scan-budget-exceeded"
    elif kind == "xml":
        with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
            body = "<Sm2SequenceContainer>" + " " * 20_000_000 + "</Sm2SequenceContainer>"
            zf.writestr("SeqContainer/big.xml", xml(body))
        kwargs = {"max_total_xml_bytes": 20_000}
        code = "xml-budget-exceeded"
    else:
        blob = _qcompress(2 * 1024 * 1024, 1024).hex()
        with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as zf:
            cache = f"<X><RenderCacheBA>{blob}</RenderCacheBA></X>" * 50
            zf.writestr(
                "MediaPool/Master/Bomb/MpFolder.xml",
                xml(f"<Sm2MpFolder DbId='z'>{cache}</Sm2MpFolder>"),
            )
        kwargs = {"check_blobs": True, "max_total_blob_output": 1024 * 1024}
        code = "blob-budget-exceeded"
    start = time.perf_counter()
    drp = read_drp(path, **kwargs)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - start
    assert elapsed < 30, f"{kind}: read took {elapsed:.1f}s"
    assert issues(drp, code), f"{kind}: the budget never tripped"


# --------------------------------------------------------------------------- task 8: X12 / X15


@pytest.mark.parametrize(
    "edit",
    [
        once(b'<?xml version="1.0"', b' <?xml version="1.0"'),  # leading whitespace
        once(b'DbPrjVer="17"-->\n', b'DbPrjVer="17"-->\n\n'),  # blank line before the root
        once(b'DbPrjVer="17"-->\n', b'DbPrjVer="17"-->\r\n'),  # CRLF at the end of the head
    ],
    ids=["leading-space", "blank-line", "crlf"],
)
def test_x12_prolog_whitespace_is_compared_without_strip(tmp_path: Path, edit) -> None:  # type: ignore[no-untyped-def]
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: edit})
    c = only_change(H264, b)
    assert (c.section, c.field_name) == ("opaque", SEQ_MEMBER)
    assert not diff(H264, b, gate=True).gate.passed


def test_x15_member_only_version_stamp_change_is_reported(tmp_path: Path) -> None:
    b = edited(
        tmp_path, "b.drp", {SEQ_MEMBER: once(b'DbAppVer="21.0.0.0048"', b'DbAppVer="21.0.0.9999"')}
    )
    d = assert_mirrors(H264, b)
    [stamp] = [c for c in d.changes if c.field_name == "version_stamp"]
    assert stamp.section == "header" and "differs from the file's main stamp" in stamp.detail


# --------------------------------------------------------------------------- task 9: DbPrjVer


@pytest.mark.parametrize("spelling", [b'DbPrjVer="017"', b'DbPrjVer="0017"'])
def test_dbprjver_spelling_change_is_opaque_and_fails_the_gate(
    tmp_path: Path, spelling: bytes
) -> None:
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: once(b'DbPrjVer="17"', spelling)})
    c = only_change(H264, b)
    assert (c.section, c.field_name) == ("opaque", SEQ_MEMBER)
    assert not diff(H264, b, gate=True).gate.passed


def test_dbprjver_non_ascii_digits_fail_the_gate(tmp_path: Path) -> None:
    arabic = b'DbPrjVer="\xd9\xa1\xd9\xa7"'  # Arabic-Indic 17 (U+0661 U+0667); parse_int rejects it
    b = edited(tmp_path, "b.drp", {SEQ_MEMBER: once(b'DbPrjVer="17"', arabic)})
    d = assert_mirrors(H264, b)
    assert any(c.section == "opaque" and c.field_name == SEQ_MEMBER for c in d.changes)
    assert not diff(H264, b, gate=True).gate.passed


# --------------------------------------------------------------------------- task 10: exit codes


def _run(args: list[str], *, stdout, stderr, close_stderr: bool = False):  # type: ignore[no-untyped-def]
    return subprocess.run(
        [sys.executable, "-c", CLI, *args],
        stdout=stdout,
        stderr=stderr,
        preexec_fn=(lambda: os.close(2)) if close_stderr else None,
        check=False,
    )


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_diff_broken_stderr_never_exits_120(tmp_path: Path) -> None:
    r19 = H264.parent / "media-clip-r19.drp"
    # (1) stdout full, stderr closed: an output error, exit 74, never 120.
    with open("/dev/full", "wb") as full:
        proc = _run(
            ["drp", "diff", str(r19), str(H264)], stdout=full, stderr=None, close_stderr=True
        )
    assert proc.returncode == 74
    # (2) missing input with stderr full: a usage/input error, exit 2, never 120.
    with open("/dev/full", "wb") as full:
        proc = _run(
            ["drp", "diff", str(H264), str(tmp_path / "missing.drp")],
            stdout=subprocess.DEVNULL,
            stderr=full,
        )
    assert proc.returncode == 2
    # (3) both full: still an output error, exit 74, never 120.
    with open("/dev/full", "wb") as out, open("/dev/full", "wb") as err:
        proc = _run(["drp", "diff", str(r19), str(H264)], stdout=out, stderr=err)
    assert proc.returncode == 74


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_inspect_broken_stderr_never_exits_120() -> None:
    # The read/inspect commands share the pattern, so they get the same fix.
    with open("/dev/full", "wb") as out, open("/dev/full", "wb") as err:
        proc = _run(["drp", "inspect", str(H264)], stdout=out, stderr=err)
    assert proc.returncode == 74


# --------------------------------------------------------------------------- task 12: short id


def test_short_id_of_a_one_character_project_id_change_is_legible(tmp_path: Path) -> None:
    import re

    pid = read_drp(H264).project.id
    assert pid and len(pid) > 12
    flipped = pid[:-1] + ("0" if pid[-1] != "0" else "1")
    b = edited(
        tmp_path,
        "b.drp",
        {"project.xml": once(f'DbId="{pid}"'.encode(), f'DbId="{flipped}"'.encode())},
    )
    c = only_change(H264, b)
    assert (c.section, c.field_name) == ("header", "id")
    m = re.search(r"project DbId (\S+) . (\S+) ", c.detail)
    assert m and m.group(1) != m.group(2), c.detail  # the two ids read differently now


# --------------------------------------------------------------------------- task 13: zstd gate


def test_gate_fails_on_zstd_unavailable_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    # H264 holds zstd-compressed blobs the reader would decode; with no backend they can only
    # be compared opaque, so the gate must not vouch for the file. Mutant N11 (a gate that
    # ignores zstd-unavailable) is caught by requiring that reason explicitly.
    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))
    d = diff(H264, H264, gate=True)
    assert not d.gate.passed
    assert any("zstd-unavailable" in reason for reason in d.gate.reasons)
