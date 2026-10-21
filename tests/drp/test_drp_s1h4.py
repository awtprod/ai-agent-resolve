"""S1h4 item 3: pin the non-equivalent mutants the S1h3 pass left surviving.

Each test asserts the current (correct) behaviour; the validator's mutant for it changes that
behaviour, so the test fails on the mutant and passes here. Mutant ids match
aar-evidence/s1h3/probes/mut.py.
"""

from __future__ import annotations

import math
import struct
import subprocess
import sys
from pathlib import Path

import pytest
from diff_testing import H264
from drp_edit import edit_zip, member_named
from test_drp_s1h2 import _timeline_members
from test_drp_s1h3 import _retimed_timemap
from test_drp_synthetic import EXTENTS, RATE_25, members

from aar.cli import main
from aar.drp import blobs
from aar.drp.diff import diff_projects
from aar.drp.diff_model import DiffError
from aar.drp.reader import read_drp
from aar.drp.scan import scan

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"
_VALID_RATE = (struct.pack("<d", 25.0) + b"\x00" * 8).hex()


def _rate_hex(value: float) -> str:
    return (struct.pack("<d", value) + b"\x00" * 8).hex()


# --------------------------------------------------------------------------- M16 frame rate


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_m16_non_finite_frame_rate_warns_and_inspect_json_exits_0(
    make_zip, capsys: pytest.CaptureFixture[str], value: float
) -> None:  # type: ignore[no-untyped-def]
    path = make_zip("r.drp", _timeline_members(_rate_hex(value), EXTENTS.hex()))
    drp = read_drp(path, check_blobs=True)
    assert drp.timelines[0].frame_rate is None
    assert any(i.code == "blob-decode" for i in drp.issues)
    code = main(["drp", "inspect", "--json", str(path)])
    out = capsys.readouterr().out
    assert code == 0 and "Infinity" not in out and "NaN" not in out


# --------------------------------------------------------------------------- M48 retime guard


@pytest.mark.parametrize("x_max", [0.0, -1.0])
def test_m48_zero_or_negative_xmax_gives_null_speed(make_zip, x_max: float) -> None:  # type: ignore[no-untyped-def]
    extra = (
        '<Element><Sm2TiVideoClip DbId="i9"><Name>r</Name><Start>0</Start><Duration>10</Duration>'
        "<In>0</In><MediaRef>clip1</MediaRef>"
        f"<MediaTimemapBA>{_retimed_timemap(x_max, 1.0)}</MediaTimemapBA>"
        "</Sm2TiVideoClip></Element>"
    )
    drp = read_drp(make_zip("tm.drp", members(extra_item=extra)))  # no ZeroDivisionError
    item = next(i for t in drp.timelines for tr in t.tracks for i in tr.items if i.id == "i9")
    assert item.speed is None


# --------------------------------------------------------------------------- M15 extents duration


@pytest.mark.parametrize("dur", [float("nan"), float("inf")])
def test_m15_non_finite_extents_duration_warns(
    make_zip, capsys: pytest.CaptureFixture[str], dur: float
) -> None:  # type: ignore[no-untyped-def]
    extents = struct.pack("<dd", 1.0, dur).hex()
    path = make_zip("e.drp", _timeline_members(RATE_25.hex(), extents))
    drp = read_drp(path, check_blobs=True)
    assert drp.timelines[0].duration_seconds is None
    assert any(i.code == "blob-decode" for i in drp.issues)
    code = main(["drp", "inspect", "--json", str(path)])
    out = capsys.readouterr().out
    assert code == 0 and "Infinity" not in out and "NaN" not in out


# ----------------------------------------------------------------------- M11 overflow start_frame


def test_m11_overflow_leaves_no_huge_start_frame(make_zip) -> None:  # type: ignore[no-untyped-def]
    # start*rate is finite (a 308-digit int) but (start+dur)*rate overflows: the except must
    # reset start_frame too, not leave the huge int behind.
    extents = struct.pack("<dd", 1e306, 1e307).hex()
    drp = read_drp(make_zip("m11.drp", _timeline_members(_rate_hex(24.0), extents)))
    tl = drp.timelines[0]
    assert tl.start_frame is None and tl.end_frame is None
    assert any(i.code == "implausible-extents" for i in drp.issues)


# --------------------------------------------------------------------------- M41b self-diff pipe


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
def test_m41b_self_diff_to_full_stdout_exits_74_small_output() -> None:
    norm = subprocess.run(
        [sys.executable, "-c", CLI, "drp", "diff", str(H264), str(H264)],
        stdout=subprocess.PIPE,
        check=False,
    )
    assert len(norm.stdout) < 4096  # buffered, so /dev/full fails only at flush
    with open("/dev/full", "wb") as full:
        proc = subprocess.run(
            [sys.executable, "-c", CLI, "drp", "diff", str(H264), str(H264)],
            stdout=full,
            stderr=subprocess.PIPE,
            check=False,
        )
    assert proc.returncode == 74 and b"Traceback" not in proc.stderr


# --------------------------------------------------------------------------- N11/N12/N15 cumulative


def _cum_pair(tmp_path: Path):  # type: ignore[no-untyped-def]
    empty = H264.parent / "empty-project.drp"
    seq = member_named(empty, "SeqContainer/")
    a = edit_zip(empty, tmp_path / "a.drp", {})
    b = edit_zip(
        empty,
        tmp_path / "b.drp",
        {
            "project.xml": lambda d: d.replace(b"</", b"<Q/></", 1),
            seq: lambda d: d.replace(
                b"<UserDefinedName/>", b"<UserDefinedName>x</UserDefinedName>", 1
            ),
        },
    )
    return read_drp(a), read_drp(b), empty, seq


def test_n11_n12_cumulative_element_budget_across_members(tmp_path: Path) -> None:
    da, db, empty, seq = _cum_pair(tmp_path)
    import zipfile

    z = zipfile.ZipFile(empty)
    counts = {n: scan(z.read(n), n).element_count for n in ("project.xml", seq)}
    # each member fits alone, but the two together do not: only a cumulative budget trips.
    budget = max(counts.values()) + 5
    assert min(counts.values()) > 5
    with pytest.raises(DiffError, match="element budget"):
        diff_projects(da, db, max_elements=budget)


def test_n15_cumulative_xml_byte_budget_across_members(tmp_path: Path) -> None:
    da, db, _empty, _seq = _cum_pair(tmp_path)
    total = sum(m.size for m in da.members)
    with pytest.raises(DiffError, match="XML budget"):
        diff_projects(da, db, max_total_xml_bytes=total - 500)


# --------------------------------------------------------------------------- N32/N33/N34 allow_nan


def test_n32_n33_inspect_json_fails_closed_on_a_non_finite(
    make_zip, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    # Force a non-finite past the decoders (a hypothetical future bug); the inspect --json path
    # must fail closed (exit 74), never emit the invalid token Infinity.
    monkeypatch.setattr(blobs, "decode_rate", lambda data: math.inf)
    path = make_zip("r.drp", _timeline_members(_VALID_RATE, EXTENTS.hex()))
    code = main(["drp", "inspect", "--json", str(path)])
    out = capsys.readouterr().out
    assert code == 74 and "Infinity" not in out and "NaN" not in out


def test_n34_diff_json_fails_closed_on_a_non_finite(
    make_zip, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    rate_a = struct.pack("<d", 11.0) + b"\x00" * 8
    orig = blobs.decode_rate
    monkeypatch.setattr(
        blobs, "decode_rate", lambda data: math.inf if data == rate_a else orig(data)
    )
    a = make_zip("a.drp", _timeline_members(rate_a.hex(), EXTENTS.hex()))
    b = make_zip("b.drp", _timeline_members(_rate_hex(22.0), EXTENTS.hex()))
    code = main(["drp", "diff", "--json", str(a), str(b)])
    out = capsys.readouterr().out
    assert code != 1 and "Infinity" not in out and "NaN" not in out
