"""Follow-ups from the S1h2 review (dispatched as slice S1h3).

Scanner and opaque-rescan budgets charge every token; the gate doc matches the code; the
duplicate-stamp mask hole; short_id_pair on a prefix extension; broken-pipe handling for aar
plan / aar node; non-finite retime values and allow_nan JSON; and the corpus pins helpers.
"""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from diff_testing import H264, SEQ_MEMBER, diff, edited, only_change
from drp_edit import once
from drp_helpers import kd, qstr
from test_drp_corpus import _marker_rows, _pin_width, _row_key
from test_drp_hardening import small_drp
from test_drp_synthetic import members

from aar.drp.diff import diff_projects
from aar.drp.diff_model import DiffError, short_id_pair
from aar.drp.reader import read_drp
from aar.drp.scan import scan

CLI = "import sys; from aar.cli import main; sys.exit(main(sys.argv[1:]))"


# --------------------------------------------------------------------------- item 1: opaque budget


def test_opaque_rescan_honours_the_element_budget(tmp_path: Path) -> None:
    # The opaque compare re-scans members; a tight budget must stop it with a structured error,
    # not scan a hostile member whole (7M elements were ~2 GB).
    a = small_drp(tmp_path / "a.drp", "<Big>" + "<a/>" * 500 + "<Zz/></Big>")
    b = edited(tmp_path, "b.drp", {"project.xml": once(b"<Zz/>", b"<Yy/>")}, src=a)
    da, db = read_drp(a), read_drp(b)  # default budget reads them whole
    with pytest.raises(DiffError, match="element budget"):
        diff_projects(da, db, max_elements=100)


def test_opaque_rescan_honours_the_xml_byte_budget(tmp_path: Path) -> None:
    a = small_drp(tmp_path / "a.drp", "<Pad>" + " " * 5000 + "</Pad>")
    b = edited(tmp_path, "b.drp", {"project.xml": once(b"<Pad>", b"<Pad>x")}, src=a)
    da, db = read_drp(a), read_drp(b)
    with pytest.raises(DiffError, match="XML budget"):
        diff_projects(da, db, max_total_xml_bytes=200)


# --------------------------------------------------------------------------- item 2: token budget


@pytest.mark.parametrize(
    ("kind", "token"),
    [
        ("close-tags", b"</x>"),
        ("comments", b"<!---->"),
        ("pis", b"<?x?>"),
        ("cdata", b"<![CDATA[]]>"),
        ("decls", b"<!x>"),
        ("lone-lt", b"< "),
    ],
)
def test_scan_charges_every_token_kind_against_the_budget(kind: str, token: bytes) -> None:
    data = b"<r>" + token * 200 + b"</r>"
    doc = scan(data, kind, max_elements=50)
    assert doc.stopped_at is not None, kind  # the budget tripped even with no start elements
    assert doc.element_count == 50, kind


def test_scan_budget_bounds_a_comment_bomb() -> None:
    # 1M tiny comments were ~2 GB because the comment list grew unbounded; the budget caps it.
    data = b"<r>" + b"<!---->" * 1_000_000 + b"</r>"
    doc = scan(data, "bomb.xml", max_elements=1000)
    assert doc.stopped_at is not None and doc.element_count == 1000
    assert len(doc.comments) <= 1000  # was one per comment (1M)
    assert doc.work <= 3 * (doc.stopped_at + 16)  # scanned only up to the stop


# --------------------------------------------------------------------------- item 3: gate doc


def test_stamp_value_change_fails_gate_but_project_dbid_change_passes(tmp_path: Path) -> None:
    # Pins the behaviour the DESIGN gate paragraph describes.
    with zipfile.ZipFile(H264) as z:
        xmls = [n for n in z.namelist() if n.endswith(".xml")]
    bump = edited(
        tmp_path,
        "bump.drp",
        {n: once(b'DbAppVer="21.0.0.0048"', b'DbAppVer="21.0.0.0049"') for n in xmls},
    )
    d = diff(H264, bump, gate=True)
    assert not d.gate.passed  # the stamp sits in Gallery.xml (not interpreted), compared opaque
    assert any(c.section == "opaque" for c in d.changes)

    pid = read_drp(H264).project.id
    flip = pid[:-1] + ("0" if pid[-1] != "0" else "1")
    idchg = edited(
        tmp_path,
        "id.drp",
        {"project.xml": once(f'DbId="{pid}"'.encode(), f'DbId="{flip}"'.encode())},
    )
    d2 = diff(H264, idchg, gate=True)
    assert d2.gate.passed  # a project DbId change is reported but only listed
    assert [(c.section, c.field_name) for c in d2.changes] == [("header", "id")]


# ------------------------------------------------------------------------ item 4: duplicate stamp


def test_edited_first_of_two_dbappver_values_fails_the_gate(tmp_path: Path) -> None:
    two = {
        SEQ_MEMBER: once(
            b'DbAppVer="21.0.0.0048"', b'DbAppVer="00.0.0.0000" DbAppVer="21.0.0.0048"'
        )
    }
    a = edited(tmp_path, "a.drp", two)
    b = edited(
        tmp_path,
        "b.drp",
        {SEQ_MEMBER: once(b'DbAppVer="00.0.0.0000"', b'DbAppVer="99.9.9.9999"')},
        src=a,
    )
    c = only_change(a, b)
    assert (c.section, c.field_name) == ("opaque", SEQ_MEMBER)
    assert not diff(a, b, gate=True).gate.passed


# --------------------------------------------------------------------------- item 5: short_id_pair


def test_short_id_pair_distinguishes_a_prefix_extension() -> None:
    a, b = "abcdefghijklm", "abcdefghijklmX"  # both > 12 chars; b extends a
    sa, sb = short_id_pair(a, b)
    assert sa != sb, (sa, sb)
    # and the usual one-character-flip case stays visible
    x, y = short_id_pair("0f98e18c1111222233334444", "0f98e18c1111222233334445")
    assert x != y


# ------------------------------------------------------------------------ item 6: plan/node pipes


def _run(args: list[str], *, stdout, stderr, close_stdout: bool = False):  # type: ignore[no-untyped-def]
    return subprocess.run(
        [sys.executable, "-c", CLI, *args],
        stdout=stdout,
        stderr=stderr,
        preexec_fn=(lambda: os.close(1)) if close_stdout else None,
        check=False,
    )


@pytest.mark.skipif(not Path("/dev/full").exists(), reason="needs /dev/full (Linux)")
@pytest.mark.parametrize("args", [["plan", "schema"], ["node", "ops"], ["node", "ops", "--json"]])
def test_plan_and_node_full_stdout_exit_74_without_traceback(args: list[str]) -> None:
    with open("/dev/full", "wb") as full:
        proc = _run(args, stdout=full, stderr=subprocess.PIPE)
    assert proc.returncode == 74, (args, proc.returncode, proc.stderr)
    assert b"Traceback" not in proc.stderr and b"Exception ignored" not in proc.stderr


@pytest.mark.parametrize("args", [["plan", "schema"], ["node", "ops"]])
def test_plan_and_node_closed_stdout_exit_74(args: list[str]) -> None:
    proc = _run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, close_stdout=True)
    assert proc.returncode == 74
    assert b"Traceback" not in proc.stderr


@pytest.mark.parametrize("args", [["plan", "schema"], ["node", "ops"]])
def test_plan_and_node_broken_pipe_exit_141_quietly(args: list[str]) -> None:
    read_end, write_end = os.pipe()
    os.close(read_end)
    try:
        proc = _run(args, stdout=write_end, stderr=subprocess.PIPE)
    finally:
        os.close(write_end)
    assert proc.returncode == 141 and proc.stderr == b""


# --------------------------------------------------------------------------- item 7 / M14: timemap


def _retimed_timemap(x_max: float, last_valid_y: float) -> str:
    return kd(
        [
            ("DbType", 0x0A, qstr("Sm2TimeMap")),
            ("XMax", 0x06, struct.pack(">d", x_max)),
            ("LastValidYOffset", 0x06, struct.pack(">d", last_valid_y)),
        ]
    ).hex()


def _read_timemap_item(make_zip, x_max: float, last_valid_y: float):  # type: ignore[no-untyped-def]
    extra = (
        '<Element><Sm2TiVideoClip DbId="i9"><Name>r</Name><Start>0</Start><Duration>10</Duration>'
        "<In>0</In><MediaRef>clip1</MediaRef>"
        f"<MediaTimemapBA>{_retimed_timemap(x_max, last_valid_y)}</MediaTimemapBA>"
        "</Sm2TiVideoClip></Element>"
    )
    drp = read_drp(make_zip("tm.drp", members(extra_item=extra)))
    item = next(i for t in drp.timelines for tr in t.tracks for i in tr.items if i.id == "i9")
    return item, drp


@pytest.mark.parametrize(
    ("x_max", "source", "expect_issue"),
    [
        (float("nan"), 1.0, True),  # NaN XMax: no longer slips the finite check (item 7)
        (1.0, float("nan"), True),  # NaN LastValidYOffset
        (float("inf"), 1.0, True),  # infinite XMax
        (5e-324, 1.0, True),  # finite ends, ratio overflows to inf (M14)
        (-1.0, 1.0, False),  # negative XMax: finite, reverse map not decoded, speed stays null
        (2.0, 1.0, False),  # valid retime: speed 0.5
    ],
)
def test_non_finite_retime_never_reaches_the_model(
    make_zip,
    x_max: float,
    source: float,
    expect_issue: bool,  # type: ignore[no-untyped-def]
) -> None:
    item, drp = _read_timemap_item(make_zip, x_max, source)
    assert item.speed is None or math.isfinite(item.speed)
    if expect_issue:
        assert item.speed is None
        assert any(i.code == "blob-decode" and "MediaTimemapBA" in i.message for i in drp.issues)
    elif x_max == 2.0:
        assert item.speed == 0.5


def test_inspect_json_uses_allow_nan_false(make_zip) -> None:  # type: ignore[no-untyped-def]
    # Defence in depth: even if a non-finite value reached the model, the JSON path fails
    # closed (allow_nan=False) rather than emit the invalid token Infinity.
    _item, drp = _read_timemap_item(make_zip, 5e-324, 1.0)
    import json

    text = json.dumps(drp.to_dict(), ensure_ascii=True, allow_nan=False)
    assert "Infinity" not in text and "NaN" not in text


# --------------------------------------------------------------------------- item 9: pins helpers


def test_pin_width_uses_the_widest_row() -> None:
    assert _pin_width([[1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]) == 6  # not 4 (the first row)
    assert _pin_width([]) == 4


def test_marker_rows_sorts_none_and_str_without_error() -> None:
    from types import SimpleNamespace

    def marker(frame, color, duration):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            frame=frame, color=color, name="m", duration=duration, note="", custom_data=""
        )

    markers = [marker(1, None, 5), marker(1, "Blue", None), marker(0, "Red", 2)]
    rows = _marker_rows(markers, 6)  # must not raise TypeError on None vs str/int
    assert rows == sorted(rows, key=_row_key) and len(rows) == 3
