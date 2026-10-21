"""S1h4 round 2, gate depth: the opaque walk is linear in how deep the XML is nested.

``diff_opaque`` used to build each element's path as ``(*parent_path, segment)``, a copy of every
segment above it, so a chain nested ``d`` deep cost O(d**2): 12.1 s at 100,000 levels (2.5 s now)
and no finish within 900 s at 1,000,000 (23.5 s now). The path is now a parent-linked chain and
its ``/``-joined text is built only for a reported change.

The scaling test measures CPU time of the diff stage alone at two depths four times apart and
asserts the ratio, so it needs no absolute time limit: linear work gives about 4, the old copy
gave about 12 at these depths.
"""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

from drp_helpers import xml

from aar.drp.diff import diff_projects
from aar.drp.reader import read_drp

_SMALL = 30_000
_LARGE = 120_000


def _nested(path: Path, depth: int, leaf: str) -> Path:
    chain = f"{'<a>' * depth}<leaf>{leaf}</leaf>{'</a>' * depth}"
    body = f"<ProjectName>P</ProjectName><Junk>{chain}</Junk>"
    members = {
        "project.xml": xml(f'<SM_Project DbId="p1">{body}</SM_Project>\n'),
        "Gallery.xml": xml('<Gallery::GyGallery DbId="g"/>\n'),
        "MediaPool/Master/MpFolder.xml": xml(
            '<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'
        ),
    }
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, text in members.items():
            zf.writestr(name, text)
    return path


def _diff_cpu_seconds(tmp_path: Path, depth: int) -> float:
    """Best of two CPU timings of the gate on a chain ``depth`` deep with a changed leaf."""
    a = read_drp(_nested(tmp_path / f"a{depth}.drp", depth, "X"))
    b = read_drp(_nested(tmp_path / f"b{depth}.drp", depth, "Y"))
    best = float("inf")
    for _ in range(2):
        start = time.process_time()
        result = diff_projects(a, b, gate=True)
        best = min(best, time.process_time() - start)
        assert len(result.changes) == 1
    return best


def test_the_gate_is_linear_in_nesting_depth(tmp_path: Path) -> None:
    small = _diff_cpu_seconds(tmp_path, _SMALL)
    large = _diff_cpu_seconds(tmp_path, _LARGE)
    ratio = large / max(small, 1e-3)
    assert ratio < 8, f"4x the depth took {ratio:.1f}x the time (linear ~4, the old copy ~12)"


def test_the_reported_path_of_a_deep_change_is_every_segment(tmp_path: Path) -> None:
    depth = 3_000
    a = read_drp(_nested(tmp_path / "a.drp", depth, "X"))
    b = read_drp(_nested(tmp_path / "b.drp", depth, "Y"))
    result = diff_projects(a, b, gate=True)
    (change,) = result.changes
    expected = "Junk/" + "a/" * depth + "leaf"
    assert change.field_name == expected
    assert f"opaque <{expected}> changed" in change.detail
    assert not result.gate.passed


def test_a_removed_deep_element_reports_the_same_path(tmp_path: Path) -> None:
    # The presence branch builds its text the same way as the changed branch.
    depth = 500
    full = _nested(tmp_path / "a.drp", depth, "X")
    body = "<ProjectName>P</ProjectName><Junk>" + "<a>" * depth + "</a>" * depth + "</Junk>"
    with zipfile.ZipFile(tmp_path / "b.drp", "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.xml", xml(f'<SM_Project DbId="p1">{body}</SM_Project>\n'))
        zf.writestr("Gallery.xml", xml('<Gallery::GyGallery DbId="g"/>\n'))
        zf.writestr(
            "MediaPool/Master/MpFolder.xml",
            xml('<Sm2MpFolder DbId="b1"><Name>Master</Name><MediaVec/></Sm2MpFolder>\n'),
        )
    result = diff_projects(read_drp(full), read_drp(tmp_path / "b.drp"), gate=True)
    removed = [c for c in result.changes if c.kind == "opaque-removed"]
    assert [c.field_name for c in removed] == ["Junk/" + "a/" * depth + "leaf"]
