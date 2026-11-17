"""Whole-frame edits serialized with Resolve 21.1's floating-point remainders."""

from copy import deepcopy

import pytest
from timeline_import_support import Row, build_timeline

from aar.plan.model import EditPlan
from aar.plan.timeline_import import import_timeline


def imported(field: str, fraction: float, carry: int = 0):
    plan = EditPlan.model_validate(
        {
            "version": 2,
            "project": "P",
            "timeline": "T",
            "kind": "longform",
            "fps": {"num": 30000, "den": 1001},
            "resolution": {"width": 1920, "height": 1080},
            "profile": "ccn",
            "sources": [{"id": "a", "media": "E:\\A.mov", "duration": 500}],
            "segments": [{"source": "a", "src_in": 100, "src_out": 200}],
        }
    )
    values = {"record_start": 5, "duration": 80, "source_in": 110}
    values[field] -= carry
    rows = [
        Row(
            track,
            kind,
            values["record_start"],
            values["duration"],
            values["source_in"],
            "E:\\A.mov",
            fractions={field: fraction},
        )
        for track, kind in (("V1", "video"), ("A1", "audio"))
    ]
    timeline, media = build_timeline(rows, rate=30000 / 1001)
    before = deepcopy(timeline)
    result = import_timeline(plan, timeline, media)
    assert timeline == before  # The forensic reader's raw values remain intact.
    return result


@pytest.mark.parametrize("field", ["record_start", "duration", "source_in"])
@pytest.mark.parametrize(
    ("fraction", "carry"),
    [(2.7675639557855902e-12, 0), (0.9999999999977263, 1)],
)
def test_native_resolve_roundoff_does_not_hide_a_trim(field, fraction, carry):
    result = imported(field, fraction, carry)
    segment = result.plan.segments[0]
    assert (segment.record_in, segment.src_in, segment.src_out) == (5, 110, 190)
    assert result.diff["not_carried"] == []


@pytest.mark.parametrize("field", ["record_start", "duration", "source_in"])
@pytest.mark.parametrize("fraction", [1e-8, 0.5, 1 - 1e-8])
def test_genuine_subframe_edits_still_require_attention(field, fraction):
    result = imported(field, fraction)
    assert "sub_frame" in [entry["code"] for entry in result.diff["not_carried"]]
    assert (result.plan.segments[0].src_in, result.plan.segments[0].src_out) == (100, 200)
