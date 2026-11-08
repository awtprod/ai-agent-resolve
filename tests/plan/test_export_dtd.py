"""DTD validation of every golden and of random exports against Apple's FCPXML 1.10 DTD.

Apple's DTD is not kept in the repo; set ``AAR_FCPXML_DTD`` to a copy (aar-evidence/s18/apple-docs
holds one). With lxml installed it validates with lxml; without lxml it uses the small validator in
``dtd_support``; without the DTD it skips.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dtd_support import load_dtd, validate
from export_support import EXPORT_FIXTURES, MEDIA_MAP, crossfade_form, permissive, random_plan

from aar.plan.export import export_fcpxml, media_map_from_mapping
from aar.plan.model import EditPlan

DTD_PATH = os.environ.get("AAR_FCPXML_DTD")
pytestmark = pytest.mark.skipif(
    not DTD_PATH or not Path(DTD_PATH).is_file(),
    reason="set AAR_FCPXML_DTD to Apple's FCPXMLv1_10.dtd to run the DTD validation",
)
GOLDEN_FILES = sorted(EXPORT_FIXTURES.glob("*.fcpxml"))


def _problems(document: str) -> list[str]:
    assert DTD_PATH is not None
    text = Path(DTD_PATH).read_text(encoding="utf-8")
    try:
        from lxml import etree  # type: ignore[import-not-found]
    except ImportError:
        return validate(load_dtd(text), document)
    dtd = etree.DTD(Path(DTD_PATH))
    doc = etree.fromstring(document.encode("utf-8"))
    return [] if dtd.validate(doc) else [str(e) for e in dtd.error_log]


def test_every_golden_is_present() -> None:
    assert len(GOLDEN_FILES) == 5


@pytest.mark.parametrize("path", GOLDEN_FILES, ids=[p.name for p in GOLDEN_FILES])
def test_golden_is_valid_against_apples_dtd(path: Path) -> None:
    assert _problems(path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("seed", range(0, 300, 3))
def test_random_export_is_valid_against_apples_dtd(seed: int) -> None:
    data, start = random_plan(seed)
    plan = EditPlan.model_validate(data)
    if plan.total_frames() == 0:
        pytest.skip("empty timeline")
    result = export_fcpxml(
        plan,
        permissive(start),
        media_map=media_map_from_mapping(MEDIA_MAP),
        audio_crossfade=crossfade_form(seed),  # type: ignore[arg-type]
    )
    assert _problems(result.fcpxml) == []


def test_the_validator_sees_dtd_violations() -> None:
    """The fallback validator is a real check: each break of the DTD is reported."""
    golden = (EXPORT_FIXTURES / "ccn_longform_v2.fcpxml").read_text(encoding="utf-8")
    cases = {
        "bogus attribute": golden.replace("<spine>", '<spine bogus="1">', 1),
        "order": golden.replace(
            '<adjust-conform type="fill" />\n              <adjust-transform',
            '<adjust-transform position="0 0" />\n              <adjust-conform type="fill" />'
            "\n              <adjust-transform",
            1,
        ),
        "enum": golden.replace('srcEnable="video"', 'srcEnable="both"', 1),
        "unknown element": golden.replace("<spine>", "<spine><bogus />", 1),
        "dangling IDREF": golden.replace('ref="r2"', 'ref="r99"', 1),
        "fixed version": golden.replace('version="1.10"', 'version="1.9"', 1),
        "missing required": golden.replace('<sequence format="r1" ', "<sequence ", 1),
    }
    for name, document in cases.items():
        assert document != golden, name
        assert _problems(document), name
