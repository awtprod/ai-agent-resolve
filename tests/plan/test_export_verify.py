"""The self-check catches a file that does not reproduce the plan: each tampering of the CCN
golden is reported with a specific problem, and hostile input is refused before parsing."""

from __future__ import annotations

import pytest
from export_support import EXPORT_FIXTURES, GOLDEN_PLANS, ccn, load_plan

from aar.plan.export import verify_plan_fcpxml
from aar.plan.export.reader import MAX_BYTES, check_structure, read_fcpxml

# The transition-form golden: its audio-only crossfade is a spine transition element.
GOLDEN = (EXPORT_FIXTURES / "ccn_longform_v2.transition.fcpxml").read_text(encoding="utf-8")
PLAN = load_plan(GOLDEN_PLANS["ccn_longform_v2"])
FORM = "transition"


def _problems(text: str) -> list[str]:
    result = verify_plan_fcpxml(text, PLAN, ccn(), audio_crossfade=FORM)
    assert not result.ok
    assert result.problems
    return result.problems


def _tamper(old: str, new: str) -> str:
    assert GOLDEN.count(old) >= 1, old
    return GOLDEN.replace(old, new, 1)


def test_the_golden_is_clean() -> None:
    assert verify_plan_fcpxml(GOLDEN, PLAN, ccn(), audio_crossfade=FORM).problems == []


@pytest.mark.parametrize(
    ("old", "new", "expect"),
    [
        # A clip one frame late breaks the spine and the segment.
        (
            'offset="86558472/24000s"',
            'offset="86559473/24000s"',
            "does not follow the previous element",
        ),
        # A source in point one frame off.
        ('start="72072/24000s"', 'start="73073/24000s"', "segment 1"),
        # A clip pointing at another file.
        ('ref="r6" offset="86558472/24000s"', 'ref="r7" offset="86558472/24000s"', "src"),
        ("file:///E:/CCN/Ep1/B001.braw", "file:///E:/CCN/Ep1/B002.braw", "assets"),
        # The crossfade's length and window.
        ('duration="12012/24000s"', 'duration="13013/24000s"', "not in the plan"),
        (
            'offset="86696610/24000s" duration',
            'offset="86697611/24000s" duration',
            "not in the plan",
        ),
        # Timing header.
        ('tcStart="86486400/24000s"', 'tcStart="86400s"', "tcStart"),
        ('frameDuration="1001/24000s"', 'frameDuration="1/24s"', "frameDuration"),
        ('duration="426426/24000s"', 'duration="427427/24000s"', "sequence duration"),
        # Sound: the split timing that stops the sound at the outro, a volume, a fade.
        (' audioStart="24024/24000s" audioDuration="96096/24000s"', "", "sound"),
        ('amount="-2dB"', 'amount="-3dB"', "volume_db"),
        ('<fadeOut type="linear" duration="2002/24000s" />', "", "fade_out"),
        ('srcEnable="audio"', 'srcEnable="all"', "enable"),
        ('lane="-4"', 'lane="-3"', "lane -4"),
        # Picture: transform.
        ('scale="1.1 1.1"', 'scale="1.2 1.2"', "transform"),
        # Titles and markers.
        ("Jane Doe\nHead Judge", "Jane Doe\nHead Judge!", "titles"),
        ('value="Act 2"', 'value="Act 3"', "markers"),
        # Names the plan gives.
        ('<project name="Baking Resistance Ep1">', '<project name="Other">', "project"),
    ],
)
def test_tampering_is_caught(old: str, new: str, expect: str) -> None:
    problems = _problems(_tamper(old, new))
    assert any(expect in p for p in problems), problems


def test_removed_transition_is_caught() -> None:
    start = GOLDEN.index('<transition name="Audio Crossfade"')
    end = GOLDEN.index("</transition>", start) + len("</transition>")
    problems = _problems(GOLDEN[:start] + GOLDEN[end:])
    assert any("transitions" in p for p in problems), problems


def test_removed_head_transition_is_caught() -> None:
    start = GOLDEN.index('<transition name="Cross Dissolve"')
    end = GOLDEN.index("</transition>", start) + len("</transition>")
    problems = _problems(GOLDEN[:start] + GOLDEN[end:])
    assert any("head: the plan's cut 0 (15 frames) is not in the file" in p for p in problems)


def test_dropped_clip_is_caught() -> None:
    start = GOLDEN.index('<asset-clip ref="r6"')
    end = GOLDEN.index("/>", start) + 2
    problems = _problems(GOLDEN[:start] + GOLDEN[end:])
    assert any("segment 1" in p for p in problems), problems


@pytest.mark.parametrize(
    ("old", "new", "expect"),
    [
        ('ref="r7"', 'ref="r99"', "does not resolve to asset (found nothing)"),
        (
            'ref="r6" offset="86558472/24000s"',
            'ref="r2" offset="86558472/24000s"',
            "does not resolve to asset (found effect)",
        ),
        ('<asset-clip ref="r6"', '<asset-clip bogus="1" ref="r6"', "not in FCPXML 1.10"),
        ('srcEnable="video"', 'srcEnable="both"', "is not one of"),
        ('<effect id="r3"', '<effect id="r2"', "used twice"),
        ('offset="86558472/24000s"', 'offset="1/48000s"', "not a whole number of frames"),
        ('offset="86558472/24000s"', 'offset="3600.5s"', "not an FCPXML rational time"),
        ('lane="-4" ', 'lane="0" ', "lane 0"),
        ('<sequence format="r1"', "<sequence", "missing required attribute 'format'"),
        ('<fcpxml version="1.10">', '<fcpxml version="1.9">', "is not one of"),
    ],
)
def test_structure_problems_are_caught(old: str, new: str, expect: str) -> None:
    problems = _problems(_tamper(old, new))
    assert any(expect in p for p in problems), problems


def test_child_order_follows_the_dtd() -> None:
    # A marker before the anchored title breaks asset-clip's content model.
    start = GOLDEN.index('<marker start="0s"')
    end = GOLDEN.index("/>", start) + 2
    marker = GOLDEN[start:end]
    text = GOLDEN[:start] + GOLDEN[end:]
    text = text.replace(
        '<title ref="r4" lane="1" offset="48048',
        marker + '<title ref="r4" lane="1" offset="48048',
        1,
    )
    problems = _problems(text)
    assert any("out of FCPXML 1.10 order" in p for p in problems), problems


def test_unknown_elements_and_text_are_caught() -> None:
    problems = _problems(_tamper("<spine>", '<spine><caption duration="1s" />'))
    assert any("<caption>" in p for p in problems), problems
    problems = _problems(_tamper("<spine>", "<spine>stray"))
    assert any("text content" in p for p in problems), problems


@pytest.mark.parametrize(
    "text",
    [
        '<?xml version="1.0"?><!DOCTYPE fcpxml [<!ENTITY a "aaaa">]><fcpxml version="1.10"/>',
        '<?xml version="1.0"?><!DOCTYPE fcpxml [<!ELEMENT fcpxml ANY>]><fcpxml version="1.10"/>',
    ],
)
def test_entities_and_internal_subsets_are_refused(text: str) -> None:
    decoded = read_fcpxml(text)
    assert decoded.problems == ["the file declares entities or a DOCTYPE internal subset (refused)"]


def test_oversized_and_malformed_input() -> None:
    assert read_fcpxml("x" * (MAX_BYTES + 1)).problems == [f"the file is over {MAX_BYTES} bytes"]
    assert read_fcpxml("<fcpxml").problems[0].startswith("not well-formed XML")
    assert read_fcpxml("<resources/>").problems == ["the root element is <resources>, not <fcpxml>"]
    assert _problems("<fcpxml").count("the file could not be decoded far enough to compare") == 1


def test_two_projects_are_refused() -> None:
    start = GOLDEN.index("<project ")
    end = GOLDEN.index("</project>") + len("</project>")
    text = GOLDEN[:end] + GOLDEN[start:end] + GOLDEN[end:]
    assert any("expected one project" in p for p in _problems(text))


def test_check_structure_on_the_golden_is_clean() -> None:
    import xml.etree.ElementTree as ET

    assert check_structure(ET.fromstring(GOLDEN)) == []


def test_verify_against_another_plan_fails() -> None:
    other = load_plan(GOLDEN_PLANS["short_reframe"])
    result = verify_plan_fcpxml(GOLDEN, other, ccn(), audio_crossfade=FORM)
    assert not result.ok
    assert any("segment 0" in p for p in result.problems)
