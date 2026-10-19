"""S1h4 item 1: the tag scanner.

The scanner reads a start tag one attribute at a time, charges every attribute against the
element budget, stores at most 4096 per element, and skips an unterminated tag instead of
re-reading it from each ``<`` inside a quoted value (which was quadratic). A test that names a
commit in its comment fails on that commit (20f89a0 is S1h3, 8d562a6 the first S1h4 scanner).
"""

from __future__ import annotations

import time
from pathlib import Path

from test_drp_hardening import small_drp

from aar.drp.reader import read_drp
from aar.drp.scan import scan


def _attr_tag(count: int, name: bytes = b"a", end: bytes = b"/>") -> bytes:
    return b"<" + name + b"".join(b' k%d="v"' % i for i in range(count)) + end


# --------------------------------------------------------------------------- attributes charged


def test_attributes_are_charged_against_the_element_budget() -> None:
    # Fails on 20f89a0: attributes were free, so 100 of them never tripped a budget of 10.
    doc = scan(b"<r>" + _attr_tag(100) + b"</r>", "m.xml", max_elements=10)
    assert doc.stopped_at == 3  # the offset of "<a", which is not materialised
    assert doc.element_count == 10  # never past the budget
    assert [i.code for i in doc.issues if i.code == "budget-exceeded"] == ["budget-exceeded"]
    assert doc.root is not None and doc.root.tag == "r" and doc.root.children == []


def test_each_attribute_is_one_token() -> None:
    doc = scan(_attr_tag(7))
    assert doc.element_count == 8  # seven attributes plus the tag itself
    assert doc.stopped_at is None


def test_an_element_keeps_at_most_4096_attributes_but_every_one_is_charged() -> None:
    # Fails on 20f89a0 (5000 stored, 1 token) and models the 10M-attribute bomb in miniature.
    doc = scan(_attr_tag(5000))
    assert doc.root is not None
    assert len(doc.root.attrs) == 4096
    assert "k0" in doc.root.attrs and "k4095" in doc.root.attrs and "k4096" not in doc.root.attrs
    assert doc.element_count == 5001


def test_reader_budget_counts_attributes(tmp_path: Path) -> None:
    # Fails on 20f89a0: 300 attributes were not tokens, so a budget of 200 was never reached.
    path = small_drp(tmp_path / "attrs.drp", _attr_tag(300, b"Big").decode())
    drp = read_drp(path, max_elements=200)
    assert "scan-budget-exceeded" in {i.code for i in drp.issues}
    assert "scan-budget-exceeded" not in {i.code for i in read_drp(path).issues}


# --------------------------------------------------------------------------- unterminated tags


def test_unterminated_tag_with_quoted_markup_is_scanned_in_linear_time() -> None:
    # An unterminated start tag whose quoted values hold "<": the S1h3 scanner restarted at each
    # of them and re-read the rest of the tag (20000 of them took ~31 s; 40000, 145 s). The
    # linear scanner needs ~0.03 s here, so a 5 s limit has two orders of margin on a slow host.
    data = b"<a " + b'b="<" ' * 20_000
    start = time.perf_counter()
    doc = scan(data, "quad.xml")
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"scan took {elapsed:.1f}s: unterminated tags are quadratic again"
    assert doc.roots == []
    assert "unparsed-markup" in {i.code for i in doc.issues}


def test_unterminated_tag_does_not_swallow_the_next_well_formed_tag() -> None:
    doc = scan(b'<r><a x="1" <b y="2"/></r>')
    assert doc.root is not None
    [b] = list(doc.root.iter("b"))
    assert b.attrs == {"y": "2"} and b.raw == b'<b y="2"/>'
    assert list(doc.root.iter("a")) == []  # the broken tag is not an element
    assert "unparsed-markup" in {i.code for i in doc.issues}


# --------------------------------------------------------------------------- duplicate attributes


def test_duplicate_attribute_is_reported_after_the_text_before_the_tag() -> None:
    # Fails on 8d562a6: the duplicate was reported before the gap's own issues.
    doc = scan(b'junk<a x="1" x="2"/>')
    assert [(i.code, i.offset) for i in doc.issues] == [
        ("text-outside-root", 0),
        ("duplicate-attribute", 4),
    ]
    assert doc.root is not None and doc.root.attrs == {"x": "2"}  # the last value wins


def test_duplicate_attribute_is_not_reported_for_an_unterminated_tag() -> None:
    # Fails on 8d562a6: it was reported as the attributes were read, tag or no tag.
    doc = scan(b'<a x="1" x="2"')
    assert doc.roots == []
    assert "duplicate-attribute" not in {i.code for i in doc.issues}


def test_duplicate_attribute_issues_are_capped_per_code() -> None:
    doc = scan(b"<a" + b' x="1"' * 1000 + b"/>")
    listed = [i for i in doc.issues if i.code == "duplicate-attribute"]
    assert len(listed) == 100  # the per-code cap
    [more] = [i for i in doc.issues if i.code == "issues-truncated"]
    assert more.message.startswith("899 more duplicate-attribute")  # 999 repeats, 100 listed
