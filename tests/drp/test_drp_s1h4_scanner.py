"""S1h4 item 1: the tag scanner.

The scanner reads a start tag one attribute at a time, charges every attribute against the
element budget, stores at most 4096 per element, and skips an unterminated tag instead of
re-reading it from each ``<`` inside a quoted value (which was quadratic). A test that names a
commit in its comment fails on that commit (20f89a0 is S1h3, 8d562a6 the first S1h4 scanner).
"""

from __future__ import annotations

import time
import tracemalloc
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
    # Models the 10M-attribute bomb in miniature (fails on 20f89a0: 5000 stored, 1 token). The
    # 410a6b3 version of this test passed while the dropped attributes were *silent*; see
    # test_an_element_past_the_attribute_cap_says_so and test_drp_s1h4_cap_gate.py.
    doc = scan(_attr_tag(5000))
    assert doc.root is not None
    assert len(doc.root.attrs) == 4096
    assert "k0" in doc.root.attrs and "k4095" in doc.root.attrs and "k4096" not in doc.root.attrs
    assert doc.element_count == 5001


def test_an_element_past_the_attribute_cap_says_so() -> None:
    # Fails on 410a6b3: attributes past 4096 were dropped with no issue, so the reader raised no
    # warning and `aar drp diff --gate` compared only the first 4096 (validator C1).
    doc = scan(b"<r>" + _attr_tag(5000, b"Big") + b"</r>", "m.xml")
    capped = [i for i in doc.issues if i.code == "attributes-truncated"]
    assert len(capped) == 1
    assert capped[0].offset == 3  # the element's "<"
    assert "5000 attributes" in capped[0].message and "first 4096" in capped[0].message
    # exactly 4096 is still whole; one more is not
    assert not [i for i in scan(_attr_tag(4096)).issues if i.code == "attributes-truncated"]
    one_over = [i for i in scan(_attr_tag(4097)).issues if i.code == "attributes-truncated"]
    assert len(one_over) == 1 and "4097 attributes" in one_over[0].message


def test_each_capped_element_gets_its_own_issue_and_the_log_stays_bounded() -> None:
    body = b"".join(_attr_tag(4100, b"E") for _ in range(150))
    doc = scan(b"<r>" + body + b"</r>", "m.xml")
    codes = [i.code for i in doc.issues]
    assert codes.count("attributes-truncated") == 100  # the per-code cap
    assert codes.count("issues-truncated") == 1
    more = next(i for i in doc.issues if i.code == "issues-truncated")
    assert "50 more attributes-truncated" in more.message


def test_a_capped_tag_that_never_closes_reports_unparsed_markup_not_truncation() -> None:
    # The cap issue belongs to an element that exists; an unterminated tag has none.
    doc = scan(b"<r>" + _attr_tag(5000, b"Big", end=b" <") + b"</r>", "m.xml")
    codes = {i.code for i in doc.issues}
    assert "attributes-truncated" not in codes
    assert "unparsed-markup" in codes


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


def test_an_unterminated_tag_costs_no_decode_of_its_attribute_values() -> None:
    # M1: a value closed by a far-off quote and then no ">" was decoded and stored before the
    # tag was known to be unterminated, on top of the copy the gap makes: twice the member in
    # memory at 410a6b3 (measured 2.0x here; 1.0x now), 2060 MB against 1048 MB at 530 MB.
    data = b'<a b="' + b"<x " * 2_000_000 + b'"c'
    tracemalloc.start()
    try:
        doc = scan(data, "m.xml")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert doc.roots == []
    assert "unparsed-markup" in {i.code for i in doc.issues}
    assert peak < 1.5 * len(data), f"peak {peak / len(data):.2f}x the member"


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


# ------------------------------------------------------------ pins for the validator's survivors
# Round 2 L4: each test below fails on one mutant that survived the round-1 suite (the id is the
# validator's, in aar-evidence/s1h4-val/mut/). They pin behaviour, not wording.


def test_an_empty_double_quoted_value_is_an_empty_string_not_a_missing_one() -> None:
    # S17: `dq or sq` read "" as absent and then decoded None.
    doc = scan(b"<a x=\"\" y=''/>")
    assert doc.root is not None and doc.root.attrs == {"x": "", "y": ""}
    assert doc.issues == []


def test_whitespace_may_precede_the_closing_angle_bracket() -> None:
    # S18: `<a b="1" >` and `<a />` are well-formed; without the leading \s* they became
    # unterminated tags and the whole element vanished into the gap.
    for data, tag, attrs in (
        (b'<a b="1" ></a>', "a", {"b": "1"}),
        (b"<a />", "a", {}),
        (b"<a\n\t>t</a>", "a", {}),
    ):
        doc = scan(data)
        assert doc.issues == [], data
        assert doc.root is not None and (doc.root.tag, doc.root.attrs) == (tag, attrs), data


def test_an_attribute_needs_whitespace_before_it() -> None:
    # S19: `<a b="1"c="2">` is not well-formed XML; reading it as two attributes would let a
    # malformed tag pass the gate as valid. It stays an unparsed tag.
    doc = scan(b'<a b="1"c="2"></a>')
    assert doc.roots == []
    assert "unparsed-markup" in {i.code for i in doc.issues}


def test_the_bytes_of_a_start_tag_are_work() -> None:
    # S21: `work` is what the reader's total byte budget adds up, so a member that is mostly
    # start tags must not be free. Every byte of a completely scanned member is examined at
    # least once, and the attributes of a tag are the tag's bytes.
    for data in (b'<a b="1">x</a>', b"<a/>", _attr_tag(40), b"<r><a k='v'/><b/></r>"):
        assert scan(data).work >= len(data), data
    one = scan(_attr_tag(1)).work
    many = scan(_attr_tag(101)).work
    assert many - one >= 100 * len(b' k0="v"'), (one, many)


def test_a_run_of_unterminated_tags_trips_the_element_budget() -> None:
    # S10: an unterminated start tag is skipped, but it is still a token that was materialised.
    doc = scan(b"<a " * 10, "m.xml", max_elements=5)
    assert doc.element_count == 5
    assert doc.stopped_at == 15  # the sixth "<a"
    assert [i.code for i in doc.issues] == ["budget-exceeded"]


def test_a_run_of_unmatched_end_tags_trips_the_element_budget() -> None:
    # S25: "</" that is not a close tag is skipped, but it is a token too.
    doc = scan(b"</" * 10, "m.xml", max_elements=5)
    assert doc.element_count == 5
    assert doc.stopped_at == 10
    assert [i.code for i in doc.issues] == ["budget-exceeded"]
    assert scan(b"</>" * 3).element_count == 3


def test_a_budget_stop_inside_a_tag_leaves_no_pending_gap() -> None:
    # S07: the budget can run out between two attributes of one tag. The rest of the member is
    # then not read, so no text before it may be reported as unparsed markup or stray text.
    for data, limit in (
        (b'<a b="1" c="2">tail<x/>', 2),
        (b'<r><a b="1" c="2" d="3"/>tail<x/></r>', 4),
    ):
        doc = scan(data, "m.xml", max_elements=limit)
        assert doc.stopped_at is not None, data
        assert [i.code for i in doc.issues] == ["budget-exceeded"], (data, doc.issues)


def test_an_attribute_name_with_invalid_utf8_keeps_a_replacement_character() -> None:
    # Round 3 (validator R14): the name is decoded with errors="replace". Dropping the bad bytes
    # ("ignore") would fold `k\xffx` into the valid name `kx` and silently lose one of the two.
    doc = scan(b'<r kx="valid" k\xffx="bad" \xff="lone"/>', "m.xml")
    assert doc.root is not None
    assert doc.root.attrs == {"kx": "valid", "k�x": "bad", "�": "lone"}
    assert [i.code for i in doc.issues] == ["invalid-utf8"]
