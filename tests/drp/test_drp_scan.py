"""Tests for the tolerant scanner in ``aar.drp.scan``."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from aar.drp.scan import scan

SAMPLE = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b'<!--DbAppVer="21.0.0.0048" DbPrjVer="17"-->\n'
    b'<Sm2Sequence DbId="s1">\n'
    b" <pLmVerTable>\n"
    b'  <ListMgt::LmVersionTable DbId="v1">\n'
    b"   <Name>A &amp; B &lt;1&gt; &#233;</Name>\n"
    b"   <In/>\n"
    b"   <Body>8128b52ffd</Body>\n"
    b"  </ListMgt::LmVersionTable>\n"
    b" </pLmVerTable>\n"
    b"</Sm2Sequence>\n"
)


def test_stock_parser_rejects_resolve_tags_but_scanner_reads_them() -> None:
    with pytest.raises(ET.ParseError):
        ET.fromstring(SAMPLE)
    doc = scan(SAMPLE, "m.xml")
    assert doc.issues == []
    assert doc.root is not None and doc.root.tag == "Sm2Sequence"
    table = doc.root.path("pLmVerTable", "ListMgt::LmVersionTable")
    assert table is not None
    assert table.attrs == {"DbId": "v1"}
    assert table.child_text("Name") == "A & B <1> é"
    assert doc.comments[0][2] == 'DbAppVer="21.0.0.0048" DbPrjVer="17"'


def test_spans_cover_exact_source_bytes() -> None:
    doc = scan(SAMPLE)
    assert doc.root is not None
    body = next(doc.root.iter("Body"))
    assert SAMPLE[body.start : body.end] == b"<Body>8128b52ffd</Body>"
    assert body.inner_start is not None and body.inner_end is not None
    assert SAMPLE[body.inner_start : body.inner_end] == b"8128b52ffd"
    name = next(doc.root.iter("Name"))
    assert name.inner_raw == b"A &amp; B &lt;1&gt; &#233;"
    in_el = next(doc.root.iter("In"))
    assert in_el.self_closing and in_el.text == "" and in_el.raw == b"<In/>"
    assert SAMPLE[doc.root.start : doc.root.end].endswith(b"</Sm2Sequence>")


def test_attributes_quotes_entities_and_cdata() -> None:
    doc = scan(b"<a x='1' y=\"&quot;q&quot;\"><b><![CDATA[<raw> & ok]]></b></a>")
    assert doc.root is not None
    assert doc.root.attrs == {"x": "1", "y": '"q"'}
    assert doc.root.child_text("b") == "<raw> & ok"


def test_elements_helper_unwraps_resolve_vectors() -> None:
    doc = scan(b"<Vec><Element><A/></Element><Element><B/></Element><Other/></Vec>")
    assert doc.root is not None
    assert [e.tag for e in doc.root.elements()] == ["A", "B"]


@pytest.mark.parametrize(
    ("data", "codes"),
    [
        (b"<a><b></a>", {"unclosed-element"}),
        (b"<a></b></a>", {"stray-end-tag"}),
        (b"<a><b>text", {"truncated"}),
        (b"<a>text<b/>more</a>", {"mixed-content"}),
        (b"<a><b x=1/></a>", {"unparsed-markup"}),
        (b"<a/><c/>", {"extra-root"}),
        (b"<a>\xff\xfe</a>", {"invalid-utf8"}),
    ],
)
def test_malformed_input_is_reported_not_raised(data: bytes, codes: set[str]) -> None:
    doc = scan(data)
    assert {i.code for i in doc.issues} >= codes
    assert doc.root is not None and doc.root.tag == "a"


def test_unclosed_child_ends_where_parent_closes() -> None:
    data = b"<a><b><c>x</c></a>"
    doc = scan(data)
    assert doc.root is not None
    b = doc.root.find("b")
    assert b is not None and b.find("c") is not None
    assert data[b.start : b.end] == b"<b><c>x</c>"
    assert doc.root.end == len(data)
