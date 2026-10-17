"""Tolerant XML scanner that keeps byte spans.

Resolve writes element names such as ``ListMgt::LmVersionTable`` and ``Gallery::GyGallery``,
which stock XML parsers reject as undeclared namespace prefixes. This scanner does not try to
validate XML. It tokenizes the raw bytes of one zip member, builds an element tree, and records
for every element where it starts and ends in those bytes, so later slices can splice edits
back into the original text without re-serializing anything.

Problems (stray end tags, unclosed elements, unparseable markup, invalid UTF-8) never raise;
they are collected as :class:`ScanIssue` records on the :class:`Document`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

__all__ = ["Document", "Element", "ScanIssue", "scan", "unescape"]

_TOKEN = re.compile(
    rb"""
      <!--(?P<comment>.*?)-->
    | <!\[CDATA\[(?P<cdata>.*?)\]\]>
    | <\?(?P<pi>.*?)\?>
    | <!(?P<decl>[^>]*)>
    | </(?P<end>[^\s<>]+)\s*>
    | <(?P<start>[^\s<>/!?][^\s<>/]*)
       (?P<attrs>(?:\s+[^\s<>=/]+\s*=\s*(?:"[^"]*"|'[^']*'))*)
       \s*(?P<close>/?)>
    """,
    re.S | re.X,
)
_ATTR = re.compile(rb"""([^\s<>=/]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_ENTITY = re.compile(r"&(lt|gt|amp|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);")
_NAMED = {"lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'"}


def unescape(text: str) -> str:
    """Replace the five XML entities and numeric character references; leave anything else."""
    if "&" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        ref = m.group(1)
        if ref in _NAMED:
            return _NAMED[ref]
        try:
            code = int(ref[2:], 16) if ref[1] in "xX" else int(ref[1:])
            return chr(code)
        except (ValueError, OverflowError):
            return m.group(0)

    return _ENTITY.sub(repl, text)


@dataclass(frozen=True, slots=True)
class ScanIssue:
    """Something the scanner tolerated rather than rejected."""

    code: str
    offset: int
    message: str


@dataclass(eq=False, slots=True)
class Element:
    """One element with its byte spans inside the member it came from.

    ``start``/``end`` cover the whole element, from ``<`` to the final ``>``.
    ``inner_start``/``inner_end`` cover the content between the start and end tags and are
    ``None`` for a self-closing element such as ``<In/>``.
    """

    tag: str
    attrs: dict[str, str]
    start: int
    end: int
    inner_start: int | None
    inner_end: int | None
    children: list[Element] = field(default_factory=list)
    _data: bytes = field(default=b"", repr=False)

    @property
    def self_closing(self) -> bool:
        return self.inner_start is None

    @property
    def raw(self) -> bytes:
        """The exact source bytes of the whole element."""
        return self._data[self.start : self.end]

    @property
    def inner_raw(self) -> bytes:
        """The exact source bytes between the tags (empty for a self-closing element)."""
        if self.inner_start is None or self.inner_end is None:
            return b""
        return self._data[self.inner_start : self.inner_end]

    @property
    def text(self) -> str | None:
        """Unescaped character data of a leaf element.

        ``""`` for an empty or self-closing leaf, ``None`` for an element with children (Resolve
        does not write mixed content; if it ever appears the scanner records an issue).
        """
        if self.children:
            return None
        raw = self.inner_raw
        if b"<" in raw:
            raw = _strip_markup(raw)
        return unescape(raw.decode("utf-8", errors="replace"))

    def find(self, tag: str) -> Element | None:
        """First direct child with this tag."""
        for child in self.children:
            if child.tag == tag:
                return child
        return None

    def findall(self, tag: str) -> list[Element]:
        """All direct children with this tag."""
        return [c for c in self.children if c.tag == tag]

    def child_text(self, tag: str) -> str | None:
        """Text of the first direct child with this tag, or ``None`` when there is no such child."""
        child = self.find(tag)
        return None if child is None else child.text

    def path(self, *tags: str) -> Element | None:
        """Follow direct children by tag, e.g. ``el.path("Video", "BtVideoInfo", "Clip")``."""
        node: Element | None = self
        for tag in tags:
            if node is None:
                return None
            node = node.find(tag)
        return node

    def iter(self, tag: str | None = None) -> Iterator[Element]:
        """Depth-first iteration over this element and its descendants (iterative, no recursion)."""
        stack: list[Element] = [self]
        while stack:
            node = stack.pop()
            if tag is None or node.tag == tag:
                yield node
            stack.extend(reversed(node.children))

    def elements(self) -> Iterator[Element]:
        """Direct children of ``<Element>`` wrappers, the way Resolve writes vectors."""
        for wrapper in self.children:
            if wrapper.tag == "Element":
                yield from wrapper.children


def _strip_markup(raw: bytes) -> bytes:
    """Keep CDATA content and drop comments/PIs inside a leaf's raw bytes."""
    out = bytearray()
    pos = 0
    for m in _TOKEN.finditer(raw):
        out += raw[pos : m.start()]
        if m.group("cdata") is not None:
            out += m.group("cdata")
        pos = m.end()
    out += raw[pos:]
    return bytes(out)


@dataclass(eq=False, slots=True)
class Document:
    """The scan result for one member."""

    member: str
    data: bytes
    roots: list[Element]
    comments: list[tuple[int, int, str]]
    issues: list[ScanIssue]

    @property
    def root(self) -> Element | None:
        return self.roots[0] if self.roots else None


def _decode_attrs(blob: bytes, offset: int, issues: list[ScanIssue]) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for m in _ATTR.finditer(blob):
        value = m.group(2) if m.group(2) is not None else m.group(3)
        name = m.group(1).decode("utf-8", errors="replace")
        if name in attrs:
            issues.append(ScanIssue("duplicate-attribute", offset, f"attribute {name!r} repeated"))
        attrs[name] = unescape(value.decode("utf-8", errors="replace"))
    return attrs


def scan(data: bytes, member: str = "") -> Document:
    """Scan one XML member into a :class:`Document` without ever raising on malformed input."""
    issues: list[ScanIssue] = []
    comments: list[tuple[int, int, str]] = []
    roots: list[Element] = []
    stack: list[Element] = []
    # Per open element: whether it has seen non-whitespace text (for mixed-content detection).
    text_seen: list[bool] = []

    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        issues.append(ScanIssue("invalid-utf8", exc.start, f"member is not valid UTF-8: {exc}"))

    def gap(start: int, end: int) -> None:
        if start >= end:
            return
        chunk = data[start:end]
        if b"<" in chunk:
            at = start + chunk.index(b"<")
            issues.append(ScanIssue("unparsed-markup", at, "markup the scanner could not parse"))
        if chunk.strip():
            if stack:
                text_seen[-1] = True
            else:
                issues.append(ScanIssue("text-outside-root", start, "text outside any element"))

    pos = 0
    for m in _TOKEN.finditer(data):
        gap(pos, m.start())
        pos = m.end()
        if m.group("comment") is not None:
            comments.append((m.start(), m.end(), m.group("comment").decode("utf-8", "replace")))
            continue
        if m.group("cdata") is not None:
            if stack:
                text_seen[-1] = True
            continue
        if m.group("pi") is not None or m.group("decl") is not None:
            continue
        end_name = m.group("end")
        if end_name is not None:
            tag = end_name.decode("utf-8", errors="replace")
            if stack and stack[-1].tag == tag:
                _close(stack.pop(), text_seen.pop(), m.start(), m.end(), issues)
            elif any(el.tag == tag for el in stack):
                while stack and stack[-1].tag != tag:
                    el = stack.pop()
                    text_seen.pop()
                    issues.append(ScanIssue("unclosed-element", el.start, f"<{el.tag}> not closed"))
                    _close(el, False, m.start(), m.start(), issues)
                _close(stack.pop(), text_seen.pop(), m.start(), m.end(), issues)
            else:
                msg = f"</{tag}> has no open element"
                issues.append(ScanIssue("stray-end-tag", m.start(), msg))
            continue
        # Start or self-closing tag.
        tag = m.group("start").decode("utf-8", errors="replace")
        attrs = _decode_attrs(m.group("attrs"), m.start(), issues) if m.group("attrs") else {}
        closed = m.group("close") == b"/"
        el = Element(
            tag=tag,
            attrs=attrs,
            start=m.start(),
            end=m.end(),
            inner_start=None if closed else m.end(),
            inner_end=None if closed else m.end(),
            _data=data,
        )
        if stack:
            stack[-1].children.append(el)
        else:
            if roots:
                issues.append(ScanIssue("extra-root", m.start(), f"second root element <{tag}>"))
            roots.append(el)
        if not closed:
            stack.append(el)
            text_seen.append(False)
    gap(pos, len(data))
    while stack:
        el = stack.pop()
        text_seen.pop()
        issues.append(ScanIssue("truncated", el.start, f"<{el.tag}> still open at end of member"))
        _close(el, False, len(data), len(data), issues)
    return Document(member=member, data=data, roots=roots, comments=comments, issues=issues)


def _close(
    el: Element, had_text: bool, end_tag_start: int, end: int, issues: list[ScanIssue]
) -> None:
    el.inner_end = end_tag_start
    el.end = end
    if had_text and el.children:
        issues.append(ScanIssue("mixed-content", el.start, f"<{el.tag}> mixes text and elements"))
