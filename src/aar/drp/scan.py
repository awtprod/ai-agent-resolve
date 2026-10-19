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

__all__ = ["Document", "Element", "ScanIssue", "parse_int", "scan", "unescape"]

# A start tag is read piece by piece -- the name, then one attribute at a time, then the
# closing ">" -- never with one regex over the whole tag. The old single regex buffered the
# entire attribute run (a 10M-attribute tag peaked at several GB) and, on a tag with no
# terminator, restarted at every "<" inside a quoted value (quadratic). Each attribute is also
# charged against the budget. End tags, comments, CDATA, PIs and declarations are found by
# bytes.find; with this, every byte is examined a bounded number of times.
_NAME_AT = re.compile(rb"<(?P<start>[^\s<>/!?][^\s<>/]*)")
_ATTR_AT = re.compile(rb"""\s+(?P<name>[^\s<>=/]+)\s*=\s*(?:"(?P<dq>[^"]*)"|'(?P<sq>[^']*)')""")
_TAGCLOSE_AT = re.compile(rb"\s*(?P<close>/?)>")
_END = re.compile(rb"</(?P<end>[^\s<>]+)\s*>")
_DELIMITED: tuple[tuple[bytes, bytes, str], ...] = (
    (b"<!--", b"-->", "comment"),
    (b"<![CDATA[", b"]]>", "cdata"),
    (b"<?", b"?>", "pi"),
    (b"<!", b">", "decl"),
)
_MAX_INT_DIGITS = 20
_MAX_ISSUES_PER_CODE = 100
# Attributes past this many on a single element are charged against the budget but not stored,
# so one element with millions of attributes cannot grow an unbounded dict. Far above any real
# element (corpus maximum is a handful), so decoded models are unaffected. An element that goes
# past it is never silent: it carries an ``attributes-truncated`` issue (see ``scan``).
_MAX_ATTRS_PER_ELEMENT = 4096
_ENTITY = re.compile(r"&(lt|gt|amp|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);")
_NAMED = {"lt": "<", "gt": ">", "amp": "&", "quot": '"', "apos": "'"}


def parse_int(text: str, *, max_digits: int = _MAX_INT_DIGITS) -> int | None:
    """Parse a decimal integer the way Resolve writes them, or return ``None``.

    Accepts optional surrounding whitespace and a leading ``-``, then ASCII digits only, at
    most ``max_digits`` of them. ``int()`` alone is too lenient (it accepts ``"1_000"`` and
    non-ASCII digits) and ``str.isdigit()`` is wrong (it accepts ``"²"``, which ``int()``
    rejects). Never raises.
    """
    body = text.strip()
    digits = body[1:] if body.startswith("-") else body
    if not digits or len(digits) > max_digits or not (digits.isascii() and digits.isdigit()):
        return None
    return int(body)


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


def _delimited(data: bytes, lt: int) -> tuple[str, int, int, int] | None:
    """If a comment/CDATA/PI/declaration starts at ``lt``: (kind, inner start, inner end, end).

    An unterminated construct runs to the end of ``data`` and its kind gets a ``!`` suffix.
    """
    for opener, closer, kind in _DELIMITED:
        if data.startswith(opener, lt):
            inner = lt + len(opener)
            close = data.find(closer, inner)
            if close < 0:
                return kind + "!", inner, len(data), len(data)
            return kind, inner, close, close + len(closer)
    return None


def _strip_markup(raw: bytes) -> bytes:
    """Keep CDATA content and drop comments/PIs inside a leaf's raw bytes (linear time)."""
    out = bytearray()
    pos = 0
    while True:
        lt = raw.find(b"<", pos)
        if lt < 0:
            out += raw[pos:]
            return bytes(out)
        out += raw[pos:lt]
        found = _delimited(raw, lt)
        if found is None:
            out += b"<"
            pos = lt + 1
            continue
        kind, inner, inner_end, pos = found
        if kind.startswith("cdata"):
            out += raw[inner:inner_end]


@dataclass(eq=False, slots=True)
class Document:
    """The scan result for one member."""

    member: str
    data: bytes
    roots: list[Element]
    comments: list[tuple[int, int, str]]
    issues: list[ScanIssue]
    element_count: int = 0
    """Scanner nodes materialised at all levels: start/self-closing and end tags, comments,
    PIs, CDATA, declarations and lone ``<``. The per-file budget is charged against this, so
    hostile input made only of close tags or comments is bounded too."""
    work: int = 0
    """Bytes examined plus tag-match attempts: a deterministic measure of scanning effort."""
    stopped_at: int | None = None
    """Offset where scanning stopped because the element budget ran out, else ``None``."""

    @property
    def root(self) -> Element | None:
        return self.roots[0] if self.roots else None


class _IssueLog(list[ScanIssue]):
    """Keeps at most ``_MAX_ISSUES_PER_CODE`` issues per code; counts the rest."""

    def __init__(self) -> None:
        super().__init__()
        self.counts: dict[str, int] = {}
        self.first_dropped: dict[str, int] = {}

    def append(self, issue: ScanIssue) -> None:
        n = self.counts.get(issue.code, 0) + 1
        self.counts[issue.code] = n
        if n <= _MAX_ISSUES_PER_CODE:
            super().append(issue)
        else:
            self.first_dropped.setdefault(issue.code, issue.offset)

    def finish(self) -> list[ScanIssue]:
        out = list(self)
        for code, offset in self.first_dropped.items():
            more = self.counts[code] - _MAX_ISSUES_PER_CODE
            out.append(
                ScanIssue("issues-truncated", offset, f"{more} more {code} issue(s) not listed")
            )
        return out


def scan(data: bytes, member: str = "", *, max_elements: int | None = None) -> Document:
    """Scan one XML member into a :class:`Document` without ever raising on malformed input.

    Scanning stops with a ``budget-exceeded`` issue before node number ``max_elements + 1`` is
    materialised (``max_elements`` counts every token, not only start elements; see
    ``Document.element_count``); what was scanned so far is kept and open elements are closed at
    the end of the member. Attributes are charged too (one each), and an element keeps at most
    4096 of them; an element with more gets one ``attributes-truncated`` issue (with the exact
    count), so a reader can warn and a diff gate can refuse rather than compare a prefix.
    Attribute names and values are decoded only after the tag's closing ``>`` has matched.

    Time is linear in ``len(data)``: a start tag is read one attribute at a time and a tag with
    no closing ``>`` is skipped past, never re-read. Measured on the worst shapes tried: 240 KB
    of ``<`` inside quoted values in one unterminated tag takes 0.09 s (the previous
    single-regex scanner needed 145 s), and 3M tokens take 1.5 to 11 s. ``Document.work``
    counts bytes and match attempts, not the work inside one regex match, so it would not have
    shown that quadratic cost; ``tests/drp/test_drp_s1h4_scanner.py`` guards it by timing.
    """
    issues = _IssueLog()
    work = 0
    element_count = 0
    stopped_at: int | None = None
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
        nonlocal work
        if start >= end:
            return
        work += end - start
        chunk = data[start:end]
        if b"<" in chunk:
            at = start + chunk.index(b"<")
            issues.append(ScanIssue("unparsed-markup", at, "markup the scanner could not parse"))
        if chunk.strip():
            if stack:
                text_seen[-1] = True
            else:
                issues.append(ScanIssue("text-outside-root", start, "text outside any element"))

    open_tags: dict[str, int] = {}

    def push(el: Element) -> None:
        stack.append(el)
        text_seen.append(False)
        open_tags[el.tag] = open_tags.get(el.tag, 0) + 1

    def pop() -> tuple[Element, bool]:
        el = stack.pop()
        open_tags[el.tag] -= 1
        return el, text_seen.pop()

    pos = 0  # end of the last construct; text from here on is not yet accounted for
    cursor = 0  # where to look for the next "<"
    while True:
        lt = data.find(b"<", cursor)
        if lt < 0:
            work += len(data) - cursor
            break
        # Every token the scanner materialises is charged, so a member that is all close tags,
        # comments or lone "<" trips the budget too, not only one full of start elements. The
        # check sits at the top so it bounds the work and memory of any token kind.
        if max_elements is not None and element_count >= max_elements:
            stopped_at = lt
            msg = f"node budget of {max_elements} reached; the rest of the member is not read"
            issues.append(ScanIssue("budget-exceeded", lt, msg))
            pos = len(data)
            break
        work += lt - cursor + 1
        nxt = data[lt + 1 : lt + 2]
        found = _delimited(data, lt) if nxt in (b"!", b"?") else None
        if found is not None:
            gap(pos, lt)
            kind, inner, inner_end, pos = found
            work += pos - lt
            cursor = pos
            element_count += 1
            if kind.endswith("!"):
                kind = kind[:-1]
                issues.append(
                    ScanIssue(
                        f"unterminated-{kind}", lt, f"{kind} never closed; rest of member skipped"
                    )
                )
            elif kind == "comment":
                comments.append((lt, pos, data[inner:inner_end].decode("utf-8", "replace")))
            if kind == "cdata" and stack:
                text_seen[-1] = True
            continue
        if nxt == b"/":
            m = _END.match(data, lt)
            work += 1 if m is None else m.end() - lt
            if m is None:
                element_count += 1
                cursor = lt + 1
                continue
            gap(pos, m.start())
            pos = cursor = m.end()
            element_count += 1
            tag = m.group("end").decode("utf-8", errors="replace")
            if stack and stack[-1].tag == tag:
                el, had_text = pop()
                _close(el, had_text, m.start(), m.end(), issues)
            elif open_tags.get(tag):
                while stack and stack[-1].tag != tag:
                    el, _ = pop()
                    issues.append(ScanIssue("unclosed-element", el.start, f"<{el.tag}> not closed"))
                    _close(el, False, m.start(), m.start(), issues)
                el, had_text = pop()
                _close(el, had_text, m.start(), m.end(), issues)
            else:
                msg = f"</{tag}> has no open element"
                issues.append(ScanIssue("stray-end-tag", m.start(), msg))
            continue
        # Start or self-closing tag: match the name, then read attributes one at a time until
        # the closing ">". Each attribute is charged, so a tag with millions of them trips the
        # budget, and nothing re-scans the attribute run, so a tag with no terminator is linear.
        mname = _NAME_AT.match(data, lt)
        if mname is None:
            work += 1
            element_count += 1
            cursor = lt + 1
            continue
        # Attribute matches are only *recorded* while the tag is read: a Match holds spans into
        # ``data``, not copies, so a tag that never closes (``<a b="`` followed by hundreds of MB)
        # costs nothing beyond the gap it leaves. Names and values are decoded once the closing
        # ">" has matched. At most ``_MAX_ATTRS_PER_ELEMENT`` are kept; the rest are charged
        # against the budget and counted, and the element then carries an
        # ``attributes-truncated`` issue (the reader turns any scan issue into a warning, so a
        # diff gate refuses the file instead of comparing only the attributes that were kept).
        found_attrs: list[re.Match[bytes]] = []
        attrs_over = 0
        p = mname.end()
        closed: bool | None = None
        over_budget = False
        while True:
            if max_elements is not None and element_count >= max_elements:
                over_budget = True
                break
            mc = _TAGCLOSE_AT.match(data, p)
            if mc is not None:
                closed = mc.group("close") == b"/"
                p = mc.end()
                break
            ma = _ATTR_AT.match(data, p)
            if ma is None:  # neither the closing ">" nor another attribute: unterminated tag
                break
            if len(found_attrs) < _MAX_ATTRS_PER_ELEMENT:
                found_attrs.append(ma)
            else:
                attrs_over += 1
            element_count += 1
            p = ma.end()
        work += p - lt
        if over_budget:
            stopped_at = lt
            msg = f"node budget of {max_elements} reached; the rest of the member is not read"
            issues.append(ScanIssue("budget-exceeded", lt, msg))
            pos = len(data)
            break
        if closed is None:
            # Unterminated start tag: the text stays in the gap, which reports it as unparsed
            # markup. Skip past what was scanned (never rewind to lt + 1, which would re-read the
            # attribute run at every inner "<" -- quadratic). Nothing was decoded, and the
            # recorded matches are dropped before the gap is ever copied.
            found_attrs = []
            element_count += 1
            cursor = max(p, lt + 1)
            continue
        gap(pos, lt)
        attrs: dict[str, str] = {}
        # Repeated attribute names are reported (after the text before the tag, as before). Keep
        # at most the per-code cap of names: more would be dropped by the issue log anyway, and a
        # tag may repeat one name thousands of times.
        repeated: list[str] = []
        repeated_extra = 0
        for ma in found_attrs:
            name = ma.group("name").decode("utf-8", errors="replace")
            raw = ma.group("dq") if ma.group("dq") is not None else ma.group("sq")
            if name in attrs:
                if len(repeated) < _MAX_ISSUES_PER_CODE:
                    repeated.append(name)
                else:
                    repeated_extra += 1
            attrs[name] = unescape(raw.decode("utf-8", errors="replace"))
        found_attrs = []
        for name in repeated:
            issues.append(ScanIssue("duplicate-attribute", lt, f"attribute {name!r} repeated"))
        for _ in range(repeated_extra):  # past the cap: counted, never listed
            issues.append(ScanIssue("duplicate-attribute", lt, "attribute repeated"))
        if attrs_over:
            total = _MAX_ATTRS_PER_ELEMENT + attrs_over
            msg = (
                f"element has {total} attributes; only the first {_MAX_ATTRS_PER_ELEMENT} "
                "were read, so attribute changes past that point cannot be seen"
            )
            issues.append(ScanIssue("attributes-truncated", lt, msg))
        pos = cursor = p
        element_count += 1
        tag = mname.group("start").decode("utf-8", errors="replace")
        el = Element(
            tag=tag,
            attrs=attrs,
            start=lt,
            end=p,
            inner_start=None if closed else p,
            inner_end=None if closed else p,
            _data=data,
        )
        if stack:
            stack[-1].children.append(el)
        else:
            if roots:
                issues.append(ScanIssue("extra-root", lt, f"second root element <{tag}>"))
            roots.append(el)
        if not closed:
            push(el)
    gap(pos, len(data))
    while stack:
        el, _ = pop()
        if stopped_at is None:
            msg = f"<{el.tag}> still open at end of member"
            issues.append(ScanIssue("truncated", el.start, msg))
        _close(el, False, len(data), len(data), issues)
    return Document(
        member=member,
        data=data,
        roots=roots,
        comments=comments,
        issues=issues.finish(),
        element_count=element_count,
        work=work,
        stopped_at=stopped_at,
    )


def _close(
    el: Element, had_text: bool, end_tag_start: int, end: int, issues: list[ScanIssue]
) -> None:
    el.inner_end = end_tag_start
    el.end = end
    if had_text and el.children:
        issues.append(ScanIssue("mixed-content", el.start, f"<{el.tag}> mixes text and elements"))
