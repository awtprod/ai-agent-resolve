"""A small DTD validator for the FCPXML DTD, used when lxml is not installed.

It reads Apple's ``FCPXMLv1_10.dtd`` (not kept in the repo: point ``AAR_FCPXML_DTD`` at a copy)
and checks a document against it: every element declared, its children matching the element's
content model, text only where the model allows it, every attribute declared with a valid
enumerated or ``#FIXED`` value, every ``#REQUIRED`` attribute present, IDs unique and every IDREF
resolving. That is what a validating parser checks for these documents.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

_COMMENT = re.compile(r"<!--.*?-->", re.S)
_ENTITY = re.compile(r'<!ENTITY\s+%\s+([\w.-]+)\s+"([^"]*)"\s*>', re.S)
_ELEMENT = re.compile(r"<!ELEMENT\s+([\w.:-]+)\s+(.*?)\s*>", re.S)
_ATTLIST = re.compile(r"<!ATTLIST\s+([\w.:-]+)\s*(.*?)\s*>", re.S)
# XML 1.0 (Fifth Edition) Name: NameStartChar NameChar* (productions [4], [4a], [5]).
_NAME_START = (
    ":A-Z_a-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u02ff\u0370-\u037d\u037f-\u1fff"
    "\u200c-\u200d\u2070-\u218f\u2c00-\u2fef\u3001-\ud7ff\uf900-\ufdcf\ufdf0-\ufffd"
    "\U00010000-\U000effff"
)
_NAME_CHAR = _NAME_START + "\\-.0-9\u00b7\u0300-\u036f\u203f-\u2040"
XML_NAME = re.compile(f"[{_NAME_START}][{_NAME_CHAR}]*")
_ATTR = re.compile(
    r"([\w.:-]+)\s+(CDATA|ID|IDREFS?|NMTOKENS?|\([^)]*\))\s+"
    r"(#REQUIRED|#IMPLIED|#FIXED\s+(?:\"[^\"]*\"|'[^']*')|\"[^\"]*\"|'[^']*')",
    re.S,
)


@dataclass
class Attribute:
    kind: str  # CDATA, ID, IDREF, IDREFS, NMTOKEN(S) or "enum"
    values: frozenset[str] = frozenset()
    required: bool = False
    fixed: str | None = None


@dataclass
class Dtd:
    models: dict[str, str] = field(default_factory=dict)  # element -> regex over child tags
    mixed: set[str] = field(default_factory=set)  # elements that may hold text
    empty: set[str] = field(default_factory=set)  # declared EMPTY: no content, not even spaces
    attributes: dict[str, dict[str, Attribute]] = field(default_factory=dict)


def _expand(text: str, entities: dict[str, str]) -> str:
    for _ in range(20):
        new = re.sub(r"%([\w.-]+);", lambda m: entities[m.group(1)], text)
        if new == text:
            return new
        text = new
    raise ValueError("entity expansion does not settle")


def _model_regex(model: str) -> tuple[str, bool]:
    """A content model as a regex over space-joined child tags, and whether text is allowed."""
    model = model.strip()
    if model == "EMPTY":
        return "", False
    if model == "ANY":
        return r"(?:\S+ )*", True
    if "#PCDATA" in model:
        names = [n for n in re.findall(r"[\w.:-]+", model) if n != "PCDATA"]
        if not names:
            return "", True
        return "(?:(?:" + "|".join(map(re.escape, names)) + ") )*", True
    out = []
    for token in re.findall(r"[\w.:-]+|[(),|?*+]", model):
        if token == ",":
            continue
        if token == "(":
            out.append("(?:")
        elif token in "|)?*+":
            out.append(token)
        else:
            out.append(f"(?:{re.escape(token)} )")
    return "".join(out), False


def load_dtd(text: str) -> Dtd:
    text = _COMMENT.sub("", text)
    entities: dict[str, str] = {}
    for name, value in _ENTITY.findall(text):
        entities[name] = _expand(value, entities)
    text = _expand(_ENTITY.sub("", text), entities)
    dtd = Dtd()
    for name, model in _ELEMENT.findall(text):
        regex, mixed = _model_regex(model)
        dtd.models[name] = regex
        if mixed:
            dtd.mixed.add(name)
        if model.strip() == "EMPTY":
            dtd.empty.add(name)
    for element, body in _ATTLIST.findall(text):
        attrs = dtd.attributes.setdefault(element, {})
        for name, kind, default in _ATTR.findall(body):
            attr = Attribute(kind)
            if kind.startswith("("):
                attr.kind = "enum"
                attr.values = frozenset(v.strip() for v in kind.strip("()").split("|"))
            if default == "#REQUIRED":
                attr.required = True
            elif default.startswith("#FIXED"):
                attr.fixed = default.split(None, 1)[1].strip("\"'")
            attrs[name] = attr
    return dtd


def validate(dtd: Dtd, document: str) -> list[str]:
    """Every way ``document`` breaks the DTD (empty when it is valid)."""
    root = ET.fromstring(document)
    problems: list[str] = []
    ids: set[str] = set()
    refs: list[tuple[str, str]] = []
    for element in root.iter():
        tag = element.tag
        if tag not in dtd.models:
            problems.append(f"undeclared element <{tag}>")
            continue
        children = "".join(f"{child.tag} " for child in element)
        if not re.fullmatch(dtd.models[tag], children):
            problems.append(f"<{tag}> children {children.strip()!r} break its content model")
        if tag in dtd.empty:
            # EMPTY means no content at all, white space included (XML 1.0, VC Element Valid).
            if element.text:
                problems.append(f"<{tag}> is declared EMPTY but holds {element.text!r}")
        else:
            has_text = (element.text or "").strip() or any((c.tail or "").strip() for c in element)
            if has_text and tag not in dtd.mixed:
                problems.append(f"<{tag}> holds text its model does not allow")
        declared = dtd.attributes.get(tag, {})
        for name, value in element.attrib.items():
            attr = declared.get(name)
            if attr is None:
                problems.append(f"<{tag}> has undeclared attribute {name!r}")
                continue
            if attr.kind == "enum" and value not in attr.values:
                problems.append(f"<{tag} {name}={value!r}> is not one of {sorted(attr.values)}")
            if attr.fixed is not None and value != attr.fixed:
                problems.append(f"<{tag} {name}={value!r}> is fixed at {attr.fixed!r}")
            if attr.kind == "ID":
                if not XML_NAME.fullmatch(value):
                    problems.append(f"<{tag} {name}={value!r}> is not an XML Name")
                if value in ids:
                    problems.append(f"duplicate ID {value!r}")
                ids.add(value)
            elif attr.kind == "IDREF":
                # An IDREF holds exactly one name; only IDREFS may be a whitespace list.
                if not value or value != value.strip() or len(value.split()) != 1:
                    problems.append(f"<{tag} {name}={value!r}> is not a single IDREF name")
                elif not XML_NAME.fullmatch(value):
                    problems.append(f"<{tag} {name}={value!r}> is not an XML Name")
                else:
                    refs.append((tag, value))
            elif attr.kind == "IDREFS":
                for token in value.split():
                    if XML_NAME.fullmatch(token):
                        refs.append((tag, token))
                    else:
                        problems.append(f"<{tag} {name}> token {token!r} is not an XML Name")
        for name, attr in declared.items():
            if attr.required and name not in element.attrib:
                problems.append(f"<{tag}> lacks required attribute {name!r}")
    problems.extend(
        f"<{tag}> IDREF {ref!r} resolves to no ID" for tag, ref in refs if ref not in ids
    )
    return problems
