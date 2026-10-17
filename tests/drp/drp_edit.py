"""Test-only helpers: copy a fixture into a temp dir and edit the XML inside the zip.

These write new files only (never the fixtures) and re-encode blobs with the codecs written
from the spec in ``drp_helpers``. They are not product code.
"""

from __future__ import annotations

import re
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path

from drp_helpers import envelope, kd, pb_len, pb_var

from aar.drp import blobs

Edit = Callable[[bytes], bytes]


def edit_zip(src: Path, dst: Path, edits: dict[str, Edit] | None = None, **kw: object) -> Path:
    """Copy ``src`` to ``dst`` applying ``edits`` (member name -> bytes transform).

    ``reverse=True`` writes members in reverse order; ``stored=True`` writes them
    uncompressed. Directory entries are copied as they are.
    """
    edits = dict(edits or {})
    with zipfile.ZipFile(src) as zin:
        infos = zin.infolist()
        if kw.get("reverse"):
            infos = list(reversed(infos))
        with zipfile.ZipFile(dst, "w") as zout:
            for info in infos:
                data = zin.read(info)
                if info.filename in edits:
                    data = edits.pop(info.filename)(data)
                compress = zipfile.ZIP_STORED if kw.get("stored") else info.compress_type
                zout.writestr(info, data, compress_type=compress)
    assert not edits, f"members not found: {sorted(edits)}"
    return dst


def member_named(src: Path, prefix: str) -> str:
    with zipfile.ZipFile(src) as z:
        [name] = [n for n in z.namelist() if n.startswith(prefix) and not n.endswith("/")]
    return name


def once(old: bytes, new: bytes) -> Edit:
    """Replace ``old`` with ``new``; it must occur exactly once."""

    def run(data: bytes) -> bytes:
        assert data.count(old) == 1, f"{old!r} occurs {data.count(old)} times"
        return data.replace(old, new)

    return run


def every(old: bytes, new: bytes, count: int) -> Edit:
    def run(data: bytes) -> bytes:
        assert data.count(old) == count, f"{old!r} occurs {data.count(old)} times"
        return data.replace(old, new)

    return run


def after(anchor: bytes, old: bytes, new: bytes) -> Edit:
    """Replace the first ``old`` after ``anchor`` (e.g. a field inside one DbId's element)."""

    def run(data: bytes) -> bytes:
        at = data.index(anchor)
        i = data.index(old, at)
        return data[:i] + new + data[i + len(old) :]

    return run


def chain(*edits: Edit) -> Edit:
    def run(data: bytes) -> bytes:
        for e in edits:
            data = e(data)
        return data

    return run


def leaf_text(data: bytes, anchor: bytes, tag: bytes) -> tuple[int, int]:
    """Byte span of the text of the first ``<tag>`` after ``anchor``."""
    at = data.index(anchor)
    start = data.index(b"<" + tag + b">", at) + len(tag) + 2
    return start, data.index(b"</" + tag + b">", start)


def flip_hex_byte(anchor: bytes, tag: bytes, index: int) -> Edit:
    """XOR byte ``index`` of the hex blob in the first ``<tag>`` after ``anchor``."""

    def run(data: bytes) -> bytes:
        start, end = leaf_text(data, anchor, tag)
        raw = bytearray(bytes.fromhex(data[start:end].decode()))
        raw[index] ^= 0xFF
        return data[:start] + raw.hex().encode() + data[end:]

    return run


_GAP = re.compile(rb">(\n *)<")


def reindent(data: bytes) -> bytes:
    """Deepen the indentation between elements; the text of leaves is left untouched.

    Whitespace between a start tag and its own end tag (``<Notes>\n</Notes>``) is a leaf's
    text, so it is kept as it is.
    """
    out = []
    pos = 0
    for m in _GAP.finditer(data):
        tag = data[data.rfind(b"<", 0, m.start()) : m.start() + 1]
        opens = not tag.startswith((b"</", b"<?", b"<!")) and not tag.endswith(b"/>")
        if opens and data.startswith(b"</", m.end() - 1):
            continue
        out.append(data[pos : m.start() + 1] + m.group(1) + m.group(1)[1:] + b"  ")
        pos = m.end() - 1
    out.append(data[pos:])
    return b"".join(out)


def rewrite_clip_blobs(
    replace: Callable[[list[blobs.PbField], bytes], bytes],
) -> Edit:
    """Re-encode every ``<Clip>`` blob: decode, let ``replace`` rebuild the protobuf, recompress.

    The envelope version and codec are kept, so only the protobuf content changes.
    """
    import zstandard

    def run(data: bytes) -> bytes:
        def repl(m: re.Match[bytes]) -> bytes:
            raw = bytes.fromhex(m.group(1).decode())
            env = blobs.parse_envelope(raw)
            assert env is not None and env.version is not None
            payload = blobs.unwrap_message(raw)
            fields = blobs.decode_protobuf(payload, nested_depth=0)
            body = replace(fields, payload)
            if env.codec == "zstd":
                out = envelope(env.version, 0x81, zstandard.ZstdCompressor().compress(body))
            else:
                out = envelope(env.version, 0x80, body)
            return b"<Clip>" + out.hex().encode() + b"</Clip>"

        new, n = re.subn(rb"<Clip>([0-9a-f]+)</Clip>", repl, data)
        assert n >= 1
        return new

    return run


def set_fields(values: dict[int, bytes]) -> Callable[[list[blobs.PbField], bytes], bytes]:
    """Protobuf rebuilder that replaces the first occurrence of some length-delimited fields."""

    def build(fields: list[blobs.PbField], payload: bytes) -> bytes:
        out = b""
        done: set[int] = set()
        for f in fields:
            if f.number in values and f.number not in done:
                out += pb_len(f.number, values[f.number])
                done.add(f.number)
            else:
                out += payload[f.offset : f.end]
        for number, value in values.items():
            if number not in done:
                out += pb_len(number, value)
        return out

    return build


def marker_entry(frame: int, color: int, name: str, note: str = "", extra: bytes = b"") -> bytes:
    """One marker list entry; ``extra`` is appended to the detail message (undecoded fields)."""
    detail = (
        pb_var(1, color)
        + pb_len(3, note.encode())
        + pb_len(3, b"1")
        + pb_len(3, name.encode())
        + extra
    )
    inner = pb_len(1, detail)
    body = struct.pack(">II", 2, len(inner)) + inner
    return pb_len(1, pb_var(1, frame) + pb_len(2, body))


def markers_fields_blob(*entries: bytes) -> str:
    blob_data = envelope(10001, 0x80, pb_len(2, b"".join(entries)))
    return kd([("BlobData", 0x0C, struct.pack(">I", len(blob_data)) + blob_data)]).hex()


def carrier(dbid: str, owner: str, fields_blob: str, tag: str = "Sm2SequenceLockableBlob") -> bytes:
    """A lockable-blob element holding markers for ``owner``, wrapped like Resolve does."""
    return (
        f'<Element>\n     <{tag} DbId="{dbid}">\n      <FieldsBlob>{fields_blob}</FieldsBlob>\n'
        f"      <BlobOwner>{owner}</BlobOwner>\n     </{tag}>\n    </Element>\n    "
    ).encode()


def add_carrier(element: bytes) -> Edit:
    """Insert a lockable-blob element at the start of ``<LocableBlobSet>``."""
    return once(b"<LocableBlobSet>\n    ", b"<LocableBlobSet>\n    " + element)
