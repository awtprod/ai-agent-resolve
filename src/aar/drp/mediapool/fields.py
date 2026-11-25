"""Compare two media clips' rows field by field, blobs decoded as far as they decode.

:func:`flatten` turns a clip's rows into ``{path: leaf}``. Paths name the row by its role
(``M`` the clip, ``V`` its video info, ``A0`` its first audio info, ``JA0`` and ``JF0`` its join
rows, ``L0`` its metadata blob, ``P0`` a preset), then the column, then the way into the blob
(``>`` enters an envelope, ``#n`` a protobuf field, ``.key`` a keyed-dict entry; a repeated
field or key gets ``[i]`` from its second occurrence on, so duplicates stay distinct entries; a
key that is not a plain word is written quoted). Ids are replaced by the role of the row they
name (``<A0>``), so a reference to the wrong row differs; any other uuid text is ``<uuid>``.

The comparison is total: two blobs that differ in any byte give different leaves, except where
this docstring says otherwise.

* A keyed dict's leaf holds its header; each entry has a ``~kd`` leaf with its type tag and its
  null flag. A bool is compared by its byte (the decoder reads any non-zero byte as True), a
  NaN by its bits. A list or map nested in an entry is compared by its raw bytes: the decoder
  does not keep the types and null flags of its items, and none occurs outside the relink
  history.
* Every keyed dict and protobuf message has a ``~order`` leaf listing its children in order.
* A protobuf message is decoded only if it re-encodes to its own bytes (canonical varints and
  lengths); otherwise it is compared as bytes. A field's leaf names its wire type.
* An envelope's leaf holds its version and its codec (``zstd``, ``protobuf``, ``none``). A zstd
  payload is compared *decoded*, not as its frame: Resolve compresses where the encoder writes
  raw blocks, so the frames differ while the messages are the same. The codec is still compared.
  A zstd frame that does not decompress, or is not exactly one frame, is compared as bytes.
* Anything that does not decode is compared byte for byte.
* Metadata entries are compared by key and their list's order leaf is sorted: their order is
  not a function of their keys (in the corpus, clips with the same set of keys store them in
  different orders), and what decides it is not known. No file is in both saved projects, so
  whether it differs between saves of one file is not shown. A key that occurs twice keeps
  both entries.

:data:`EXCLUDED` is the reviewed list of fields that are not compared, each with its reason;
nothing else is left out. An order leaf ignores the children that are excluded.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.mediapool.blobtree import MAX_BLOB_OUTPUT
from aar.drp.mediapool.templates import ClipRows
from aar.drp.modify import encode

__all__ = ["EXCLUDED", "Difference", "Exclusion", "compare", "excluded", "flatten"]

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_MAX_DEPTH = 12
Leaf = tuple[Any, ...]


@dataclass(frozen=True)
class Exclusion:
    """A field the gate does not compare: ``category`` is ``identity`` or ``project-local``."""

    pattern: str
    category: str
    reason: str
    media: frozenset[str] = frozenset({"video", "still", "audio"})

    def applies(self, path: str, media: str) -> bool:
        return media in self.media and re.fullmatch(self.pattern, path) is not None


#: Every field the gate leaves out, and why. Identities are not here: they are compared as the
#: role they name (``<A0>``) or as a fresh uuid (``<uuid>``).
EXCLUDED: tuple[Exclusion, ...] = (
    Exclusion(
        r"M\.(MpFolder|Sm2MpFolder_id)|JF0\.(DbOwner|DbIndex)",
        "project-local",
        "the bin the clip is in and its position there",
    ),
    Exclusion(
        r"M\.(CurPlayheadPosition|PTZRPreset|PinsBA|VideoMetadata|MediaMetadata)",
        "project-local",
        "UI state and per-clip notes set once the clip is viewed or edited "
        "(a fresh import has none)",
    ),
    Exclusion(
        r"P\d+\..*",
        "project-local",
        "the pan/tilt/zoom preset Resolve adds once the clip is viewed",
    ),
    Exclusion(
        r"(V|A\d+)\.MediaMetadata",
        "project-local",
        "NULL or empty depending on what was done to the clip",
    ),
    Exclusion(
        r"M\.FieldsBlob>#1(\.#1)?\.#1\.#1\.#2", "project-local", "the date the clip was added"
    ),
    Exclusion(
        r"M\.FieldsBlob>#1\.#3",
        "project-local",
        "a project age in ms written by a later edit of the clip (a fresh import has none)",
    ),
    Exclusion(
        r"M\.FieldsBlob>#2(\..*|\[.*)?",
        "project-local",
        "the infos the clip had before it was relinked (a fresh import has none)",
        frozenset({"video", "still"}),
    ),
    Exclusion(
        r"M\.FieldsBlob>#2\.#1",
        "project-local",
        "the project's frame rate when the sound clip was imported",
        frozenset({"audio"}),
    ),
    Exclusion(
        r"V\.Time\.FrameRate",
        "project-local",
        "the project's frame rate when the still was imported",
        frozenset({"still"}),
    ),
    Exclusion(
        r"(V|A\d+)\.Clip>#(14|26)",
        "project-local",
        "set after import (#14 a frame number, #26 the folder without its drive; "
        "a fresh import has neither)",
    ),
    Exclusion(
        r"L\d+\.(DbSavedTime|BtLockableBlobMap_id)",
        "project-local",
        "when the metadata was saved, and the project's lockable map",
    ),
)


def excluded(path: str, media: str) -> Exclusion | None:
    """The exclusion that covers ``path`` (its field, without a ``~`` marker), if any."""
    field = base_path(path)
    return next((e for e in EXCLUDED if e.applies(field, media)), None)


def _roles(clip: ClipRows) -> dict[str, str]:
    roles = {clip.media_id: "<M>"}
    pool = clip.media.get("UniqueMediaPoolItemId")
    if isinstance(pool, str):
        roles[pool] = "<pool-item>"
    if clip.video is not None:
        roles[str(clip.video["BtVideoInfo_id"])] = "<V>"
    for i, a in enumerate(clip.audio):
        roles[str(a["BtAudioInfo_id"])] = f"<A{i}>"
    for i, r in enumerate(clip.lockables):
        roles[str(r["BtLockableBlob_id"])] = f"<L{i}>"
    for i, r in enumerate(clip.presets):
        roles[str(r["SmPreset_id"])] = f"<P{i}>"
    return roles


_PLAIN_KEY = re.compile(r"[A-Za-z0-9_]+\Z")
#: Keyed-dict types whose value the decoder returns without its items' types and null flags.
_NESTED = frozenset({blobs.KD_MAP, blobs.KD_LIST, blobs.KD_STRING_LIST})
ORDER = "~order"
KD = "~kd"


def base_path(path: str) -> str:
    """The field a leaf belongs to: the path without its ``~`` marker (``~kd``, ``~order``)."""
    return path.split("~", 1)[0]


def _key_name(key: str) -> str:
    return key if _PLAIN_KEY.match(key) else json.dumps(key)


def _kd_value_bytes(data: bytes, entry: blobs.KdEntry) -> bytes:
    """The raw bytes of an entry's value (after its key, type and null flag)."""
    (klen,) = struct.unpack_from(">I", data, entry.offset)
    return data[entry.offset + 4 + klen + 5 : entry.end]


def _protobuf(data: bytes) -> list[blobs.PbField] | None:
    """The fields of ``data`` if it is a canonical protobuf message, else None."""
    try:
        fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=10_000)
        if encode.protobuf([(f.number, f.wire, f.value) for f in fields]) != data:
            return None
    except DrpError:  # EncodeError is a DrpError
        return None
    return fields


def _keyed(data: bytes) -> blobs.KeyedDict | None:
    if len(data) < 8 or data[:4] != b"\x00\x00\x00\x01":
        return None
    try:
        return blobs.decode_keyed_dict(data, strict=True)
    except DrpError:
        return None


def _unwrapped(data: bytes, env: blobs.Envelope) -> bytes | None:
    """The message bytes of an envelope; None if a zstd payload is not exactly one frame that
    decompresses within the bound."""
    try:
        if env.codec == "zstd":
            encode.check_single_frame(env.payload)
        return blobs.unwrap_message(data, max_output=MAX_BLOB_OUTPUT)
    except DrpError:
        return None


def _float_leaf(tag: str, value: float, raw: bytes) -> Leaf:
    # float.hex is exact for every finite value and both infinities; a NaN keeps its bits.
    return (tag, value.hex() if not math.isnan(value) else "nan:" + raw.hex())


class _Flat:
    def __init__(self, roles: dict[str, str]) -> None:
        self.roles = roles
        self.out: dict[str, Leaf] = {}

    def text(self, value: str) -> Leaf:
        if value in self.roles:
            return ("ref", self.roles[value])
        if _UUID.match(value):
            return ("ref", "<uuid>")
        return ("text", value)

    def value(self, v: object, path: str, depth: int = 0) -> None:
        if v is None:
            self.out[path] = ("null",)
        elif isinstance(v, bool):
            self.out[path] = ("bool", v)
        elif isinstance(v, int):
            self.out[path] = ("int", v)
        elif isinstance(v, float):
            self.out[path] = _float_leaf("float", v, struct.pack(">d", v))
        elif isinstance(v, str):
            self.out[path] = self.text(v)
        elif isinstance(v, bytes):
            self.blob(v, path, depth)
        else:
            self.out[path] = ("other", type(v).__name__, repr(v))

    def blob(self, b: bytes, path: str, depth: int) -> None:
        """A blob: a keyed dict, an envelope holding a message, or bytes."""
        if not b:
            self.out[path] = ("bytes", "")
            return
        if depth <= _MAX_DEPTH:
            kd = _keyed(b)
            if kd is not None:
                self.keyed(kd, b, path, depth)
                return
            env = blobs.parse_envelope(b)
            if env is not None:
                payload = _unwrapped(b, env)
                if payload is not None and self.inner(payload, path + ">", depth + 1):
                    self.out[path] = ("envelope", env.version, env.codec)
                    return
        self.out[path] = ("fixed16" if len(b) == 16 else "bytes", b.hex())

    def inner(self, data: bytes, path: str, depth: int) -> bool:
        """``data`` as a keyed dict or a canonical protobuf message at ``path``; False (and
        nothing written) if it is neither."""
        kd = _keyed(data)
        if kd is not None:
            self.keyed(kd, data, path, depth)
            return True
        fields = _protobuf(data)
        if fields is None:
            return False
        self.message(fields, path, depth)
        return True

    def keyed(self, kd: blobs.KeyedDict, data: bytes, path: str, depth: int) -> None:
        self.out[path] = ("keyed", kd.header)
        seen: dict[str, int] = {}
        children: list[str] = []
        for e in kd.entries:
            name = _key_name(e.key)
            n = seen.get(name, 0)
            seen[name] = n + 1
            p = f"{path}.{name}" + (f"[{n}]" if n else "")
            children.append(p)
            self.out[p + KD] = ("kd", e.type, e.subtype)
            if e.type in _NESTED:
                self.out[p] = ("nested", _kd_value_bytes(data, e).hex())
            elif e.type == blobs.KD_BOOL:
                self.out[p] = ("bool", _kd_value_bytes(data, e).hex())  # any non-zero is True
            elif e.type in (blobs.KD_DOUBLE, blobs.KD_FLOAT):
                self.out[p] = _float_leaf("float", e.value, _kd_value_bytes(data, e))
            else:
                self.value(e.value, p, depth + 1)
        self.out[path + ORDER] = tuple(children)

    def message(self, fields: list[blobs.PbField], path: str, depth: int) -> None:
        """A decoded message whose fields go under ``path`` (``path#n``)."""
        self.out[path] = ("message",)
        seen: dict[int, int] = {}
        children: list[str] = []
        for f in fields:
            n = seen.get(f.number, 0)
            seen[f.number] = n + 1
            p = f"{path}#{f.number}" + (f"[{n}]" if n else "")
            children.append(p)
            if f.wire == 0:
                self.out[p] = ("varint", f.value)
            elif f.wire in (1, 5):
                assert isinstance(f.value, bytes)
                fmt, tag = ("<d", "f64") if f.wire == 1 else ("<f", "f32")
                self.out[p] = _float_leaf(tag, struct.unpack(fmt, f.value)[0], f.value)
            else:
                assert isinstance(f.value, bytes)
                self.length_delimited(f.value, p, depth)
        self.out[path + ORDER] = tuple(children)

    def length_delimited(self, v: bytes, path: str, depth: int) -> None:
        try:
            text = v.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None and text and all(32 <= ord(c) < 127 or ord(c) > 159 for c in text):
            self.out[path] = self.text(text)
            return
        if v and depth < _MAX_DEPTH:
            kd = _keyed(v)
            if kd is not None:
                self.keyed(kd, v, path, depth + 1)
                return
            fields = _protobuf(v)
            if fields is not None:
                self.out[path] = ("nested message",)
                self.message(fields, path + ".", depth + 1)
                return
        self.blob(v, path, depth + 1)


_META = re.compile(r"(L\d+\.FieldsBlob\.BlobData>#2>)#2(\[\d+\])?\Z")


def _metadata_by_key(out: dict[str, Leaf]) -> dict[str, Leaf]:
    """Metadata entries keyed by their key, not their position (their order is not a function
    of their keys: clips with the same keys store them in different orders). ``meta{K}`` is the
    entry's value; ``meta{K}~e...`` the rest of the entry (its message, its key field, its
    order); a key that occurs twice gets ``[1]``."""
    renamed: dict[str, str] = {}
    containers: set[str] = set()
    seen: dict[tuple[str, object], int] = {}
    for path, leaf in out.items():
        m = _META.match(path)
        if m is None or leaf != ("nested message",):
            continue
        key_leaf = out.get(path + ".#1")
        if key_leaf is None or key_leaf[0] != "varint":
            continue  # not a key and a value: compared by position, as it is
        prefix = m.group(1)
        containers.add(prefix)
        count = seen.get((prefix, key_leaf[1]), 0)
        seen[(prefix, key_leaf[1])] = count + 1
        name = f"{prefix}meta{{{key_leaf[1]}}}" + (f"[{count}]" if count else "")
        renamed[path] = name
    if not renamed:
        return out

    def rename(path: str) -> str:
        for prefix_path, name in renamed.items():
            if path == prefix_path or path.startswith((prefix_path + ".", prefix_path + "~")):
                rest = path[len(prefix_path) :]
                return name if rest == ".#2" else f"{name}~e{rest}"
        return path

    result: dict[str, Leaf] = {}
    for path, leaf in out.items():
        if path.endswith(ORDER):
            leaf = tuple(rename(str(c)) for c in leaf)
        result[rename(path)] = leaf
    for prefix in containers:
        order = result.get(prefix + ORDER)
        if isinstance(order, tuple):
            result[prefix + ORDER] = tuple(sorted(str(c) for c in order))
    return result


def flatten(clip: ClipRows) -> dict[str, Leaf]:
    """Every field of a clip's rows as ``{path: leaf}`` (see the module docstring)."""
    flat = _Flat(_roles(clip))
    named: list[tuple[str, dict[str, Any] | None]] = [("M", clip.media), ("V", clip.video)]
    named += [(f"A{i}", r) for i, r in enumerate(clip.audio)]
    named += [(f"JA{i}", r) for i, r in enumerate(clip.audio_links)]
    named += [("JF0", clip.folder_link)]
    named += [(f"L{i}", r) for i, r in enumerate(clip.lockables)]
    named += [(f"P{i}", r) for i, r in enumerate(clip.presets)]
    for role, row in named:
        if row is None:
            continue
        for column, value in row.items():
            flat.value(value, f"{role}.{column}")
    return _metadata_by_key(flat.out)


@dataclass(frozen=True)
class Difference:
    path: str
    expected: Leaf | None
    got: Leaf | None


def media_class(clip: ClipRows, still_extensions: frozenset[str]) -> str:
    if clip.video is None:
        return "audio"
    name = str(clip.media.get("Name") or "")
    return "still" if name.rpartition(".")[2].lower() in still_extensions else "video"


def compare(
    expected: ClipRows, got: ClipRows, media: str
) -> tuple[int, list[Difference], list[tuple[Exclusion, Difference]]]:
    """``(fields compared, differences, differences in excluded fields)`` between Resolve's rows
    and authored rows of one file; :data:`EXCLUDED` fields are not compared, but where they
    differ they are returned with the exclusion that hid them (so each exclusion is shown to be
    needed). An order leaf is compared without the children that are excluded."""
    a, b = flatten(expected), flatten(got)
    compared = 0
    found: list[Difference] = []
    hidden: list[tuple[Exclusion, Difference]] = []
    for path in sorted(set(a) | set(b)):
        left, right = a.get(path), b.get(path)
        if path.endswith(ORDER):
            left, right = _kept(left, media), _kept(right, media)
        rule = excluded(path, media)
        if rule is not None:
            if left != right:
                hidden.append((rule, Difference(path, left, right)))
            continue
        compared += 1
        if left != right:
            found.append(Difference(path, left, right))
    return compared, found, hidden


def _kept(order: Leaf | None, media: str) -> Leaf | None:
    if order is None:
        return None
    return tuple(c for c in order if excluded(str(c), media) is None)
