"""Compare two media clips' rows field by field, blobs decoded as far as they decode.

:func:`flatten` turns a clip's rows into ``{path: leaf}``. Paths name the row by its role
(``M`` the clip, ``V`` its video info, ``A0`` its first audio info, ``JA0`` and ``JF0`` its join
rows, ``L0`` its metadata blob, ``P0`` a preset), then the column, then the way into the blob
(``>`` enters an envelope, ``#n`` a protobuf field, ``.key`` a keyed-dict entry, ``[i]`` a list
item or a repeated field). Ids are replaced by the role of the row they name (``<A0>``), so a
reference to the wrong row differs; any other uuid text is ``<uuid>``. An envelope's codec (zstd
or plain) is not compared, only what it holds. Metadata entries are compared by key, as their
order differs between saves of the same file.

:data:`EXCLUDED` is the reviewed list of fields that are not compared, each with its reason;
nothing else is left out.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any

from aar.drp import blobs
from aar.drp.errors import DrpError
from aar.drp.mediapool.blobtree import MAX_BLOB_OUTPUT
from aar.drp.mediapool.templates import ClipRows

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
    return next((e for e in EXCLUDED if e.applies(path, media)), None)


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
            self.out[path] = ("float", v.hex() if math.isfinite(v) else repr(v))
        elif isinstance(v, str):
            self.out[path] = self.text(v)
        elif isinstance(v, list):
            self.out[path] = ("list", len(v))
            for i, item in enumerate(v):
                self.value(item, f"{path}[{i}]", depth + 1)
        elif isinstance(v, dict):
            self.out[path] = ("map", len(v))
            for key, item in v.items():
                self.value(item, f"{path}.{key}", depth + 1)
        elif isinstance(v, bytes):
            self.blob(v, path, depth)
        else:
            self.out[path] = ("other", repr(v)[:80])

    def blob(self, b: bytes, path: str, depth: int) -> None:
        if not b:
            self.out[path] = ("bytes", "")
            return
        if depth > _MAX_DEPTH:
            self.out[path] = ("bytes", b.hex())
            return
        if len(b) >= 8 and b[:4] == b"\x00\x00\x00\x01":
            try:
                kd = blobs.decode_keyed_dict(b, strict=True)
            except DrpError:
                kd = None
            if kd is not None:
                self.out[path] = ("keyed", len(kd.entries))
                for e in kd.entries:
                    self.value(e.value, f"{path}.{e.key}", depth + 1)
                return
        env = blobs.parse_envelope(b)
        if env is not None and env.version is not None:
            try:
                msg = blobs.unwrap_message(b, max_output=MAX_BLOB_OUTPUT)
            except DrpError as exc:
                self.out[path] = ("undecodable", str(exc)[:80])
                return
            if self.message(msg, path + ">", depth + 1):
                self.out[path] = ("envelope", env.version)
                return
        elif (
            env is not None
            and env.codec == "protobuf"
            and self.message(env.payload, path + ">", depth + 1)
        ):
            self.out[path] = ("flagged",)
            return
        if len(b) == 16:
            self.out[path] = ("fixed16", b.hex())
            return
        self.out[path] = ("bytes", b.hex())

    def message(self, data: bytes, path: str, depth: int) -> bool:
        if data[:4] == b"\x00\x00\x00\x01" and len(data) >= 8:
            before = len(self.out)
            self.blob(data, path.rstrip("."), depth)
            return len(self.out) > before
        try:
            fields = blobs.decode_protobuf(data, nested_depth=0, max_fields=10_000)
        except DrpError:
            return False
        seen: dict[int, int] = {}
        for f in fields:
            n = seen.get(f.number, 0)
            seen[f.number] = n + 1
            p = f"{path}#{f.number}" + (f"[{n}]" if n else "")
            if f.wire == 0:
                self.out[p] = ("varint", f.value)
            elif f.wire in (1, 5):
                assert isinstance(f.value, bytes)
                fmt = "<d" if f.wire == 1 else "<f"
                self.out[p] = (
                    "f64" if f.wire == 1 else "f32",
                    struct.unpack(fmt, f.value)[0].hex(),
                )
            else:
                assert isinstance(f.value, bytes)
                self.length_delimited(f.value, p, depth)
        return True

    def length_delimited(self, v: bytes, path: str, depth: int) -> None:
        try:
            text = v.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        if text is not None and text and all(32 <= ord(c) < 127 or ord(c) > 159 for c in text):
            self.out[path] = self.text(text)
            return
        if v and depth < _MAX_DEPTH:
            before = dict(self.out)
            if self.message(v, path + ".", depth + 1) and len(self.out) > len(before):
                return
            self.out = before
        self.blob(v, path, depth + 1)


_META = re.compile(r"(L\d+\.FieldsBlob\.BlobData>#2>)#2(\[\d+\])?\.#1\Z")


def _metadata_by_key(out: dict[str, Leaf]) -> dict[str, Leaf]:
    """Metadata entries keyed by their key, not their position."""
    keyed: dict[str, Leaf] = {}
    drop: set[str] = set()
    for path, leaf in out.items():
        m = _META.match(path)
        if m is None or leaf[0] != "varint":
            continue
        base = path[: -len(".#1")]
        drop.update(p for p in out if p == base or p.startswith(base + "."))
        keyed[f"{m.group(1)}meta{{{leaf[1]}}}"] = out.get(base + ".#2", ("absent",))
    result = {p: v for p, v in out.items() if p not in drop}
    result.update(keyed)
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
    needed)."""
    a, b = flatten(expected), flatten(got)
    compared = 0
    found: list[Difference] = []
    hidden: list[tuple[Exclusion, Difference]] = []
    for path in sorted(set(a) | set(b)):
        rule = excluded(path, media)
        if rule is not None:
            if a.get(path) != b.get(path):
                hidden.append((rule, Difference(path, a.get(path), b.get(path))))
            continue
        compared += 1
        if a.get(path) != b.get(path):
            found.append(Difference(path, a.get(path), b.get(path)))
    return compared, found, hidden
