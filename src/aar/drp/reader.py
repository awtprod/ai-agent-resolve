"""Build the :mod:`aar.drp.model` from a container.

Kept apart from ``model.py`` so the dataclasses stay import-light for the diff (S2) and
modify (S8) slices, which consume the model but not this builder.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from aar.drp import blobs
from aar.drp.container import (
    DEFAULT_MAX_MEMBER_BYTES,
    Container,
    open_container,
    parse_version_header,
)
from aar.drp.errors import BlobError, ContainerError, ZstdUnavailableError
from aar.drp.model import (
    AudioTrackInfo,
    Bin,
    BlobStats,
    DrpFile,
    Issue,
    Marker,
    MediaClip,
    MemberSummary,
    ProjectInfo,
    Provenance,
    Spans,
    Timeline,
    TimelineItem,
    Track,
    VersionInfo,
)
from aar.drp.scan import Document, Element, scan

__all__ = ["MARKER_COLORS", "read_drp"]

# Marker colour bits, from drp-format's timeline-markers-blob.js (measured there on 19.1.3.7).
MARKER_COLORS = {
    2: "Blue",
    4: "Cyan",
    8: "Green",
    16: "Yellow",
    32: "Red",
    64: "Pink",
    128: "Purple",
    512: "Fuchsia",
    1024: "Rose",
    2048: "Lavender",
    4096: "Sky",
    8192: "Mint",
    16384: "Lemon",
    32768: "Sand",
    65536: "Cocoa",
    131072: "Cream",
}

_MEDIA_TYPES = {
    "Sm2MpVideoClip": "video",
    "Sm2MpAudioClip": "audio",
    "Sm2MpTimelineClip": "timeline",
    "Sm2MpCompoundClip": "compound",
    "Sm2MpMulticamClip": "multicam",
    "Sm2MpStillClip": "still",
}
_ITEM_KINDS = {
    "Sm2TiVideoClip": "video",
    "Sm2TiAudioClip": "audio",
    "Sm2TiGenerator": "generator",
    "Sm2TiTransition": "transition",
}
_TRACK_TYPE_CODES = {0: "video", 1: "audio", 2: "subtitle"}
_SEQ_MEMBER = re.compile(r"(^|/)SeqContainer(\d*\.xml|/[^/]+\.xml)$")
_BIN_MEMBER = re.compile(r"^MediaPool/(.+/)?MpFolder\.xml$")
_MAX_BIN_DEPTH = 256


def read_drp(
    path: str | Path,
    *,
    check_blobs: bool = False,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_blob_output: int = blobs.DEFAULT_MAX_OUTPUT,
) -> DrpFile:
    """Read a ``.drp``/``.drt``/``.drb`` file into a :class:`DrpFile`.

    Raises :class:`~aar.drp.errors.ContainerError` only when the file itself cannot be read.
    Everything inside it that cannot be understood becomes an :class:`Issue` instead.
    ``check_blobs`` fully decodes every hex blob to fill :attr:`DrpFile.blob_stats` (slower).
    """
    with open_container(path, max_member_bytes=max_member_bytes) as container:
        return _Reader(container, check_blobs, max_blob_output).run()


def member_role(name: str) -> str:
    """What a member is, judged from its name."""
    lower = name.lower()
    if name == "project.xml":
        return "project"
    if name == "Gallery.xml":
        return "gallery"
    if _BIN_MEMBER.match(name):
        return "bin"
    if _SEQ_MEMBER.search(name):
        return "timeline"
    if name.startswith("stills/"):
        return "gallery-still" if lower.endswith(".xml") else "gallery-image"
    if lower.endswith(".dat"):
        return "data"
    if lower.endswith((".cube", ".3dl", ".dat", ".lut", ".olut")) or lower.startswith("lut"):
        return "lut"
    return "unknown"


_INTERPRETED_ROLES = frozenset({"project", "bin", "timeline"})


def _text_prov(member: str, el: Element) -> Provenance:
    if el.inner_start is None or el.inner_end is None:
        return Provenance(member, el.start, el.end, "element")
    return Provenance(member, el.inner_start, el.inner_end, "text")


def _el_prov(member: str, el: Element) -> Provenance:
    return Provenance(member, el.start, el.end, "element")


def _blob_prov(member: str, el: Element, blob_path: str) -> Provenance:
    base = _text_prov(member, el)
    return Provenance(member, base.start, base.end, "blob", blob_path)


def _iter_skipping(el: Element, skip: frozenset[str]) -> Iterator[Element]:
    """Descendants of ``el`` (not ``el`` itself), not descending into tags in ``skip``."""
    stack = list(reversed(el.children))
    while stack:
        node = stack.pop()
        yield node
        if node.tag not in skip:
            stack.extend(reversed(node.children))


def _join_path(directory: str | None, filename: str | None) -> str | None:
    """Join the Clip blob's directory and file name with the separator the directory uses."""
    if not filename:
        return directory or None
    if not directory:
        return filename
    if directory.endswith(("\\", "/")):
        return directory + filename
    sep = "\\" if "\\" in directory and "/" not in directory else "/"
    return directory + sep + filename


class _Reader:
    def __init__(self, container: Container, check_blobs: bool, max_blob_output: int) -> None:
        self.c = container
        self.check_blobs = check_blobs
        self.max_out = max_blob_output
        self.issues: list[Issue] = []
        self.docs: dict[str, Document] = {}
        self.media: dict[str, MediaClip] = {}
        self.bins: dict[str, Bin] = {}
        # Sm2Sequence DbId -> (owning media clip, Sm2Timeline element or None, sequence, member)
        self.sequences: dict[str, tuple[MediaClip, Element | None, Element, str]] = {}
        self.markers: dict[str, list[Marker]] = defaultdict(list)
        self.stale_paths = 0

    # ------------------------------------------------------------------ helpers

    def issue(
        self, severity: str, code: str, message: str, where: Provenance | None = None
    ) -> None:
        self.issues.append(Issue(severity, code, message, where))

    def text(self, member: str, el: Element, tag: str, spans: Spans, key: str) -> str | None:
        child = el.find(tag)
        if child is None:
            return None
        spans[key] = _text_prov(member, child)
        value = child.text
        return value if value else None

    def integer(self, member: str, el: Element, tag: str, spans: Spans, key: str) -> int | None:
        raw = self.text(member, el, tag, spans, key)
        if raw is None:
            return None
        try:
            return int(raw.strip())
        except ValueError:
            self.issue("warning", "bad-integer", f"<{tag}> is not an integer", spans[key])
            return None

    def frames(
        self,
        member: str,
        el: Element,
        tag: str,
        spans: Spans,
        key: str,
        fractions: dict[str, float],
    ) -> int | None:
        raw = self.text(member, el, tag, spans, key)
        if raw is None:
            return None
        try:
            value, fraction = blobs.parse_frames(raw)
        except BlobError as exc:
            self.issue("warning", "bad-frame-count", f"<{tag}>: {exc}", spans[key])
            return None
        if fraction is not None:
            fractions[key] = fraction
        return value

    def blob(self, member: str, el: Element | None) -> bytes | None:
        if el is None:
            return None
        text = el.text
        if not text:
            return None
        try:
            return blobs.parse_hex(text)
        except BlobError as exc:
            self.issue("warning", "bad-hex", f"<{el.tag}>: {exc}", _text_prov(member, el))
            return None

    def keyed(self, member: str, el: Element | None) -> blobs.KeyedDict | None:
        data = self.blob(member, el)
        if data is None or el is None:
            return None
        try:
            kd = blobs.decode_keyed_dict(data)
        except BlobError as exc:
            self.issue("warning", "blob-decode", f"<{el.tag}>: {exc}", _text_prov(member, el))
            return None
        if kd.error is not None:
            self.issue(
                "warning",
                "blob-partial",
                f"<{el.tag}> keyed dict decoded only partly: {kd.error}",
                _text_prov(member, el),
            )
        return kd

    def message(self, member: str, el: Element, data: bytes, what: str) -> bytes | None:
        try:
            return blobs.unwrap_message(data, max_output=self.max_out)
        except ZstdUnavailableError as exc:
            self.issue("warning", "zstd-unavailable", f"{what}: {exc}", _text_prov(member, el))
        except BlobError as exc:
            self.issue("warning", "blob-decode", f"{what}: {exc}", _text_prov(member, el))
        return None

    # ------------------------------------------------------------------ run

    def run(self) -> DrpFile:
        c = self.c
        for warning in c.warnings:
            self.issue("warning", "container", warning)
        members: list[MemberSummary] = []
        raw_members: list[str] = []
        for m in c.members:
            role = "directory" if m.is_dir else member_role(m.name)
            interpreted = role in _INTERPRETED_ROLES
            members.append(MemberSummary(m.name, m.size, m.compressed_size, role, interpreted))
            if not m.is_dir and not interpreted:
                raw_members.append(m.name)
        if not c.is_zip:
            self.issue(
                "warning",
                "not-zip",
                "not a zip archive; only the size, hash and leading bytes are reported "
                "(.drb support is best effort)",
            )
        elif c.kind == "drb":
            self.issue(
                "info",
                "drb-best-effort",
                "bin exports (.drb) are read best effort: members are listed and any "
                "MediaPool folders are parsed",
            )
        for m in members:
            if m.interpreted:
                self._load(m.name)
        versions = self._versions(members) if c.is_zip else None
        if c.is_zip and c.kind in ("drp", "drt") and "project.xml" not in self.docs:
            self.issue("error", "missing-member", "project.xml is missing")
        if c.kind in ("drp", "drt") and not any(m.role == "bin" for m in members):
            self.issue("warning", "missing-member", "no MediaPool/.../MpFolder.xml member")
        if c.kind in ("drp", "drt") and not any(m.role == "timeline" for m in members):
            self.issue("info", "no-timelines", "no SeqContainer members")

        project = self._project()
        self._lockable_blobs()
        for name in sorted(n for n, d in self.docs.items() if member_role(n) == "bin"):
            self._bin(name)
        self._bin_paths()
        timelines = self._timelines(project)
        self._finish_media(timelines)
        stats = self._blob_stats() if self.check_blobs else BlobStats(checked=False)
        return DrpFile(
            path=str(c.path),
            kind=c.kind,
            kind_source=c.kind_source,
            sniffed_kind=c.sniffed_kind,
            size=c.size,
            sha256=c.sha256,
            is_zip=c.is_zip,
            signature=c.signature,
            head_hex=c.head_hex,
            versions=versions,
            members=members,
            project=project,
            bins=sorted(self.bins.values(), key=lambda b: b.path),
            media=list(self.media.values()),
            timelines=timelines,
            issues=self.issues,
            blob_stats=stats,
            raw_members=raw_members,
            zstd_backend=blobs.zstd_backend(),
        )

    def _load(self, name: str) -> None:
        try:
            data = self.c.read(name)
        except ContainerError as exc:
            self.issue("error", "unreadable-member", str(exc))
            return
        doc = scan(data, name)
        self.docs[name] = doc
        for si in doc.issues:
            where = Provenance(name, si.offset, si.offset, "offset")
            self.issue("warning", f"scan-{si.code}", si.message, where)
        if doc.root is None:
            self.issue("error", "empty-member", f"{name} has no root element")

    def _versions(self, members: list[MemberSummary]) -> VersionInfo | None:
        found = {}
        for m in members:
            if m.role == "directory" or not m.name.lower().endswith(".xml"):
                continue
            doc = self.docs.get(m.name)
            try:
                head = doc.data[:1024] if doc is not None else self.c.head(m.name)
            except ContainerError as exc:
                self.issue("warning", "unreadable-member", str(exc))
                continue
            v = parse_version_header(head, m.name)
            if v is not None:
                found[m.name] = v
        if not found:
            self.issue("warning", "no-version-header", "no DbAppVer/DbPrjVer header found")
            return None
        main = found.get("project.xml") or next(iter(found.values()))
        mismatched = sorted(
            n
            for n, v in found.items()
            if (v.app_version, v.project_version) != (main.app_version, main.project_version)
        )
        if mismatched:
            self.issue(
                "warning",
                "version-mismatch",
                f"{len(mismatched)} member(s) carry a different version header than {main.member}",
            )
        return VersionInfo(
            app_version=main.app_version,
            project_version=main.project_version,
            source=Provenance(main.member, main.offset, main.end, "comment"),
            mismatched_members=mismatched,
        )

    # ------------------------------------------------------------------ project

    def _project(self) -> ProjectInfo | None:
        doc = self.docs.get("project.xml")
        if doc is None or doc.root is None:
            return None
        root, member = doc.root, doc.member
        if root.tag != "SM_Project":
            self.issue("warning", "unexpected-root", f"project.xml root is <{root.tag}>")
        spans: Spans = {}
        timeline_ids: list[str] = []
        handles = root.find("TimelineHandleVec")
        if handles is not None:
            for i, el in enumerate(handles.findall("Element")):
                if el.text:
                    timeline_ids.append(el.text.strip())
                    spans[f"timeline_ids[{i}]"] = _text_prov(member, el)
        last_mod = self.integer(member, root, "LastModTimeInSecs", spans, "last_modified_secs")
        info = ProjectInfo(
            id=root.attrs.get("DbId"),
            name=self.text(member, root, "ProjectName", spans, "name"),
            project_version=self.text(member, root, "ProjectVersion", spans, "project_version"),
            current_timeline_index=self.integer(
                member, root, "CurrentTimelineIndex", spans, "current_timeline_index"
            ),
            timeline_ids=timeline_ids,
            last_modified_secs=last_mod,
            frame_rate_hint=None,
            settings_decoded=False,
            source=_el_prov(member, root),
            spans=spans,
        )
        self._setup_hint(member, root, info)
        return info

    def _setup_hint(self, member: str, root: Element, info: ProjectInfo) -> None:
        """Project settings live in ``SM_Config``'s ``SetupBA``, an unnamed protobuf.

        Only one field is surfaced, as a hint: field 69 holds a decimal string that matched
        the timeline frame rate in every corpus file that had it. That is a correlation, not a
        decoded schema, hence the name.
        """
        fb = root.path("CommonConfig", "SM_Config", "FieldsBlob")
        kd = self.keyed(member, fb)
        if kd is None or fb is None:
            return
        setup = kd.get("SetupBA")
        if not isinstance(setup, bytes):
            return
        payload = self.message(member, fb, setup, "SM_Config SetupBA")
        if payload is None:
            return
        try:
            fields = blobs.decode_protobuf(payload, nested_depth=0)
        except BlobError as exc:
            self.issue("warning", "blob-decode", f"SetupBA protobuf: {exc}", _text_prov(member, fb))
            return
        for f in fields:
            if f.number == 69 and f.as_text and re.fullmatch(r"\d+(\.\d+)?", f.as_text):
                info.frame_rate_hint = f.as_text
                info.spans["frame_rate_hint"] = _blob_prov(
                    member, fb, "FieldsBlob>SetupBA>message>protobuf#69"
                )
                return

    # ------------------------------------------------------------------ markers

    def _lockable_blobs(self) -> None:
        doc = self.docs.get("project.xml")
        if doc is None or doc.root is None:
            return
        member = doc.member
        undecoded: Counter[str] = Counter()
        for el in doc.root.iter():
            if not el.tag.endswith("LockableBlob") or el.find("BlobOwner") is None:
                continue
            owner = (el.child_text("BlobOwner") or "").strip()
            fb = el.find("FieldsBlob")
            kd = self.keyed(member, fb)
            if kd is None or fb is None:
                continue
            data = kd.get("BlobData")
            if not isinstance(data, bytes):
                continue
            owner_kind = {
                "Sm2SequenceLockableBlob": "timeline",
                "Sm2TiItemLockableBlob": "item",
                "Sm2MpItemLockableBlob": "media",
            }.get(el.tag, el.tag)
            if owner_kind not in ("timeline", "item"):
                undecoded[el.tag] += 1  # payload layout not decoded yet; kept raw
                continue
            payload = self.message(member, fb, data, f"{el.tag} BlobData")
            if payload is None:
                continue
            prov = _blob_prov(member, fb, "FieldsBlob>BlobData>message>protobuf#2")
            try:
                self.markers[owner].extend(_decode_markers(payload, owner, owner_kind, prov))
            except BlobError as exc:
                self.issue("warning", "marker-decode", f"{el.tag}: {exc}", prov)
        for tag, count in sorted(undecoded.items()):
            self.issue("info", "uninterpreted", f"{count} <{tag}> payload(s) not interpreted")

    # ------------------------------------------------------------------ media pool

    def _bin(self, member: str) -> None:
        doc = self.docs[member]
        root = doc.root
        if root is None:
            return
        if root.tag != "Sm2MpFolder":
            self.issue("warning", "unexpected-root", f"{member} root is <{root.tag}>")
            return
        spans: Spans = {}
        bin_id = root.attrs.get("DbId") or member
        if bin_id in self.bins:
            self.issue(
                "warning", "duplicate-id", f"bin {bin_id} appears twice", _el_prov(member, root)
            )
            return
        parent = self.text(member, root, "MpFolder", spans, "parent_id")
        b = Bin(
            id=bin_id,
            name=self.text(member, root, "Name", spans, "name"),
            parent_id=parent.strip() if parent else None,
            path="",
            member=member,
            source=_el_prov(member, root),
            spans=spans,
        )
        self.bins[bin_id] = b
        vec = root.find("MediaVec")
        for el in vec.elements() if vec is not None else ():
            clip = self._media_clip(member, el, bin_id)
            if clip is None:
                continue
            if clip.id in self.media:
                self.issue("warning", "duplicate-id", f"media {clip.id} appears twice", clip.source)
                continue
            self.media[clip.id] = clip
            b.clip_ids.append(clip.id)

    def _bin_paths(self) -> None:
        for b in self.bins.values():
            if b.parent_id and b.parent_id in self.bins:
                self.bins[b.parent_id].child_ids.append(b.id)
            elif b.parent_id:
                self.issue("warning", "orphan-bin", f"bin {b.id} has an unknown parent", b.source)
        for b in self.bins.values():
            names: list[str] = []
            node: Bin | None = b
            seen: set[str] = set()
            while node is not None:
                if node.id in seen or len(seen) > _MAX_BIN_DEPTH:
                    self.issue("warning", "bin-cycle", f"bin {b.id} has a cyclic parent chain")
                    break
                seen.add(node.id)
                names.append(node.name or "?")
                node = self.bins.get(node.parent_id) if node.parent_id else None
            b.path = "/".join(reversed(names))

    def _media_clip(self, member: str, el: Element, bin_id: str) -> MediaClip | None:
        if not el.tag.startswith("Sm2Mp"):
            self.issue("info", "unknown-media", f"media pool entry <{el.tag}> not interpreted")
            return None
        spans: Spans = {}
        clip_id = el.attrs.get("DbId")
        if not clip_id:
            self.issue("warning", "missing-id", f"<{el.tag}> without DbId", _el_prov(member, el))
            return None
        folder = self.text(member, el, "MpFolder", spans, "bin_id")
        clip = MediaClip(
            id=clip_id,
            element=el.tag,
            type=_MEDIA_TYPES.get(el.tag, "other"),
            name=self.text(member, el, "Name", spans, "name"),
            bin_id=folder.strip() if folder else bin_id,
            pool_item_id=self.text(member, el, "UniqueMediaPoolItemId", spans, "pool_item_id"),
            path=None,
            path_source="none",
            blob_path=None,
            timeline_paths=[],
            codec=None,
            frames=None,
            start_frame=None,
            frame_rate=None,
            resolution=None,
            audio_tracks=[],
            sequence_id=None,
            timeline_id=None,
            source=_el_prov(member, el),
            spans=spans,
        )
        skip = frozenset({"Sequence", "TimelineSharedHandle"})
        for node in _iter_skipping(el, skip):
            if node.tag == "BtVideoInfo":
                self._video_info(member, node, clip)
            elif node.tag == "BtAudioInfo":
                self._audio_info(member, node, clip)
        seq = el.path("TimelineSharedHandle", "Sm2Timeline", "Sequence", "Sm2Sequence")
        timeline_el = el.path("TimelineSharedHandle", "Sm2Timeline")
        if seq is None:
            seq = el.path("Sequence", "Sm2Sequence")
            timeline_el = None
        if seq is not None and seq.attrs.get("DbId"):
            clip.sequence_id = seq.attrs["DbId"]
            clip.timeline_id = timeline_el.attrs.get("DbId") if timeline_el is not None else None
            if clip.sequence_id in self.sequences:
                self.issue("warning", "duplicate-id", f"sequence {clip.sequence_id} twice")
            else:
                self.sequences[clip.sequence_id] = (clip, timeline_el, seq, member)
        return clip

    def _clip_blob(self, member: str, info: Element, clip: MediaClip, what: str) -> None:
        clip_el = info.find("Clip")
        data = self.blob(member, clip_el)
        if data is None or clip_el is None:
            return
        payload = self.message(member, clip_el, data, f"{clip.element} {what} Clip blob")
        if payload is None:
            return
        try:
            fields = blobs.decode_protobuf(payload, nested_depth=0)
        except BlobError as exc:
            self.issue("warning", "blob-decode", f"Clip blob: {exc}", _text_prov(member, clip_el))
            return
        first = {}
        for f in fields:
            first.setdefault(f.number, f)
        directory = first[1].as_text if 1 in first else None
        filename = first[2].as_text if 2 in first else None
        path = _join_path(directory, filename)
        if path and clip.blob_path is None:
            clip.blob_path = path
            clip.spans["blob_path"] = _blob_prov(
                member, clip_el, f"{what}/Clip>message>protobuf#1+#2"
            )
        elif path and path != clip.blob_path and path not in clip.linked_paths:
            clip.linked_paths.append(path)
            clip.spans[f"linked_paths[{len(clip.linked_paths) - 1}]"] = _blob_prov(
                member, clip_el, f"{what}/Clip>message>protobuf#1+#2"
            )
        if clip.codec is None and 5 in first and first[5].as_text:
            clip.codec = first[5].as_text
            clip.spans["codec"] = _blob_prov(member, clip_el, f"{what}/Clip>message>protobuf#5")

    def _video_info(self, member: str, info: Element, clip: MediaClip) -> None:
        self._clip_blob(member, info, clip, "video")
        time_el = info.find("Time")
        kd = self.keyed(member, time_el)
        if kd is not None and time_el is not None:
            frames, start = kd.get("NumFrames"), kd.get("StartFrame")
            if isinstance(frames, int):
                clip.frames = frames
                clip.spans["frames"] = _blob_prov(member, time_el, "Time>keyed:NumFrames")
            if isinstance(start, int):
                clip.start_frame = start
                clip.spans["start_frame"] = _blob_prov(member, time_el, "Time>keyed:StartFrame")
            rate = kd.get("FrameRate")
            if isinstance(rate, bytes):
                try:
                    clip.frame_rate = blobs.decode_rate(rate)
                    clip.spans["frame_rate"] = _blob_prov(member, time_el, "Time>keyed:FrameRate")
                except BlobError as exc:
                    prov = _text_prov(member, time_el)
                    self.issue("warning", "blob-decode", f"Time FrameRate: {exc}", prov)
        geo_el = info.find("Geometry")
        kd = self.keyed(member, geo_el)
        if kd is not None and geo_el is not None:
            res = kd.get("Resolution")
            if isinstance(res, bytes):
                try:
                    clip.resolution = blobs.decode_resolution(res)
                    clip.spans["resolution"] = _blob_prov(
                        member, geo_el, "Geometry>keyed:Resolution"
                    )
                except BlobError as exc:
                    prov = _text_prov(member, geo_el)
                    self.issue("warning", "blob-decode", f"Geometry Resolution: {exc}", prov)

    def _audio_info(self, member: str, info: Element, clip: MediaClip) -> None:
        self._clip_blob(member, info, clip, "audio")
        tracks_el = info.find("TracksBA")
        kd = self.keyed(member, tracks_el)
        if kd is None:
            return
        for entry in kd.entries:
            if not isinstance(entry.value, bytes):
                continue
            try:
                inner = blobs.decode_keyed_dict(entry.value, strict=True)
            except BlobError as exc:
                self.issue("warning", "blob-decode", f"TracksBA[{entry.key}]: {exc}")
                continue
            if inner.get("DbType") != "BtAudioTrack":
                continue
            rate, channels = inner.get("SampleRate"), inner.get("NumChannels")
            codec, duration = inner.get("CodecName"), inner.get("Duration")
            clip.audio_tracks.append(
                AudioTrackInfo(
                    index=entry.key,
                    sample_rate=rate if isinstance(rate, int) else None,
                    channels=channels if isinstance(channels, int) else None,
                    codec=codec if isinstance(codec, str) else None,
                    duration_samples=duration if isinstance(duration, int) else None,
                )
            )

    # ------------------------------------------------------------------ timelines

    def _sequence_fields(self, member: str, seq: Element) -> blobs.KeyedDict | None:
        fb = seq.find("FieldsBlob")
        if fb is None or not fb.text:
            return None
        data = self.blob(member, fb)
        if data is None or data[:4] != b"\x00\x00\x00\x01":
            return None
        return self.keyed(member, fb)

    def _timelines(self, project: ProjectInfo | None) -> list[Timeline]:
        containers: dict[str, tuple[str, Element]] = {}
        for name, doc in sorted(self.docs.items()):
            if member_role(name) != "timeline" or doc.root is None:
                continue
            if doc.root.tag != "Sm2SequenceContainer":
                self.issue("warning", "unexpected-root", f"{name} root is <{doc.root.tag}>")
                continue
            cid = doc.root.attrs.get("DbId") or name
            containers[cid] = (name, doc.root)

        # Join media-pool sequences to containers: SeqRef in the sequence's FieldsBlob first,
        # then the <Sequence> reference every track carries.
        seq_for_container: dict[str, str] = {}
        for seq_id, (_clip, _tl, seq, member) in self.sequences.items():
            kd = self._sequence_fields(member, seq)
            ref = kd.get("SeqRef") if kd is not None else None
            if isinstance(ref, str) and ref in containers:
                seq_for_container[ref] = seq_id
        for cid, (_name, root) in containers.items():
            refs = {
                (t.child_text("Sequence") or "").strip()
                for vec in root.children
                if vec.tag.endswith("TrackVec")
                for t in vec.elements()
            } - {""}
            if cid in seq_for_container:
                if refs and refs != {seq_for_container[cid]}:
                    self.issue(
                        "warning",
                        "sequence-link-conflict",
                        f"container {cid}: tracks reference a different sequence than SeqRef",
                    )
                continue
            if len(refs) > 1:
                self.issue("warning", "sequence-link-conflict", f"container {cid}: mixed refs")
            for ref in sorted(refs):
                if ref in self.sequences:
                    seq_for_container[cid] = ref
                    break

        container_for_seq = {s: c for c, s in seq_for_container.items()}
        timelines: list[Timeline] = []
        for seq_id, (clip, tl_el, seq, member) in self.sequences.items():
            cid = container_for_seq.get(seq_id)
            timelines.append(self._timeline(clip, tl_el, seq, member, cid, containers.get(cid)))
        for cid, (name, root) in containers.items():
            if cid in seq_for_container:
                continue
            self.issue(
                "warning",
                "orphan-timeline",
                f"{name} is not referenced by any media-pool sequence",
                _el_prov(name, root),
            )
            timelines.append(self._timeline(None, None, None, None, cid, (name, root)))

        order = {tid: i for i, tid in enumerate(project.timeline_ids if project else [])}
        kind_rank = {"timeline": 0, "compound": 1, "multicam": 2}
        timelines.sort(
            key=lambda t: (
                kind_rank.get(t.kind, 3),
                order.get(t.id or "", len(order)),
                t.name or "",
            )
        )
        return timelines

    def _timeline(
        self,
        clip: MediaClip | None,
        tl_el: Element | None,
        seq: Element | None,
        member: str | None,
        cid: str | None,
        container: tuple[str, Element] | None,
    ) -> Timeline:
        spans: Spans = {}
        name = None
        if tl_el is not None and member is not None:
            name = self.text(member, tl_el, "Name", spans, "name")
        if name is None and clip is not None:
            name = clip.name
        kind = (
            clip.type if clip is not None and clip.type in ("compound", "multicam") else "timeline"
        )
        tl = Timeline(
            id=(tl_el.attrs.get("DbId") if tl_el is not None else None) or cid,
            kind=kind,
            name=name,
            media_clip_id=clip.id if clip else None,
            sequence_id=seq.attrs.get("DbId") if seq is not None else None,
            container_id=cid,
            member=container[0] if container else None,
            frame_rate=None,
            resolution=None,
            start_seconds=None,
            duration_seconds=None,
            start_frame=None,
            end_frame=None,
            source=_el_prov(member, seq) if seq is not None and member else None,
            spans=spans,
        )
        if seq is not None and member is not None:
            self._sequence_props(member, seq, tl)
        if container is not None:
            self._tracks(container[0], container[1], tl)
        elif kind == "timeline":
            self.issue("warning", "missing-container", f"timeline {tl.name!r} has no SeqContainer")
        if tl.sequence_id:
            tl.markers = self.markers.pop(tl.sequence_id, [])
        return tl

    def _sequence_props(self, member: str, seq: Element, tl: Timeline) -> None:
        for tag, key in (("FrameRate", "frame_rate"), ("Resolution", "resolution")):
            el = seq.find(tag)
            data = self.blob(member, el)
            if data is None or el is None:
                continue
            try:
                if tag == "FrameRate":
                    tl.frame_rate = blobs.decode_rate(data)
                else:
                    tl.resolution = blobs.decode_resolution(data)
                tl.spans[key] = _blob_prov(member, el, tag)
            except BlobError as exc:
                self.issue("warning", "blob-decode", f"{tag}: {exc}", _text_prov(member, el))
        el = seq.find("MediaExtents")
        data = self.blob(member, el)
        if data is not None and el is not None:
            try:
                tl.start_seconds, tl.duration_seconds = blobs.decode_extents(data)
                tl.spans["extents"] = _blob_prov(member, el, "MediaExtents")
            except BlobError as exc:
                self.issue("warning", "blob-decode", f"MediaExtents: {exc}", _text_prov(member, el))
        if tl.frame_rate and tl.start_seconds is not None:
            tl.start_frame = round(tl.start_seconds * tl.frame_rate)
            if tl.duration_seconds is not None:
                tl.end_frame = round((tl.start_seconds + tl.duration_seconds) * tl.frame_rate)

    def _tracks(self, member: str, root: Element, tl: Timeline) -> None:
        for vec in root.children:
            if not vec.tag.endswith("TrackVec"):
                continue
            vec_type = vec.tag[: -len("TrackVec")].lower() or "other"
            index = 0
            for el in vec.elements():
                if el.tag != "Sm2TiTrack":
                    self.issue("info", "unknown-track", f"<{el.tag}> in {vec.tag} not interpreted")
                    continue
                index += 1
                tl.tracks.append(self._track(member, el, vec_type, index))

    def _track(self, member: str, el: Element, vec_type: str, index: int) -> Track:
        spans: Spans = {}
        type_code = self.integer(member, el, "Type", spans, "type_code")
        if type_code is not None and _TRACK_TYPE_CODES.get(type_code, vec_type) != vec_type:
            self.issue(
                "warning",
                "track-type-mismatch",
                f"track {el.attrs.get('DbId')} has Type {type_code} inside a {vec_type} vector",
                spans.get("type_code"),
            )
        track = Track(
            id=el.attrs.get("DbId", ""),
            type=vec_type,
            index=index,
            type_code=type_code,
            subtype=self.integer(member, el, "SubType", spans, "subtype"),
            flags=self.integer(member, el, "Flags", spans, "flags"),
            name=self.text(member, el, "UserDefinedName", spans, "name"),
            source=_el_prov(member, el),
            spans=spans,
        )
        items = el.find("Items")
        for item_el in items.elements() if items is not None else ():
            track.items.append(self._item(member, item_el))
        holder = el.find("FusionCompHolderItems")
        if holder is not None and holder.children:
            self.issue(
                "info",
                "uninterpreted",
                "FusionCompHolderItems has content that is not interpreted",
                _el_prov(member, holder),
            )
        return track

    def _item(self, member: str, el: Element) -> TimelineItem:
        spans: Spans = {}
        fractions: dict[str, float] = {}
        start = self.frames(member, el, "Start", spans, "record_start", fractions)
        duration = self.frames(member, el, "Duration", spans, "duration", fractions)
        pretty = self.text(member, el, "PrettyType", spans, "pretty_type")
        kind = _ITEM_KINDS.get(el.tag, el.tag)
        if el.tag == "Sm2TiVideoClip" and pretty == "Fusion Title":
            kind = "title"
        source_in: int | None = None
        in_el = el.find("In")
        if in_el is not None:
            source_in = self.frames(member, el, "In", spans, "source_in", fractions)
            if source_in is None and not (in_el.text or "").strip():
                source_in = 0
        media_ref = self.text(member, el, "MediaRef", spans, "media_ref")
        media_track = self.integer(member, el, "MediaTrackIdx", spans, "media_track_index")
        speed = None
        form = None
        tm_el = el.find("MediaTimemapBA")
        tm = self.blob(member, tm_el)
        if tm is not None and tm_el is not None:
            try:
                decoded = blobs.decode_timemap(tm)
                speed, form = decoded.speed, decoded.form
                spans["speed"] = _blob_prov(member, tm_el, "MediaTimemapBA")
            except BlobError as exc:
                self.issue(
                    "warning", "blob-decode", f"MediaTimemapBA: {exc}", _text_prov(member, tm_el)
                )
        source_out = None
        if source_in is not None and duration is not None and form != "retimed":
            source_out = source_in + duration
        item = TimelineItem(
            id=el.attrs.get("DbId", ""),
            element=el.tag,
            kind=kind,
            name=self.text(member, el, "Name", spans, "name"),
            pretty_type=pretty,
            record_start=start,
            record_end=start + duration if start is not None and duration is not None else None,
            duration=duration,
            source_in=source_in,
            source_out=source_out,
            media_ref=media_ref.strip() if media_ref else None,
            media_clip_id=None,
            media_path=None,
            path_source="none",
            media_file_path=self.text(member, el, "MediaFilePath", spans, "media_file_path"),
            media_track_index=media_track,
            speed=speed,
            timemap_form=form,
            fractions=fractions,
            source=_el_prov(member, el),
            spans=spans,
        )
        item.markers = self.markers.pop(item.id, []) if item.id else []
        return item

    # ------------------------------------------------------------------ cross references

    def _finish_media(self, timelines: list[Timeline]) -> None:
        unresolved = Counter()
        for tl in timelines:
            for track in tl.tracks:
                for item in track.items:
                    if item.media_ref is None:
                        item.media_path = item.media_file_path
                        item.path_source = "timeline_item" if item.media_file_path else "none"
                        continue
                    clip = self.media.get(item.media_ref)
                    if clip is None:
                        unresolved[tl.name or tl.id or "?"] += 1
                        item.media_path = item.media_file_path
                        item.path_source = "timeline_item" if item.media_file_path else "none"
                        continue
                    item.media_clip_id = clip.id
                    if item.media_file_path and item.media_file_path not in clip.timeline_paths:
                        clip.timeline_paths.append(item.media_file_path)
        for clip in self.media.values():
            if clip.blob_path:
                clip.path, clip.path_source = clip.blob_path, "media_pool_blob"
                if any(p != clip.blob_path for p in clip.timeline_paths):
                    self.stale_paths += 1
            elif clip.timeline_paths:
                clip.path, clip.path_source = clip.timeline_paths[0], "timeline_item"
        for tl in timelines:
            for track in tl.tracks:
                for item in track.items:
                    clip = self.media.get(item.media_clip_id or "")
                    if clip is not None and clip.path:
                        item.media_path, item.path_source = clip.path, clip.path_source
                    elif clip is not None:
                        item.media_path = item.media_file_path
                        item.path_source = "timeline_item" if item.media_file_path else "none"
        for name, count in sorted(unresolved.items()):
            self.issue(
                "warning",
                "unresolved-media",
                f"timeline {name!r}: {count} item(s) reference media not in this file",
            )
        linked = sum(1 for clip in self.media.values() if clip.linked_paths)
        if linked:
            self.issue(
                "info",
                "linked-files",
                f"{linked} media clip(s) name more than one file (e.g. synced external audio)",
            )
        if self.stale_paths:
            self.issue(
                "info",
                "stale-media-file-path",
                f"{self.stale_paths} media clip(s): timeline <MediaFilePath> differs from the "
                "Media Pool path (the Media Pool path is used)",
            )
        leftover = sum(len(v) for v in self.markers.values())
        if leftover:
            self.issue(
                "info",
                "orphan-markers",
                f"{leftover} marker(s) belong to owners not found in this file",
            )

    # ------------------------------------------------------------------ blob stats

    def _blob_stats(self) -> BlobStats:
        stats = BlobStats(checked=True)
        by_tag: dict[str, Counter[str]] = defaultdict(Counter)
        failures: Counter[str] = Counter()
        undecoded: Counter[str] = Counter()
        for name in self.c.xml_members():
            doc = self.docs.get(name)
            if doc is None:
                try:
                    doc = scan(self.c.read(name), name)
                except ContainerError as exc:
                    self.issue("warning", "unreadable-member", str(exc))
                    continue
            if doc.root is None:
                continue
            for el in doc.root.iter():
                if el.children or not blobs.is_blob_tag(el.tag):
                    continue
                text = el.text
                if not text:
                    continue
                stats.total += 1
                try:
                    data = blobs.parse_hex(text)
                except BlobError as exc:
                    by_tag[el.tag][blobs.BlobKind.INVALID_HEX] += 1
                    failures[f"{el.tag}: {_generic(str(exc))}"] += 1
                    continue
                result = blobs.check(el.tag, data, max_output=self.max_out)
                by_tag[el.tag][result.kind] += 1
                if result.undecoded:
                    undecoded[el.tag] += 1
                elif not result.ok:
                    failures[f"{el.tag} ({result.kind}): {_generic(result.error or '')}"] += 1
        stats.by_tag = {t: dict(sorted(c.items())) for t, c in sorted(by_tag.items())}
        stats.failures = dict(failures.most_common())
        stats.undecoded = dict(sorted(undecoded.items()))
        return stats


def _generic(message: str) -> str:
    """Collapse numbers so failures with the same cause aggregate."""
    return re.sub(r"\b\d+\b", "N", message)


def _decode_markers(payload: bytes, owner: str, owner_kind: str, prov: Provenance) -> list[Marker]:
    """Decode a lockable-blob marker list.

    ``payload`` = protobuf ``field 2 { repeated field 1 entry }``; each entry is
    ``{1: frame varint, 2: [u32 2][u32 len] message {1: {1: colour bit, 3: note,
    3: duration, 3: name, 6: custom data}}}``. Layout from drp-format's
    timeline-markers-blob.js, checked against the corpus (frames are plain varints: both odd
    and even values occur, so they are not zigzag-encoded).
    """
    markers: list[Marker] = []
    top = blobs.decode_protobuf(payload, nested_depth=0)
    for holder in top:
        if holder.number != 2 or not isinstance(holder.value, bytes):
            raise BlobError(f"marker list: unexpected top-level field {holder.number}")
        for entry in blobs.decode_protobuf(holder.value, nested_depth=0):
            if entry.number != 1 or not isinstance(entry.value, bytes):
                raise BlobError(f"marker list: unexpected entry field {entry.number}")
            parts = blobs.decode_protobuf(entry.value, nested_depth=0)
            frame = next((p.value for p in parts if p.number == 1 and p.wire == 0), None)
            body = next((p.value for p in parts if p.number == 2 and p.wire == 2), None)
            if not isinstance(frame, int) or not isinstance(body, bytes):
                raise BlobError("marker entry without frame or body")
            env = blobs.parse_envelope(body)
            if env is None or env.version is None:
                raise BlobError("marker body without its [u32][u32] header")
            inner = blobs.decode_protobuf(env.payload, nested_depth=0)
            detail = next((p.value for p in inner if p.number == 1 and p.wire == 2), b"")
            if not isinstance(detail, bytes):
                raise BlobError("marker body without detail message")
            fields = blobs.decode_protobuf(detail, nested_depth=0)
            color_bit = next((f.value for f in fields if f.number == 1 and f.wire == 0), None)
            strings = [f.value for f in fields if f.number == 3 and isinstance(f.value, bytes)]
            custom = next((f.value for f in fields if f.number == 6 and f.wire == 2), b"")
            texts = [s.decode("utf-8", "replace") for s in strings] + ["", "", ""]
            duration = int(texts[1]) if texts[1].strip().isdigit() else None
            markers.append(
                Marker(
                    owner_id=owner,
                    owner_kind=owner_kind,
                    frame=frame,
                    frame_space="timeline-relative" if owner_kind == "timeline" else "item",
                    color=MARKER_COLORS.get(color_bit) if isinstance(color_bit, int) else None,
                    name=texts[2],
                    note=texts[0],
                    duration=duration,
                    custom_data=custom.decode("utf-8", "replace")
                    if isinstance(custom, bytes)
                    else "",
                    source=prov,
                )
            )
    return markers
