"""Typed, read-only model of a DRP/DRT/DRB file.

Every value that came from the file carries a :class:`Provenance`: the zip member and the byte
span of the source text it was read from. For a value read from a leaf element such as
``<Name>clip.mov</Name>`` the span is the text between the tags (``kind="text"``); for a
self-closing leaf such as ``<In/>`` it is the whole element (``kind="element"``); for a value
decoded out of a hex blob it is the blob's hex text plus ``blob_path``, a short description of
where inside the decoded blob the value sits. Later slices splice edits through these spans.

The model never drops what it does not understand: each model object keeps the span of its
whole element (``source``), members that are not interpreted are listed in
``DrpFile.raw_members``, and anything that failed to decode is reported in ``DrpFile.issues``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "AudioTrackInfo",
    "Bin",
    "BlobStats",
    "DrpFile",
    "Issue",
    "Marker",
    "MediaClip",
    "MemberSummary",
    "ProjectInfo",
    "Provenance",
    "Timeline",
    "TimelineItem",
    "Track",
    "VersionInfo",
]


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a value came from: member name, byte span, and what the span covers."""

    member: str
    start: int
    end: int
    kind: str = "text"
    blob_path: str | None = None


Spans = dict[str, Provenance]


@dataclass(slots=True)
class Issue:
    """A warning or error found while reading. ``code`` is stable; ``message`` is for people."""

    severity: str
    code: str
    message: str
    where: Provenance | None = None


@dataclass(slots=True)
class VersionInfo:
    """The ``DbAppVer``/``DbPrjVer`` stamp and where it was read."""

    app_version: str | None
    project_version: int | None
    source: Provenance | None
    mismatched_members: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MemberSummary:
    """One zip member and the role the reader gave it."""

    name: str
    size: int
    compressed_size: int
    role: str
    interpreted: bool


@dataclass(slots=True)
class ProjectInfo:
    """Project-level facts from ``project.xml``."""

    id: str | None
    name: str | None
    project_version: str | None
    current_timeline_index: int | None
    timeline_ids: list[str]
    last_modified_secs: int | None
    frame_rate_hint: str | None
    settings_decoded: bool
    source: Provenance | None
    spans: Spans = field(default_factory=dict)


@dataclass(slots=True)
class Bin:
    """A Media Pool folder."""

    id: str
    name: str | None
    parent_id: str | None
    path: str
    member: str
    clip_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    source: Provenance | None = None
    spans: Spans = field(default_factory=dict)


@dataclass(slots=True)
class AudioTrackInfo:
    """One embedded audio track of a media clip (from ``TracksBA``)."""

    index: str
    sample_rate: int | None
    channels: int | None
    codec: str | None
    duration_samples: int | None


@dataclass(slots=True)
class MediaClip:
    """A Media Pool item: file clip, timeline, compound or multicam clip.

    ``path`` is the file Resolve links to. ``path_source`` says where it came from:
    ``media_pool_blob`` (the ``Clip`` blob of the video info, else the first audio info),
    ``timeline_item`` (only a timeline ``<MediaFilePath>`` was available; it can be stale) or
    ``none``. ``linked_paths`` lists other files named by the clip's other ``Clip`` blobs, for
    example the external recorder file of a clip with synced audio.
    """

    id: str
    element: str
    type: str
    name: str | None
    bin_id: str | None
    pool_item_id: str | None
    path: str | None
    path_source: str
    blob_path: str | None
    timeline_paths: list[str]
    codec: str | None
    frames: int | None
    start_frame: int | None
    frame_rate: float | None
    resolution: tuple[int, int] | None
    audio_tracks: list[AudioTrackInfo]
    sequence_id: str | None
    timeline_id: str | None
    linked_paths: list[str] = field(default_factory=list)
    source: Provenance | None = None
    spans: Spans = field(default_factory=dict)


@dataclass(slots=True)
class Marker:
    """A marker. ``frame_space`` says what ``frame`` counts from.

    ``timeline-relative``: frame 0 is the first frame of the timeline. ``item``: a clip marker;
    whether it counts from the source media or the item is not verified yet.
    """

    owner_id: str
    owner_kind: str
    frame: int
    frame_space: str
    color: str | None
    name: str
    note: str
    duration: int | None
    custom_data: str
    source: Provenance | None = None


@dataclass(slots=True)
class TimelineItem:
    """One item on a track: clip, generator, title or transition.

    ``record_start``/``record_end`` are timeline frames with an exclusive end.
    ``source_in``/``source_out`` come from ``<In>`` (frames, exclusive end); ``None`` when the
    item has no source (transitions) or is retimed. ``fractions`` holds the sub-frame
    remainders Resolve writes as ``<n>|<hex double>`` (key = field name). ``enabled`` is
    ``None``: the flag has not been located.
    """

    id: str
    element: str
    kind: str
    name: str | None
    pretty_type: str | None
    record_start: int | None
    record_end: int | None
    duration: int | None
    source_in: int | None
    source_out: int | None
    media_ref: str | None
    media_clip_id: str | None
    media_path: str | None
    path_source: str
    media_file_path: str | None
    media_track_index: int | None
    speed: float | None
    timemap_form: str | None
    enabled: bool | None = None
    fractions: dict[str, float] = field(default_factory=dict)
    markers: list[Marker] = field(default_factory=list)
    source: Provenance | None = None
    spans: Spans = field(default_factory=dict)


@dataclass(slots=True)
class Track:
    """One track. ``index`` is 1-based within its type (V1, A1, S1)."""

    id: str
    type: str
    index: int
    type_code: int | None
    subtype: int | None
    flags: int | None
    name: str | None
    items: list[TimelineItem] = field(default_factory=list)
    source: Provenance | None = None
    spans: Spans = field(default_factory=dict)

    @property
    def label(self) -> str:
        prefix = {"video": "V", "audio": "A", "subtitle": "S"}.get(self.type, "T")
        return f"{prefix}{self.index}"


@dataclass(slots=True)
class Timeline:
    """A timeline, or the sequence inside a compound/multicam clip.

    ``kind`` is ``timeline``, ``compound`` or ``multicam``. ``start_frame`` is
    ``round(start_seconds * frame_rate)``, which matches item ``Start`` values in the corpus.
    """

    id: str | None
    kind: str
    name: str | None
    media_clip_id: str | None
    sequence_id: str | None
    container_id: str | None
    member: str | None
    frame_rate: float | None
    resolution: tuple[int, int] | None
    start_seconds: float | None
    duration_seconds: float | None
    start_frame: int | None
    end_frame: int | None
    tracks: list[Track] = field(default_factory=list)
    markers: list[Marker] = field(default_factory=list)
    source: Provenance | None = None
    spans: Spans = field(default_factory=dict)


@dataclass(slots=True)
class BlobStats:
    """Blob statistics, filled only when blob checking ran.

    ``by_tag`` counts layouts per element tag. ``failures`` counts blobs whose layout was
    recognised but did not decode, grouped by cause. ``undecoded`` counts blobs whose layout is
    not interpreted (carried raw) per tag.
    """

    checked: bool
    total: int = 0
    by_tag: dict[str, dict[str, int]] = field(default_factory=dict)
    failures: dict[str, int] = field(default_factory=dict)
    undecoded: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class DrpFile:
    """Everything the reader learned about one file."""

    path: str
    kind: str
    kind_source: str
    sniffed_kind: str
    size: int
    sha256: str
    is_zip: bool
    signature: str | None
    head_hex: str
    versions: VersionInfo | None
    members: list[MemberSummary]
    project: ProjectInfo | None
    bins: list[Bin]
    media: list[MediaClip]
    timelines: list[Timeline]
    issues: list[Issue]
    blob_stats: BlobStats
    raw_members: list[str]
    zstd_backend: str | None

    def media_by_id(self) -> dict[str, MediaClip]:
        return {m.id: m for m in self.media}

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict (tuples become lists; nothing else changes)."""
        return _jsonable(dataclasses.asdict(self))


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return value.hex()
    return value
