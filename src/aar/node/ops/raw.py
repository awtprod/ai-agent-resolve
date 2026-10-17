"""Plain readers over the Resolve scripting objects, returning dicts of builtins.

These functions use nothing but builtins (no imports, no annotations evaluated at run time),
so the same code can be replayed inside a restricted scripting sandbox to check it against a
live Resolve. The op layer wraps their output in typed models.

Every call here is a ``Get*`` read; nothing changes in Resolve.
"""

from __future__ import annotations

from typing import Any

STILL_EXTENSIONS = frozenset(
    {"png", "jpg", "jpeg", "tif", "tiff", "bmp", "gif", "psd", "heic", "webp", "tga"}
)
# Clip-property values as read from a live 21.1 media pool: 'Type' is one of 'Video + Audio',
# 'Video', 'Audio', 'Timeline', 'Multicam', 'Compound'; BRAW reads as Format 'BRAW' and Video
# Codec 'Blackmagic RAW'. A still's 'Type' was not observed ('Still' is assumed), so stills are
# also recognised by file extension.
STILL_TYPES = frozenset({"Still"})
MULTICAM_TYPES = frozenset({"Multicam"})
NO_VIDEO_TYPES = frozenset({"Audio"})


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_media(props: dict[str, Any]) -> dict[str, Any]:
    """Facts about a media-pool clip from its ``GetClipProperty()`` dict."""
    kind = str(props.get("Type") or "")
    fmt = str(props.get("Format") or "")
    codec = str(props.get("Video Codec") or "")
    file_name = str(props.get("File Name") or "")
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    is_braw = fmt == "BRAW" or codec.startswith("Blackmagic RAW") or ext == "braw"
    is_still = kind in STILL_TYPES or ext in STILL_EXTENSIONS
    audio_ch = _int_or_none(props.get("Audio Ch"))
    return {
        "type": kind,
        "format": fmt,
        "video_codec": codec,
        "file_name": file_name,
        "fps": _float_or_none(props.get("FPS")),
        "frames": _int_or_none(props.get("Frames")),
        "is_braw": is_braw,
        "is_still": is_still,
        "is_multicam": kind in MULTICAM_TYPES,
        "has_video": kind not in NO_VIDEO_TYPES and kind != "",
        "has_audio": "Audio" in kind or (audio_ch or 0) > 0,
        "online_status": str(props.get("Online Status") or ""),
        "proxy": str(props.get("Proxy") or ""),
        "input_color_space": str(props.get("Input Color Space") or ""),
    }


def read_media(mpi: Any) -> dict[str, Any]:
    """Identity and facts of one ``MediaPoolItem``."""
    props = mpi.GetClipProperty() or {}
    out = classify_media(props)
    out["media_id"] = str(mpi.GetUniqueId())
    out["name"] = str(mpi.GetName())
    out["file_path"] = str(props.get("File Path") or "")
    return out


def read_item(item: Any, timeline_start: int) -> dict[str, Any]:
    """Positions, source range, fades and gain of one ``TimelineItem``."""
    start = int(item.GetStart())
    end = int(item.GetEnd())
    fades = item.GetFades() or {}
    props = item.GetProperties() or {}
    mpi = item.GetMediaPoolItem()
    volume = props.get("AudioVolume")
    fade_in = fades.get("FadeIn")
    fade_out = fades.get("FadeOut")
    return {
        "item_id": str(item.GetUniqueId()),
        "name": str(item.GetName()),
        "type": str(item.GetType() or ""),
        "start": start,
        "end": end,
        "duration": end - start,
        "record_in": start - timeline_start,
        "record_out": end - timeline_start,
        "source_start": _int_or_none(item.GetSourceStartFrame()),
        "source_end": _int_or_none(item.GetSourceEndFrame()),
        "media_id": str(mpi.GetUniqueId()) if mpi is not None else None,
        "fade_in": round(fade_in) if fade_in is not None else None,
        "fade_out": round(fade_out) if fade_out is not None else None,
        "audio_volume_db": float(volume) if volume is not None else None,
        "enabled": bool(item.GetClipEnabled()),
    }


def find_gaps(spans: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int]:
    """Interior gaps between ``[start, end)`` spans, and the number of overlapping pairs.

    Gaps before the first span and after the last are not gaps: tracks need not start at the
    timeline start or run to its end.
    """
    gaps: list[tuple[int, int]] = []
    overlaps = 0
    reach: int | None = None
    for start, end in sorted(spans):
        if reach is not None:
            if start > reach:
                gaps.append((reach, start))
            elif start < reach:
                overlaps += 1
        reach = end if reach is None else max(reach, end)
    return gaps, overlaps


def read_markers(owner: Any) -> list[dict[str, Any]]:
    """``GetMarkers()`` of a timeline (or item) as a list sorted by frame."""
    markers = owner.GetMarkers() or {}
    out = []
    for frame in sorted(markers):
        info = markers[frame] or {}
        out.append(
            {
                "frame": int(frame),
                "color": str(info.get("color") or ""),
                "name": str(info.get("name") or ""),
                "note": str(info.get("note") or ""),
                "duration": _int_or_none(info.get("duration")) or 1,
                "custom_data": str(info.get("customData") or ""),
            }
        )
    return out


def read_timeline(
    timeline: Any,
    track_types: tuple[str, ...] = ("video", "audio", "subtitle"),
    max_items_per_track: int = 1000,
    include_media: bool = True,
) -> dict[str, Any]:
    """Everything ``timeline.describe`` reports, as plain data."""
    settings = timeline.GetSettings() or {}
    start = int(timeline.GetStartFrame())
    end = int(timeline.GetEndFrame())
    info = {
        "name": str(timeline.GetName()),
        "unique_id": str(timeline.GetUniqueId()),
        "fps": _float_or_none(settings.get("timelineFrameRate")),
        "drop_frame": str(settings.get("timelineDropFrameTimecode") or "0") == "1",
        "start_frame": start,
        "end_frame": end,
        "duration": end - start,
        "start_timecode": str(timeline.GetStartTimecode() or ""),
        "width": _int_or_none(settings.get("timelineResolutionWidth")),
        "height": _int_or_none(settings.get("timelineResolutionHeight")),
    }
    tracks = []
    media: dict[str, Any] = {}
    for track_type in track_types:
        count = int(timeline.GetTrackCount(track_type) or 0)
        for index in range(1, count + 1):
            items = list(timeline.GetItemListInTrack(track_type, index) or [])
            rows = []
            for item in items[:max_items_per_track]:
                row = read_item(item, start)
                rows.append(row)
                if include_media and row["media_id"] is not None and row["media_id"] not in media:
                    media[row["media_id"]] = read_media(item.GetMediaPoolItem())
            # Transitions are listed as items of their own that straddle the cut; they are
            # neither gaps nor overlaps.
            clips = [r for r in rows if r["type"] != "transition"]
            gaps, overlaps = find_gaps([(r["start"], r["end"]) for r in clips])
            tracks.append(
                {
                    "type": track_type,
                    "index": index,
                    "name": str(timeline.GetTrackName(track_type, index) or ""),
                    "enabled": bool(timeline.GetIsTrackEnabled(track_type, index)),
                    "locked": bool(timeline.GetIsTrackLocked(track_type, index)),
                    "subtype": (
                        str(timeline.GetTrackSubType(track_type, index) or "")
                        if track_type == "audio"
                        else ""
                    ),
                    "item_count": len(items),
                    "transitions": len(rows) - len(clips),
                    "truncated": len(items) > max_items_per_track,
                    "items": rows,
                    "gaps": [
                        {
                            "start": g0,
                            "end": g1,
                            "frames": g1 - g0,
                            "record_in": g0 - start,
                            "record_out": g1 - start,
                        }
                        for g0, g1 in gaps
                    ],
                    "overlaps": overlaps,
                }
            )
    return {
        "timeline": info,
        "tracks": tracks,
        "markers": read_markers(timeline),
        "media": [media[k] for k in sorted(media)],
    }
