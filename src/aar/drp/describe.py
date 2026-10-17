"""Plain-text description of a :class:`~aar.drp.model.DrpFile` for an LLM agent to read."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from aar.drp.model import DrpFile, Marker, MediaClip, Timeline, TimelineItem

__all__ = ["describe", "frames_to_timecode", "printable"]

DEFAULT_MAX_ITEMS = 200
_MAX_ISSUES = 25


def printable(text: str) -> str:
    """Escape characters a terminal would act on rather than show.

    Names and paths come from the file, so they can hold NUL, ESC (terminal control
    sequences), bidi overrides such as U+202E, line separators or lone surrogates. Every
    character for which ``str.isprintable()`` is false, other than an ordinary space, becomes
    a ``\\xNN``, ``\\uNNNN`` or ``\\UNNNNNNNN`` escape.
    """
    if text.isprintable():
        return text
    out = []
    for ch in text:
        if ch.isprintable() or ch == " ":
            out.append(ch)
        else:
            code = ord(ch)
            out.append(
                f"\\x{code:02x}"
                if code < 0x100
                else f"\\u{code:04x}"
                if code < 0x10000
                else f"\\U{code:08x}"
            )
    return "".join(out)


def frames_to_timecode(frame: int, fps: float | None) -> str:
    """Non-drop-frame ``HH:MM:SS:FF`` at the nominal (rounded) rate; ``f<frame>`` if no rate."""
    if not fps or fps <= 0:
        return f"f{frame}"
    nominal = max(1, round(fps))
    sign = "-" if frame < 0 else ""
    frame = abs(frame)
    ff = frame % nominal
    total = frame // nominal
    return f"{sign}{total // 3600:02d}:{total // 60 % 60:02d}:{total % 60:02d}:{ff:02d}"


def _fmt_rate(fps: float | None) -> str:
    if fps is None:
        return "unknown fps"
    return f"{fps:.3f}".rstrip("0").rstrip(".") + " fps"


def _fmt_res(res: tuple[int, int] | None) -> str:
    return f"{res[0]}x{res[1]}" if res else "unknown resolution"


def describe(drp: DrpFile, *, max_items: int = DEFAULT_MAX_ITEMS) -> str:
    """Render ``drp`` as compact text.

    ``max_items`` caps the media clips listed and the items listed per track (0 = all).
    """
    lines: list[str] = []
    out = lines.append
    v = drp.versions
    stamp = f"Resolve {v.app_version or '?'} (DbPrjVer {v.project_version})" if v else "no version"
    out(
        f"{drp.kind.upper()} file ({drp.kind_source}): {stamp}; {drp.size} bytes; "
        f"{sum(1 for m in drp.members if m.role != 'directory')} members"
    )
    if not drp.is_zip:
        out(f"not a zip archive; leading bytes {drp.head_hex} ({drp.signature or 'unknown'})")
    if drp.project is not None:
        p = drp.project
        hint = f"; project frame rate {p.frame_rate_hint} (inferred)" if p.frame_rate_hint else ""
        out(f'project "{p.name or "?"}"{hint}; project settings blob not decoded')

    media = drp.media_by_id()
    file_clips = [m for m in drp.media if m.type in ("video", "audio", "still", "other")]
    type_counts = Counter(m.type for m in drp.media)
    counts = ", ".join(f"{n} {t}" for t, n in sorted(type_counts.items())) or "none"
    out(f"media pool: {len(drp.bins)} bin(s), {len(drp.media)} item(s) ({counts})")
    for b in drp.bins:
        out(f"  bin {b.path}: {len(b.clip_ids)} item(s)")
    shown = file_clips if max_items <= 0 else file_clips[:max_items]
    for clip in shown:
        out("  " + _media_line(clip))
    if len(shown) < len(file_clips):
        out(f"  ... {len(file_clips) - len(shown)} more media clip(s) (raise --max-items)")

    main = [t for t in drp.timelines if t.kind == "timeline"]
    nested = [t for t in drp.timelines if t.kind != "timeline"]
    out(
        f"timelines: {len(main)}"
        + (f"; compound/multicam sequences: {len(nested)}" if nested else "")
    )
    for tl in main + nested:
        lines.extend(_timeline_lines(tl, media, max_items))

    lines.extend(_media_problems(drp))
    problems = [i for i in drp.issues if i.severity in ("warning", "error")]
    if problems:
        out(f"warnings: {len(problems)}")
        for i in problems[:_MAX_ISSUES]:
            out(f"  [{i.severity}] {i.code}: {i.message}")
        if len(problems) > _MAX_ISSUES:
            out(f"  ... and {len(problems) - _MAX_ISSUES} more (see inspect --json)")
    if drp.raw_members:
        raw = set(drp.raw_members)  # built once: members can number in the tens of thousands
        roles = Counter(m.role for m in drp.members if m.name in raw)
        summary = ", ".join(f"{n} {r}" for r, n in sorted(roles.items()))
        out(f"not interpreted (kept raw): {len(drp.raw_members)} member(s): {summary}")
    return "\n".join(printable(line) for line in lines) + "\n"


def _media_line(clip: MediaClip) -> str:
    bits = [f"[{clip.type}] {clip.name or '?'}"]
    if clip.frames is not None:
        bits.append(f"{clip.frames} frames @ {_fmt_rate(clip.frame_rate)}")
    if clip.resolution:
        bits.append(_fmt_res(clip.resolution))
    if clip.audio_tracks:
        a = clip.audio_tracks[0]
        bits.append(f"audio {len(clip.audio_tracks)} trk {a.sample_rate or '?'} Hz {a.channels}ch")
    if clip.codec:
        bits.append(f"codec {clip.codec}")
    elif clip.type in ("video", "audio"):
        bits.append("codec unknown")
    bits.append(f"path {clip.path or '(none)'} [{clip.path_source}]")
    return "; ".join(bits)


def _timeline_lines(tl: Timeline, media: dict[str, MediaClip], max_items: int) -> list[str]:
    fps = tl.frame_rate

    def tc(frame: int) -> str:
        return frames_to_timecode(frame, fps)

    head = f'{tl.kind.capitalize()} "{tl.name or "?"}": {_fmt_rate(fps)}, {_fmt_res(tl.resolution)}'
    if tl.start_frame is not None:
        head += f", start {tc(tl.start_frame)} (frame {tl.start_frame})"
    if tl.start_frame is not None and tl.end_frame is not None:
        length = tl.end_frame - tl.start_frame
        head += f", length {tc(length)} ({length} frames)"
    if fps and abs(fps - round(fps)) > 1e-6:
        head += "; timecodes shown non-drop-frame"
    lines = [head]
    if not tl.tracks:
        lines.append("  no tracks" + ("" if tl.member else " (no SeqContainer member)"))
    for track in tl.tracks:
        name = f' "{track.name}"' if track.name else ""
        lines.append(f"  {track.label}{name}: {len(track.items)} item(s)")
        shown = track.items if max_items <= 0 else track.items[:max_items]
        for item in shown:
            lines.append("    " + _item_line(item, media, tc))
            for marker in item.markers:
                lines.append("      " + _marker_line(marker, None))
        if len(shown) < len(track.items):
            lines.append(
                f"    ... {len(track.items) - len(shown)} more item(s) (raise --max-items)"
            )
    if tl.markers:
        lines.append(f"  timeline markers: {len(tl.markers)}")
        base = tl.start_frame or 0
        for marker in sorted(tl.markers, key=lambda m: m.frame):
            lines.append("    " + _marker_line(marker, (base, fps)))
    return lines


def _item_line(item: TimelineItem, media: dict[str, MediaClip], tc: Callable[[int], str]) -> str:
    rec = (
        f"{tc(item.record_start)}-{tc(item.record_end)}"
        if item.record_start is not None and item.record_end is not None
        else "?"
    )
    label = item.name or item.pretty_type or item.kind
    parts = [rec, f"{item.kind}", f'"{label}"']
    if item.source_in is not None and item.source_out is not None and item.kind != "transition":
        parts.append(f"src {tc(item.source_in)}-{tc(item.source_out)}")
    if item.speed is not None and abs(item.speed - 1.0) > 1e-9:
        parts.append(f"speed {item.speed:.4g}x")
    if item.media_ref:
        clip = media.get(item.media_ref)
        parts.append(f"media {clip.name!r}" if clip else f"UNRESOLVED media {item.media_ref}")
    if item.kind == "audio" and item.media_track_index is not None:
        parts.append(f"ch {item.media_track_index}")
    return " ".join(parts)


def _marker_line(marker: Marker, timeline: tuple[int, float | None] | None) -> str:
    where = f"frame {marker.frame}"
    if timeline is not None:
        base, fps = timeline
        where += f" ({frames_to_timecode(base + marker.frame, fps)})"
    else:
        where = f"marker at {marker.frame} ({marker.frame_space})"
    text = " ".join(x for x in (marker.color or "", repr(marker.name), marker.note) if x)
    dur = f" dur {marker.duration}" if marker.duration and marker.duration != 1 else ""
    return f"{where} {text}{dur}"


def _media_problems(drp: DrpFile) -> list[str]:
    lines: list[str] = []
    unresolved = sum(
        1
        for tl in drp.timelines
        for tr in tl.tracks
        for it in tr.items
        if it.media_ref and it.media_clip_id is None
    )
    no_path = [m for m in drp.media if m.type in ("video", "audio") and not m.path]
    unverified = [m for m in drp.media if m.path_source == "timeline_item"]
    stale = [
        m for m in drp.media if m.blob_path and any(p != m.blob_path for p in m.timeline_paths)
    ]
    if unresolved or no_path or unverified or stale:
        lines.append("media problems (file existence is not checked):")
        if unresolved:
            lines.append(f"  {unresolved} timeline item(s) reference media missing from this file")
        if no_path:
            lines.append(f"  {len(no_path)} media clip(s) have no path at all")
        if unverified:
            lines.append(
                f"  {len(unverified)} media clip(s) only have the timeline <MediaFilePath>, "
                "which can be stale"
            )
        if stale:
            lines.append(
                f"  {len(stale)} media clip(s) have a timeline <MediaFilePath> that differs from "
                "the Media Pool path (Media Pool path used)"
            )
    return lines
