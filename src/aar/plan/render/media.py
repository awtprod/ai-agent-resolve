"""Hub media for the render: every used source must be a readable file on the hub (S18b).

A source resolves to a hub file in one of two ways, and anything else is refused as
``media_not_on_hub`` (with a ``reason``):

* ``plan-path``: the plan's media is a hub path (absolute, or a ``file:`` URL of one; a relative
  name resolves against ``media_root``, the plan file's folder in the CLI) and the file exists;
* ``media-map``: the plan's media is an edit-PC path (``E:\\...``, ``\\\\host\\share\\...`` or a
  ``file:`` URL of one) and the media map (the same ``linux_root -> windows_root`` JSON the FCPXML
  export takes, read the other way) maps it to a hub file that exists.

A media-pool name (a multicam clip) has no file and is refused. Each file is then probed with
:func:`aar.media.probe`; a file ffprobe cannot read is ``media_unreadable``, one with no video
stream ``no_video_stream``, a frame rate other than the plan's (beyond ``[render]
fps_tolerance``) ``fps_not_renderable``, an unknown picture size ``size_not_renderable``, and a
``Source.frame_size`` whose aspect is not the file's ``frame_size_mismatch``. Nothing here writes.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal
from urllib.parse import unquote, urlsplit

from aar.media import MediaError, MediaInfo, probe
from aar.plan.apply.inputs import classify_media
from aar.plan.export.errors import ExportError
from aar.plan.export.media import MediaMap, media_ref
from aar.plan.model import EditPlan, Fps, Source
from aar.plan.render.errors import RenderError

HubRule = Literal["plan-path", "media-map"]

#: How far a plan frame size's aspect may sit from the file's display aspect (a face track's
#: sampled size rounds, e.g. 853x480 for 16:9).
ASPECT_TOLERANCE = 0.005
_DRIVE_URL_PATH = re.compile(r"^/[A-Za-z]:/")


@dataclass(frozen=True)
class HubSource:
    """One source's hub file and what ffprobe says about it."""

    source_id: str
    plan_media: str
    path: Path
    rule: HubRule
    info: MediaInfo
    #: The picture as decoded and shown (pixel aspect applied, display rotation applied).
    display: tuple[int, int]
    #: The pixel aspect when it is not square: the render scales to ``display`` first.
    sample_aspect: Fraction | None
    fps: Fraction
    #: Whole frames the video holds (its duration at its rate); None when ffprobe gives no length.
    frames: int | None
    #: The first audio stream's channel count; None when the file has no sound.
    channels: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source_id,
            "plan_media": self.plan_media,
            "hub_path": str(self.path),
            "rule": self.rule,
            "fps": f"{self.fps.numerator}/{self.fps.denominator}",
            "display_size": list(self.display),
            "frames": self.frames,
            "audio_channels": self.channels,
        }


def _refuse(source: Source, why: str, hint: str, reason: str, **extra: Any) -> RenderError:
    return RenderError(
        "media_not_on_hub",
        f"source {source.id!r} media {source.media!r} {why}",
        hint,
        source=source.id,
        media=source.media,
        reason=reason,
        **extra,
    )


_MAP_HINT = (
    "give the hub copy's path in the plan (aar media ingest prints it), or pass --media-map with "
    "a JSON object of linux_root -> windows_root that covers this path"
)


def _edit_path(source: Source) -> str:
    """The edit-PC path of a Windows path or a ``file:`` URL of one (S18's own rules)."""
    try:
        return media_ref(source).windows_path
    except ExportError as exc:
        raise _refuse(
            source,
            f"is not a hub path or an edit-PC path the media map can read ({exc.message})",
            _MAP_HINT,
            str(exc.details.get("reason", exc.code)),
        ) from None


def _mapped(source: Source, edit: str, media_map: MediaMap | None) -> Path:
    if media_map is None:
        raise _refuse(
            source, "is an edit-PC path and no media map was given", _MAP_HINT, "no-media-map"
        )
    result = media_map.pathmap.edit_to_hub(edit)
    if result.new is None:
        raise _refuse(
            source,
            f"is an edit-PC path the media map does not cover ({result.reason})",
            _MAP_HINT,
            "not-in-media-map",
        )
    return Path(result.new)


def hub_path(
    source: Source, media_map: MediaMap | None, media_root: Path | None
) -> tuple[Path, HubRule]:
    """Where the source's media is on the hub, by the rules in the module docstring (the file's
    existence is checked by :func:`resolve_source`)."""
    media = source.media
    if media[:5].lower() == "file:":
        try:
            parts = urlsplit(media)
        except ValueError:
            raise _refuse(source, "is a malformed file: URL", _MAP_HINT, "malformed-url") from None
        path = unquote(parts.path)
        local = parts.netloc in ("", "localhost") and not parts.query and not parts.fragment
        hub_url = path.startswith("/") and not path.startswith("//")
        if local and hub_url and not _DRIVE_URL_PATH.match(path):
            return Path(path), "plan-path"
        return _mapped(source, _edit_path(source), media_map), "media-map"
    if media.startswith("/") and not media.startswith("//"):
        return Path(media), "plan-path"
    kind = classify_media(media)
    if kind == "import":
        return _mapped(source, _edit_path(source), media_map), "media-map"
    if kind == "pool":
        raise _refuse(
            source,
            "is a media-pool name (e.g. a multicam clip), not a file",
            "give the underlying camera file's hub path, or its edit-PC path with --media-map",
            "pool-name",
        )
    if media_root is None:
        raise _refuse(
            source,
            "is a relative name and there is no media root to resolve it against",
            "give an absolute hub path, or render from the CLI (relative to the plan's folder)",
            "relative",
        )
    return media_root / media, "plan-path"


def _probe(source: Source, path: Path) -> MediaInfo:
    try:
        return probe(path)
    except MediaError as exc:
        raise RenderError(
            "media_unreadable",
            f"source {source.id!r}: ffprobe cannot read {str(path)!r}: {exc.summary()}",
            "check the hub copy (aar media availability), or re-ingest it",
            source=source.id,
            path=str(path),
        ) from None


def resolve_source(
    source: Source,
    plan_fps: Fps,
    *,
    media_map: MediaMap | None,
    media_root: Path | None,
    fps_tolerance: float,
) -> HubSource:
    """The source's hub file, probed and checked (module docstring)."""
    path, rule = hub_path(source, media_map, media_root)
    path = Path(os.path.abspath(path))
    if not path.is_file() or not os.access(path, os.R_OK):
        what = "is missing" if not path.exists() else "is not a readable regular file"
        raise _refuse(
            source,
            f"resolves to {str(path)!r} on the hub, which {what}",
            "ingest the media on the hub (aar media ingest) and give that path, or fix the map",
            "missing" if not path.exists() else "unreadable",
            hub_path=str(path),
        )
    info = _probe(source, path)
    video = info.video
    if video is None:
        raise RenderError(
            "no_video_stream",
            f"source {source.id!r}: {str(path)!r} has no video stream",
            "give a file with picture",
            source=source.id,
            path=str(path),
        )
    want = Fraction(plan_fps.num, plan_fps.den)
    if video.fps is None or abs(video.fps / want - 1) > fps_tolerance:
        got = "unknown" if video.fps is None else f"{video.fps} ({float(video.fps):.4f})"
        raise RenderError(
            "fps_not_renderable",
            f"source {source.id!r}: {str(path)!r} runs at {got} fps, not the plan's {want} "
            f"({float(want):.4f}); plan positions are frames at the plan's rate",
            "conform the hub copy to the plan's frame rate (or raise [render] fps_tolerance for "
            "a near-identical rate)",
            source=source.id,
            path=str(path),
            fps=None if video.fps is None else str(video.fps),
        )
    display = video.display_size
    if display is None or min(display) < 2:
        raise RenderError(
            "size_not_renderable",
            f"source {source.id!r}: {str(path)!r} has no usable picture size ({display})",
            "give a file whose video stream has a known width and height",
            source=source.id,
            path=str(path),
        )
    if source.frame_size is not None:
        fw, fh = source.frame_size[0], source.frame_size[1]
        off = abs(fw * display[1] - fh * display[0]) / (fw * display[1])
        if off > ASPECT_TOLERANCE:
            raise RenderError(
                "frame_size_mismatch",
                f"source {source.id!r}: the plan's frame_size {fw}x{fh} does not have the hub "
                f"file's display aspect ({display[0]}x{display[1]})",
                "set frame_size to the media's display size (aar plan produce reads it)",
                source=source.id,
                frame_size=[fw, fh],
                display_size=list(display),
            )
    duration = video.duration_s if video.duration_s is not None else info.duration_s
    frames = None if duration is None else round(Fraction(duration) * video.fps)
    audio = info.audio
    aspect = video.sample_aspect if video.sample_aspect not in (None, Fraction(1)) else None
    return HubSource(
        source.id,
        source.media,
        path,
        rule,
        info,
        display,
        aspect,
        video.fps,
        frames,
        None if audio is None else (audio.channels or 1),
    )


def used_source_ids(plan: EditPlan) -> list[str]:
    """Ids of the sources the timeline plays, in plan order (as the export resolves them)."""
    used = {seg.source for seg in plan.segments}
    if plan.outro is not None:
        used.add(plan.outro.source)
    seen: list[str] = []
    for source in plan.sources:
        if source.id in used and source.id not in seen:
            seen.append(source.id)
    return seen
