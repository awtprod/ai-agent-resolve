"""Media path rules for FCPXML export: each plan source becomes a ``file:`` URL Resolve on the edit
PC can open, or the export is refused.

* An absolute edit-PC (Windows) path, ``E:\\Show\\A001.braw`` or ``\\\\host\\share\\x.mov``, is
  checked with the build ops' own rule (:func:`aar.plan.apply.inputs.is_edit_path`) and written as
  a percent-encoded ``file:`` URL (``file:///E:/Show/A001.braw``, ``file://host/share/x.mov``).
* A ``file:`` URL whose path is a Windows drive or UNC path is used as is.
* A hub (Linux) path, or a ``file:`` URL of one, needs a media map (``linux_root -> windows_root``,
  the S13f :class:`aar.host.pathmap.HostPathMap`). Without a map, or outside it, it is refused as
  ``media_not_on_edit_pc``.
* A media-pool name (a multicam clip) cannot be referenced by FCPXML; it is refused as
  ``pool_source_not_exportable``.

Nothing here touches the network or a file other than the media map itself.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import quote, unquote, urlsplit

from aar.host.errors import ConfigError
from aar.host.pathmap import HostPathMap
from aar.plan.apply.inputs import classify_media, is_edit_path
from aar.plan.export.errors import ExportError
from aar.plan.model import Source

MediaRuleName = Literal["windows-path", "file-url", "media-map"]

MEDIA_MAP_MAX_BYTES = 1_000_000
_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.S)
_UNC = re.compile(r"^\\\\([^\\/]+)[\\/](.*)$", re.S)
# Characters a percent-encoded URL may carry (RFC 3986 unreserved, reserved and '%').
_URL_CHARS = re.compile(r"[A-Za-z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*")
_PCT = re.compile(r"%(?![0-9A-Fa-f]{2})")


@dataclass(frozen=True)
class MediaRef:
    """Where one source's media is for FCPXML: the ``src`` URL and which rule produced it."""

    source_id: str
    plan_media: str
    src: str
    rule: MediaRuleName
    windows_path: str
    name: str

    def as_dict(self) -> dict[str, str]:
        return {
            "source": self.source_id,
            "plan_media": self.plan_media,
            "src": self.src,
            "rule": self.rule,
            "windows_path": self.windows_path,
        }


def load_media_map(path: Path) -> HostPathMap:
    """Read a media map JSON file: an object of ``"linux_root": "windows_root"`` pairs."""
    try:
        size = path.stat().st_size
        if size > MEDIA_MAP_MAX_BYTES:
            raise ExportError(
                "bad_media_map",
                f"media map {str(path)!r} is {size} bytes, over the {MEDIA_MAP_MAX_BYTES} limit",
                "a media map lists a few root pairs; check the file",
            )
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExportError(
            "bad_media_map",
            f"media map {str(path)!r} could not be read as JSON: {exc}",
            'give a JSON object such as {"/srv/media": "E:\\\\Media"}',
        ) from None
    return media_map_from_mapping(data)


def media_map_from_mapping(data: object) -> HostPathMap:
    """A :class:`HostPathMap` from ``{"linux_root": "windows_root", ...}``."""
    if not isinstance(data, dict) or not data:
        raise ExportError(
            "bad_media_map",
            "a media map must be a non-empty JSON object of linux_root -> windows_root",
            'give a JSON object such as {"/srv/media": "E:\\\\Media"}',
        )
    table: list[tuple[str, str]] = []
    for linux_root, windows_root in data.items():
        if not isinstance(linux_root, str) or not isinstance(windows_root, str):
            raise ExportError(
                "bad_media_map",
                f"media map entry {linux_root!r}: both roots must be strings",
                'give a JSON object such as {"/srv/media": "E:\\\\Media"}',
            )
        table.append((windows_root, linux_root))
    try:
        return HostPathMap(tuple(table))
    except ConfigError as exc:
        raise ExportError(
            "bad_media_map",
            f"media map is invalid: {exc}",
            "each key is an absolute Linux prefix, each value a Windows drive or UNC prefix",
        ) from None


def windows_file_url(path: str) -> str:
    """A valid absolute edit-PC path as a percent-encoded ``file:`` URL."""
    drive = _DRIVE.match(path)
    if drive is not None:
        parts = re.split(r"[\\/]", drive.group(2))
        return f"file:///{drive.group(1)}:/" + "/".join(quote(p, safe="") for p in parts)
    unc = _UNC.match(path)
    assert unc is not None, "callers pass a path is_edit_path accepted"
    parts = re.split(r"[\\/]", unc.group(2))
    return f"file://{quote(unc.group(1), safe='')}/" + "/".join(quote(p, safe="") for p in parts)


def _basename(windows_path: str) -> str:
    return re.split(r"[\\/]", windows_path)[-1]


def _not_on_edit_pc(source: Source, why: str, hint: str | None = None) -> ExportError:
    return ExportError(
        "media_not_on_edit_pc",
        f"source {source.id!r} media {source.media!r} {why}",
        hint
        or "give the media's edit-PC path (E:\\...), or a hub path with --media-map "
        "linux_root -> windows_root (stage it first with `aar host push`, docs/MEDIA-STORE.md)",
        source=source.id,
        media=source.media,
    )


def _from_linux(source: Source, linux_path: str, media_map: HostPathMap | None) -> MediaRef:
    if media_map is None:
        raise _not_on_edit_pc(source, "is a hub (Linux) path and no media map was given")
    mapped = media_map.hub_to_edit(linux_path)
    if mapped.new is None:
        raise _not_on_edit_pc(
            source, f"is a hub (Linux) path the media map does not cover ({mapped.reason})"
        )
    if not is_edit_path(mapped.new):
        raise _not_on_edit_pc(source, f"maps to {mapped.new!r}, which is not a valid edit-PC path")
    return MediaRef(
        source.id,
        source.media,
        windows_file_url(mapped.new),
        "media-map",
        mapped.new,
        _basename(mapped.new),
    )


def _from_file_url(source: Source, media_map: HostPathMap | None) -> MediaRef:
    text = source.media
    bad = "is a file: URL that is not percent-encoded"
    if not text.isascii() or not _URL_CHARS.fullmatch(text) or _PCT.search(text):
        raise _not_on_edit_pc(
            source, bad, "give the edit-PC path instead (E:\\...); the exporter encodes it"
        )
    parts = urlsplit(text)
    if not text.lower().startswith("file://") or parts.query or parts.fragment:
        raise _not_on_edit_pc(
            source,
            "is a file: URL that is not file:///X:/... or file://host/share/... "
            "(no query or fragment)",
        )
    path = unquote(parts.path)
    host = parts.netloc
    if host in ("", "localhost"):
        if re.match(r"^/[A-Za-z]:/", path):
            windows = path[1:]
        elif path.startswith("/") and not path.startswith("//"):
            return _from_linux(source, path, media_map)  # a hub path written as a URL
        else:
            raise _not_on_edit_pc(source, "is a file: URL without a drive or UNC path")
    else:
        windows = f"\\\\{unquote(host)}" + path.replace("/", "\\")
    if not is_edit_path(windows):
        raise _not_on_edit_pc(source, f"is a file: URL of {windows!r}, not a valid edit-PC path")
    return MediaRef(source.id, text, text, "file-url", windows, _basename(windows))


def media_ref(source: Source, media_map: HostPathMap | None = None) -> MediaRef:
    """The FCPXML ``src`` for one source, by the rules in the module docstring."""
    media = source.media
    if media[:5].lower() == "file:":
        return _from_file_url(source, media_map)
    kind = classify_media(media)
    if kind == "import":
        return MediaRef(
            source.id, media, windows_file_url(media), "windows-path", media, _basename(media)
        )
    if kind == "pool":
        raise ExportError(
            "pool_source_not_exportable",
            f"source {source.id!r} media {media!r} is a media-pool name (e.g. a multicam clip); "
            "FCPXML can only reference media files",
            "give the underlying camera file instead: its edit-PC path (E:\\...), or its hub "
            "path with --media-map",
            source=source.id,
            media=media,
        )
    if media.startswith("/") and not media.startswith("//"):
        return _from_linux(source, media, media_map)
    raise _not_on_edit_pc(source, "is not an absolute edit-PC path, file: URL or hub path")
