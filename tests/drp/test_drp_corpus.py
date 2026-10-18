"""Opt-in checks over a private corpus of real project files.

Set ``AAR_DRP_CORPUS`` to a directory of ``.drp``/``.drt``/``.drb`` files (searched
recursively). The files are never committed. Test ids are numbers, not file names, so no
project names reach test output. When the directory has a ``MANIFEST.tsv`` with
``local_path`` and ``original_windows_path`` columns, files whose original path contains
``validation_output`` are treated as tool-written (not Resolve exports): they may hold
malformed blobs, so only "reads without raising" is required of them.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from aar.drp import blobs
from aar.drp.describe import describe
from aar.drp.reader import read_drp
from aar.drp.scan import unescape

CORPUS = os.environ.get("AAR_DRP_CORPUS")
SUFFIXES = {".drp", ".drt", ".drb"}


def _files() -> list[Path]:
    if not CORPUS:
        return []
    return sorted(p for p in Path(CORPUS).rglob("*") if p.suffix.lower() in SUFFIXES)


def _tool_written() -> set[str]:
    if not CORPUS:
        return set()
    manifest = Path(CORPUS) / "MANIFEST.tsv"
    if not manifest.is_file():
        return set()
    with manifest.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {
        Path(row["local_path"]).name
        for row in rows
        if "validation_output" in (row.get("original_windows_path") or "").lower()
    }


FILES = _files()
TOOL_WRITTEN = _tool_written()


@pytest.mark.skipif(not CORPUS, reason="set AAR_DRP_CORPUS to run the corpus checks")
@pytest.mark.parametrize("path", FILES, ids=[f"corpus-{i:03d}" for i in range(len(FILES))])
def test_corpus_file(path: Path) -> None:
    drp = read_drp(path, check_blobs=True)
    describe(drp)
    if path.name in TOOL_WRITTEN or not drp.is_zip:
        return
    assert drp.versions is not None
    assert not [i for i in drp.issues if i.severity == "error"]
    if drp.zstd_backend is None:
        # Without zstandard, compressed zstd blobs are expected to fail, and nothing else.
        assert all("zstd-unavailable" in f for f in drp.blob_stats.failures)
    else:
        assert drp.blob_stats.failures == {}
    for clip in drp.media:
        # A clip's path never comes from another stream's Clip blob: it is the main stream's
        # blob path, or a timeline <MediaFilePath>, or nothing.
        if clip.path_source == "media_pool_blob":
            assert clip.path == clip.blob_path
        elif clip.path_source == "timeline_item":
            assert clip.blob_path is None and clip.path in clip.timeline_paths
        else:
            assert clip.path is None
    # Spans must point at the exact source text of the values they describe.
    with zipfile.ZipFile(path) as zf:
        cache: dict[str, bytes] = {}
        for tl in drp.timelines:
            for track in tl.tracks:
                for item in track.items:
                    for key in ("name", "media_ref", "media_file_path"):
                        prov = item.spans.get(key)
                        value = getattr(item, key)
                        if prov is None or value is None or prov.kind != "text":
                            continue
                        if prov.member not in cache:
                            cache[prov.member] = zf.read(prov.member)
                        data = cache[prov.member]
                        assert unescape(data[prov.start : prov.end].decode()).strip() == value
                    if item.media_ref:
                        assert item.media_clip_id or any(
                            i.code == "unresolved-media" for i in drp.issues
                        )


# Marker pins live in the corpus, never in the repo: $AAR_DRP_CORPUS/marker_pins.json maps
# a file's sha256 to its decoded markers, each row [frame, colour, name, duration] and, in the
# richer form, also [note, custom_data], sorted. The pin's own row width decides how many
# fields are checked, so a 4-field pins file and a 6-field one both work. The frame/colour/
# name/duration values were cross-checked against an independent decoder (drp-format's
# timeline-markers-blob.js, run under node), which agreed on every one.
PINS_FILE = "marker_pins.json"
_MARKER_FIELDS = ("frame", "color", "name", "duration", "note", "custom_data")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pins() -> dict[str, dict[str, list[list[object]]]]:
    pins_path = Path(CORPUS or "") / PINS_FILE
    if not CORPUS or not pins_path.is_file():
        pytest.skip(f"no {PINS_FILE} in $AAR_DRP_CORPUS: marker values are not pinned")
    data = json.loads(pins_path.read_text(encoding="utf-8"))
    files = data.get("files")
    if not isinstance(files, dict) or not files:
        pytest.fail(f"{PINS_FILE} has no 'files' mapping")
    return files


def _marker_rows(markers: list, width: int) -> list[list[object]]:  # type: ignore[type-arg]
    fields = _MARKER_FIELDS[:width]
    return sorted([getattr(m, f) for f in fields] for m in markers)


def _pin_width(rows: list[list[object]]) -> int:
    """How many marker fields this pins file records (4 for the original form, 6 richer)."""
    return len(rows[0]) if rows else 4


@pytest.mark.skipif(not CORPUS, reason="set AAR_DRP_CORPUS to run the corpus checks")
def test_corpus_markers_match_pinned_values() -> None:
    pins = _load_pins()
    if blobs.zstd_backend() is None:
        pytest.skip("some pinned marker blobs are zstd-compressed; needs zstandard")
    pinned = {sha: path for path in FILES if (sha := _sha256(path)) in pins}
    if not pinned:
        pytest.skip(f"none of the files in {PINS_FILE} is in this corpus")
    for sha, path in pinned.items():
        pin = pins[sha]
        drp = read_drp(path)
        timeline_markers = [m for t in drp.timelines for m in t.markers]
        item_markers = [
            m for t in drp.timelines for tr in t.tracks for i in tr.items for m in i.markers
        ]
        assert _marker_rows(timeline_markers, _pin_width(pin["timeline"])) == sorted(pin["timeline"])
        assert _marker_rows(item_markers, _pin_width(pin["items"])) == sorted(pin["items"])
        assert all(m.frame_space == "timeline-relative" for m in timeline_markers)
        assert all(m.owner_kind == "item" for m in item_markers)
        assert not any(i.code in ("marker-decode", "orphan-markers") for i in drp.issues)


def test_corpus_is_skipped_or_found() -> None:
    if CORPUS:
        assert FILES, f"AAR_DRP_CORPUS is set but holds no {sorted(SUFFIXES)} files"
