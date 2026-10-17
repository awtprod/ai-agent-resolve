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
import os
import zipfile
from pathlib import Path

import pytest

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
    assert drp.blob_stats.failures == {}
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


def test_corpus_is_skipped_or_found() -> None:
    if CORPUS:
        assert FILES, f"AAR_DRP_CORPUS is set but holds no {sorted(SUFFIXES)} files"
