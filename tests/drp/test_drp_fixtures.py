"""Reader tests against the four MIT-licensed drp-format templates in tests/fixtures/drp."""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from drp_helpers import needs_zstandard

from aar.drp import blobs
from aar.drp.model import DrpFile, Provenance
from aar.drp.reader import read_drp
from aar.drp.scan import unescape

FixturePath = Callable[[str], Path]
# The h264 template's Media Pool blob points at a file on the upstream author's machine.
# Compare a digest rather than repeating that path here.
H264_BLOB_PATH_SHA256 = "eca8a8d98731cdc140e8aecde2ab28dd386b94091cc181d67e19d9f25032c487"


def span_bytes(path: Path, prov: Provenance) -> bytes:
    with zipfile.ZipFile(path) as zf:
        return zf.read(prov.member)[prov.start : prov.end]


@needs_zstandard
@pytest.mark.parametrize(
    ("name", "app", "prj", "project"),
    [
        ("empty-project.drp", "21.0.0.0048", 17, "_drt_template_src"),
        ("empty-project-r19.drp", "19.1.3.0007", 14, "MCP_TPL_EMPTY"),
        ("media-clip-h264.drp", "21.0.0.0048", 17, "_media_tmpl"),
        ("media-clip-r19.drp", "19.1.3.0007", 14, "_mcp_e89_tpl16"),
    ],
)
def test_version_header_and_project_name(
    fixture_path: FixturePath, name: str, app: str, prj: int, project: str
) -> None:
    path = fixture_path(name)
    drp = read_drp(path, check_blobs=True)
    assert drp.kind == "drp" and drp.sniffed_kind == "drp"
    assert drp.versions is not None
    assert (drp.versions.app_version, drp.versions.project_version) == (app, prj)
    assert drp.versions.mismatched_members == []
    assert drp.versions.source is not None
    assert (
        span_bytes(path, drp.versions.source)
        == f'<!--DbAppVer="{app}" DbPrjVer="{prj}"-->'.encode()
    )
    assert drp.project is not None and drp.project.name == project
    assert drp.raw_members == ["Gallery.xml"]
    assert drp.blob_stats.checked and drp.blob_stats.failures == {}
    assert not [i for i in drp.issues if i.severity in ("warning", "error")]


@pytest.mark.parametrize("name", ["empty-project.drp", "empty-project-r19.drp"])
def test_empty_project_has_one_empty_24fps_timeline(fixture_path: FixturePath, name: str) -> None:
    drp = read_drp(fixture_path(name))
    assert [t.name for t in drp.timelines] == ["Timeline 1"]
    tl = drp.timelines[0]
    assert tl.kind == "timeline"
    assert tl.frame_rate == 24.0
    assert tl.resolution == (1920, 1080)
    assert (tl.start_frame, tl.end_frame) == (86400, 86400)
    assert [(t.label, len(t.items)) for t in tl.tracks] == [("V1", 0), ("A1", 0)]
    assert drp.project is not None and drp.project.timeline_ids == [tl.id]
    assert [m.type for m in drp.media] == ["timeline"]


def _video_clip(drp: DrpFile):  # type: ignore[no-untyped-def]
    clips = [m for m in drp.media if m.type == "video"]
    assert len(clips) == 1
    return clips[0]


@needs_zstandard
def test_media_clip_h264_media_pool(fixture_path: FixturePath) -> None:
    path = fixture_path("media-clip-h264.drp")
    drp = read_drp(path)
    assert sorted(m.type for m in drp.media) == ["timeline", "video"]
    clip = _video_clip(drp)
    assert clip.name == "sample.mp4"
    assert clip.frames == 4576
    assert clip.frame_rate == pytest.approx(30000 / 1001)
    assert clip.resolution == (352, 262)
    assert clip.codec == "avc1"
    assert [(a.sample_rate, a.channels, a.codec) for a in clip.audio_tracks] == [(44100, 2, "AAC")]
    # The timeline's <MediaFilePath> is stale; the Media Pool blob wins.
    assert clip.timeline_paths == ["/Media/sample.mp4"]
    assert clip.path_source == "media_pool_blob"
    assert clip.path is not None and clip.path != "/Media/sample.mp4"
    assert clip.path.startswith("/") and clip.path.endswith(".mp4")
    assert hashlib.sha256(clip.path.encode()).hexdigest() == H264_BLOB_PATH_SHA256
    prov = clip.spans["blob_path"]
    assert prov.kind == "blob" and prov.member == "MediaPool/Master/MpFolder.xml"
    assert blobs.classify("Clip", blobs.parse_hex(span_bytes(path, prov).decode())) == (
        blobs.BlobKind.ENVELOPE_ZSTD
    )
    assert any(i.code == "stale-media-file-path" for i in drp.issues)


@needs_zstandard
def test_media_clip_h264_timeline(fixture_path: FixturePath) -> None:
    path = fixture_path("media-clip-h264.drp")
    drp = read_drp(path)
    clip = _video_clip(drp)
    [tl] = drp.timelines
    assert (tl.name, tl.frame_rate, tl.resolution) == ("MediaTemplate", 24.0, (1920, 1080))
    assert (tl.start_frame, tl.end_frame) == (86400, 86400 + 3664)
    assert tl.member == "SeqContainer/106d9f2b-2951-4ab9-b961-0aa934df20e9.xml"
    assert tl.container_id == "106d9f2b-2951-4ab9-b961-0aa934df20e9"
    assert [(t.label, len(t.items)) for t in tl.tracks] == [("V1", 1), ("A1", 1)]
    video, audio = tl.tracks[0].items[0], tl.tracks[1].items[0]
    for item, kind in ((video, "video"), (audio, "audio")):
        assert item.kind == kind
        assert (item.record_start, item.record_end, item.duration) == (86400, 90064, 3664)
        assert (item.source_in, item.source_out) == (0, 3664)
        assert item.media_clip_id == clip.id
        assert item.media_path == clip.path and item.path_source == "media_pool_blob"
        assert item.media_file_path == "/Media/sample.mp4"
        assert (item.timemap_form, item.speed) == ("identity", 1.0)
        assert span_bytes(path, item.spans["name"]) == b"sample.mp4"
        assert span_bytes(path, item.spans["record_start"]) == b"86400"
        # <In/> is self-closing, so its span is the whole element (for a splice to replace).
        assert item.spans["source_in"].kind == "element"
        assert span_bytes(path, item.spans["source_in"]) == b"<In/>"
        assert span_bytes(path, item.source).startswith(
            f'<Sm2Ti{kind.capitalize()}Clip DbId="'.encode()
        )
    assert audio.media_track_index == 0


def test_media_clip_r19_bins_and_paths(fixture_path: FixturePath) -> None:
    drp = read_drp(fixture_path("media-clip-r19.drp"))
    assert [(b.path, len(b.clip_ids)) for b in drp.bins] == [("Master", 2), ("Master/Archive", 0)]
    master, archive = drp.bins
    assert archive.parent_id == master.id and master.child_ids == [archive.id]
    assert archive.member == "MediaPool/Master/000_Archive/MpFolder.xml"
    clip = _video_clip(drp)
    assert clip.path == "/tmp/r19_template_media.mp4"
    assert clip.timeline_paths == ["/tmp/r19_template_media.mp4"]
    assert [(a.channels, a.duration_samples) for a in clip.audio_tracks] == [(1, 6732306)]
    [tl] = drp.timelines
    assert [t.type for t in tl.tracks] == ["video"] + ["audio"] * 16
    assert [t.label for t in tl.tracks][-1] == "A16"
    assert not any(i.code == "stale-media-file-path" for i in drp.issues)


def test_every_text_span_reproduces_its_value(fixture_path: FixturePath) -> None:
    """A splice relies on spans: the bytes at each text span must decode to the model value."""
    path = fixture_path("media-clip-h264.drp")
    drp = read_drp(path)
    checked = 0
    for tl in drp.timelines:
        for track in tl.tracks:
            for item in track.items:
                for key in ("name", "media_file_path", "media_ref", "duration"):
                    prov = item.spans[key]
                    value = getattr(item, key)
                    assert unescape(span_bytes(path, prov).decode()) == str(value)
                    checked += 1
    assert drp.project is not None
    assert span_bytes(path, drp.project.spans["name"]) == b"_media_tmpl"
    assert checked == 8


def test_without_zstd_library_raw_block_blobs_still_decode(
    fixture_path: FixturePath, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The video Clip blob is a compressed zstd frame; the audio one is a raw-block frame."""
    monkeypatch.setattr(blobs, "_zstd_backend", lambda: (None, None))
    drp = read_drp(fixture_path("media-clip-h264.drp"))
    clip = _video_clip(drp)
    # Nothing is borrowed from the audio stream: not its codec, and not its Clip blob path
    # (here it names the same file, but for synced audio it would not). The path falls back to
    # the video item's <MediaFilePath>, labelled as such.
    assert (clip.path, clip.path_source) == ("/Media/sample.mp4", "timeline_item")
    assert clip.blob_path is None and clip.linked_paths == []
    assert clip.codec is None
    assert clip.frames == 4576  # keyed dicts need no zstd
    [unavailable] = [i for i in drp.issues if i.code == "zstd-unavailable"]
    assert "zstandard" in unavailable.message
    assert drp.zstd_backend is None
