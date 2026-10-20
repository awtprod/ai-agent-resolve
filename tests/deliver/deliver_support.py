"""Shared fakes and builders for the ``aar.deliver`` tests.

Real ffmpeg generates small synthetic sources into ``tmp_path``. A fake ``EditHostLike`` serves
them from a tmp "edit PC" directory and verifies sha256 like the real one. A fake ``Transcoder``
writes an output of a chosen size (so the retry and cap paths are deterministic). A real hub
(``EventLog``, ``ArtifactStore``, ``request_review``) runs on ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aar.deliver.model import RenderOutput
from aar.hub.artifacts import ArtifactStore
from aar.hub.events import Actor, EventLog
from aar.hub.graph import WorkGraph
from aar.hub.registry import Registry
from aar.media import run_tool
from aar.plan.profile import Profile, QATargets
from aar.qa.report import QACheck, QAReport, build_report

HUMAN = Actor("human", "andrew")
AGENT = Actor("agent", "recipe-1")


@dataclass
class Hub:
    log: EventLog
    store: ArtifactStore
    project_id: str
    subject_id: str
    root: Path


def make_hub(tmp_path: Path) -> Hub:
    """A fresh hub with one registered project and an ``op`` node to review."""
    log = EventLog(str(tmp_path / "hub.db"))
    store = ArtifactStore(str(tmp_path / "artifacts"), log)
    project_id = Registry(log).create_project("acme-show", HUMAN).id
    subject_id = WorkGraph(log, project_id).add_node("op", AGENT, title="build the acme cut").id
    return Hub(log, store, project_id, subject_id, tmp_path)


def make_source(
    path: Path | str,
    *,
    width: int,
    height: int,
    duration: float = 2.0,
    fps: int = 25,
    video_kbps: int = 800,
    volume: float = 0.5,
) -> Path:
    """Generate a small real mp4 (testsrc2 + a quiet sine) at ``path`` with real ffmpeg."""
    path = Path(path)
    run_tool(
        "ffmpeg",
        [
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size={width}x{height}:rate={fps}:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-filter:a",
            f"volume={volume}",
            "-c:v",
            "libx264",
            "-b:v",
            f"{video_kbps}k",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-shortest",
            str(path),
        ],
        timeout_s=120.0,
    )
    return path


@dataclass(frozen=True)
class FakeTransfer:
    """What the fake edit host returns: size and an upper-case sha256, like the real transport."""

    size: int
    sha256: str
    op: str = "pull"
    remote: str | None = None
    local: str | None = None


class FakeEditHost:
    """Serves files from a tmp "edit PC" directory by basename; records every call."""

    def __init__(self, edit_pc: Path) -> None:
        self.edit_pc = Path(edit_pc)
        self.pull_calls: list[tuple[str, str]] = []
        self.hash_calls: list[str] = []

    def _src(self, remote: str) -> Path:
        return self.edit_pc / Path(remote).name

    def file_hash(self, remote: str) -> FakeTransfer:
        self.hash_calls.append(remote)
        data = self._src(remote).read_bytes()
        return FakeTransfer(
            len(data), hashlib.sha256(data).hexdigest().upper(), op="hash", remote=remote
        )

    def pull(self, remote: str, local: str, *, overwrite: bool = False) -> FakeTransfer:
        self.pull_calls.append((remote, local))
        shutil.copyfile(self._src(remote), local)
        data = Path(local).read_bytes()
        return FakeTransfer(
            len(data), hashlib.sha256(data).hexdigest().upper(), remote=remote, local=local
        )


class RaisingEditHost:
    """A fake edit host whose ``pull`` raises, for the transport exit-code test."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def file_hash(self, remote: str) -> FakeTransfer:  # pragma: no cover - not reached
        raise self.exc

    def pull(self, remote: str, local: str, *, overwrite: bool = False) -> FakeTransfer:
        raise self.exc


class FakeTranscoder:
    """Writes an output of the next size in ``sizes`` (the last repeats); records every call."""

    def __init__(self, sizes: list[int]) -> None:
        self.sizes = sizes
        self.calls: list[dict[str, Any]] = []

    def transcode(
        self,
        src: str,
        out: str,
        *,
        width: int,
        height: int,
        video_kbps: int,
        audio_kbps: int,
        passlogfile: str,
        timeout_s: float,
    ) -> None:
        self.calls.append(
            {
                "src": src,
                "out": out,
                "width": width,
                "height": height,
                "video_kbps": video_kbps,
                "audio_kbps": audio_kbps,
                "passlogfile": passlogfile,
            }
        )
        size = self.sizes[min(len(self.calls) - 1, len(self.sizes) - 1)]
        Path(out).write_bytes(b"\x00" * size)


@dataclass
class QaStub:
    """A QA callable that returns a fixed report; records the (path, kind) it was called with."""

    report: QAReport
    calls: list[tuple[str, str]] = field(default_factory=list)

    def __call__(self, path: Any, *, kind: str, profile: Any = None, **_: Any) -> QAReport:
        self.calls.append((str(path), kind))
        return self.report


def qa_report(checks: list[QACheck], *, kind: str = "review") -> QAReport:
    return build_report(file="x", kind=kind, profile="test", media=None, checks=checks)


def qa_pass(kind: str = "review") -> QaStub:
    return QaStub(qa_report([QACheck(id="decode", status="pass", message="ok")], kind=kind))


def qa_fail(kind: str = "review") -> QaStub:
    return QaStub(
        qa_report(
            [QACheck(id="decode", status="fail", message="not readable")],
            kind=kind,
        )
    )


def make_profile(
    *,
    review_mp4_max_mb: float = 5.0,
    lufs_tolerance: float = 60.0,
    true_peak_max_dbtp: float = 0.0,
) -> Profile:
    """A permissive profile: the cap is set per test; loudness tolerance is wide for synthetic
    sine audio."""
    return Profile(
        name="test",
        qa=QATargets(
            longform_lufs=-22.0,
            short_lufs=-14.5,
            lufs_tolerance=lufs_tolerance,
            review_mp4_max_mb=review_mp4_max_mb,
            true_peak_max_dbtp=true_peak_max_dbtp,
        ),
        rules={},
    )


def render_output(
    remote: Path | str,
    *,
    duration: float = 6.0,
    frame_rate: float = 25.0,
    frame_count: int = 150,
    render_job_id: str = "rj-1",
) -> RenderOutput:
    return RenderOutput.from_measured(
        {
            "output_path": str(remote),
            "frame_count": frame_count,
            "frame_rate": frame_rate,
            "duration_seconds": duration,
            "render_job_id": render_job_id,
        }
    )
