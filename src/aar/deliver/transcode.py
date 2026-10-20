"""The review transcoder: a protocol and a default two-pass H.264 ffmpeg implementation.

The default runs ffmpeg through :mod:`aar.media` (an argv list, never a shell; a timeout on every
call; stdin closed) and confirms the output decodes before it returns, so the caller can rename a
``.part`` into place knowing ffmpeg exited 0 and the file is readable. A missing ffmpeg raises the
existing :class:`~aar.media.MediaToolMissing`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from aar.deliver.errors import DeliverError
from aar.media import MediaError, MediaToolMissing, input_url, probe, run_tool
from aar.media.probe import PROBE_TIMEOUT_S

AUDIO_KBPS = 128
"""The review's AAC bit rate in kbit/s (the size budget subtracts this from the video rate)."""


def _scale(width: int, height: int) -> str:
    return f"scale={width}:{height}"


def pass1_argv(
    src: str, *, width: int, height: int, video_kbps: int, passlogfile: str
) -> list[str]:
    """ffmpeg argv for the two-pass analysis pass (video only, output discarded)."""
    return [
        "-nostdin",
        "-y",
        "-i",
        input_url(src),
        "-map",
        "0:v:0",
        "-vf",
        _scale(width, height),
        "-c:v",
        "libx264",
        "-b:v",
        f"{video_kbps}k",
        "-preset",
        "medium",
        "-pass",
        "1",
        "-passlogfile",
        passlogfile,
        "-an",
        "-f",
        "null",
        "-",
    ]


def pass2_argv(
    src: str,
    out: str,
    *,
    width: int,
    height: int,
    video_kbps: int,
    audio_kbps: int,
    passlogfile: str,
) -> list[str]:
    """ffmpeg argv for the encode pass: H.264 video, AAC audio, +faststart, mp4 to ``out``."""
    return [
        "-nostdin",
        "-y",
        "-i",
        input_url(src),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0?",
        "-vf",
        _scale(width, height),
        "-c:v",
        "libx264",
        "-b:v",
        f"{video_kbps}k",
        "-preset",
        "medium",
        "-pass",
        "2",
        "-passlogfile",
        passlogfile,
        "-c:a",
        "aac",
        "-b:a",
        f"{audio_kbps}k",
        "-movflags",
        "+faststart",
        "-f",
        "mp4",
        out,
    ]


@runtime_checkable
class Transcoder(Protocol):
    """Produce a decodable H.264/AAC mp4 at ``out`` (a ``.part`` name) or raise.

    The implementation owns the ffmpeg run and the "it decodes" guarantee; the caller owns the
    ``.part`` naming, the size check against the cap, and the rename into place.
    """

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
    ) -> None: ...


class FfmpegTranscoder:
    """A two-pass H.264 + AAC ffmpeg transcoder. Records every argv it ran in ``argv_log``."""

    def __init__(self) -> None:
        self.argv_log: list[list[str]] = []

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
        one = pass1_argv(
            src, width=width, height=height, video_kbps=video_kbps, passlogfile=passlogfile
        )
        two = pass2_argv(
            src,
            out,
            width=width,
            height=height,
            video_kbps=video_kbps,
            audio_kbps=audio_kbps,
            passlogfile=passlogfile,
        )
        self._run(one, timeout_s)
        self._run(two, timeout_s)
        self._confirm_decodes(out, timeout_s)

    def _run(self, argv: list[str], timeout_s: float) -> None:
        self.argv_log.append(argv)
        try:
            run_tool("ffmpeg", argv, timeout_s=timeout_s, check=True)
        except MediaToolMissing:
            raise  # the CLI maps a missing tool to its own exit code
        except MediaError as exc:
            raise DeliverError(
                "transcode_failed",
                f"ffmpeg failed: {exc.summary()}",
                "check the source render decodes and the output size and bit rate are valid for "
                "H.264",
                exit_code=exc.exit_code,
            ) from exc

    def _confirm_decodes(self, out: str, timeout_s: float) -> None:
        try:
            info = probe(out, timeout_s=min(PROBE_TIMEOUT_S, timeout_s))
        except MediaError as exc:
            raise DeliverError(
                "transcode_failed",
                f"the transcoded review did not decode: {exc.summary()}",
                "re-run the delivery; if it recurs, check the source render and ffmpeg build",
                path=out,
            ) from exc
        if info.video is None:
            raise DeliverError(
                "transcode_failed",
                "the transcoded review has no video stream",
                "re-run the delivery; check the source render has a video stream",
                path=out,
            )
