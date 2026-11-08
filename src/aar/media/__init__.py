"""A small ffmpeg/ffprobe layer: probing, measuring and decoding local media files, and a
seek-based QuickTime reader for the start timecode of files ffprobe may not open (BRAW).

Binaries come from ``AAR_FFMPEG`` / ``AAR_FFPROBE`` or ``PATH``. Calls use argument lists
(never a shell), a timeout each, bounded stderr capture and typed errors (``MediaError`` with
the exit code and the stderr tail). Stdlib only.
"""

from aar.media.measure import (
    AudioMeasurements,
    ClipCount,
    DecodeResult,
    count_clipped,
    decode_check,
    measure_audio,
    read_pcm,
)
from aar.media.parse import (
    AStats,
    Ebur128Summary,
    ParseError,
    Silence,
    parse_astats,
    parse_ebur128,
    parse_silencedetect,
)
from aar.media.pcm import Pcm, QuietRun, QuietScan, quiet_runs
from aar.media.probe import MediaInfo, Stream, frames_timecode, parse_rational, probe
from aar.media.quicktime import QuickTimeTimecode, TimecodeUnreadable, read_timecode
from aar.media.tools import (
    MediaError,
    MediaInputError,
    MediaTimeout,
    MediaToolMissing,
    OutputTooLarge,
    RunResult,
    input_url,
    resolve_tool,
    run_tool,
    timeout_for,
)

__all__ = [
    "AStats",
    "AudioMeasurements",
    "ClipCount",
    "DecodeResult",
    "Ebur128Summary",
    "MediaError",
    "MediaInfo",
    "MediaInputError",
    "MediaTimeout",
    "MediaToolMissing",
    "OutputTooLarge",
    "ParseError",
    "Pcm",
    "QuickTimeTimecode",
    "QuietRun",
    "QuietScan",
    "RunResult",
    "Silence",
    "Stream",
    "TimecodeUnreadable",
    "count_clipped",
    "decode_check",
    "frames_timecode",
    "input_url",
    "measure_audio",
    "parse_astats",
    "parse_ebur128",
    "parse_rational",
    "parse_silencedetect",
    "probe",
    "quiet_runs",
    "read_pcm",
    "read_timecode",
    "resolve_tool",
    "run_tool",
    "timeout_for",
]
