"""Every value of a media clip's rows that is derived from the measured facts.

Each rule was found on Resolve 21.1 saves and is checked against Resolve's own rows by the
corpus gate (``tests/drp/test_mediapool_gate.py``); docs/DESIGN.md lists them with their evidence.
A value that cannot be derived exactly is refused (:class:`MediaPoolError`), never guessed.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from fractions import Fraction

from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import TIMECODE_TAG, MediaFacts, ProbeStream
from aar.media.probe import STANDARD_RATES, parse_probe_json, stream_rotation, timecode_frames

__all__ = [
    "PICTURE_CLASSES",
    "audio_bit_depth",
    "audio_codec_name",
    "audio_samples",
    "audio_streams",
    "channel_layout",
    "channels",
    "creation_us",
    "frame_size",
    "picture_class",
    "qt_local_time",
    "resolution",
    "sample_rate",
    "start_seconds",
    "start_timecode",
    "video_frames",
    "video_rate",
    "video_stream",
]

_DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_HINT = "measure the file again with the facts script on the edit PC"
#: ``pix_fmt`` -> (class name, bytes per pixel as a fraction): ``FrameSize`` is
#: ``width * even(height) * bytes per pixel + 8192``.
PICTURE_CLASSES: dict[str, tuple[str, Fraction]] = {
    "yuv420p": ("8-bit 4:2:0", Fraction(3, 2)),
    "yuvj420p": ("8-bit 4:2:0", Fraction(3, 2)),
    "yuv420p10le": ("10-bit 4:2:0", Fraction(3)),
    "yuv422p10le": ("10-bit 4:2:2", Fraction(8, 3)),
    "yuva444p12le": ("12-bit 4:4:4:4", Fraction(8)),
}
_FRAME_SIZE_PAD = 8192
#: The longest file the writer authors: 24 hours. A longer frame or sample count is refused
#: (no recording of the corpus comes near it, and a larger number is a measuring error).
MAX_SECONDS = 24 * 3600
#: Sample rates the writer accepts (Resolve was seen storing 44.1 and 48 kHz).
MIN_SAMPLE_RATE, MAX_SAMPLE_RATE = 1000, 384000
#: Creation times the writer accepts, in microseconds since 1970: from 1970 up to the year 9999.
_MAX_US = int(datetime(9999, 12, 31, 23, 59, 59, tzinfo=UTC).timestamp()) * 1_000_000
_ISO = re.compile(
    r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.([0-9]{1,9}))?Z\Z"
)
#: ffprobe audio codec -> (Resolve's codec name, its BitDepth code).
_AUDIO_CODECS: dict[str, tuple[str, int]] = {
    "pcm_s16le": ("Linear PCM", 1),
    "pcm_s16be": ("Linear PCM", 1),
    "pcm_s24le": ("Linear PCM", 2),
    "pcm_s24be": ("Linear PCM", 2),
    "aac": ("AAC", 3),
    "mp3": ("MP3", 3),
}
#: ffprobe ``channel_layout`` -> Resolve's ``ChannelLayout`` (absent when ffprobe names none).
_LAYOUTS = {"mono": 1, "stereo": 2}


def _refuse(code: str, message: str, hint: str = _HINT) -> MediaPoolError:
    return MediaPoolError(code, message, hint)


def qt_local_time(value: datetime) -> str:
    """The modification time as Resolve writes it in a clip's ``Clip`` blob (field 3): the edit
    PC's wall clock in C ``ctime`` form, the day padded with a space (``Tue Mar  3 04:05:06 2026``).
    """
    return (
        f"{_DAYS[value.weekday()]} {_MONTHS[value.month - 1]} {value.day:2d} "
        f"{value.hour:02d}:{value.minute:02d}:{value.second:02d} {value.year}"
    )


def video_stream(facts: MediaFacts) -> ProbeStream | None:
    """The first video stream that is not cover art."""
    return next(
        (s for s in facts.streams if s.codec_type == "video" and not s.is_attached_picture), None
    )


def audio_streams(facts: MediaFacts) -> list[ProbeStream]:
    return [s for s in facts.streams if s.codec_type == "audio"]


def _rational(text: str | None, what: str) -> Fraction:
    if text is None:
        raise _refuse("cannot_derive", f"{what} is missing")
    num, _, den = text.partition("/")
    if int(den) == 0 or int(num) <= 0 or int(den) < 0:
        raise _refuse("cannot_derive", f"{what} is {text!r}, not a positive rate")
    return Fraction(int(num), int(den))


def video_rate(facts: MediaFacts) -> Fraction:
    """The clip's frame rate: a QuickTime timecode track's average rate when there is one (a
    variable-frame-rate camera file's own rates are not what Resolve shows), else the video
    stream's rate when it is constant (``r_frame_rate`` equals ``avg_frame_rate``). It must be a
    standard rate exactly: Resolve stores it as a double."""
    video = video_stream(facts)
    if video is None:
        raise _refuse("cannot_derive", "the file has no video stream to take a frame rate from")
    tmcd = next((s for s in facts.streams if s.codec_tag_string == "tmcd"), None)
    if tmcd is not None:
        rate = _rational(tmcd.avg_frame_rate, "the timecode track's avg_frame_rate")
    else:
        base = _rational(video.r_frame_rate, "the video stream's r_frame_rate")
        avg = _rational(video.avg_frame_rate, "the video stream's avg_frame_rate")
        if base != avg:
            raise _refuse(
                "vfr_without_timecode",
                f"the video's frame rate varies (r_frame_rate {base}, avg_frame_rate {avg}) and "
                "the file has no timecode track to give the rate Resolve uses",
                "transcode the file to a constant frame rate first",
            )
        rate = avg
    if rate not in STANDARD_RATES:
        raise _refuse(
            "not_exact",
            f"the frame rate {rate} is not one this writer has seen Resolve store",
            "transcode the file to a standard frame rate first",
        )
    return rate


def video_frames(facts: MediaFacts, rate: Fraction) -> int:
    """``Time.NumFrames``: the video stream's duration times ``rate``
    (``duration_ts * time_base * rate``). Every file of a supported kind gives a whole number;
    one that does not is refused, as only one variable-frame-rate camera file (a kind this
    writer refuses) showed the rounding: down, not to the nearest, and not ``nb_frames``."""
    video = video_stream(facts)
    if video is None or video.duration_ts is None:
        raise _refuse("cannot_derive", "the video stream has no duration_ts")
    base = _rational(video.time_base, "the video stream's time_base")
    exact = video.duration_ts * base * rate
    if exact.denominator != 1:
        raise _refuse(
            "not_exact",
            f"the video lasts {exact.numerator}/{exact.denominator} frames at {rate} fps, "
            "not a whole number",
            "transcode the file to a constant frame rate first",
        )
    if exact < 1:
        raise _refuse("cannot_derive", "the video stream is shorter than one frame")
    if exact > rate * MAX_SECONDS:
        raise _refuse(
            "cannot_derive",
            f"the video lasts {int(exact)} frames, more than {MAX_SECONDS // 3600} hours",
        )
    return int(exact)


def start_timecode(facts: MediaFacts) -> str | None:
    """The start timecode as written: the video stream's tag, else another stream's (a timecode
    track or a camera's metadata track), else the container's
    (:class:`aar.media.probe.MediaInfo`)."""
    info = parse_probe_json(facts.probe.model_dump(), facts.path, facts.size)
    timecode = info.timecode
    if timecode is not None and not TIMECODE_TAG.match(timecode):
        raise _refuse(
            "cannot_derive", f"the start timecode {timecode[:40]!r} is not HH:MM:SS:FF in ASCII"
        )
    return timecode


def start_seconds(timecode: str | None, rate: Fraction) -> float:
    """``StartTime`` and the start of ``MediaExtents``: the timecode's frame number (counted at
    the nominal rate, 30 for 29.97) divided by the rate as a double; 0 without a timecode."""
    if timecode is None:
        return 0.0
    try:
        frames = timecode_frames(timecode, rate)
    except ValueError as exc:
        raise _refuse("cannot_derive", f"the start timecode cannot be read: {exc}") from None
    return frames / float(rate)


def audio_samples(stream: ProbeStream) -> int:
    """``TracksBA.Duration``: the stream's length in samples, ``floor(duration_ts * time_base *
    sample_rate)`` (exact for PCM, whose time base is 1/sample rate)."""
    if stream.duration_ts is None or stream.sample_rate is None:
        raise _refuse(
            "cannot_derive", f"audio stream {stream.index} has no duration_ts or sample_rate"
        )
    rate = sample_rate(stream)
    base = _rational(stream.time_base, f"audio stream {stream.index}'s time_base")
    samples = math.floor(stream.duration_ts * base * rate)
    if samples < 1:
        raise _refuse("cannot_derive", f"audio stream {stream.index} holds no samples")
    if samples > rate * MAX_SECONDS:
        raise _refuse(
            "cannot_derive",
            f"audio stream {stream.index} holds {samples} samples, "
            f"more than {MAX_SECONDS // 3600} hours",
        )
    return samples


def sample_rate(stream: ProbeStream) -> int:
    """The stream's sample rate in Hz (ASCII digits, checked by the facts model), within
    :data:`MIN_SAMPLE_RATE`..:data:`MAX_SAMPLE_RATE`."""
    if stream.sample_rate is None:
        raise _refuse("cannot_derive", f"audio stream {stream.index} has no sample_rate")
    rate = int(stream.sample_rate)
    if not MIN_SAMPLE_RATE <= rate <= MAX_SAMPLE_RATE:
        raise _refuse("cannot_derive", f"audio stream {stream.index} has a sample rate of {rate}")
    return rate


def channels(stream: ProbeStream) -> int:
    """The stream's channel count; a stream without one is refused."""
    if stream.channels is None:
        raise _refuse("cannot_derive", f"audio stream {stream.index} has no channel count")
    return stream.channels


def audio_codec_name(stream: ProbeStream) -> str:
    """``CodecName`` and the audio ``Clip`` field 5: ``Linear PCM``, ``AAC`` or ``MP3``."""
    known = _AUDIO_CODECS.get(stream.codec_name or "")
    if known is None:
        raise _refuse(
            "cannot_derive",
            f"audio codec {stream.codec_name!r} has no known Resolve codec name",
            "transcode the audio to PCM first",
        )
    return known[0]


def audio_bit_depth(stream: ProbeStream) -> int:
    """``BitDepth``: 1 for 16-bit PCM, 2 for 24-bit PCM, 3 for a codec decoded to floats."""
    known = _AUDIO_CODECS.get(stream.codec_name or "")
    if known is None:
        raise _refuse("cannot_derive", f"audio codec {stream.codec_name!r} has no known bit depth")
    return known[1]


def channel_layout(stream: ProbeStream) -> int | None:
    """``ChannelLayout``: 1 (mono) or 2 (stereo) when ffprobe names a layout, absent otherwise
    (a WAV without a channel mask, PCM in a QuickTime file)."""
    if stream.channel_layout is None:
        return None
    layout = _LAYOUTS.get(stream.channel_layout)
    if layout is None:
        raise _refuse("cannot_derive", f"channel layout {stream.channel_layout!r} is not known")
    if layout != stream.channels:
        raise _refuse(
            "bad_facts",
            f"audio stream {stream.index}: layout {stream.channel_layout} "
            f"but {stream.channels} channels",
        )
    return layout


def _iso_us(text: str) -> int | None:
    """An ISO UTC time in microseconds since 1970; None if it is not one, names a date that
    does not exist, or lies outside 1970..9999."""
    match = _ISO.match(text)
    if match is None:
        return None
    try:
        whole = datetime(*(int(match.group(n)) for n in range(1, 7)), tzinfo=UTC)
    except ValueError:
        return None
    fraction = (match.group(7) or "0").ljust(6, "0")[:6]
    value = int(whole.timestamp()) * 1_000_000 + int(fraction)
    return value if 0 <= value <= _MAX_US else None


def creation_us(facts: MediaFacts) -> int:
    """The ``Clip`` blob's field 13, in microseconds since 1970: the container's
    ``creation_time`` tag when it has one, else the file's creation time on the edit PC (UTC,
    truncated to the microsecond; not its modification time, which a re-render changes)."""
    tag = facts.probe.format.tags.get("creation_time")
    if tag is not None:
        value = _iso_us(tag)
        if value is None:
            raise _refuse(
                "cannot_derive",
                f"the creation_time tag {tag[:40]!r} is not a UTC time from 1970 to 9999",
            )
        return value
    if facts.ctime_utc_ns is None:
        raise _refuse(
            "cannot_derive",
            "the file has no creation_time tag and its creation time was not measured",
        )
    value = facts.ctime_utc_ns // 1000
    if value > _MAX_US:
        raise _refuse("cannot_derive", "the file's creation time lies after the year 9999")
    return value


def picture_class(stream: ProbeStream) -> tuple[str, Fraction]:
    """The pixel layout of a video stream, with its bytes per pixel."""
    known = PICTURE_CLASSES.get(stream.pix_fmt or "")
    if known is None:
        raise _refuse(
            "cannot_derive",
            f"pixel format {stream.pix_fmt!r} has no known frame size rule",
            "transcode the file to a pixel format this writer knows",
        )
    return known


def frame_size(width: int, height: int, bytes_per_pixel: Fraction) -> int:
    """``Geometry.FrameSize`` of a moving picture: ``width * even(height) * bytes per pixel +
    8192`` (an odd height counts one more line). A still's ``FrameSize`` is its file size."""
    value = width * (height + (height & 1)) * bytes_per_pixel
    if value.denominator != 1:
        raise _refuse("not_exact", f"a {width}x{height} frame is not a whole number of bytes")
    return int(value) + _FRAME_SIZE_PAD


def resolution(stream: ProbeStream) -> tuple[int, int, bool]:
    """``Geometry.Resolution`` (width, height) and whether it is transposed: a display rotation
    of 90 or 270 degrees swaps the two and sets ``IsResolutionTransposed``."""
    if stream.width is None or stream.height is None:
        raise _refuse("cannot_derive", f"video stream {stream.index} has no width or height")
    rotation = rotation_of(stream)
    if rotation in (90, 270):
        return stream.height, stream.width, True
    return stream.width, stream.height, False


def rotation_of(stream: ProbeStream) -> int:
    """The display rotation in degrees, 0-359 (:func:`aar.media.probe.stream_rotation`)."""
    return stream_rotation(stream.model_dump())
