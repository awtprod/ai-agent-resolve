"""What a timeline item takes from its media clip's rows.

A picture or sound item copies a few facts of the clip it plays into its own row (its name, its
file's path, where the file's time starts, how long it is, its frame rate, how many channels a
sound has). :func:`media_view` reads them from the clip's rows (:class:`ClipRows`, Resolve's own
or authored by :mod:`aar.drp.mediapool`), with the rules the item gate checked against every
real item (docs/DESIGN.md, "Timeline items (S25c2)"):

* a picture file: the rate is its ``Time.FrameRate``; its length ``NumFrames / rate - 1 / rate``
  (the time of its last frame, ``MediaTimemapBA``); its start the start timecode's frame number
  divided by the rate (``MediaStartTime``, 0 without a timecode);
* a sound file: the rate is the project's when the clip was imported (``FieldsBlob`` #2.#1);
  its length in frames is ``Duration / SampleRate * rate`` of its first audio track (a double),
  and the time of its last frame that over the rate minus ``1 / rate``, as for a picture; its
  start that track's ``StartTime``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from fractions import Fraction

from aar.drp import blobs
from aar.drp.items.errors import ItemError
from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.errors import refusing
from aar.drp.mediapool.templates import ClipRows, check_clip
from aar.media.probe import STANDARD_RATES, snap_rate

__all__ = ["MediaView", "media_view"]

_HINT = "use a media clip Resolve saved, or one authored by aar.drp.mediapool"
#: The channel counts an item's ``VirtualAudioTrackBA`` was seen with (mono, stereo).
CHANNELS = (1, 2)


@dataclass(frozen=True)
class MediaView:
    """The values an item copies from its media clip."""

    media_id: str
    name: str
    #: ``Clip`` #1 + ``\\`` + #2 of the clip's video info (its first audio info for a sound).
    path: str
    picture: bool
    #: Channels of each audio info, in ``EmbeddedAudioVec`` order.
    channels: tuple[int, ...]
    #: The rate as Resolve stores it (16 bytes: a little-endian double and 8 zero bytes).
    rate_bytes: bytes
    rate: Fraction
    start_seconds: float
    #: The time of the file's last frame (sample), in seconds.
    last_seconds: float
    #: Frames of the picture (None for a sound file): the most a picture item may read.
    frames: int | None
    #: Samples of the first audio track over its rate (None without sound).
    sound_seconds: float | None


def _bad(why: str) -> ItemError:
    return ItemError("bad_media", why, _HINT)


def _kd(value: object, what: str) -> dict[str, object]:
    entries = bt.kd_decode(value, what)
    out: dict[str, object] = {}
    for key, _, _, item in entries:
        if key in out:
            raise _bad(f"{what} holds {key!r} twice")
        out[key] = item
    return out


def _rate(data: object, what: str) -> tuple[bytes, Fraction]:
    if not isinstance(data, bytes) or len(data) != 16 or any(data[8:]):
        raise _bad(f"{what} is not a frame rate (a double and 8 zero bytes)")
    value = blobs.decode_rate(data)
    rate = snap_rate(Fraction(value))
    if rate not in STANDARD_RATES or float(rate) != value:
        raise _bad(f"{what} {value!r} is not exactly a standard rate")
    return data, rate


def _first_track(clip: ClipRows) -> dict[str, object]:
    tracks = _kd(clip.audio[0]["TracksBA"], "TracksBA")
    if len(tracks) != 1:
        raise _bad(f"the first audio info has {len(tracks)} tracks, not one")
    return _kd(next(iter(tracks.values())), "TracksBA track")


def _number(value: object, what: str, kind: type) -> int | float:
    if not isinstance(value, kind) or isinstance(value, bool):
        raise _bad(f"{what} is not a {kind.__name__}")
    return value  # type: ignore[return-value]


def _channels(clip: ClipRows) -> tuple[int, ...]:
    out = []
    for info in clip.audio:
        tracks = _kd(info["TracksBA"], "TracksBA")
        if len(tracks) != 1:
            raise _bad(f"an audio info has {len(tracks)} tracks, not one")
        track = _kd(next(iter(tracks.values())), "TracksBA track")
        out.append(int(_number(track.get("NumChannels"), "NumChannels", int)))
    return tuple(out)


def _sound_rate(clip: ClipRows) -> tuple[bytes, Fraction]:
    """The project's rate when a sound file was imported: ``Sm2MpMedia.FieldsBlob`` #2.#1."""
    _, payload = bt.message_of(clip.media["FieldsBlob"], "Sm2MpMedia.FieldsBlob")
    fields = bt.pb_decode(payload, "Sm2MpMedia.FieldsBlob")
    inner = bt.pb_get(fields, 2, "Sm2MpMedia.FieldsBlob")
    if inner[1] != 2:
        raise _bad("Sm2MpMedia.FieldsBlob #2 is not a message")
    rate_field = bt.pb_get(bt.pb_decode(inner[2], "FieldsBlob #2"), 1, "FieldsBlob #2")
    if rate_field[1] != 1:
        raise _bad("Sm2MpMedia.FieldsBlob #2.#1 is not a double")
    return _rate(rate_field[2] + bytes(8), "the sound clip's import rate")


@refusing("bad_media", "reading what an item needs from its media clip", _HINT, error=ItemError)
def media_view(clip: ClipRows) -> MediaView:
    """The values an item takes from ``clip`` (checked with
    :func:`aar.drp.mediapool.templates.check_clip` first)."""
    clip = check_clip(clip)
    row = clip.video if clip.video is not None else clip.audio[0]
    _, payload = bt.message_of(row["Clip"], "Clip")
    fields = bt.pb_decode(payload, "Clip")
    folder = bt.pb_get(fields, 1, "Clip")[2]
    name = bt.pb_get(fields, 2, "Clip")[2]
    if not isinstance(folder, bytes) or not isinstance(name, bytes):
        raise _bad("the clip's folder or file name is not text")
    path = folder.decode("utf-8") + "\\" + name.decode("utf-8")
    channels = _channels(clip)
    sound_seconds: float | None = None
    if clip.audio:
        track = _first_track(clip)
        samples = int(_number(track.get("Duration"), "Duration", int))
        sample_rate = int(_number(track.get("SampleRate"), "SampleRate", int))
        if sample_rate <= 0 or samples < 0:
            raise _bad("the first audio track has no samples or no sample rate")
        sound_seconds = samples / sample_rate
    if clip.video is not None:
        time = _kd(clip.video["Time"], "Time")
        rate_bytes, rate = _rate(time.get("FrameRate"), "Time.FrameRate")
        frames = int(_number(time.get("NumFrames"), "Time.NumFrames", int))
        if frames < 1:
            raise _bad("Time.NumFrames is below 1")
        timecode = time.get("Timecode")
        if timecode is not None and not isinstance(timecode, str):
            raise _bad("Time.Timecode is not text")
        double = float(rate)
        start = fm.start_seconds(timecode, rate)
        last = frames / double - 1 / double
        picture_frames: int | None = frames
    else:
        rate_bytes, rate = _sound_rate(clip)
        double = float(rate)
        track = _first_track(clip)
        start = float(_number(track.get("StartTime"), "StartTime", float))
        assert sound_seconds is not None
        # The length in frames at the rate, as a double, then the same rule as a picture's
        # (``(Duration / SampleRate) * rate / rate - 1 / rate``): the plain ``seconds - 1 / rate``
        # differs from Resolve's value in the last bit on 16 of 4,615 real sound items.
        last = sound_seconds * double / double - 1 / double
        picture_frames = None
    name_value = clip.media.get("Name")
    if not isinstance(name_value, str):
        raise _bad("Sm2MpMedia.Name is not text")
    return MediaView(
        media_id=clip.media_id,
        name=name_value,
        path=path,
        picture=clip.video is not None,
        channels=channels,
        rate_bytes=rate_bytes,
        rate=rate,
        start_seconds=start,
        last_seconds=last,
        frames=picture_frames,
        sound_seconds=sound_seconds,
    )


def timemap(seconds: float) -> bytes:
    """``MediaTimemapBA`` of a file played at full speed: ``0x02`` and the time of its last
    frame as a big-endian double (the form every full-speed item in the corpus has)."""
    return b"\x02" + struct.pack(">d", seconds)
