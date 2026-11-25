"""The kind of a media file: which template clips may serve it.

Two files share a kind when every property below is equal. Each is read from the measured facts
(:func:`kind_of_facts`) and, where Resolve records it, from a saved clip's rows
(:func:`kind_of_rows`); the rows decide which template may serve which file.

* ``media``: ``audio`` (no picture), ``still`` (an image file) or ``video``.
* ``container``: the file name's extension, lower case.
* ``video_codec``: the picture's four-character code as Resolve stores it (``Clip`` field 5;
  ffprobe's ``codec_tag_string``), None for a still.
* ``picture``: the pixel layout class (:data:`aar.drp.mediapool.formulas.PICTURE_CLASSES`); on
  the rows it is the class whose frame size rule gives the clip's ``FrameSize``.
* ``transposed``: a 90 or 270 degree display rotation (``IsResolutionTransposed``).
* ``timecode``: whether the file has a start timecode (``Time.Timecode`` present).
* ``audio``: per audio stream, Resolve's codec name, ``BitDepth``, channels, sample rate and
  whether a channel layout is named (``ChannelLayout`` present).
* ``look``: the properties Resolve keeps only inside the unexplained ``Radiometry`` blob
  (``pix_fmt``, the colour tags, the rotation). The rows cannot be read for them, so a template
  proves its ``look`` only through its own file's facts; without them :func:`matches` says the
  look is unverified.

:data:`KINDS` lists the kinds whose rows the writer may author, with the evidence: ``verified``
kinds reproduced Resolve's own rows leave-one-out on the private answer key (the number of files
checked), ``unverified`` kinds had a single member, so nothing could be checked, or have an
unexplained field (said in the note). Every other kind is refused.

The kind key leaves out what does not change which fields are copied (resolution, frame rate,
profile, sample rate...), but the evidence for a kind covers only the files it was checked on.
Each verified kind records that **envelope** (:func:`envelope_of`, derived from the corpus by
``scripts/media_scoreboard.py``): a file of a verified kind outside it is authored only as
unverified, with the property named.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from aar.drp.mediapool import blobtree as bt
from aar.drp.mediapool import formulas as fm
from aar.drp.mediapool.errors import MediaPoolError
from aar.drp.mediapool.facts import MediaFacts
from aar.drp.mediapool.templates import ClipRows
from aar.media.probe import is_drop_frame

__all__ = [
    "ENVELOPE_PROPERTIES",
    "KINDS",
    "STILL_EXTENSIONS",
    "AudioKind",
    "KindKey",
    "KindStatus",
    "envelope_of",
    "kind_of_facts",
    "kind_of_rows",
    "matches",
    "outside_envelope",
]

STILL_EXTENSIONS = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp", "dpx", "exr"})


@dataclass(frozen=True, order=True)
class AudioKind:
    codec: str
    bit_depth: int
    channels: int
    sample_rate: int
    layout_named: bool


@dataclass(frozen=True)
class KindKey:
    media: str
    container: str
    video_codec: str | None
    picture: str | None
    transposed: bool
    timecode: bool
    audio: tuple[AudioKind, ...]
    look: tuple[str, ...] | None = None

    def without_look(self) -> KindKey:
        return replace(self, look=None)

    def label(self) -> str:
        parts = [self.media, self.container]
        if self.video_codec:
            parts.append(self.video_codec)
        if self.picture:
            parts.append(self.picture)
        if self.transposed:
            parts.append("transposed")
        parts.append("timecode" if self.timecode else "no timecode")
        for a in self.audio:
            layout = "+layout" if a.layout_named else ""
            parts.append(f"{a.codec}/{a.bit_depth}/{a.channels}ch/{a.sample_rate}{layout}")
        if self.look is not None:
            parts.append("look=" + ",".join(self.look))
        return " ".join(parts)


#: The properties a kind's evidence is recorded for (:func:`envelope_of`).
ENVELOPE_PROPERTIES = (
    "resolution",
    "frame_rate",
    "drop_frame",
    "bit_depth",
    "chroma",
    "video_profile",
    "audio",
    "audio_profile",
)
#: ``pix_fmt`` -> (bit depth, chroma class).
_PIXELS: dict[str, tuple[str, str]] = {
    "yuv420p": ("8", "4:2:0"),
    "yuvj420p": ("8", "4:2:0"),
    "yuv420p10le": ("10", "4:2:0"),
    "yuv422p10le": ("10", "4:2:2"),
    "yuva444p12le": ("12", "4:4:4:4"),
    "rgba": ("8", "RGBA"),
    "rgb24": ("8", "RGB"),
    "gray": ("8", "gray"),
}

Envelope = Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class KindStatus:
    """A kind's evidence. ``envelope`` maps each of :data:`ENVELOPE_PROPERTIES` to the values
    the checked files had; None means no envelope is recorded (the gate's own runs), which
    :data:`KINDS` never uses for a verified kind."""

    status: str  # "verified" or "unverified"
    files: int
    note: str = ""
    envelope: Envelope | None = field(default=None, compare=False)


def envelope_of(facts: MediaFacts) -> dict[str, str]:
    """The values of :data:`ENVELOPE_PROPERTIES` for a measured file (``-`` where a property
    does not apply). Rates are fractions as ffprobe gives them (``30000/1001``)."""
    video = fm.video_stream(facts)
    out = dict.fromkeys(ENVELOPE_PROPERTIES, "-")
    if video is not None:
        if video.width is not None and video.height is not None:
            out["resolution"] = f"{video.width}x{video.height}"
        depth, chroma = _PIXELS.get(video.pix_fmt or "", (f"?{video.pix_fmt}", f"?{video.pix_fmt}"))
        out["bit_depth"], out["chroma"] = depth, chroma
        out["video_profile"] = video.profile or "-"
        if facts.extension not in STILL_EXTENSIONS:
            rate = fm.video_rate(facts)
            out["frame_rate"] = f"{rate.numerator}/{rate.denominator}"
            timecode = fm.start_timecode(facts)
            if timecode is not None:
                out["drop_frame"] = "drop-frame" if is_drop_frame(timecode) else "non-drop"
    audio = fm.audio_streams(facts)
    if audio:
        out["audio"] = "+".join(f"{s.sample_rate} Hz {s.channels} ch" for s in audio)
        out["audio_profile"] = "+".join(s.profile or "-" for s in audio)
    return out


def outside_envelope(status: KindStatus, facts: MediaFacts) -> list[str]:
    """Why ``facts`` lies outside the evidence for its kind, one reason per property; empty if
    it lies inside (or no envelope is recorded)."""
    if status.envelope is None:
        return []
    values = envelope_of(facts)
    reasons = []
    for prop in ENVELOPE_PROPERTIES:
        allowed = status.envelope.get(prop)
        if allowed is None or values[prop] not in allowed:
            checked = ", ".join(sorted(allowed)) if allowed else "nothing"
            reasons.append(
                f"the file's {prop.replace('_', ' ')} {values[prop]} lies outside the evidence "
                f"for its kind (checked: {checked})"
            )
    return reasons


def _look(facts: MediaFacts) -> tuple[str, ...]:
    video = fm.video_stream(facts)
    if video is None:
        return ()
    return (
        video.pix_fmt or "-",
        video.color_space or "-",
        video.color_transfer or "-",
        video.color_primaries or "-",
        video.color_range or "-",
        str(fm.rotation_of(video)),
    )


def _audio_kinds_of_facts(facts: MediaFacts) -> tuple[AudioKind, ...]:
    out = []
    for stream in fm.audio_streams(facts):
        try:
            codec, depth = fm.audio_codec_name(stream), fm.audio_bit_depth(stream)
        except MediaPoolError:
            codec, depth = f"?{stream.codec_name}", 0
        rate = int(stream.sample_rate) if stream.sample_rate is not None else 0
        out.append(
            AudioKind(codec, depth, stream.channels or 0, rate, stream.channel_layout is not None)
        )
    return tuple(out)


def kind_of_facts(facts: MediaFacts) -> KindKey:
    """The kind of a measured file. Properties it cannot read become values no template has
    (``?<codec>``), so such a file finds no template rather than a wrong one."""
    container = facts.extension
    video = fm.video_stream(facts)
    if video is None:
        media = "audio"
    elif container in STILL_EXTENSIONS:
        media = "still"
    else:
        media = "video"
    codec = picture = None
    transposed = False
    if video is not None:
        tag = video.codec_tag_string or ""
        if media == "video":
            plain = len(tag) == 4 and tag.isascii() and tag.isalnum()
            codec = tag if plain else f"?{video.codec_name}"
            known = fm.PICTURE_CLASSES.get(video.pix_fmt or "")
            picture = known[0] if known else f"?{video.pix_fmt}"
        transposed = fm.rotation_of(video) in (90, 270)
    return KindKey(
        media=media,
        container=container,
        video_codec=codec,
        picture=picture,
        transposed=transposed,
        timecode=media == "video" and fm.start_timecode(facts) is not None,
        audio=_audio_kinds_of_facts(facts),
        look=_look(facts),
    )


def _clip_fields(blob: object, what: str) -> dict[int, object]:
    _, payload = bt.message_of(blob, what)
    out: dict[int, object] = {}
    for number, _wire, value in bt.pb_decode(payload, what):
        out.setdefault(number, value)
    return out


def _file_name(clip: ClipRows) -> str:
    if clip.video is None and not clip.audio:
        raise _bad_rows("the clip has neither a video nor an audio info")
    row = clip.video if clip.video is not None else clip.audio[0]
    name = _clip_fields(row.get("Clip"), "Clip").get(2)
    if not isinstance(name, bytes):
        raise _bad_rows("the Clip blob has no file name")
    try:
        return name.decode("utf-8")
    except UnicodeDecodeError:
        raise _bad_rows("the Clip blob's file name is not UTF-8") from None


def _bad_rows(why: str) -> MediaPoolError:
    return MediaPoolError("bad_template", why, "choose another template")


def _picture_of_rows(geometry: bt.KdEntries) -> tuple[str | None, bool]:
    res = bt.kd_get(geometry, "Resolution")
    size = bt.kd_get(geometry, "FrameSize")
    transposed = bt.kd_get(geometry, "IsResolutionTransposed")
    is_transposed = transposed is not None and transposed[3] is True
    if (
        res is None
        or size is None
        or not isinstance(res[3], bytes)
        or len(res[3]) != 16
        or isinstance(size[3], bool)
        or not isinstance(size[3], int)
    ):
        raise _bad_rows("Geometry has no Resolution or FrameSize")
    from aar.drp import blobs

    width, height = blobs.decode_resolution(res[3])
    if is_transposed:
        width, height = height, width
    for name, bpp in {v[0]: v[1] for v in fm.PICTURE_CLASSES.values()}.items():
        try:
            if width and height and fm.frame_size(width, height, bpp) == size[3]:
                return name, is_transposed
        except MediaPoolError:
            continue
    return f"?{size[3]}/{width}x{height}", is_transposed


def kind_of_rows(clip: ClipRows) -> KindKey:
    """The kind of a saved clip, from its rows (``look`` is None: rows do not record it)."""
    name = _file_name(clip)
    stem, dot, ext = name.rpartition(".")
    container = ext.lower() if dot and stem else ""
    codec = picture = None
    transposed = timecode = False
    if clip.video is None:
        media = "audio"
    else:
        media = "still" if container in STILL_EXTENSIONS else "video"
        geometry = bt.kd_decode(clip.video.get("Geometry"), "Geometry")
        picture, transposed = _picture_of_rows(geometry)
        time = bt.kd_decode(clip.video.get("Time"), "Time")
        if media == "video":
            tag = _clip_fields(clip.video.get("Clip"), "Clip").get(5)
            codec = tag.decode("ascii", "replace") if isinstance(tag, bytes) else None
            timecode = bt.kd_get(time, "Timecode") is not None
        else:
            picture = None
    audio = []
    for info in clip.audio:
        tracks = bt.kd_decode(info.get("TracksBA"), "TracksBA")
        if [e[0] for e in tracks] != ["0"] or not isinstance(tracks[0][3], bytes):
            raise _bad_rows("TracksBA does not hold one track")
        track = bt.kd_decode(tracks[0][3], "TracksBA track")

        def number(key: str, entries: bt.KdEntries = track) -> int:
            entry = bt.kd_get(entries, key)
            value = None if entry is None else entry[3]
            if isinstance(value, bool) or not isinstance(value, int):
                raise _bad_rows(f"the audio track's {key} is not a whole number")
            return value

        codec_name = bt.kd_get(track, "CodecName")
        if codec_name is None or not isinstance(codec_name[3], str):
            raise _bad_rows("the audio track has no codec name")
        audio.append(
            AudioKind(
                codec=codec_name[3],
                bit_depth=number("BitDepth"),
                channels=number("NumChannels"),
                sample_rate=number("SampleRate"),
                layout_named=bt.kd_get(track, "ChannelLayout") is not None,
            )
        )
    return KindKey(media, container, codec, picture, transposed, timecode, tuple(audio))


def matches(
    facts_kind: KindKey, template_rows_kind: KindKey, template_facts: MediaFacts | None
) -> str | None:
    """Whether a template may serve a file: None if it may with every property proven; the
    reason if it may but its ``look`` is not proven (no facts for the template's file). Raises
    ``kind_mismatch`` when the rows, or the template's facts, show another kind."""
    if facts_kind.without_look() != template_rows_kind.without_look():
        raise MediaPoolError(
            "kind_mismatch",
            f"the template is a {template_rows_kind.label()}, "
            f"the file a {facts_kind.without_look().label()}",
            "give a template clip of the same kind",
        )
    if template_facts is None:
        if not facts_kind.look:
            return None  # no picture: nothing is kept in Radiometry
        return (
            "the template's colour and pixel format (Radiometry) are not proven: "
            "no facts for its file"
        )
    other = kind_of_facts(template_facts)
    if other != facts_kind:
        raise MediaPoolError(
            "kind_mismatch",
            f"the template's file is a {other.label()}, the file a {facts_kind.label()}",
            "give a template clip whose file has the same pixel format, colour tags and rotation",
        )
    return None


_BT709 = ("bt709", "bt709", "bt709", "tv", "0")


def _pcm(depth: int, channels: int, layout: bool) -> AudioKind:
    return AudioKind("Linear PCM", depth, channels, 48000, layout)


def _audio(container: str, *streams: AudioKind) -> KindKey:
    return KindKey("audio", container, None, None, False, False, streams, ())


def _video(
    codec: str, picture: str, timecode: bool, look: tuple[str, ...], *audio: AudioKind
) -> KindKey:
    return KindKey(
        "video", "mp4" if codec == "avc1" else "mov", codec, picture, False, timecode, audio, look
    )


_AAC_44 = AudioKind("AAC", 3, 2, 44100, True)
_AAC_48 = AudioKind("AAC", 3, 2, 48000, True)


def _envelope(
    *,
    resolution: str = "-",
    frame_rate: str = "-",
    drop_frame: str = "-",
    bit_depth: str = "-",
    chroma: str = "-",
    video_profile: str = "-",
    audio: str = "-",
    audio_profile: str = "-",
) -> Envelope:
    """An envelope of one value per property (every verified kind's evidence is that narrow)."""
    values = {
        "resolution": resolution,
        "frame_rate": frame_rate,
        "drop_frame": drop_frame,
        "bit_depth": bit_depth,
        "chroma": chroma,
        "video_profile": video_profile,
        "audio": audio,
        "audio_profile": audio_profile,
    }
    return {prop: frozenset({values[prop]}) for prop in ENVELOPE_PROPERTIES}


def _sound(audio: str) -> Envelope:
    return _envelope(audio=audio)


#: Envelope values the scoreboard derived from the checked files (generic values only).
_UHD, _HD, _NTSC, _FILM = "3840x2160", "1920x1080", "30000/1001", "24000/1001"
#: Why ProRes 4444 is not verified although its files compare equal.
_PRORES_4444_NOTE = (
    "Sm2MpMedia.FieldsBlob field #1.#3 is on every ProRes 4444 clip of the answer key and on "
    "almost no other clip, with values no rule derives; the writer leaves it out and the gate "
    "excludes it, so until it is explained the kind is not verified"
)

#: The kinds the writer may author, with their evidence: the corpus gate on two Resolve 21.1
#: saves (docs/DESIGN.md, "Media-pool rows (S25c1)"). ``files`` is the number of files whose rows
#: were authored leave-one-out and equalled Resolve's on every compared field (files changed
#: after Resolve read them are not counted). ``envelope`` is what those files covered, as the
#: scoreboard derives it (:func:`envelope_of`). An ``unverified`` kind had one file (its derived
#: fields equalled Resolve's when it was its own template, which proves nothing about the fields
#: copied from a template) or holds a field nothing explains.
KINDS: dict[KindKey, KindStatus] = {
    _video("avc1", "8-bit 4:2:0", True, ("yuv420p", *_BT709), _pcm(1, 2, False)): KindStatus(
        "verified",
        68,
        envelope=_envelope(
            resolution=_UHD,
            frame_rate=_FILM,
            drop_frame="non-drop",
            bit_depth="8",
            chroma="4:2:0",
            video_profile="High",
            audio="48000 Hz 2 ch",
        ),
    ),
    _audio("wav", _pcm(2, 2, False)): KindStatus("verified", 29, envelope=_sound("48000 Hz 2 ch")),
    _audio("wav", _pcm(2, 2, True)): KindStatus("verified", 29, envelope=_sound("48000 Hz 2 ch")),
    _video("apch", "10-bit 4:2:2", True, ("yuv422p10le", *_BT709)): KindStatus(
        "verified",
        17,
        envelope=_envelope(
            resolution=_HD,
            frame_rate=_NTSC,
            drop_frame="non-drop",
            bit_depth="10",
            chroma="4:2:2",
            video_profile="HQ",
        ),
    ),
    _video("ap4h", "12-bit 4:4:4:4", False, ("yuva444p12le", "-", "-", "-", "tv", "0")): KindStatus(
        "unverified",
        7,
        _PRORES_4444_NOTE,
        envelope=_envelope(
            resolution=_HD, frame_rate=_NTSC, bit_depth="12", chroma="4:4:4:4", video_profile="4444"
        ),
    ),
    _audio("wav", _pcm(2, 1, False)): KindStatus("verified", 6, envelope=_sound("48000 Hz 1 ch")),
    _audio("wav", _pcm(2, 1, True)): KindStatus("verified", 4, envelope=_sound("48000 Hz 1 ch")),
    KindKey(
        "still", "png", None, None, False, False, (), ("rgba", "gbr", "-", "-", "pc", "0")
    ): KindStatus(
        "verified",
        3,
        "a still whose name holds a digit is refused",
        envelope=_envelope(resolution=_UHD, bit_depth="8", chroma="RGBA"),
    ),
    _audio("mp3", AudioKind("MP3", 3, 2, 44100, True)): KindStatus(
        "verified", 2, envelope=_sound("44100 Hz 2 ch")
    ),
    _audio("m4a", _AAC_44): KindStatus("unverified", 1, "one AAC .m4a file (with cover art)"),
    _audio("mp3", AudioKind("MP3", 3, 2, 48000, True)): KindStatus(
        "unverified", 1, "one 48 kHz .mp3 file (with cover art)"
    ),
    _audio("wav", _pcm(1, 2, False)): KindStatus("unverified", 1, "one 16-bit .wav file"),
    _video(
        "apch", "10-bit 4:2:2", False, ("yuv422p10le", "bt709", "-", "-", "tv", "0")
    ): KindStatus(
        "unverified", 1, "one ProRes file without a timecode or transfer and primaries tags"
    ),
    _video("hvc1", "10-bit 4:2:0", True, ("yuv420p10le", *_BT709), _AAC_48): KindStatus(
        "unverified", 1, "one 10-bit HEVC file Resolve rendered"
    ),
}
