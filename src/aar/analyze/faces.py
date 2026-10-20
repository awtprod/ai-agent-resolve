"""``analyze.faces``: faces in sampled video frames, from OpenCV's YuNet detector.

**Model.** ``--model`` or ``$AAR_YUNET_MODEL``: an existing regular ``.onnx`` file, never
downloaded. Its sha256 and size go in the output header.

**Decode.** ffmpeg writes ``rawvideo`` ``bgr24`` to a pipe through
``fps=<sample fps>:start_time=0:round=up`` and a scale to the frame size: the display size
(ffprobe's size with the sample aspect ratio applied, and width and height swapped for a 90 or
270 degree rotation, because ffmpeg autorotates) shrunk so the long side is at most
``max_side``, aspect kept, rounded down to even numbers, never enlarged. Exactly ``w * h * 3``
bytes make a frame; a short final read is reported as ``truncated_tail``, never parsed. The
call has a timeout that grows with the media length, and a failed or killed ffmpeg is a typed
error with stderr's tail.

**Detect.** ``cv2.FaceDetectorYN.create(model, "", (w, h), score, nms, top_k)``, then
``detect(frame)``, which gives None or an N x 15 array: the box (x, y, w, h), five landmarks
(right eye, left eye, nose tip, right and left mouth corner) as x, y pairs, and the score. The
detector is injectable, so tests need no OpenCV.

**Normalise.** Coordinates are divided by the frame size and clamped to [0, 1]; a face whose
clamped box has no area is dropped (``zero-area``), and a row with NaN or inf is dropped
(``non-finite``); both are counted. Faces are sorted largest first and rounded to 4 decimals.

**Time.** Sample i is at file time ``i / sample_fps`` (an exact fraction); recording time adds
the ``RecordingOffset``, as the transcript does. Sample i is the source frame on screen at that
time: the latest frame whose timestamp is at or before it (``round=up``). The ``fps`` filter's
default rounding would instead take the last frame within half a sample interval either side,
up to 0.25 s late at 2 fps. The picture can be up to one source frame older than the sample
time, so the header records ``time_tolerance_s``, one source frame.

**Resume.** One checkpoint record per frame (``aar.analyze.common``). A resumed run seeks the
input to ``k / sample_fps`` and numbers on from ``k``, the first frame not on file; the output
is byte-identical to an uninterrupted run's.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from aar.analyze.common import (
    RecordingOffset,
    append_record,
    check_header,
    checkpoint_header,
    checkpoint_path,
    finish_checkpoint,
    input_identity,
    is_number,
    read_checkpoint,
    start_checkpoint,
    trim_checkpoint,
    write_atomic,
)
from aar.analyze.errors import AnalyzeError, from_media_error, need_cv2, need_numpy
from aar.media import MediaError, timeout_for
from aar.media.probe import parse_rational
from aar.media.tools import input_url, require_regular_file, run_tool

SCHEMA = "aar.faces/1"
MODEL_ENV = "AAR_YUNET_MODEL"
_KIND = "aar.analyze.faces"
_DIGITS = 4
_DROP_REASONS = ("non-finite", "zero-area")
_BASE = ["-hide_banner", "-nostdin", "-nostats", "-v", "error"]


@dataclass(frozen=True)
class FaceSettings:
    sample_fps: Fraction = Fraction(2)
    max_side: int = 640
    score: float = 0.6
    nms: float = 0.3
    top_k: int = 50
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not (isinstance(self.sample_fps, Fraction) and 0 < self.sample_fps <= 120):
            raise ValueError("sample_fps must be a fraction above 0 and at most 120")
        if not (isinstance(self.max_side, int) and self.max_side >= 16):
            raise ValueError("max_side must be a whole number of at least 16")
        for name, value in (("score", self.score), ("nms", self.nms)):
            if not (math.isfinite(value) and 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")
        if not (isinstance(self.top_k, int) and self.top_k >= 1):
            raise ValueError("top_k must be a whole number of at least 1")
        if self.timeout_s is not None and not (
            math.isfinite(self.timeout_s) and self.timeout_s > 0
        ):
            raise ValueError("timeout_s must be a finite number of seconds above 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_fps": _ratio(self.sample_fps),
            "max_side": self.max_side,
            "score": self.score,
            "nms": self.nms,
            "top_k": self.top_k,
        }


def _ratio(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


# ---------------------------------------------------------------- model


@dataclass(frozen=True)
class YunetModel:
    path: Path
    sha256: str
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "size": self.size}


def resolve_yunet(model: str | os.PathLike[str] | None) -> YunetModel:
    """The YuNet model from ``model`` or ``$AAR_YUNET_MODEL``: an existing regular file."""
    text = os.fspath(model) if model is not None else os.environ.get(MODEL_ENV, "").strip()
    hint = f"pass --model PATH or set {MODEL_ENV} to a YuNet .onnx file; aar never downloads"
    if not text:
        raise AnalyzeError("model-missing", "no YuNet model given", hint=hint)
    path = Path(text).expanduser()
    if not path.is_file():
        raise AnalyzeError("model-missing", f"{path} is not an existing regular file", hint=hint)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return YunetModel(path=path.resolve(), sha256=digest.hexdigest(), size=path.stat().st_size)


class Detector(Protocol):
    def detect(self, frame: Any) -> Any:
        """None, or an N x 15 array in frame pixels (box, five landmarks, score)."""
        ...


#: (model path, (width, height), score, nms, top_k) -> a detector for frames of that size.
DetectorFactory = Callable[[Path, tuple[int, int], float, float, int], Detector]


class _YunetDetector:
    def __init__(self, inner: Any) -> None:
        self.inner = inner

    def detect(self, frame: Any) -> Any:
        _, faces = self.inner.detect(frame)  # OpenCV returns (retval, faces)
        return faces


def yunet_factory(
    model: Path, size: tuple[int, int], score: float, nms: float, top_k: int
) -> Detector:
    cv2 = need_cv2()
    try:
        inner = cv2.FaceDetectorYN.create(str(model), "", size, score, nms, top_k)
    except Exception as exc:  # OpenCV raises cv2.error on a bad model
        raise AnalyzeError("model-invalid", f"cannot load the YuNet model {model}: {exc}") from exc
    return _YunetDetector(inner)


# ---------------------------------------------------------------- video


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    rotation: int
    sample_aspect: Fraction
    fps: Fraction | None
    duration_s: float | None

    @property
    def display_size(self) -> tuple[int, int]:
        """Width and height as ffmpeg emits them: the aspect ratio applied, then rotated."""
        width = max(1, round(self.width * self.sample_aspect))
        height = self.height
        if self.rotation % 180 == 90:
            width, height = height, width
        return width, height

    def to_dict(self) -> dict[str, Any]:
        width, height = self.display_size
        return {
            "width": self.width,
            "height": self.height,
            "rotation": self.rotation,
            "sample_aspect_ratio": _ratio(self.sample_aspect),
            "display_width": width,
            "display_height": height,
            "fps": None if self.fps is None else _ratio(self.fps),
            "duration_s": self.duration_s,
        }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def probe_video(path: Path, timeout_s: float = 30.0) -> VideoInfo:
    """Size, rotation, aspect, frame rate and duration of the first video stream."""
    entries = (
        "stream=width,height,sample_aspect_ratio,r_frame_rate,avg_frame_rate,duration"
        ":stream_side_data=rotation:stream_tags=rotate:format=duration"
    )
    result = run_tool(
        "ffprobe",
        [
            *("-hide_banner", "-v", "error", "-select_streams", "v:0"),
            *("-show_entries", entries, "-of", "json", input_url(path)),
        ],
        timeout_s=timeout_s,
        stdout_limit=1024 * 1024,
    )
    try:
        data = json.loads(result.stdout.decode("utf-8", errors="replace") or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe printed invalid JSON: {exc}", tool="ffprobe") from None
    streams = data.get("streams") if isinstance(data, dict) else None
    if not streams or not isinstance(streams[0], dict):
        raise AnalyzeError("no-video", f"{path} has no video stream", hint="faces need video")
    stream = streams[0]
    width, height = stream.get("width"), stream.get("height")
    if not (isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0):
        raise AnalyzeError("no-video", f"{path}: the video stream has no size")
    rotation = 0
    for side in stream.get("side_data_list") or []:
        if isinstance(side, dict) and _number(side.get("rotation")) is not None:
            rotation = round(float(side["rotation"]))
    tags = stream.get("tags")
    if rotation == 0 and isinstance(tags, dict) and _number(tags.get("rotate")) is not None:
        rotation = round(float(tags["rotate"]))
    sar = parse_rational(stream.get("sample_aspect_ratio"))  # "0:1" (unknown) gives None
    if sar is None and isinstance(stream.get("sample_aspect_ratio"), str):
        sar = parse_rational(stream["sample_aspect_ratio"].replace(":", "/"))
    fps = parse_rational(stream.get("avg_frame_rate")) or parse_rational(stream.get("r_frame_rate"))
    fmt = data.get("format") if isinstance(data.get("format"), dict) else {}
    duration = _number(stream.get("duration")) or _number(fmt.get("duration"))
    return VideoInfo(
        width=width,
        height=height,
        rotation=rotation % 360,
        sample_aspect=sar or Fraction(1),
        fps=fps,
        duration_s=duration,
    )


def frame_size(display: tuple[int, int], max_side: int) -> tuple[int, int]:
    """The long side at most ``max_side``, aspect kept, even, never larger than the display."""
    width, height = display
    scale = min(Fraction(1), Fraction(max_side, max(width, height)))
    return (max(2, math.floor(width * scale / 2) * 2), max(2, math.floor(height * scale / 2) * 2))


class _Frames:
    """Cuts ffmpeg's rawvideo stream into whole frames; keeps the short tail, never parses it."""

    def __init__(self, frame_bytes: int, on_frame: Callable[[bytes], None]) -> None:
        self.frame_bytes = frame_bytes
        self.on_frame = on_frame
        self.buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self.buf += chunk
        size = self.frame_bytes
        while len(self.buf) >= size:
            frame = bytes(self.buf[:size])
            del self.buf[:size]
            self.on_frame(frame)

    @property
    def tail(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------- normalise


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def normalise(rows: Any, width: int, height: int) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Detector rows (frame pixels) to faces in [0, 1], largest first; drops counted by reason."""
    dropped: Counter[str] = Counter()
    faces: list[tuple[float, float, dict[str, Any]]] = []
    for row in [] if rows is None else rows:
        values = [float(v) for v in row[:15]]
        if len(values) < 15 or not all(math.isfinite(v) for v in values):
            dropped["non-finite"] += 1
            continue
        x, y, w, h = values[:4]
        x0, y0 = _clamp(x / width), _clamp(y / height)
        x1, y1 = _clamp((x + w) / width), _clamp((y + h) / height)
        if x1 <= x0 or y1 <= y0:
            dropped["zero-area"] += 1
            continue
        marks = [
            [round(_clamp(values[4 + 2 * k] / width), _DIGITS),
             round(_clamp(values[5 + 2 * k] / height), _DIGITS)]
            for k in range(5)
        ]  # fmt: skip
        face = {
            "box": [round(v, _DIGITS) for v in (x0, y0, x1 - x0, y1 - y0)],
            "score": round(values[14], _DIGITS),
            "landmarks": marks,
        }
        faces.append(((x1 - x0) * (y1 - y0), values[14], face))
    faces.sort(key=lambda item: (-item[0], -item[1]))
    return [face for _, _, face in faces], dropped


# ---------------------------------------------------------------- checkpoint records


def _valid_face(face: Any) -> bool:
    if not (isinstance(face, dict) and is_number(face.get("score"))):
        return False
    box, marks = face.get("box"), face.get("landmarks")
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(is_number(v) for v in box)
        and isinstance(marks, list)
        and len(marks) == 5
        and all(isinstance(m, list) and len(m) == 2 and all(is_number(v) for v in m) for m in marks)
    )


def _check_frame(record: Any, previous: Any) -> str | None:
    if not (isinstance(record, dict) and isinstance(record.get("index"), int)):
        return "is not a frame record"
    want = 0 if previous is None else previous["index"] + 1
    if record["index"] != want:
        return f"has index {record['index']}, expected {want}"
    faces, dropped = record.get("faces"), record.get("dropped")
    if not (isinstance(faces, list) and all(_valid_face(f) for f in faces)):
        return "has malformed faces"
    if not (
        isinstance(dropped, dict)
        and all(k in _DROP_REASONS and isinstance(v, int) for k, v in dropped.items())
    ):
        return "has malformed drop counts"
    return None


# ---------------------------------------------------------------- the worker


def _seconds(value: Fraction) -> str:
    return f"{float(value):.9f}".rstrip("0").rstrip(".") or "0"


def analyse_faces(
    media: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    model: str | os.PathLike[str] | None = None,
    settings: FaceSettings | None = None,
    offset: RecordingOffset | None = None,
    recording: str | None = None,
    resume: bool = False,
    detector_factory: DetectorFactory | None = None,
) -> dict[str, Any]:
    """Detect faces in ``media`` and write ``aar.faces/1`` to ``out``; return a short report.

    Raises ``AnalyzeError`` (``aar.analyze.errors`` has the types and exit codes).
    """
    s = settings or FaceSettings()
    offset = offset or RecordingOffset()
    out_path = Path(out)
    ckpt = checkpoint_path(out_path)
    try:
        media_path = require_regular_file(media, tool="ffmpeg")
    except MediaError as exc:
        raise from_media_error(exc, "input") from None
    yunet = resolve_yunet(model)
    factory = detector_factory or yunet_factory
    if detector_factory is None:
        need_cv2()
    np = need_numpy()
    try:
        info = probe_video(media_path)
    except MediaError as exc:
        raise from_media_error(exc, f"input {media_path}") from None
    width, height = frame_size(info.display_size, s.max_side)
    header = checkpoint_header(
        _KIND,
        media_path,
        {**s.to_dict(), "model_sha256": yunet.sha256, "frame_size": [width, height]},
    )
    records: list[dict[str, Any]] = []
    good = 0
    if ckpt.exists():
        if not resume:
            raise AnalyzeError(
                "checkpoint-exists",
                f"checkpoint {ckpt} exists from an earlier run",
                hint=f"pass --resume to continue it, or delete {ckpt} to start over",
            )
        found, records, good = read_checkpoint(ckpt, kind=_KIND, what="frame", check=_check_frame)
        check_header(ckpt, found, header)
    resumed = ckpt.exists()
    first = len(records)
    detector = factory(yunet.path, (width, height), s.score, s.nms, s.top_k)
    if resumed:
        trim_checkpoint(ckpt, good)
    else:
        start_checkpoint(ckpt, header)
    frame_bytes = width * height * 3
    timeout = s.timeout_s or timeout_for(info.duration_s, per_media_s=1.0)
    args = [*_BASE]
    if first:
        args += ["-ss", _seconds(Fraction(first) / s.sample_fps)]
    vf = f"fps=fps={_ratio(s.sample_fps)}:start_time=0:round=up,scale={width}:{height},setsar=1"
    args += ["-i", input_url(media_path), "-map", "0:v:0", "-an", "-sn", "-dn"]
    args += ["-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
    new: list[dict[str, Any]] = []
    with open(ckpt, "ab") as fh:

        def on_frame(data: bytes) -> None:
            frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
            try:
                rows = detector.detect(frame)
            except Exception as exc:
                raise AnalyzeError(
                    "detection-failed",
                    f"the face detector failed on frame {first + len(new)}: "
                    f"{type(exc).__name__}: {exc}",
                    hint=f"the checkpoint {ckpt} keeps the frames before it; run again with "
                    "--resume",
                ) from exc
            faces, dropped = normalise(rows, width, height)
            record = {"index": first + len(new), "faces": faces, "dropped": dict(dropped)}
            append_record(fh, record)
            new.append(record)

        frames = _Frames(frame_bytes, on_frame)
        try:
            run_tool("ffmpeg", args, timeout_s=timeout, stdout_sink=frames.feed)
        except MediaError as exc:
            raise from_media_error(exc, f"input {media_path}") from None
    records += new
    document = _document(records, s, info, (width, height), yunet, media_path, offset, recording)
    document["summary"]["truncated_tail"] = {"bytes": frames.tail} if frames.tail else None
    try:
        write_atomic(out_path, json.dumps(document, indent=1) + "\n")
    except OSError as exc:
        raise AnalyzeError(
            "output-error",
            f"cannot write {out_path}: {exc}",
            hint=f"the checkpoint {ckpt} keeps every frame; fix the output and run --resume",
        ) from exc
    finish_checkpoint(ckpt)
    return {
        "ok": True,
        "output": str(out_path),
        "input": str(media_path),
        "model": yunet.to_dict(),
        "frame_size": [width, height],
        "summary": document["summary"],
        "resume": {"resumed": resumed, "from_index": first if resumed else None},
    }


def _document(
    records: list[dict[str, Any]],
    s: FaceSettings,
    info: VideoInfo,
    size: tuple[int, int],
    yunet: YunetModel,
    media: Path,
    offset: RecordingOffset,
    recording: str | None,
) -> dict[str, Any]:
    frames = []
    dropped: Counter[str] = Counter()
    for record in records:
        file_s = Fraction(record["index"]) / s.sample_fps
        frames.append(
            {
                "file_s": round(float(file_s), 6),
                "recording_s": round(float(file_s) + offset.offset_s, 6),
                "faces": record["faces"],
            }
        )
        dropped.update(record["dropped"])
    tolerance = None if info.fps is None else round(float(1 / info.fps), 6)
    return {
        "schema": SCHEMA,
        "header": {
            "model": yunet.to_dict(),
            "settings": s.to_dict(),
            "input": {"path": str(media), **input_identity(media)},
            "source": info.to_dict(),
            "sample_fps": _ratio(s.sample_fps),
            "frame_size": list(size),
            "time_tolerance_s": tolerance,
            "offset": offset.to_dict(),
            "recording": recording if recording is not None else offset.recording,
        },
        "frames": frames,
        "summary": {
            "frames": len(frames),
            "frames_with_faces": sum(1 for f in frames if f["faces"]),
            "max_faces": max((len(f["faces"]) for f in frames), default=0),
            "dropped": {reason: dropped.get(reason, 0) for reason in _DROP_REASONS},
            "truncated_tail": None,
        },
    }


__all__ = [
    "MODEL_ENV",
    "SCHEMA",
    "DetectorFactory",
    "FaceSettings",
    "VideoInfo",
    "YunetModel",
    "analyse_faces",
    "frame_size",
    "normalise",
    "probe_video",
    "resolve_yunet",
    "yunet_factory",
]
