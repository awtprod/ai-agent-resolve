"""``analyze.transcript``: a word-level transcript in recording time, from faster-whisper.

Two entry points share one adapter:

- ``transcribe_file(media, out, ...)`` transcribes a whole file into an
  ``aar.plan.inputs.Transcript`` (written atomically) and returns a JSON-ready report;
- ``WhisperTranscriber`` implements ``aar.qa.transcriber.Transcriber``: it transcribes one short
  window and returns words in file seconds, for QA's join re-transcription.

**Models are never downloaded.** ``model`` is a name (``medium.en``) or a path to a CTranslate2
model directory (one holding ``model.bin``). A name is looked up under ``--model-dir``, else
``$AAR_MODEL_DIR``, in this order: ``<dir>/<name>/``, ``<dir>/faster-whisper-<name>/``, and the
Hugging Face cache layout ``<dir>/models--Systran--faster-whisper-<name>/snapshots/<rev>/``
(``refs/main``, else the only snapshot). The model directory is passed to ``WhisperModel`` with
``local_files_only=True``. A missing model is ``AnalyzeError("model-missing")``. The cache is
shared by every checkout and lives outside the worktrees; aar never creates one.

**CCN preset** (``CCN_PRESET``, every field overridable): medium.en on the CPU in int8, beam 5,
word timestamps, VAD on with a 400 ms minimum silence, ``condition_on_previous_text=False`` (one
misheard line cannot snowball), English. ``cpu_threads`` 0 is faster-whisper's default.

**Recording time.** ``rec_s = file_s + offset_s``; from a plan, ``offset_s =
recording_origin / fps`` with the exact rational fps. The ``Transcript`` model itself carries no
offset (it is ``extra="forbid"``): the report says what was applied.

**Word hygiene.** Text is stripped and empty words dropped; times are clamped to >= 0 and
``end_s >= start_s``; words are sorted by start; ``speaker`` is None.

**Hallucination guard.** A segment is dropped when ``no_speech_prob > 0.6`` and
``avg_logprob < -1.0`` (silence decoded into words), or when ``compression_ratio > 2.4`` (a
repeated phrase). These are faster-whisper's own thresholds, as parameters. Every dropped
segment is listed in the report with its times and reason.

**Resume.** Segments are appended to ``<out>.ckpt.jsonl`` (flushed and fsynced one by one). Its
header records the input's size and mtime_ns and a hash of the decode settings and the model.
``resume=True`` continues after the last complete segment; a checkpoint whose header does not
match is refused (``checkpoint-mismatch``): two runs are never spliced. Without ``resume`` an
existing checkpoint is refused (``checkpoint-exists``) rather than overwritten. With VAD off the
rest is decoded with ``clip_timestamps``. With VAD on the audio is sliced at the resume point
instead, because faster-whisper ignores ``vad_filter`` whenever ``clip_timestamps`` is set (it
only runs the VAD when ``clip_timestamps == "0"``), so a resumed run would silently lose its VAD.
The output is written to a temporary file and moved into place; the checkpoint is removed only
after that.

Audio is decoded by ffmpeg (``aar.analyze.audio``: ``AAR_FFMPEG``, a timeout, a duration cap)
to 16 kHz mono float32 and handed to faster-whisper as an array, so a clean run and a resumed
run see the same samples.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

from aar.analyze.audio import read_mono
from aar.analyze.errors import AnalyzeError, from_media_error, need_faster_whisper
from aar.media import MediaError, probe, timeout_for
from aar.media.tools import require_regular_file
from aar.plan.inputs import Transcript, Word
from aar.plan.model import EditPlan

MODEL_DIR_ENV = "AAR_MODEL_DIR"
WHISPER_RATE = 16000
CHECKPOINT_SUFFIX = ".ckpt.jsonl"
CHECKPOINT_VERSION = 1
_KIND = "aar.analyze.transcript"


@dataclass(frozen=True)
class WhisperSettings:
    """faster-whisper settings plus the hallucination guard and the input caps."""

    model: str = "medium.en"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int = 0
    beam_size: int = 5
    language: str | None = "en"
    word_timestamps: bool = True
    vad_filter: bool = True
    vad_min_silence_ms: int = 400
    condition_on_previous_text: bool = False
    no_speech_prob_max: float = 0.6
    avg_logprob_min: float = -1.0
    compression_ratio_max: float = 2.4
    max_duration_s: float = 4 * 3600.0
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be blank")
        if not self.word_timestamps:
            raise ValueError("word_timestamps must stay on: the transcript is word-level")
        for name, value, low in (
            ("cpu_threads", self.cpu_threads, 0),
            ("beam_size", self.beam_size, 1),
            ("vad_min_silence_ms", self.vad_min_silence_ms, 0),
        ):
            if not (isinstance(value, int) and not isinstance(value, bool) and value >= low):
                raise ValueError(f"{name} must be a whole number of at least {low}")
        for name, value in (
            ("no_speech_prob_max", self.no_speech_prob_max),
            ("avg_logprob_min", self.avg_logprob_min),
            ("compression_ratio_max", self.compression_ratio_max),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if not (math.isfinite(self.max_duration_s) and self.max_duration_s > 0):
            raise ValueError("max_duration_s must be a finite number of seconds above 0")
        if self.timeout_s is not None and not (
            math.isfinite(self.timeout_s) and self.timeout_s > 0
        ):
            raise ValueError("timeout_s must be a finite number of seconds above 0")

    def model_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``WhisperModel(model_dir, ...)``."""
        return {
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "local_files_only": True,
        }

    def transcribe_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for ``WhisperModel.transcribe(audio, ...)``."""
        kwargs: dict[str, Any] = {
            "language": self.language,
            "beam_size": self.beam_size,
            "word_timestamps": self.word_timestamps,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
        }
        if self.vad_filter:
            kwargs["vad_parameters"] = {"min_silence_duration_ms": self.vad_min_silence_ms}
        return kwargs

    def decode_settings(self) -> dict[str, Any]:
        """What changes the decoded segments (the checkpoint hash): not the thread count, the
        guard thresholds (applied when the output is written) or the caps."""
        return {
            "model": self.model,
            "device": self.device,
            "compute_type": self.compute_type,
            **self.transcribe_kwargs(),
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CCN_PRESET = WhisperSettings()


# ---------------------------------------------------------------- models


def _has_model(path: Path) -> bool:
    return (path / "model.bin").is_file()


def _hf_snapshot(repo: Path) -> Path | None:
    ref = repo / "refs" / "main"
    if ref.is_file():
        snap = repo / "snapshots" / ref.read_text(encoding="utf-8").strip()
        if snap.is_dir():
            return snap
    snaps = [p for p in (repo / "snapshots").glob("*") if p.is_dir()]
    return snaps[0] if len(snaps) == 1 else None


def _looks_like_path(model: str) -> bool:
    seps = {os.sep, os.altsep or os.sep, "/"}
    return any(sep in model for sep in seps) or model.startswith((".", "~"))


def resolve_model(model: str, model_dir: str | os.PathLike[str] | None = None) -> Path:
    """The local directory of ``model`` (a name or a path); never downloads.

    Raises ``AnalyzeError("model-missing")`` with a hint naming ``AAR_MODEL_DIR`` and the
    layout expected.
    """
    layout = (
        f"<dir>/{model}/model.bin, <dir>/faster-whisper-{model}/model.bin or the Hugging Face "
        f"cache layout <dir>/models--Systran--faster-whisper-{model}/snapshots/<rev>/"
    )
    hint = (
        f"set {MODEL_DIR_ENV} (or --model-dir) to the shared model directory holding {layout}, "
        "or pass the model directory itself; aar never downloads models"
    )
    if _looks_like_path(model) or Path(model).is_absolute():
        path = Path(model).expanduser()
        if _has_model(path):
            return path.resolve()
        raise AnalyzeError(
            "model-missing",
            f"{path} is not a faster-whisper model directory (no model.bin)",
            hint="point at a CTranslate2 model directory holding model.bin; aar never downloads",
        )
    root_text = os.fspath(model_dir) if model_dir is not None else os.environ.get(MODEL_DIR_ENV)
    if not root_text:
        raise AnalyzeError(
            "model-missing", f"no model directory given for model {model!r}", hint=hint
        )
    root = Path(root_text).expanduser()
    looked: list[Path] = [root / model, root / f"faster-whisper-{model}"]
    for path in looked:
        if _has_model(path):
            return path.resolve()
    repo = root / f"models--Systran--faster-whisper-{model}"
    snap = _hf_snapshot(repo) if repo.is_dir() else None
    if snap is not None and _has_model(snap):
        return snap.resolve()
    looked.append(repo / "snapshots")
    raise AnalyzeError(
        "model-missing",
        f"model {model!r} not found under {root} (looked in "
        + ", ".join(str(p) for p in looked)
        + ")",
        hint=hint,
    )


def load_model(settings: WhisperSettings, model_dir: str | os.PathLike[str] | None) -> Any:
    """``(WhisperModel, model directory)``. Raises ``missing-extra``, ``model-missing`` or
    ``model-invalid``."""
    faster_whisper = need_faster_whisper()
    path = resolve_model(settings.model, model_dir)
    try:
        model = faster_whisper.WhisperModel(str(path), **settings.model_kwargs())
    except Exception as exc:  # CTranslate2 raises plain RuntimeError/ValueError on bad models
        raise AnalyzeError(
            "model-invalid",
            f"cannot load the model at {path}: {type(exc).__name__}: {exc}",
            hint="the directory must hold a complete CTranslate2 faster-whisper model",
        ) from exc
    return model, path


# ---------------------------------------------------------------- segments and words


def segment_record(segment: Any, base_s: float) -> dict[str, Any]:
    """A faster-whisper ``Segment`` as plain JSON, times shifted by ``base_s`` to file time."""
    words = [
        [base_s + float(w.start), base_s + float(w.end), str(w.word), float(w.probability)]
        for w in (segment.words or [])
    ]
    return {
        "start": base_s + float(segment.start),
        "end": base_s + float(segment.end),
        "text": str(segment.text),
        "avg_logprob": float(segment.avg_logprob),
        "compression_ratio": float(segment.compression_ratio),
        "no_speech_prob": float(segment.no_speech_prob),
        "words": words,
    }


def guard_reason(record: dict[str, Any], settings: WhisperSettings) -> str | None:
    """Why the hallucination guard drops a segment, or None to keep it."""
    reasons = []
    nsp, logprob = record["no_speech_prob"], record["avg_logprob"]
    if nsp > settings.no_speech_prob_max and logprob < settings.avg_logprob_min:
        reasons.append(
            f"no-speech: no_speech_prob {nsp:.3f} > {settings.no_speech_prob_max:g} and "
            f"avg_logprob {logprob:.3f} < {settings.avg_logprob_min:g}"
        )
    ratio = record["compression_ratio"]
    if ratio > settings.compression_ratio_max:
        reasons.append(
            f"repetitive: compression_ratio {ratio:.3f} > {settings.compression_ratio_max:g}"
        )
    return "; ".join(reasons) or None


@dataclass
class WordCounts:
    words: int = 0
    empty: int = 0
    invalid: int = 0
    clamped: int = 0


def segment_words(record: dict[str, Any], offset_s: float, counts: WordCounts) -> list[Word]:
    """The words of one kept segment in recording time, cleaned (see the module docstring)."""
    out = []
    for start, end, text, _probability in record["words"]:
        text = str(text).strip()
        if not text:
            counts.empty += 1
            continue
        if not (math.isfinite(start) and math.isfinite(end)):
            counts.invalid += 1
            continue
        rec_start, rec_end = start + offset_s, end + offset_s
        if rec_start < 0 or rec_end < 0:
            counts.clamped += 1
        rec_start = max(0.0, rec_start)
        rec_end = max(rec_start, rec_end)
        out.append(Word(text=text, start_s=rec_start, end_s=rec_end, speaker=None))
    counts.words += len(out)
    return out


def _sorted(words: Iterable[Word]) -> list[Word]:
    return sorted(words, key=lambda w: (w.start_s, w.end_s))


# ---------------------------------------------------------------- recording time


@dataclass(frozen=True)
class RecordingOffset:
    """Seconds added to file time to get recording time, and where they came from."""

    offset_s: float = 0.0
    source: Literal["none", "offset-s", "plan"] = "none"
    recording: str | None = None
    detail: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"offset_s": self.offset_s, "from": self.source}
        if self.detail:
            out.update(self.detail)
        if self.source == "none":
            out["note"] = "no plan source or --offset-s was given: times are file seconds"
        return out


def offset_from_plan(plan: EditPlan, source_id: str) -> RecordingOffset:
    """``recording_origin / fps`` of a plan source, with the exact rational fps."""
    source = plan.source_map().get(source_id)
    if source is None:
        known = ", ".join(repr(s.id) for s in plan.sources) or "none"
        raise AnalyzeError(
            "invalid-argument", f"the plan has no source {source_id!r} (sources: {known})"
        )
    fps = Fraction(plan.fps.num, plan.fps.den)
    offset = float(Fraction(source.recording_origin) / fps)
    return RecordingOffset(
        offset_s=offset,
        source="plan",
        recording=source.recording_id if source.recording_id is not None else source.id,
        detail={
            "source": source.id,
            "recording_origin": source.recording_origin,
            "fps": f"{plan.fps.num}/{plan.fps.den}",
        },
    )


def explicit_offset(offset_s: float) -> RecordingOffset:
    if not math.isfinite(offset_s):
        raise AnalyzeError("invalid-argument", "--offset-s must be a finite number of seconds")
    return RecordingOffset(offset_s=float(offset_s), source="offset-s")


# ---------------------------------------------------------------- checkpoint


def checkpoint_path(out: Path) -> Path:
    return out.with_name(out.name + CHECKPOINT_SUFFIX)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - not every platform opens directories
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def _header(media: Path, settings: WhisperSettings, model_path: Path) -> dict[str, Any]:
    st = media.stat()
    decode = {**settings.decode_settings(), "model_path": str(model_path)}
    digest = hashlib.sha256(json.dumps(decode, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "aar_checkpoint": CHECKPOINT_VERSION,
        "kind": _KIND,
        "input": {"size": st.st_size, "mtime_ns": st.st_mtime_ns},
        "settings_sha256": digest,
        "settings": decode,
    }


_SEGMENT_KEYS = {"start", "end", "text", "avg_logprob", "compression_ratio", "no_speech_prob"}


def read_checkpoint(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """(header, complete segments, byte length of the complete lines). A trailing line without
    its newline (a run killed mid-write) is not complete and is ignored."""
    data = path.read_bytes()
    complete = data[: data.rfind(b"\n") + 1]
    lines = complete.splitlines()

    def corrupt(why: str) -> AnalyzeError:
        return AnalyzeError(
            "checkpoint-corrupt",
            f"checkpoint {path} is not usable: {why}",
            hint=f"delete {path} to start over",
        )

    if not lines:
        raise corrupt("it has no complete header line")
    try:
        header = json.loads(lines[0])
        segments = [json.loads(line) for line in lines[1:]]
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise corrupt(f"a complete line is not JSON ({exc})") from None
    if not (isinstance(header, dict) and header.get("kind") == _KIND):
        raise corrupt("its first line is not an aar transcript checkpoint header")
    last = -math.inf
    for i, seg in enumerate(segments, start=1):
        if not (isinstance(seg, dict) and seg.keys() >= _SEGMENT_KEYS):
            raise corrupt(f"segment line {i} is missing fields")
        if not isinstance(seg.get("words"), list) or seg["start"] < last:
            raise corrupt(f"segment line {i} is malformed or out of order")
        last = seg["start"]
    return header, segments, len(complete)


def _check_header(path: Path, found: dict[str, Any], expected: dict[str, Any]) -> None:
    differs = [
        key
        for key in ("aar_checkpoint", "input", "settings_sha256")
        if found.get(key) != expected[key]
    ]
    if differs:
        what = {
            "aar_checkpoint": "checkpoint format",
            "input": "input file (size or mtime)",
            "settings_sha256": "model or decode settings",
        }
        raise AnalyzeError(
            "checkpoint-mismatch",
            f"checkpoint {path} was written for a different "
            + " and ".join(what[k] for k in differs)
            + "; refusing to splice two runs",
            hint=f"re-run with the original input and settings, or delete {path} to start over",
        )


def _append(handle: Any, record: dict[str, Any]) -> None:
    handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"))
    handle.flush()
    os.fsync(handle.fileno())


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temporary file beside ``path``, fsync it and move it into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)


# ---------------------------------------------------------------- full file


def _decode_timeout(settings: WhisperSettings, duration_s: float | None) -> float:
    if settings.timeout_s is not None:
        return settings.timeout_s
    return timeout_for(duration_s, per_media_s=0.5)


def transcribe_file(
    media: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    settings: WhisperSettings = CCN_PRESET,
    model_dir: str | os.PathLike[str] | None = None,
    offset: RecordingOffset | None = None,
    recording: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Transcribe ``media`` into the Transcript JSON ``out``; return the report.

    ``recording`` names the transcript's recording (default: the plan source's recording).
    Raises ``AnalyzeError`` (see ``aar.analyze.errors`` for the types and exit codes).
    """
    offset = offset or RecordingOffset()
    out_path = Path(out)
    ckpt = checkpoint_path(out_path)
    try:
        media_path = require_regular_file(media, tool="ffmpeg")
    except MediaError as exc:
        raise from_media_error(exc, "input") from None
    need_faster_whisper()
    model_path = resolve_model(settings.model, model_dir)
    try:
        info = probe(media_path)
    except MediaError as exc:
        raise from_media_error(exc, f"input {media_path}") from None
    if info.audio is None:
        raise AnalyzeError("no-audio", f"{media_path} has no audio stream")
    duration = info.audio.duration_s or info.duration_s
    if duration is not None and duration > settings.max_duration_s:
        raise AnalyzeError(
            "input-too-long",
            f"{media_path} lasts {duration:.0f} s, more than the {settings.max_duration_s:g} s cap",
            hint="raise --max-duration-s (audio takes 230 MB per hour in memory)",
        )
    header = _header(media_path, settings, model_path)
    segments: list[dict[str, Any]] = []
    resumed = False
    good = 0
    if ckpt.exists():
        if not resume:
            raise AnalyzeError(
                "checkpoint-exists",
                f"checkpoint {ckpt} exists from an earlier run",
                hint=f"pass --resume to continue it, or delete {ckpt} to start over",
            )
        found, segments, good = read_checkpoint(ckpt)
        _check_header(ckpt, found, header)
        resumed = True
    from_checkpoint = len(segments)
    resume_s = segments[-1]["end"] if segments else 0.0
    model, _ = load_model(settings, model_dir)
    try:
        audio = read_mono(
            media_path,
            rate=WHISPER_RATE,
            max_duration_s=settings.max_duration_s,
            timeout_s=_decode_timeout(settings, duration),
        ).samples
    except MediaError as exc:
        raise from_media_error(exc, f"input {media_path}") from None
    kwargs = settings.transcribe_kwargs()
    base_s = 0.0
    method = None
    if resume_s > 0:
        if settings.vad_filter:
            first = round(resume_s * WHISPER_RATE)
            audio, base_s, method = audio[first:], first / WHISPER_RATE, "slice"
        else:
            kwargs["clip_timestamps"] = [resume_s]
            method = "clip_timestamps"
    # The checkpoint is created (or trimmed) only now, so a run that could not load the model
    # or decode the input leaves nothing behind.
    if resumed:
        with open(ckpt, "r+b") as fh:  # drop a half-written last line
            fh.truncate(good)
    else:
        with open(ckpt, "wb") as fh:
            _append(fh, header)
        _fsync_dir(ckpt.parent)
    skipped = 0
    try:
        stream, _info = model.transcribe(audio, **kwargs)
        with open(ckpt, "ab") as fh:
            for segment in stream:
                record = segment_record(segment, base_s)
                if record["end"] <= resume_s and resume_s > 0:
                    skipped += 1  # the resumed decode repeated a segment we already have
                    continue
                _append(fh, record)
                segments.append(record)
    except Exception as exc:
        raise AnalyzeError(
            "transcription-failed",
            f"transcription stopped after {len(segments)} segments: {type(exc).__name__}: {exc}",
            hint=f"the checkpoint {ckpt} keeps them; run again with --resume to continue",
        ) from exc
    counts = WordCounts()
    words: list[Word] = []
    dropped: list[dict[str, Any]] = []
    for record in segments:
        reason = guard_reason(record, settings)
        if reason is not None:
            dropped.append(
                {
                    "start_s": round(record["start"] + offset.offset_s, 3),
                    "end_s": round(record["end"] + offset.offset_s, 3),
                    "file_start_s": round(record["start"], 3),
                    "file_end_s": round(record["end"], 3),
                    "text": record["text"].strip(),
                    "reason": reason,
                }
            )
            continue
        words.extend(segment_words(record, offset.offset_s, counts))
    name = recording if recording is not None else offset.recording
    transcript = Transcript(recording=name, words=_sorted(words))
    write_atomic(out_path, json.dumps(transcript.model_dump(mode="json"), indent=1) + "\n")
    ckpt.unlink()
    _fsync_dir(ckpt.parent)
    return {
        "ok": True,
        "output": str(out_path),
        "input": str(media_path),
        "duration_s": duration,
        "recording": name,
        "offset": offset.to_dict(),
        "model": {"name": settings.model, "path": str(model_path)},
        "settings": settings.to_dict(),
        "counts": {
            "segments": len(segments),
            "kept_segments": len(segments) - len(dropped),
            "dropped_segments": len(dropped),
            "words": counts.words,
            "empty_words_dropped": counts.empty,
            "invalid_words_dropped": counts.invalid,
            "clamped_words": counts.clamped,
        },
        "dropped": dropped,
        "resume": {
            "resumed": resumed,
            "from_s": resume_s if resumed else None,
            "segments_from_checkpoint": from_checkpoint,
            "method": method,
            "repeated_segments_skipped": skipped,
        },
        "checkpoint": {"path": str(ckpt), "removed": True},
    }


# ---------------------------------------------------------------- QA transcriber


class WhisperTranscriber:
    """``aar.qa.Transcriber`` on faster-whisper: one short window at a time.

    The window is decoded by ffmpeg to 16 kHz mono float32 (bounded by its length), transcribed
    with ``settings`` and returned in file seconds, cleaned and guarded like the full
    transcript. Segments the guard drops are kept in ``last_dropped``. Raises on failure.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        model_dir: str | os.PathLike[str] | None = None,
        settings: WhisperSettings | None = None,
        timeout_s: float = 60.0,
    ) -> None:
        base = settings or CCN_PRESET
        self.settings = replace(base, model=model) if model is not None else base
        self.model, self.model_path = load_model(self.settings, model_dir)
        self.timeout_s = timeout_s
        self.last_dropped: list[dict[str, Any]] = []

    def transcribe(self, path: Path, start_s: float, end_s: float) -> Sequence[Word]:
        if not (math.isfinite(start_s) and math.isfinite(end_s) and end_s > start_s >= 0):
            raise ValueError(f"bad window [{start_s}, {end_s})")
        mono = read_mono(
            path,
            rate=WHISPER_RATE,
            start_s=start_s,
            duration_s=end_s - start_s,
            max_duration_s=end_s - start_s,
            timeout_s=self.timeout_s,
        )
        stream, _info = self.model.transcribe(mono.samples, **self.settings.transcribe_kwargs())
        counts = WordCounts()
        words: list[Word] = []
        self.last_dropped = []
        for segment in stream:
            record = segment_record(segment, mono.start_s)
            reason = guard_reason(record, self.settings)
            if reason is not None:
                self.last_dropped.append(
                    {"start_s": record["start"], "end_s": record["end"], "reason": reason}
                )
                continue
            words.extend(segment_words(record, 0.0, counts))
        return _sorted(words)


__all__ = [
    "CCN_PRESET",
    "CHECKPOINT_SUFFIX",
    "MODEL_DIR_ENV",
    "RecordingOffset",
    "WhisperSettings",
    "WhisperTranscriber",
    "checkpoint_path",
    "explicit_offset",
    "guard_reason",
    "load_model",
    "offset_from_plan",
    "read_checkpoint",
    "resolve_model",
    "segment_record",
    "segment_words",
    "transcribe_file",
    "write_atomic",
]
