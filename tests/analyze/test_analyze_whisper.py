"""``analyze.transcript`` and the QA transcriber, against a fake ``faster_whisper`` module.

The fake records every ``WhisperModel(...)`` and ``transcribe(...)`` call and plays back fixed
segments in file time, honouring both resume mechanisms (an audio slice, or
``clip_timestamps``), so a clean run and an interrupted-then-resumed run can be compared.
Media is real (a short WAV through ffmpeg); no model is needed except by the live test, which
runs only with ``AAR_WHISPER_MODEL`` set and faster-whisper installed.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from analyze_support import needs_ffmpeg, tone, write_wav

from aar.analyze import CCN_PRESET, AnalyzeError, WhisperTranscriber, transcribe_file
from aar.analyze.whisper import (
    checkpoint_path,
    explicit_offset,
    guard_reason,
    read_checkpoint,
    resolve_model,
)
from aar.cli import main
from aar.plan.inputs import Transcript, load_transcript

RATE = 16000
SECONDS = 10.0


# ---------------------------------------------------------------- the fake library


@dataclass
class FakeWord:
    start: float
    end: float
    word: str
    probability: float = 0.9


@dataclass
class FakeSegment:
    start: float
    end: float
    text: str
    words: list[FakeWord]
    avg_logprob: float = -0.3
    compression_ratio: float = 1.4
    no_speech_prob: float = 0.05
    id: int = 0
    seek: int = 0
    tokens: list[int] = field(default_factory=list)
    temperature: float = 0.0


def seg(start: float, end: float, *words: str, **kw: Any) -> FakeSegment:
    """A segment whose words share its span evenly."""
    step = (end - start) / max(1, len(words))
    spans = [(start + i * step, start + (i + 1) * step) for i in range(len(words))]
    return FakeSegment(
        start=start,
        end=end,
        text="".join(f" {w}" for w in words),
        words=[FakeWord(a, b, f" {w}") for (a, b), w in zip(spans, words, strict=True)],
        **kw,
    )


class Fake:
    """The registry behind one fake ``faster_whisper`` module."""

    def __init__(self, segments: list[FakeSegment], total_s: float = SECONDS) -> None:
        self.segments = segments
        self.total_s = total_s
        self.models: list[tuple[str, dict[str, Any]]] = []
        self.calls: list[dict[str, Any]] = []
        self.fail_after: int | None = None
        self.raise_on_call: BaseException | None = None
        self.window: tuple[float, float] | None = None  # set when a QA window is decoded
        fake = self

        class WhisperModel:
            def __init__(self, model_size_or_path: str, **kwargs: Any) -> None:
                fake.models.append((model_size_or_path, kwargs))

            def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Any, Any]:
                fake.calls.append({"audio": audio, **kwargs})
                if fake.raise_on_call is not None:
                    raise fake.raise_on_call
                return fake._play(audio, kwargs), types.SimpleNamespace(language="en")

        self.module = types.ModuleType("faster_whisper")
        self.module.WhisperModel = WhisperModel  # type: ignore[attr-defined]

    def _play(self, audio: Any, kwargs: dict[str, Any]) -> Any:
        if self.window is not None:  # a QA window: the audio starts at the window
            base, stop = self.window
        else:  # a whole file, or its tail from a resume point
            base, stop = self.total_s - len(audio) / RATE, self.total_s
        clip = kwargs.get("clip_timestamps", "0")
        start = base + (clip[0] if isinstance(clip, list) else 0.0)
        playing = [x for x in self.segments if x.start >= start - 1e-9 and x.end <= stop + 1e-9]
        for n, s in enumerate(playing):
            if self.fail_after is not None and n == self.fail_after:
                raise RuntimeError("simulated crash")
            words = [FakeWord(w.start - base, w.end - base, w.word, w.probability) for w in s.words]
            yield FakeSegment(
                start=s.start - base,
                end=s.end - base,
                text=s.text,
                words=words,
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
                no_speech_prob=s.no_speech_prob,
            )


SEGMENTS = [
    seg(0.5, 1.5, "hello", "there"),
    seg(1.6, 2.4, "how", "are", "you"),
    seg(2.5, 3.5, "thanks", "for", "coming", no_speech_prob=0.7, avg_logprob=-1.2),  # dropped
    seg(3.6, 4.4, "la", "la", "la", compression_ratio=2.6),  # dropped
    seg(4.5, 5.5, "edge", "case", no_speech_prob=0.6, avg_logprob=-1.1),  # kept: 0.6 is not > 0.6
    seg(5.6, 6.4, "loud", "room", no_speech_prob=0.9, avg_logprob=-0.4),  # kept: logprob is fine
    seg(6.5, 7.5, "compression", "edge", compression_ratio=2.4),  # kept: 2.4 is not > 2.4
    seg(7.6, 8.8, "good", "night"),
]
KEPT_WORDS = ["hello", "there", "how", "are", "you", "edge", "case", "loud", "room"]
KEPT_WORDS += ["compression", "edge", "good", "night"]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Fake:
    registry = Fake(list(SEGMENTS))
    monkeypatch.setitem(sys.modules, "faster_whisper", registry.module)
    return registry


@pytest.fixture
def models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "models"
    (root / "medium.en").mkdir(parents=True)
    (root / "medium.en" / "model.bin").write_bytes(b"")
    monkeypatch.setenv("AAR_MODEL_DIR", str(root))
    return root


@pytest.fixture
def media(tmp_path: Path) -> Path:
    return write_wav(tmp_path / "talk.wav", tone(SECONDS, 300.0, peak=0.1))


def run_cli(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["analyze", "transcript", *(str(a) for a in argv)])
    return code, json.loads(capsys.readouterr().out)


TRANSCRIBE_KWARGS = {
    "language": "en",
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 400},
    "condition_on_previous_text": False,
}


# ---------------------------------------------------------------- kwargs


@needs_ffmpeg
def test_ccn_preset_kwargs_reach_faster_whisper_exactly(tmp_path, fake, models, media, capsys):
    out = tmp_path / "talk.json"
    code, report = run_cli(capsys, media, "--out", out)
    assert code == 0 and report["ok"]
    assert fake.models == [
        (
            str((models / "medium.en").resolve()),
            {"device": "cpu", "compute_type": "int8", "cpu_threads": 0, "local_files_only": True},
        )
    ]
    [call] = fake.calls
    audio = call.pop("audio")
    assert call == TRANSCRIBE_KWARGS
    assert isinstance(audio, np.ndarray) and audio.dtype == np.float32 and audio.ndim == 1
    assert len(audio) == int(SECONDS * RATE)
    assert report["model"] == {"name": "medium.en", "path": str((models / "medium.en").resolve())}
    assert report["settings"]["beam_size"] == 5


@needs_ffmpeg
def test_every_setting_can_be_overridden(tmp_path, fake, models, media, capsys):
    other = tmp_path / "elsewhere" / "small"
    other.mkdir(parents=True)
    (other / "model.bin").write_bytes(b"")
    code, _ = run_cli(
        capsys, media, "--out", tmp_path / "t.json", "--model", other, "--threads", "3",
        "--beam-size", "2", "--language", "", "--no-vad", "--condition-on-previous-text",
        "--device", "cuda", "--compute-type", "float16",
    )  # fmt: skip
    assert code == 0
    loaded = {"device": "cuda", "compute_type": "float16", "cpu_threads": 3}
    assert fake.models == [(str(other.resolve()), {**loaded, "local_files_only": True})]
    [call] = fake.calls
    call.pop("audio")
    assert call == {
        "language": None,
        "beam_size": 2,
        "word_timestamps": True,
        "vad_filter": False,
        "condition_on_previous_text": True,
    }
    code, _ = run_cli(capsys, media, "--out", tmp_path / "v.json", "--vad-min-silence-ms", "250")
    assert fake.calls[-1]["vad_parameters"] == {"min_silence_duration_ms": 250}


# ---------------------------------------------------------------- offset, hygiene, guard


def _plan(tmp_path: Path, origin: int, fps: str = "24000/1001") -> Path:
    plan = {
        "version": 1,
        "project": "Acme Show",
        "timeline": "t",
        "kind": "longform",
        "fps": fps,
        "resolution": {"width": 1920, "height": 1080},
        "profile": "ccn",
        "sources": [
            {"id": "cam-a", "media": "a.mov", "recording_id": "set-1", "recording_origin": origin},
            {"id": "cam-b", "media": "b.mov"},
        ],
        "segments": [],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


@needs_ffmpeg
def test_recording_time_from_a_plan_uses_the_exact_rational_fps(
    tmp_path, fake, models, media, capsys
):
    out = tmp_path / "t.json"
    plan = _plan(tmp_path, origin=1001)
    code, report = run_cli(capsys, media, "--out", out, "--plan", plan, "--source", "cam-a")
    assert code == 0
    offset = float(Fraction(1001) / Fraction(24000, 1001))  # 41.750416... s
    assert report["offset"] == {
        "offset_s": offset,
        "from": "plan",
        "source": "cam-a",
        "recording_origin": 1001,
        "fps": "24000/1001",
    }
    transcript = load_transcript(out)
    assert transcript.recording == "set-1" and report["recording"] == "set-1"
    first = transcript.words[0]
    assert first.text == "hello"
    assert first.start_s == pytest.approx(0.5 + offset, abs=1e-12)
    assert first.end_s == pytest.approx(1.0 + offset, abs=1e-12)
    # A source without a recording id is its own recording.
    run_cli(capsys, media, "--out", out, "--plan", plan, "--source", "cam-b")
    assert load_transcript(out).recording == "cam-b"
    assert load_transcript(out).words[0].start_s == pytest.approx(0.5)


@needs_ffmpeg
def test_explicit_offset_and_the_no_offset_note(tmp_path, fake, models, media, capsys):
    out = tmp_path / "t.json"
    code, report = run_cli(capsys, media, "--out", out, "--offset-s", "2.5", "--recording", "r")
    assert code == 0 and report["offset"] == {"offset_s": 2.5, "from": "offset-s"}
    words = load_transcript(out).words
    assert words[0].start_s == pytest.approx(3.0) and load_transcript(out).recording == "r"
    code, report = run_cli(capsys, media, "--out", out)
    assert report["offset"]["from"] == "none" and report["offset"]["offset_s"] == 0.0
    assert "file seconds" in report["offset"]["note"]
    assert load_transcript(out).recording is None


@pytest.mark.parametrize(
    ("argv", "kind"),
    [
        (["--offset-s", "1", "--plan", "{plan}", "--source", "cam-a"], "invalid-argument"),
        (["--plan", "{plan}"], "invalid-argument"),
        (["--source", "cam-a"], "invalid-argument"),
        (["--plan", "{plan}", "--source", "nope"], "invalid-argument"),
        (["--plan", "{tmp}/bad.json", "--source", "cam-a"], "invalid-plan"),
        (["--offset-s", "nan"], "invalid-argument"),
        (["--beam-size", "0"], "invalid-argument"),
    ],
)
def test_offset_and_setting_errors_exit_2(tmp_path, fake, models, media, capsys, argv, kind):
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    names = {"plan": _plan(tmp_path, 10), "tmp": tmp_path}
    code, body = run_cli(
        capsys, media, "--out", tmp_path / "t.json", *(a.format(**names) for a in argv)
    )
    assert code == 2 and body["error"]["type"] == kind
    assert not fake.calls


@needs_ffmpeg
def test_hallucination_guard_lists_every_dropped_segment(tmp_path, fake, models, media, capsys):
    out = tmp_path / "t.json"
    code, report = run_cli(capsys, media, "--out", out, "--offset-s", "100")
    assert code == 0
    assert [w.text for w in load_transcript(out).words] == KEPT_WORDS
    assert report["counts"] == {
        "segments": 8,
        "kept_segments": 6,
        "dropped_segments": 2,
        "words": 13,
        "empty_words_dropped": 0,
        "invalid_words_dropped": 0,
        "clamped_words": 0,
    }
    first, second = report["dropped"]
    assert (first["start_s"], first["end_s"]) == (102.5, 103.5)
    assert (first["file_start_s"], first["file_end_s"]) == (2.5, 3.5)
    assert first["text"] == "thanks for coming"
    assert first["reason"].startswith("no-speech: no_speech_prob 0.700 > 0.6")
    assert second["reason"] == "repetitive: compression_ratio 2.600 > 2.4"
    assert (second["start_s"], second["end_s"]) == (103.6, 104.4)


@pytest.mark.parametrize(
    ("nsp", "logprob", "ratio", "dropped"),
    [
        (0.61, -1.01, 1.0, True),
        (0.60, -1.01, 1.0, False),
        (0.61, -1.00, 1.0, False),
        (0.99, -0.50, 1.0, False),
        (0.10, -3.00, 1.0, False),
        (0.10, -0.10, 2.41, True),
        (0.10, -0.10, 2.40, False),
    ],
)
def test_guard_thresholds_are_strict_and_parameters(nsp, logprob, ratio, dropped):
    record = {"no_speech_prob": nsp, "avg_logprob": logprob, "compression_ratio": ratio}
    assert (guard_reason(record, CCN_PRESET) is not None) is dropped
    looser = CCN_PRESET.__class__(no_speech_prob_max=0.99, compression_ratio_max=9.0)
    assert guard_reason(record, looser) is None


@needs_ffmpeg
def test_word_hygiene(tmp_path, fake, models, media):
    fake.segments = [
        seg(3.0, 4.0, "late"),
        FakeSegment(
            0.2,
            1.0,
            " a  b",
            [
                FakeWord(0.2, 0.4, "  spaced  "),
                FakeWord(0.4, 0.5, "   "),
                FakeWord(0.6, 0.55, "backwards"),
                FakeWord(float("nan"), 0.7, "nan"),
            ],
        ),
        seg(1.0, 2.0, "early"),
    ]
    out = tmp_path / "t.json"
    report = transcribe_file(media, out, model_dir=models, offset=explicit_offset(-0.5))
    words = load_transcript(out).words
    assert [(w.text, w.start_s, w.end_s) for w in words] == [
        ("spaced", 0.0, 0.0),  # -0.3/-0.1 clamped to 0
        ("backwards", pytest.approx(0.1), pytest.approx(0.1)),  # end raised to start
        ("early", pytest.approx(0.5), pytest.approx(1.5)),
        ("late", pytest.approx(2.5), pytest.approx(3.5)),
    ]
    assert all(w.speaker is None for w in words)
    counts = report["counts"]
    assert (counts["empty_words_dropped"], counts["invalid_words_dropped"]) == (1, 1)
    assert counts["clamped_words"] == 1
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert Transcript.model_validate(raw).words == words


# ---------------------------------------------------------------- checkpoint and resume


@needs_ffmpeg
@pytest.mark.parametrize("vad", [True, False])
def test_interrupted_then_resumed_equals_a_clean_run(tmp_path, fake, models, media, capsys, vad):
    flags = [] if vad else ["--no-vad"]
    clean = tmp_path / "clean.json"
    assert run_cli(capsys, media, "--out", clean, *flags)[0] == 0
    out = tmp_path / "resumed.json"
    ckpt = checkpoint_path(out)
    fake.fail_after = 3
    code, body = run_cli(capsys, media, "--out", out, *flags)
    assert code == 1 and body["error"]["type"] == "transcription-failed"
    assert "--resume" in body["error"]["hint"] and not out.exists()
    header, saved, _ = read_checkpoint(ckpt)
    assert len(saved) == 3 and header["input"]["size"] == media.stat().st_size
    # A run killed mid-write leaves half a line: it is not a complete segment.
    with open(ckpt, "ab") as fh:
        fh.write(b'{"start": 5.6, "end": 6.4, "te')
    fake.fail_after = None
    code, report = run_cli(capsys, media, "--out", out, *flags, "--resume")
    assert code == 0
    assert out.read_bytes() == clean.read_bytes()
    assert not ckpt.exists() and not list(tmp_path.glob(".*.tmp"))
    resume = report["resume"]
    assert resume["resumed"] and resume["segments_from_checkpoint"] == 3
    assert resume["from_s"] == pytest.approx(3.5)
    call = fake.calls[-1]
    if vad:
        # faster-whisper ignores vad_filter when clip_timestamps is set, so slice instead.
        assert resume["method"] == "slice" and "clip_timestamps" not in call
        assert len(call["audio"]) == int(SECONDS * RATE) - round(3.5 * RATE)
        assert call["vad_filter"] is True
    else:
        assert resume["method"] == "clip_timestamps"
        assert call["clip_timestamps"] == [pytest.approx(3.5)]
        assert len(call["audio"]) == int(SECONDS * RATE)
    assert report["counts"]["segments"] == 8 and report["counts"]["dropped_segments"] == 2


@needs_ffmpeg
def test_checkpoint_is_written_segment_by_segment(tmp_path, fake, models, media):
    out = tmp_path / "t.json"
    seen: list[int] = []
    original = fake._play

    def watching(audio: Any, kwargs: dict[str, Any]) -> Any:
        for segment in original(audio, kwargs):
            seen.append(len(read_checkpoint(checkpoint_path(out))[1]))
            yield segment

    fake._play = watching  # type: ignore[method-assign]
    transcribe_file(media, out, model_dir=models)
    assert seen == list(range(8))  # before each segment, every earlier one is on disk


@needs_ffmpeg
def test_a_mismatched_checkpoint_is_refused(tmp_path, fake, models, media, capsys):
    out = tmp_path / "t.json"
    fake.fail_after = 2
    assert run_cli(capsys, media, "--out", out)[0] == 1
    fake.fail_after = None
    before = checkpoint_path(out).read_bytes()
    code, body = run_cli(capsys, media, "--out", out, "--resume", "--beam-size", "4")
    assert code == 2 and body["error"]["type"] == "checkpoint-mismatch"
    assert "model or decode settings" in body["error"]["message"]
    code, body = run_cli(capsys, media, "--out", out, "--resume", "--no-vad")
    assert code == 2 and body["error"]["type"] == "checkpoint-mismatch"
    os.utime(media, ns=(1, 1))
    code, body = run_cli(capsys, media, "--out", out, "--resume")
    assert code == 2 and body["error"]["type"] == "checkpoint-mismatch"
    assert "input file" in body["error"]["message"]
    assert checkpoint_path(out).read_bytes() == before and not out.exists()
    assert len(fake.calls) == 1  # refused before any decode


@needs_ffmpeg
def test_threads_and_guard_thresholds_do_not_invalidate_a_checkpoint(
    tmp_path, fake, models, media, capsys
):
    out = tmp_path / "t.json"
    fake.fail_after = 2
    run_cli(capsys, media, "--out", out)
    fake.fail_after = None
    code, report = run_cli(
        capsys, media, "--out", out, "--resume", "--threads", "2", "--compression-ratio-max", "9"
    )
    assert code == 0 and report["resume"]["segments_from_checkpoint"] == 2
    assert report["counts"]["dropped_segments"] == 1


@needs_ffmpeg
def test_an_existing_checkpoint_is_not_overwritten_without_resume(
    tmp_path, fake, models, media, capsys
):
    out = tmp_path / "t.json"
    fake.fail_after = 1
    run_cli(capsys, media, "--out", out)
    fake.fail_after = None
    before = checkpoint_path(out).read_bytes()
    code, body = run_cli(capsys, media, "--out", out)
    assert code == 2 and body["error"]["type"] == "checkpoint-exists"
    assert "--resume" in body["error"]["hint"]
    assert checkpoint_path(out).read_bytes() == before


@needs_ffmpeg
def test_a_corrupt_checkpoint_is_refused(tmp_path, fake, models, media, capsys):
    out = tmp_path / "t.json"
    fake.fail_after = 2
    run_cli(capsys, media, "--out", out)
    ckpt = checkpoint_path(out)
    lines = ckpt.read_bytes().splitlines(keepends=True)
    ckpt.write_bytes(lines[0] + b"not json\n" + lines[2])
    code, body = run_cli(capsys, media, "--out", out, "--resume")
    assert code == 2 and body["error"]["type"] == "checkpoint-corrupt"


@needs_ffmpeg
def test_a_failed_output_write_keeps_the_checkpoint(
    tmp_path, fake, models, media, capsys, monkeypatch
):
    import aar.analyze.whisper as whisper

    def full_disk(path: Path, text: str) -> None:
        raise OSError(28, "No space left on device")

    out = tmp_path / "t.json"
    original = whisper.write_atomic
    monkeypatch.setattr(whisper, "write_atomic", full_disk)
    code, body = run_cli(capsys, media, "--out", out)
    assert code == 74 and body["error"]["type"] == "output-error"
    assert "--resume" in body["error"]["hint"]
    assert len(read_checkpoint(checkpoint_path(out))[1]) == 8 and not out.exists()
    monkeypatch.setattr(whisper, "write_atomic", original)
    code, report = run_cli(capsys, media, "--out", out, "--resume")
    assert code == 0 and report["resume"]["segments_from_checkpoint"] == 8
    assert [w.text for w in load_transcript(out).words] == KEPT_WORDS
    assert not checkpoint_path(out).exists()


@needs_ffmpeg
def test_a_failed_model_load_leaves_no_checkpoint(tmp_path, fake, models, media, capsys):
    def broken(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Unable to open file 'model.bin'")

    fake.module.WhisperModel = broken  # type: ignore[attr-defined]
    out = tmp_path / "t.json"
    code, body = run_cli(capsys, media, "--out", out)
    assert code == 2 and body["error"]["type"] == "model-invalid"
    assert not checkpoint_path(out).exists() and not out.exists()


# ---------------------------------------------------------------- models and extras


def test_model_resolution_layouts(tmp_path, monkeypatch):
    monkeypatch.delenv("AAR_MODEL_DIR", raising=False)
    root = tmp_path / "shared"
    with pytest.raises(AnalyzeError) as missing:
        resolve_model("medium.en")
    assert missing.value.type == "model-missing" and "AAR_MODEL_DIR" in missing.value.hint
    root.mkdir()
    with pytest.raises(AnalyzeError) as empty:
        resolve_model("medium.en", root)
    assert empty.value.type == "model-missing"
    assert "faster-whisper-medium.en" in empty.value.hint and "never downloads" in empty.value.hint
    named = root / "faster-whisper-small.en"
    named.mkdir()
    (named / "model.bin").write_bytes(b"")
    assert resolve_model("small.en", root) == named.resolve()
    repo = root / "models--Systran--faster-whisper-medium.en"
    for rev in ("aaa", "bbb"):
        (repo / "snapshots" / rev).mkdir(parents=True)
        (repo / "snapshots" / rev / "model.bin").write_bytes(b"")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("bbb\n", encoding="utf-8")
    monkeypatch.setenv("AAR_MODEL_DIR", str(root))
    assert resolve_model("medium.en") == (repo / "snapshots" / "bbb").resolve()
    with pytest.raises(AnalyzeError) as not_a_model:
        resolve_model(str(root))  # a path without model.bin
    assert not_a_model.value.type == "model-missing"


def test_missing_faster_whisper_is_a_typed_error_naming_the_extra(
    tmp_path, monkeypatch, capsys, models, media
):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    code, body = run_cli(capsys, media, "--out", tmp_path / "t.json")
    assert code == 2
    assert body["error"]["type"] == "missing-extra"
    assert body["error"]["hint"] == "pip install 'ai-agent-resolve[whisper]'"


def test_missing_model_exits_2(tmp_path, fake, monkeypatch, capsys, media):
    monkeypatch.delenv("AAR_MODEL_DIR", raising=False)
    code, body = run_cli(capsys, media, "--out", tmp_path / "t.json")
    assert code == 2 and body["error"]["type"] == "model-missing"
    assert "AAR_MODEL_DIR" in body["error"]["hint"]
    assert not fake.models


def test_missing_input_exits_2(tmp_path, fake, models, capsys):
    code, body = run_cli(capsys, tmp_path / "nope.wav", "--out", tmp_path / "t.json")
    assert code == 2 and body["error"]["type"] == "input-missing"


@needs_ffmpeg
def test_media_that_does_not_decode_exits_1(tmp_path, fake, models, capsys):
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"RIFF not really" * 40)
    code, body = run_cli(capsys, junk, "--out", tmp_path / "t.json")
    assert code == 1 and body["error"]["type"] == "decode-failed"


@needs_ffmpeg
def test_input_longer_than_the_cap_is_refused(tmp_path, fake, models, media, capsys):
    code, body = run_cli(capsys, media, "--out", tmp_path / "t.json", "--max-duration-s", "5")
    assert code == 2 and body["error"]["type"] == "input-too-long"


@needs_ffmpeg
def test_input_of_unknown_length_past_the_cap_is_refused(tmp_path, fake, models, monkeypatch):
    import aar.analyze.whisper as whisper
    from aar.media.probe import MediaInfo, Stream

    def unknown_length(path: Any, **kwargs: Any) -> MediaInfo:
        audio = Stream(index=0, kind="audio", codec="pcm_s16le", duration_s=None, channels=1)
        return MediaInfo(
            path=str(path),
            format="wav",
            duration_s=None,
            size_bytes=1,
            bit_rate=None,
            streams=(audio,),
        )

    media = write_wav(tmp_path / "long.wav", tone(6.0))
    monkeypatch.setattr(whisper, "probe", unknown_length)
    settings = CCN_PRESET.__class__(max_duration_s=5.0)
    with pytest.raises(AnalyzeError) as error:
        transcribe_file(media, tmp_path / "t.json", settings=settings, model_dir=models)
    assert error.value.type == "input-too-long" and error.value.exit_code == 2
    assert not checkpoint_path(tmp_path / "t.json").exists() and not fake.calls


# ---------------------------------------------------------------- the QA transcriber


@needs_ffmpeg
def test_qa_transcriber_returns_file_seconds_for_one_window(fake, models, media):
    transcriber = WhisperTranscriber("medium.en", model_dir=models)
    fake.window = (4.0, 8.0)
    words = transcriber.transcribe(media, 4.0, 8.0)
    [call] = fake.calls
    audio = call.pop("audio")
    assert call == TRANSCRIBE_KWARGS
    assert len(audio) == 4 * RATE  # the window only, decoded to 16 kHz
    # The fake plays the segments that start inside the window; times come back in file time.
    assert [w.text for w in words] == ["edge", "case", "loud", "room", "compression", "edge"]
    assert words[0].start_s == pytest.approx(4.5) and words[-1].end_s == pytest.approx(7.5)
    assert transcriber.last_dropped == []
    fake.segments[4].compression_ratio = 3.0
    words = transcriber.transcribe(media, 4.0, 8.0)
    assert [w.text for w in words][:2] == ["loud", "room"]
    assert transcriber.last_dropped[0]["reason"].startswith("repetitive")


@needs_ffmpeg
def test_qa_transcriber_raises_on_failure(fake, models, media):
    transcriber = WhisperTranscriber("medium.en", model_dir=models)
    fake.raise_on_call = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        transcriber.transcribe(media, 1.0, 3.0)
    with pytest.raises(ValueError):
        transcriber.transcribe(media, 3.0, 3.0)


def _qa_plan(tmp_path: Path) -> Path:
    other = write_wav(tmp_path / "o.wav", tone(1.0, 200.0, peak=0.1))
    incoming = write_wav(tmp_path / "a.wav", tone(1.0, 300.0, peak=0.1))
    plan = {
        "version": 1,
        "project": "Acme Show",
        "timeline": "QA fixture",
        "kind": "short",
        "fps": "50",
        "resolution": {"width": 160, "height": 90},
        "profile": "ccn",
        "sources": [{"id": "o", "media": other.name}, {"id": "a", "media": incoming.name}],
        "segments": [
            {"track": 1, "source": "o", "src_in": 0, "src_out": 25},
            {"track": 1, "source": "a", "src_in": 0, "src_out": 25},
        ],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def _qa(capsys: pytest.CaptureFixture[str], *argv: object) -> tuple[int, dict[str, Any]]:
    code = main(["qa", "check", *(str(a) for a in argv), "--json"])
    return code, json.loads(capsys.readouterr().out)


@needs_ffmpeg
def test_qa_check_runs_join_retranscribe_with_whisper_model(tmp_path, fake, models, capsys):
    fake.segments = []
    render = write_wav(tmp_path / "render.wav", tone(1.0, 250.0, peak=0.2))
    plan = _qa_plan(tmp_path)
    transcript = tmp_path / "words.json"
    transcript.write_text(json.dumps({"words": []}), encoding="utf-8")
    base = [render, "--kind", "short", "--plan", plan, "--transcript", transcript]
    _, without = _qa(capsys, *base)
    retranscribe = next(c for c in without["checks"] if c["id"] == "join-retranscribe")
    assert retranscribe["status"] == "skipped" and retranscribe["skip_reason"] == "no-transcriber"
    assert not fake.models
    _, with_model = _qa(capsys, *base, "--whisper-model", "medium.en", "--model-dir", models)
    retranscribe = next(c for c in with_model["checks"] if c["id"] == "join-retranscribe")
    assert retranscribe["status"] != "skipped"
    assert len(fake.models) == 1 and len(fake.calls) == 3  # three windows around the one cut
    for call in fake.calls:
        assert len(call["audio"]) <= RATE + 1  # each window is clipped to the 1 s render
    # Everything else in the report is the same as without the model.
    strip = {"join-retranscribe"}
    assert [c for c in without["checks"] if c["id"] not in strip] == [
        c for c in with_model["checks"] if c["id"] not in strip
    ]


def test_qa_check_whisper_errors_exit_2(tmp_path, monkeypatch, models, capsys):
    render = write_wav(tmp_path / "render.wav", tone(1.0))
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    code, body = _qa(capsys, render, "--kind", "short", "--whisper-model", "medium.en")
    assert code == 2 and body["error"]["type"] == "missing-extra"
    assert "[whisper]" in body["error"]["message"]
    code, body = _qa(capsys, render, "--kind", "short", "--model-dir", models)
    assert code == 2 and body["error"]["type"] == "invalid-argument"


def test_qa_check_missing_model_exits_2(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.delenv("AAR_MODEL_DIR", raising=False)
    render = write_wav(tmp_path / "render.wav", tone(1.0))
    code, body = _qa(capsys, render, "--kind", "short", "--whisper-model", "medium.en")
    assert code == 2 and body["error"]["type"] == "model-missing"
    assert "AAR_MODEL_DIR" in body["error"]["message"]


# ---------------------------------------------------------------- live


LIVE_MODEL = os.environ.get("AAR_WHISPER_MODEL", "")


@needs_ffmpeg
@pytest.mark.skipif(
    not LIVE_MODEL or importlib.util.find_spec("faster_whisper") is None,
    reason="set AAR_WHISPER_MODEL (a model name or directory) and install the whisper extra",
)
def test_live_whisper_gives_a_valid_transcript(tmp_path):
    clip = write_wav(tmp_path / "clip.wav", tone(3.0, 220.0, peak=0.2))
    out = tmp_path / "clip.json"
    report = transcribe_file(clip, out, settings=CCN_PRESET.__class__(model=LIVE_MODEL))
    transcript = load_transcript(out)  # validates; an empty transcript is fine for a tone
    assert report["counts"]["words"] == len(transcript.words)
    assert not checkpoint_path(out).exists()
