"""``analyze.laughs``: audience laughs from Gemini, checked against the audio envelope.

**Pipeline.** The whole file's first audio stream is extracted to FLAC, mono, 16 kHz into the
work dir (atomically; its sha256 and duration are recorded), so the times Gemini returns are
file times. The FLAC is uploaded, waited on, prompted with ``LAUGH_PROMPT_V1`` and deleted (in a
``finally``, once the upload succeeded). The raw ``generateContent`` reply is saved to
``raw.json`` in the work dir; ``--from-raw`` re-parses a saved reply with no network and no key.

**Cost and credential.** A run that would call the API needs ``--allow-paid`` or exits 2 before
any network use. The key comes only from ``$AAR_GEMINI_API_KEY`` (there is no flag), and a
missing or blank one exits 2 naming the variable. It is sent only in a header and is scrubbed
from every message, the saved reply, the report and the map. Audio longer than
``max_audio_s`` is refused (exit 2) before it is uploaded. The upload's display name is
``aar-laughs-<first 8 hex of the audio sha256>``, so no show or person name leaves the machine
except inside the audio.

**Strict parser.** The reply is a JSON array (one surrounding ```` ```json ```` fence may be
stripped; anything else is a typed ``bad-response`` error and the raw reply is kept). Each item
is ``{start, end, level, kind}``. Every dropped item is reported with its index, its reason and
at most 200 characters of it: ``bad-item`` (not an object with those four keys), ``bare-number``
(an int or a numeric string where a time belongs), ``bad-time`` (anything but ``[h:]m:ss[.fff]``
in ASCII digits, seconds below 60 and minutes below 60 when hours are given), ``bad-level`` (not
the int 1, 2 or 3; a bool or ``"2"`` is refused), ``unknown-kind`` (not ``laugh`` or
``applause``), ``reversed`` (end before start) and ``out-of-range`` (a start at or past the
duration, or an end more than 1 s past it). An end equal to the start becomes start + 1 s
(flag ``point``); an end up to 1 s past the duration is clamped (flag ``clamped``). Strength is
``round(level / 3, 3)``. Applause never enters a laugh map; it is listed on its own.

**Envelope cross-check** (a port of the hand-made CCN method). RMS in dBFS at 10 Hz (100 ms
frames, floor -120 dB). For a laugh [a, b] the window is [a - 1, b + 2], clipped to the file;
the baseline is the median over the 8 s before it. Less than 2 s of baseline is ``unchecked``
(``no-baseline``); a baseline and window both below -80 dBFS are ``unchecked`` (``silent``).
``bump90`` is the window's 90th percentile minus the baseline and ``frac6`` the fraction of
window frames more than 6 dB above it. A laugh is ``confirmed`` when ``bump90 >= 6.0`` dB and
``frac6 >= 0.10``, else ``unconfirmed``. Both thresholds are starting values to tune on real
material. The map holds confirmed laughs only, unless ``include_unconfirmed``.

**Refinement** on a 50 Hz envelope with the same baseline. The onset is the first 20 ms frame in
[a - 1, min(a + 1.5, b)] more than 6 dB above the baseline that stays more than 3 dB above it
for 200 ms (else ``a``, flag ``onset-unrefined``). The end is the end of the last frame in
[b - 1, b + 2] more than 3 dB above the baseline (else ``b``). Start stays below end. Ending on
the laugh lands at its onset (DESIGN, end-on-laugh), so the onset is what matters most.

**Merge.** After refinement, laughs that overlap or sit within 0.25 s of each other merge,
keeping the larger strength and the best status (confirmed, then unconfirmed, then unchecked).
Every merge is reported. Recording time is file time plus the ``RecordingOffset``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aar.analyze.common import RecordingOffset, input_identity, write_atomic
from aar.analyze.errors import AnalyzeError, from_media_error, need_numpy
from aar.analyze.gemini import KEY_ENV, GeminiTransport, Scrubber, UrllibGeminiTransport
from aar.media import MediaError, probe, timeout_for
from aar.media.tools import input_url, require_regular_file, run_tool
from aar.plan.inputs import Laugh, LaughMap

SCHEMA = "aar.laughs/1"
DEFAULT_MODEL = "gemini-3.8-flash"
AUDIO_RATE = 16000
PROMPT_VERSION = "v1"
LAUGH_PROMPT_V1 = """\
This is the audio of a live stand-up comedy set. List every moment of audience laughter and \
every round of applause, and nothing else.

Answer with a JSON array only. Each element is an object with exactly these keys:
- "start": when it starts, as a string "m:ss" (use "h:mm:ss" after the first hour)
- "end": when it ends, in the same form
- "level": 1 for a small laugh, 2 for a clear laugh, 3 for a big laugh (for applause, its size)
- "kind": "laugh" or "applause"

Times are measured from the start of this audio. Do not list the performer's own laughter, \
speech or music. If there is no audience laughter or applause, answer [].
"""
PROMPT_SHA256 = hashlib.sha256(LAUGH_PROMPT_V1.encode("utf-8")).hexdigest()
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "start": {"type": "STRING"},
            "end": {"type": "STRING"},
            "level": {"type": "INTEGER"},
            "kind": {"type": "STRING", "enum": ["laugh", "applause"]},
        },
        "required": ["start", "end", "level", "kind"],
    },
}

ENV_FRAME_S = 0.1  # cross-check envelope: 10 Hz
REFINE_FRAME_S = 0.02  # refinement envelope: 50 Hz
FLOOR_DB = -120.0
SILENT_DB = -80.0
BASELINE_S = 8.0
MIN_BASELINE_S = 2.0
WINDOW_BEFORE_S, WINDOW_AFTER_S = 1.0, 2.0
ONSET_DB, SUSTAIN_DB, SUSTAIN_S = 6.0, 3.0, 0.2
ONSET_BEFORE_S, ONSET_AFTER_S = 1.0, 1.5
END_BEFORE_S, END_AFTER_S = 1.0, 2.0
MERGE_GAP_S = 0.25
RAW_EXCERPT = 200
_STATUS_RANK = {"confirmed": 2, "unconfirmed": 1, "unchecked": 0}
_TIME = re.compile(r"(?:([0-9]{1,2}):)?([0-9]{1,2}):([0-9]{2})(?:\.([0-9]{1,3}))?")
_BARE = re.compile(r"[+-]?[0-9]+(?:\.[0-9]*)?")
_FENCE = re.compile(r"\A\s*```json[ \t]*\n(.*)\n[ \t]*```\s*\Z", re.DOTALL)


@dataclass(frozen=True)
class LaughSettings:
    model: str = DEFAULT_MODEL
    confirm_bump_db: float = 6.0
    confirm_frac: float = 0.10
    include_unconfirmed: bool = False
    max_audio_s: float = 14400.0
    wait_timeout_s: float = 600.0
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if not (math.isfinite(self.confirm_bump_db) and self.confirm_bump_db >= 0):
            raise ValueError("confirm_bump_db must be a finite number of dB, at least 0")
        if not (math.isfinite(self.confirm_frac) and 0 <= self.confirm_frac <= 1):
            raise ValueError("confirm_frac must be between 0 and 1")
        for name, value in (
            ("max_audio_s", self.max_audio_s),
            ("wait_timeout_s", self.wait_timeout_s),
        ):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a finite number of seconds above 0")
        if self.timeout_s is not None and not (
            math.isfinite(self.timeout_s) and self.timeout_s > 0
        ):
            raise ValueError("timeout_s must be a finite number of seconds above 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------- strict parser


class BadResponse(AnalyzeError):
    def __init__(self, message: str) -> None:
        super().__init__(
            "bad-response",
            message,
            hint="the raw reply is kept in the work dir; re-run, or fix it and use --from-raw",
        )


@dataclass
class Item:
    """One parsed laugh or applause, in file seconds."""

    id: int
    kind: str
    start: float
    end: float
    level: int
    flags: list[str] = field(default_factory=list)

    @property
    def strength(self) -> float:
        return round(self.level / 3, 3)


def parse_time(value: Any) -> float | str:
    """Seconds, or the drop reason (``bare-number`` or ``bad-time``)."""
    if isinstance(value, bool):
        return "bad-time"
    if isinstance(value, (int, float)):
        return "bare-number"
    if not isinstance(value, str):
        return "bad-time"
    if _BARE.fullmatch(value):
        return "bare-number"
    match = _TIME.fullmatch(value)
    if match is None:
        return "bad-time"
    hours, minutes, seconds, fraction = match.groups()
    if int(seconds) >= 60 or (hours is not None and int(minutes) >= 60):
        return "bad-time"
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
    return total + (int(fraction) / 10 ** len(fraction) if fraction else 0.0)


def response_text(response: Any) -> str:
    """The text of a ``generateContent`` reply (all text parts of the first candidate)."""
    try:
        parts = response["candidates"][0]["content"]["parts"]
        texts = [p["text"] for p in parts if isinstance(p, dict) and "text" in p]
    except (KeyError, IndexError, TypeError):
        texts = None
    if not texts or not all(isinstance(t, str) for t in texts):
        raise BadResponse("the reply has no candidate text (blocked, empty or not a reply)")
    return "".join(texts)


def _excerpt(item: Any) -> str:
    try:
        text = json.dumps(item, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(item)
    return text[:RAW_EXCERPT]


def parse_items(text: str, duration_s: float) -> tuple[list[Item], list[dict[str, Any]]]:
    """Kept items (laughs and applause) and drops, from the reply text."""
    fenced = _FENCE.fullmatch(text)
    body = fenced.group(1) if fenced else text
    data: Any = None
    ok = True
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        ok = False
    if not ok or not isinstance(data, list):
        raise BadResponse("the reply is not a JSON array")
    kept: list[Item] = []
    drops: list[dict[str, Any]] = []
    for index, raw in enumerate(data):
        item, reason = _item(index, raw, duration_s)
        if item is None:
            drops.append({"index": index, "reason": reason, "raw": _excerpt(raw)})
        else:
            kept.append(item)
    return kept, drops


def _item(index: int, raw: Any, duration_s: float) -> tuple[Item | None, str]:
    if not (isinstance(raw, dict) and {"start", "end", "level", "kind"} <= raw.keys()):
        return None, "bad-item"
    start, end = parse_time(raw["start"]), parse_time(raw["end"])
    for value in (start, end):
        if isinstance(value, str):
            return None, value
    assert isinstance(start, float | int) and isinstance(end, float | int)
    level = raw["level"]
    if type(level) is not int or not 1 <= level <= 3:
        return None, "bad-level"
    kind = raw["kind"]
    if kind not in ("laugh", "applause"):
        return None, "unknown-kind"
    flags: list[str] = []
    if end < start:
        return None, "reversed"
    if end == start:
        end = start + 1.0
        flags.append("point")
    if start >= duration_s or end > duration_s + 1.0:
        return None, "out-of-range"
    if end > duration_s:
        end = duration_s
        flags.append("clamped")
    return Item(index, kind, float(start), float(end), level, flags), ""


# ---------------------------------------------------------------- envelopes


class _Powers:
    """Mean square per 20 ms frame of a mono f32le stream, accumulated as it arrives."""

    def __init__(self, np: Any) -> None:
        self.np = np
        self.frame = round(REFINE_FRAME_S * AUDIO_RATE)
        self.rest = b""
        self.carry = np.zeros(0, dtype=np.float32)
        self.chunks: list[Any] = []
        self.samples = 0

    def feed(self, chunk: bytes) -> None:
        np = self.np
        data = self.rest + chunk
        usable = len(data) - len(data) % 4
        self.rest = data[usable:]
        new = np.frombuffer(data[:usable], dtype="<f4")
        self.samples += len(new)
        x = np.concatenate([self.carry, new])
        whole = len(x) - len(x) % self.frame
        self.carry = x[whole:].copy()
        if whole:
            frames = x[:whole].astype(np.float64).reshape(-1, self.frame)
            self.chunks.append(np.nan_to_num(np.mean(frames * frames, axis=1), nan=0.0))

    def powers(self) -> Any:
        np = self.np
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0)


def to_db(np: Any, power: Any) -> Any:
    with np.errstate(divide="ignore"):
        return np.maximum(10.0 * np.log10(power), FLOOR_DB)


def envelopes(np: Any, power50: Any) -> tuple[Any, Any]:
    """(10 Hz dB, 50 Hz dB): five 20 ms frames make a 100 ms frame (mean of mean squares)."""
    whole = len(power50) - len(power50) % 5
    power10 = power50[:whole].reshape(-1, 5).mean(axis=1) if whole else np.zeros(0)
    return to_db(np, power10), to_db(np, power50)


def _frames(t0: float, t1: float, frame_s: float, n: int) -> tuple[int, int]:
    """Frames overlapping [t0, t1), clipped to [0, n)."""
    first = max(0, math.floor(t0 / frame_s + 1e-9))
    last = min(n, math.ceil(t1 / frame_s - 1e-9))
    return first, max(first, last)


@dataclass
class Check:
    status: str  # confirmed, unconfirmed, unchecked
    reason: str | None = None
    baseline_db: float | None = None
    bump90_db: float | None = None
    frac6: float | None = None


def cross_check(
    np: Any, env10: Any, start: float, end: float, duration_s: float, s: LaughSettings
) -> Check:
    w0, w1 = max(0.0, start - WINDOW_BEFORE_S), min(duration_s, end + WINDOW_AFTER_S)
    j0, j1 = _frames(w0, w1, ENV_FRAME_S, len(env10))
    b0, b1 = _frames(max(0.0, w0 - BASELINE_S), w0, ENV_FRAME_S, len(env10))
    b1 = min(b1, j0)
    if (b1 - b0) * ENV_FRAME_S < MIN_BASELINE_S - 1e-9 or j1 <= j0:
        return Check("unchecked", "no-baseline")
    baseline = float(np.median(env10[b0:b1]))
    window = env10[j0:j1]
    p90 = float(np.percentile(window, 90))
    if baseline < SILENT_DB and p90 < SILENT_DB:
        return Check("unchecked", "silent", baseline_db=baseline)
    bump = p90 - baseline
    frac = float(np.mean(window > baseline + ONSET_DB))
    confirmed = bump >= s.confirm_bump_db and frac >= s.confirm_frac
    return Check(
        "confirmed" if confirmed else "unconfirmed",
        baseline_db=baseline,
        bump90_db=bump,
        frac6=frac,
    )


def refine(
    np: Any, env50: Any, start: float, end: float, baseline: float | None, duration_s: float
) -> tuple[float, float, list[str]]:
    """(onset, end, flags) on the 50 Hz envelope; see the module docstring."""
    flags: list[str] = []
    n = len(env50)
    sustain = round(SUSTAIN_S / REFINE_FRAME_S)
    onset = None
    if baseline is not None:
        lo = max(0.0, start - ONSET_BEFORE_S)
        hi = min(start + ONSET_AFTER_S, end)
        first = max(0, math.ceil(lo / REFINE_FRAME_S - 1e-9))
        last = min(n - 1, math.floor(hi / REFINE_FRAME_S + 1e-9))
        for f in range(first, last + 1):
            if (
                env50[f] > baseline + ONSET_DB
                and f + sustain <= n
                and bool(np.all(env50[f : f + sustain] > baseline + SUSTAIN_DB))
            ):
                onset = f * REFINE_FRAME_S
                break
    if onset is None:
        onset = start
        flags.append("onset-unrefined")
    new_end = None
    if baseline is not None:
        first = max(0, math.ceil((end - END_BEFORE_S) / REFINE_FRAME_S - 1e-9))
        last = min(n - 1, math.floor((end + END_AFTER_S) / REFINE_FRAME_S + 1e-9))
        above = [f for f in range(first, last + 1) if env50[f] > baseline + SUSTAIN_DB]
        if above:
            new_end = (above[-1] + 1) * REFINE_FRAME_S
    if new_end is None:
        new_end = end
        flags.append("end-unrefined")
    new_end = min(new_end, duration_s)
    if new_end <= onset:
        new_end = min(duration_s, max(end, onset + REFINE_FRAME_S))
    return onset, new_end, flags


# ---------------------------------------------------------------- merge


@dataclass
class Found:
    ids: list[int]
    start: float
    end: float
    original: tuple[float, float]
    level: int
    check: Check
    flags: list[str]

    @property
    def strength(self) -> float:
        return round(self.level / 3, 3)


def merge(found: list[Found]) -> tuple[list[Found], list[dict[str, Any]]]:
    """Merge laughs that overlap or are within ``MERGE_GAP_S``; report every merge."""
    out: list[Found] = []
    merges: list[dict[str, Any]] = []
    for item in sorted(found, key=lambda f: (f.start, f.end)):
        if out and item.start <= out[-1].end + MERGE_GAP_S + 1e-9:
            last = out[-1]
            keep = max(
                (last, item),
                key=lambda f: (_STATUS_RANK[f.check.status], f.check.bump90_db or -math.inf),
            )
            out[-1] = Found(
                ids=last.ids + item.ids,
                start=min(last.start, item.start),
                end=max(last.end, item.end),
                original=(
                    min(last.original[0], item.original[0]),
                    max(last.original[1], item.original[1]),
                ),
                level=max(last.level, item.level),
                check=keep.check,
                flags=sorted(set(last.flags) | set(item.flags)),
            )
        else:
            out.append(item)
    for index, item in enumerate(out):
        if len(item.ids) > 1:
            merges.append(
                {
                    "into": index,
                    "ids": item.ids,
                    "file_start_s": round(item.start, 3),
                    "file_end_s": round(item.end, 3),
                }
            )
    return out, merges


# ---------------------------------------------------------------- audio


def _extract(media: Path, work_dir: Path, timeout_s: float) -> Path:
    """Mono 16 kHz FLAC of the whole first audio stream, written atomically."""
    target = work_dir / "audio.flac"
    tmp = work_dir / f".audio.{os.getpid()}.flac.tmp"
    try:
        run_tool(
            "ffmpeg",
            [
                *("-hide_banner", "-nostdin", "-nostats", "-v", "error", "-y"),
                *("-i", input_url(media), "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_RATE)),
                *("-c:a", "flac", "-fflags", "+bitexact", "-flags:a", "+bitexact"),
                *("-f", "flac", str(tmp)),
            ],
            timeout_s=timeout_s,
            stdout_limit=1024 * 1024,
        )
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _powers(np: Any, flac: Path, timeout_s: float, max_s: float) -> tuple[Any, int]:
    sink = _Powers(np)
    run_tool(
        "ffmpeg",
        [
            *("-hide_banner", "-nostdin", "-nostats", "-v", "error"),
            *("-t", f"{max_s + 1:.3f}", "-i", input_url(flac), "-map", "0:a:0"),
            *("-ac", "1", "-ar", str(AUDIO_RATE), "-c:a", "pcm_f32le", "-f", "f32le", "-"),
        ],
        timeout_s=timeout_s,
        stdout_sink=sink.feed,
    )
    return sink.powers(), sink.samples


# ---------------------------------------------------------------- the worker


TransportFactory = Callable[[str], GeminiTransport]


def analyse_laughs(
    media: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    work_dir: str | os.PathLike[str],
    allow_paid: bool = False,
    from_raw: str | os.PathLike[str] | None = None,
    settings: LaughSettings | None = None,
    offset: RecordingOffset | None = None,
    recording: str | None = None,
    map_out: str | os.PathLike[str] | None = None,
    environ: Mapping[str, str] | None = None,
    transport_factory: TransportFactory | None = None,
) -> dict[str, Any]:
    """Find laughs in ``media``; write ``aar.laughs/1`` to ``out`` (and a LaughMap to
    ``map_out``); return a short report. Raises ``AnalyzeError``."""
    s = settings or LaughSettings()
    offset = offset or RecordingOffset()
    env = os.environ if environ is None else environ
    if allow_paid and from_raw is not None:
        raise AnalyzeError("invalid-argument", "--allow-paid and --from-raw do not go together")
    if not allow_paid and from_raw is None:
        raise AnalyzeError(
            "paid-not-allowed",
            "this run would call the Gemini API, which is paid",
            hint="pass --allow-paid to call it, or --from-raw RAW.json to re-parse a saved reply",
        )
    try:
        media_path = require_regular_file(media, tool="ffmpeg")
    except MediaError as exc:
        raise from_media_error(exc, "input") from None
    raw_path = Path(from_raw) if from_raw is not None else None
    if raw_path is not None and not raw_path.is_file():
        raise AnalyzeError("input-missing", f"--from-raw {raw_path} is not a file")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    transport: GeminiTransport | None = None
    scrub: Callable[[str], str] = str
    if allow_paid:
        key = env.get(KEY_ENV, "")
        if not key.strip():
            raise AnalyzeError(
                "missing-credential",
                f"{KEY_ENV} is not set (or blank)",
                hint=f"export {KEY_ENV} in the environment; there is no flag for the key",
            )
        scrub = Scrubber(key.strip())
        transport = (transport_factory or _real_transport)(key.strip())
    np = need_numpy()
    try:
        info = probe(media_path)
    except MediaError as exc:
        raise from_media_error(exc, f"input {media_path}") from None
    if info.audio is None:
        raise AnalyzeError("no-audio", f"{media_path} has no audio stream")
    known = info.audio.duration_s or info.duration_s
    _check_length(known, s, media_path)
    timeout = s.timeout_s or timeout_for(known, per_media_s=0.5)
    try:
        flac = _extract(media_path, work, timeout)
        power50, samples = _powers(np, flac, timeout, s.max_audio_s)
    except MediaError as exc:
        raise from_media_error(exc, f"input {media_path}") from None
    duration = samples / AUDIO_RATE
    _check_length(duration, s, media_path)
    audio_sha = _sha256(flac)
    delete: dict[str, Any] = {"status": "skipped"}
    if transport is not None:
        response, delete = _ask_gemini(transport, flac, audio_sha, s, scrub)
        raw_text = scrub(json.dumps(response, ensure_ascii=False, indent=1))
        write_atomic(work / "raw.json", raw_text + "\n")
        response = json.loads(raw_text)
        source = "gemini"
    else:
        assert raw_path is not None
        try:
            response = json.loads(raw_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BadResponse(f"--from-raw {raw_path} is not readable JSON ({exc})") from None
        source = "raw"
    items, drops = parse_items(response_text(response), duration)
    env10, env50 = envelopes(np, power50)
    found: list[Found] = []
    applause: list[dict[str, Any]] = []
    for item in items:
        if item.kind == "applause":
            applause.append(_applause(item, offset))
            continue
        check = cross_check(np, env10, item.start, item.end, duration, s)
        start, end, flags = refine(np, env50, item.start, item.end, check.baseline_db, duration)
        original = (item.start, item.end)
        found.append(Found([item.id], start, end, original, item.level, check, item.flags + flags))
    merged, merges = merge(found)
    name = recording if recording is not None else offset.recording
    laughs = [_laugh(f, offset, s) for f in merged]
    in_map = [laugh for laugh in laughs if laugh["in_map"]]
    document = {
        "schema": SCHEMA,
        "header": {
            "model": s.model,
            "prompt": {"version": PROMPT_VERSION, "sha256": PROMPT_SHA256},
            "audio": {
                "sha256": audio_sha,
                "duration_s": round(duration, 6),
                "format": "flac",
                "rate": AUDIO_RATE,
                "channels": 1,
            },
            "settings": s.to_dict(),
            "input": {"path": str(media_path), **input_identity(media_path)},
            "offset": offset.to_dict(),
            "recording": name,
            "source": source,
        },
        "laughs": laughs,
        "applause": applause,
        "drops": drops,
        "merges": merges,
        "delete": delete,
        "summary": {
            "returned": len(items) + len(drops),
            "laughs": len(laughs),
            "confirmed": sum(1 for x in laughs if x["status"] == "confirmed"),
            "unconfirmed": sum(1 for x in laughs if x["status"] == "unconfirmed"),
            "unchecked": sum(1 for x in laughs if x["status"] == "unchecked"),
            "applause": len(applause),
            "dropped": len(drops),
            "merged": len(merges),
            "in_map": len(in_map),
        },
    }
    out_path = Path(out)
    try:
        write_atomic(out_path, scrub(json.dumps(document, ensure_ascii=False, indent=1)) + "\n")
        if map_out is not None:
            laugh_map = LaughMap(
                recording=name,
                laughs=[
                    Laugh(start_s=x["start_s"], end_s=x["end_s"], strength=x["strength"])
                    for x in in_map
                ],
            )
            write_atomic(Path(map_out), scrub(laugh_map.model_dump_json(indent=1)) + "\n")
    except OSError as exc:
        raise AnalyzeError("output-error", scrub(f"cannot write the output: {exc}")) from None
    return {
        "ok": True,
        "output": str(out_path),
        "map": None if map_out is None else str(map_out),
        "raw": str(work / "raw.json") if source == "gemini" else str(raw_path),
        "audio": str(flac),
        "delete": delete,
        "summary": document["summary"],
    }


def _real_transport(key: str) -> GeminiTransport:
    return UrllibGeminiTransport(key)


def _check_length(duration: float | None, s: LaughSettings, media: Path) -> None:
    if duration is not None and duration > s.max_audio_s:
        raise AnalyzeError(
            "input-too-long",
            f"{media} has {duration:.0f} s of audio, more than --max-audio-s {s.max_audio_s:g}",
            hint="nothing was uploaded; raise --max-audio-s if the cost is intended",
        )


def _ask_gemini(
    transport: GeminiTransport,
    flac: Path,
    audio_sha: str,
    s: LaughSettings,
    scrub: Callable[[str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Upload, wait, generate; delete in ``finally`` once uploaded. Errors are scrubbed and
    raised outside the handler, so the exception chain holds no transport object."""
    ref = None
    response: dict[str, Any] | None = None
    error: AnalyzeError | None = None
    delete: dict[str, Any] = {"status": "skipped"}
    try:
        ref = transport.upload(flac, "audio/flac", f"aar-laughs-{audio_sha[:8]}")
        active = transport.wait_active(ref, s.wait_timeout_s)
        response = transport.generate(s.model, active, LAUGH_PROMPT_V1, RESPONSE_SCHEMA)
    except AnalyzeError as exc:
        error = AnalyzeError(exc.type, scrub(str(exc)), hint=scrub(exc.hint))
    except Exception as exc:  # a transport bug must not leak the key either
        error = AnalyzeError("gemini-failed", scrub(f"{type(exc).__name__}: {exc}"))
    finally:
        if ref is not None:
            delete = _delete(transport, ref, scrub)
    if error is not None:
        if delete["status"] == "failed":
            error.hint = (error.hint + "; " if error.hint else "") + (
                f"the uploaded file could not be deleted: {delete['message']}"
            )
        raise error
    if not isinstance(response, dict):
        raise BadResponse("the generateContent reply is not a JSON object")
    return response, delete


def _delete(transport: GeminiTransport, ref: Any, scrub: Callable[[str], str]) -> dict[str, Any]:
    message = ""
    try:
        transport.delete(ref)
    except Exception as exc:
        message = scrub(f"{type(exc).__name__}: {exc}")
    if message:
        return {"status": "failed", "file": ref.name, "message": message}
    return {"status": "ok", "file": ref.name}


def _rec(seconds: float, offset: RecordingOffset) -> float:
    return round(max(0.0, seconds + offset.offset_s), 3)


def _applause(item: Item, offset: RecordingOffset) -> dict[str, Any]:
    return {
        "id": item.id,
        "file_start_s": round(item.start, 3),
        "file_end_s": round(item.end, 3),
        "start_s": _rec(item.start, offset),
        "end_s": _rec(item.end, offset),
        "level": item.level,
        "flags": item.flags,
    }


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _laugh(f: Found, offset: RecordingOffset, s: LaughSettings) -> dict[str, Any]:
    in_map = f.check.status == "confirmed" or s.include_unconfirmed
    out: dict[str, Any] = {
        "ids": f.ids,
        "file_start_s": round(f.start, 3),
        "file_end_s": round(f.end, 3),
        "start_s": _rec(f.start, offset),
        "end_s": _rec(f.end, offset),
        "original": {
            "file_start_s": round(f.original[0], 3),
            "file_end_s": round(f.original[1], 3),
        },
        "level": f.level,
        "strength": f.strength,
        "status": f.check.status,
        "baseline_db": _round(f.check.baseline_db, 2),
        "bump90_db": _round(f.check.bump90_db, 2),
        "frac6": _round(f.check.frac6, 3),
        "flags": f.flags,
        "in_map": in_map,
    }
    if f.check.reason:
        out["unchecked_reason"] = f.check.reason
    return out


__all__ = [
    "DEFAULT_MODEL",
    "LAUGH_PROMPT_V1",
    "PROMPT_SHA256",
    "PROMPT_VERSION",
    "RESPONSE_SCHEMA",
    "SCHEMA",
    "BadResponse",
    "LaughSettings",
    "analyse_laughs",
    "cross_check",
    "envelopes",
    "merge",
    "parse_items",
    "parse_time",
    "refine",
    "response_text",
]
