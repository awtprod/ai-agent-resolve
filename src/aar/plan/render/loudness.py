"""Loudness of a render's mix, to the profile's ``[qa]`` target (S18b, R6 decision).

* **Long-form**: two-pass ``loudnorm``. Pass one measures the mix; pass two normalizes with those
  measurements (``linear=true``, so one gain when the target LRA and peak allow it; ffmpeg reports
  ``dynamic`` otherwise, and the report says which), to ``[qa] longform_lufs``.
* **Short**: gain plus a limiter, never loudnorm, which flattened the laugh range on real shorts.
  The mix is measured, the gain is ``target - integrated``, and ``alimiter`` (run at 192 kHz, so
  its sample peak follows the true peak) holds the peaks. The result is measured again and the
  gain corrected (at most twice) while it lands more than ``[render] short_gain_tolerance_lu`` away.

Both process to ``[qa] true_peak_max_dbtp`` less ``[render] true_peak_margin_db``, so the AAC
encode stays at or under the maximum, and both keep the exact sample length (``apad`` +
``atrim``: loudnorm and the resamplers must not pad the tail). A silent mix is not normalized
(there is no loudness to measure); the report says so and QA's loudness check fails it.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aar.media import MediaError, MediaTimeout, input_url, measure_audio
from aar.plan.profile import QATargets, RenderSettings
from aar.plan.render.errors import RenderError
from aar.plan.render.runner import run_ffmpeg

Method = Literal["loudnorm-2pass", "gain-limiter", "none"]
#: Below this integrated loudness (or -inf) the mix is treated as silent.
SILENT_LUFS = -70.0
#: A short is rendered at most this many times: the gain, then up to two corrections (limiting
#: a loud laugh lowers the loudness, so one correction can land short of the target).
SHORT_GAIN_PASSES = 3
_JSON_BLOCK = re.compile(r"\{[^{}]*\}", re.S)


@dataclass
class LoudnessResult:
    method: Method
    target_lufs: float
    process_true_peak_dbtp: float
    mix: dict[str, float | None]
    output: dict[str, float | None] = field(default_factory=dict)
    gain_db: float | None = None
    passes: int = 0
    normalization_type: str | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "target_lufs": self.target_lufs,
            "process_true_peak_dbtp": self.process_true_peak_dbtp,
            "mix": self.mix,
            "output": self.output,
            "gain_db": self.gain_db,
            "passes": self.passes,
            "normalization_type": self.normalization_type,
            "note": self.note,
        }


def _finite(value: float | None) -> float | None:
    return value if value is not None and math.isfinite(value) else None


def _measure(path: Path, qa: QATargets, timeout_s: float) -> dict[str, float | None]:
    try:
        m = measure_audio(
            path, silence_db=qa.silence_db, silence_min_s=qa.silence_min_s, timeout_s=timeout_s
        )
    except MediaTimeout as exc:
        raise RenderError(
            "render_timeout",
            f"measuring the loudness of {path.name} ran past {exc.timeout_s:g} s",
            "raise [render] timeout_per_media_s or timeout_base_s for this machine",
            stage="loudness-measure",
            timeout_s=exc.timeout_s,
        ) from None
    except MediaError as exc:
        raise RenderError(
            "render_failed",
            f"measuring the loudness of {path.name} failed: {exc.summary()}",
            "check ffmpeg; the render's work dir is removed, so rerun to reproduce",
            stage="loudness-measure",
        ) from None
    return {
        "integrated_lufs": _finite(m.loudness.integrated_lufs),
        "true_peak_dbtp": _finite(m.loudness.true_peak_dbtp),
        "lra_lu": _finite(m.loudness.lra_lu),
    }


def _loudnorm_json(stderr: str) -> dict[str, Any]:
    """loudnorm's ``print_format=json`` block: the last JSON object in its log."""
    blocks = _JSON_BLOCK.findall(stderr[stderr.rfind("Parsed_loudnorm") :])
    if not blocks:
        raise RenderError(
            "render_failed",
            "loudnorm printed no measurement",
            "check ffmpeg's loudnorm filter",
            stage="loudnorm",
        )
    try:
        data = json.loads(blocks[-1])
    except json.JSONDecodeError as exc:
        raise RenderError(
            "render_failed",
            f"loudnorm printed unreadable JSON: {exc}",
            "check ffmpeg's loudnorm filter",
            stage="loudnorm",
        ) from None
    if not isinstance(data, dict):
        raise RenderError(
            "render_failed", "loudnorm JSON is not an object", "check ffmpeg", stage="loudnorm"
        )
    return data


def _number(data: dict[str, Any], key: str) -> float:
    try:
        value = float(data[key])
    except (KeyError, TypeError, ValueError):
        raise RenderError(
            "render_failed",
            f"loudnorm's measurement has no number for {key!r}",
            "check ffmpeg's loudnorm filter",
            stage="loudnorm",
        ) from None
    return value


def _exact_length(samples: int) -> str:
    return f"apad=whole_len={samples},atrim=end_sample={samples}"


def _write(mix: Path, out: Path, chain: str, stage: str, timeout_s: float, cwd: Path) -> str:
    return run_ffmpeg(
        [
            *("-v", "info", "-y", "-i", input_url(mix), "-map", "0:a:0", "-af", chain),
            *("-c:a", "pcm_f32le", "-rf64", "auto", "-f", "wav", input_url(out)),
        ],
        stage,
        timeout_s,
        cwd,
    )


def normalize(
    mix: Path,
    out: Path,
    *,
    kind: str,
    qa: QATargets,
    settings: RenderSettings,
    samples: int,
    timeout_s: float,
    cwd: Path,
) -> LoudnessResult:
    """Write ``out``: the mix brought to the kind's loudness target (module docstring)."""
    target = qa.short_lufs if kind == "short" else qa.longform_lufs
    peak = qa.true_peak_max_dbtp - settings.true_peak_margin_db
    measured = _measure(mix, qa, timeout_s)
    method: Method = "gain-limiter" if kind == "short" else "loudnorm-2pass"
    result = LoudnessResult(method, target, peak, measured)
    level = measured["integrated_lufs"]
    if level is None or level <= SILENT_LUFS:
        _write(mix, out, _exact_length(samples), "copy", timeout_s, cwd)
        result.method = "none"
        result.note = "the mix is silent: nothing to measure, so it is not normalized"
        result.output = _measure(out, qa, timeout_s)
        return result
    if kind == "short":
        return _gain_limiter(mix, out, result, qa, settings, samples, timeout_s, cwd)
    lra = settings.longform_lra
    head = f"loudnorm=I={target:g}:TP={peak:g}:LRA={lra:g}"
    first = _loudnorm_json(
        run_ffmpeg(
            [
                *("-v", "info", "-i", input_url(mix), "-map", "0:a:0"),
                *("-af", f"{head}:print_format=json", "-f", "null", "-"),
            ],
            "loudnorm-measure",
            timeout_s,
            cwd,
        )
    )
    values = {key: _number(first, key) for key in ("input_i", "input_tp", "input_lra")}
    thresh, offset = _number(first, "input_thresh"), _number(first, "target_offset")
    chain = (
        f"{head}:measured_I={values['input_i']:g}:measured_TP={values['input_tp']:g}"
        f":measured_LRA={values['input_lra']:g}:measured_thresh={thresh:g}:offset={offset:g}"
        f":linear=true:print_format=json,aresample=48000,{_exact_length(samples)}"
    )
    second = _loudnorm_json(_write(mix, out, chain, "loudnorm", timeout_s, cwd))
    result.passes = 2
    result.normalization_type = str(second.get("normalization_type"))
    result.output = _measure(out, qa, timeout_s)
    return result


def _gain_limiter(
    mix: Path,
    out: Path,
    result: LoudnessResult,
    qa: QATargets,
    settings: RenderSettings,
    samples: int,
    timeout_s: float,
    cwd: Path,
) -> LoudnessResult:
    level = result.mix["integrated_lufs"]
    assert level is not None
    gain = result.target_lufs - level
    limit = 10 ** (result.process_true_peak_dbtp / 20)
    for attempt in range(1, SHORT_GAIN_PASSES + 1):
        chain = (
            f"volume=volume={gain:.4f}dB,aresample=192000,"
            f"alimiter=limit={limit:.6f}:attack={settings.limiter_attack_ms:g}"
            f":release={settings.limiter_release_ms:g}:level=0:latency=1,"
            f"aresample=48000,{_exact_length(samples)}"
        )
        _write(mix, out, chain, "gain-limiter", timeout_s, cwd)
        result.output = _measure(out, qa, timeout_s)
        result.gain_db = round(gain, 4)
        result.passes = attempt
        got = result.output["integrated_lufs"]
        if got is None or abs(got - result.target_lufs) <= settings.short_gain_tolerance_lu:
            break
        gain += result.target_lufs - got
    return result
