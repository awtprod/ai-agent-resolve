"""Loudness of a render's mix, to the profile's ``[qa]`` target (S18b, R6 decision, round 2).

Both kinds get the same treatment, with their own target (``[qa] short_lufs`` or
``longform_lufs``): the mix is measured, brought there by **one linear gain** (``target -
integrated``), and a true-peak limiter (``alimiter`` at 192 kHz, so its sample peak follows the
true peak, latency compensated) holds the peaks at ``[qa] true_peak_max_dbtp`` less ``[render]
true_peak_margin_db`` (so the AAC encode stays at or under the maximum). Nothing compresses the
programme: loudnorm's dynamic mode flattened the laugh range on shorts and missed the target on
quiet long-forms with sharp peaks, so it is not used for either kind. The result is measured
again and the gain corrected (at most twice) while it lands more than ``[render]
gain_tolerance_lu`` from the target, since limiting loud peaks lowers the loudness.

The report gives the achieved loudness and the limiter's gain reduction (the true peak the gain
alone would give, less the true peak measured after the limiter), and a warning when that is more
than ``[render] limiter_warn_db`` (3 dB): the material needs a louder recording or a manual
limiter, not more of this one. The exact sample length is kept (``apad`` + ``atrim``: the
resamplers must not pad the tail), and checked. A silent mix is not normalized (there is no
loudness to measure); the report says so and QA's loudness check fails it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from aar.media import MediaError, MediaTimeout, input_url, measure_audio, probe
from aar.plan.profile import QATargets, RenderSettings
from aar.plan.render.errors import RenderError
from aar.plan.render.runner import run_ffmpeg

Method = Literal["gain-limiter", "none"]
#: Below this integrated loudness (or -inf) the mix is treated as silent.
SILENT_LUFS = -70.0
#: The mix is rendered at most this many times: the gain, then up to two corrections.
GAIN_PASSES = 3


@dataclass
class LoudnessResult:
    method: Method
    target_lufs: float
    process_true_peak_dbtp: float
    mix: dict[str, float | None]
    output: dict[str, float | None] = field(default_factory=dict)
    gain_db: float | None = None
    passes: int = 0
    limiter_reduction_db: float | None = None
    output_samples: int | None = None
    warning: str | None = None
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
            "limiter_reduction_db": self.limiter_reduction_db,
            "output_samples": self.output_samples,
            "warning": self.warning,
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


def _samples(path: Path) -> int | None:
    """The sample count of a written WAV (its duration at 48 kHz)."""
    try:
        info = probe(path)
    except MediaError:
        return None
    stream = info.audio
    duration = None if stream is None else stream.duration_s
    return None if duration is None else round(duration * 48000)


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
    result = LoudnessResult("gain-limiter", target, peak, measured)
    level = measured["integrated_lufs"]
    if level is None or level <= SILENT_LUFS:
        _write(mix, out, _exact_length(samples), "copy", timeout_s, cwd)
        result.method = "none"
        result.note = "the mix is silent: nothing to measure, so it is not normalized"
        result.output = _measure(out, qa, timeout_s)
        result.output_samples = _samples(out)
        return result
    gain = target - level
    limit = 10 ** (peak / 20)
    for attempt in range(1, GAIN_PASSES + 1):
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
        if got is None or abs(got - target) <= settings.gain_tolerance_lu:
            break
        gain += target - got
    mix_peak, out_peak = measured["true_peak_dbtp"], result.output["true_peak_dbtp"]
    if mix_peak is not None and out_peak is not None and result.gain_db is not None:
        result.limiter_reduction_db = round(max(0.0, mix_peak + result.gain_db - out_peak), 2)
        if result.limiter_reduction_db > settings.limiter_warn_db:
            result.warning = (
                f"the limiter took {result.limiter_reduction_db:g} dB off the peaks to reach "
                f"{target:g} LUFS (more than {settings.limiter_warn_db:g} dB): the peaks are far "
                "above the programme; check the transients, or limit them by hand"
            )
    result.output_samples = _samples(out)
    return result
