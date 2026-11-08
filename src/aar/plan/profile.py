"""Rule profiles: a show's house rules and QA targets, stored as TOML data.

A profile lists the rules to run and their thresholds. Code never hard-codes a threshold.
Built-in profiles live in ``aar/plan/profiles/<name>.toml``; a path to any other ``.toml``
file also works, so other shows can ship their own.
"""

from __future__ import annotations

import re
import tomllib
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Severity = Literal["error", "warn", "info"]

_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*")


class ProfileError(ValueError):
    """A profile is missing, malformed, or names unknown rules or thresholds."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuleConfig(_Model):
    """How one rule runs in a profile.

    ``severity`` overrides the severity of the rule's error and warn findings (the info
    "skipped" findings keep theirs). ``params`` are the rule's thresholds.
    """

    enabled: bool = True
    severity: Severity | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class QATargets(_Model):
    """Targets ``aar.qa.check`` applies to a render; the plan linter does not use them.

    The first four keys are required. Every later key has a default, so a profile written
    before it existed still loads. See docs/DESIGN.md "Media QA" for what each check measures.
    TOML allows ``nan`` and ``inf``; they are refused here, because ``lufs_tolerance = inf``
    would pass every loudness and ``short_lufs = nan`` would fail every one.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    longform_lufs: float
    short_lufs: float
    lufs_tolerance: float = Field(ge=0)
    review_mp4_max_mb: float = Field(gt=0)
    true_peak_max_dbtp: float = Field(default=-1.0, le=0)
    clip_level_dbfs: float = Field(default=-0.01, le=0)
    silence_db: float = Field(default=-42.0, lt=0)
    silence_min_s: float = Field(default=0.2, gt=0)
    longest_silence_warn_s: float | None = Field(default=None, gt=0)
    duration_tolerance_frames: int = Field(default=1, ge=0)
    join_continuity_tolerance_frames: int = Field(default=1, ge=0)
    join_quiet_db: float = Field(default=-40.0, lt=0)
    join_quiet_min_ms: float = Field(default=60.0, gt=0)
    join_quiet_search_s: float = Field(default=0.10, gt=0)
    join_quiet_slack_ms: float = Field(default=10.0, ge=0)
    join_quiet_window_ms: float = Field(default=10.0, gt=0)
    join_windows: int = Field(default=3, ge=2, le=3)
    join_window_s: float = Field(default=4.0, gt=0)
    join_window_step_s: float = Field(default=0.5, gt=0)
    join_window_edge_s: float = Field(default=0.25, ge=0)
    join_agreement_min: float = Field(default=0.8, ge=0, le=1)
    word_straddle_s: float = Field(default=0.03, ge=0)

    @model_validator(mode="after")
    def _consistent(self) -> QATargets:
        spread = self.join_window_step_s * (self.join_windows - 1)
        if spread + 2 * self.join_window_edge_s >= self.join_window_s:
            raise ValueError(
                "join windows leave no region to compare: join_window_s must be longer than "
                "join_window_step_s * (join_windows - 1) + 2 * join_window_edge_s"
            )
        if self.join_quiet_window_ms > self.join_quiet_min_ms:
            raise ValueError("join_quiet_window_ms must not exceed join_quiet_min_ms")
        if self.join_quiet_min_ms + self.join_quiet_slack_ms > self.join_quiet_search_s * 1000:
            raise ValueError(
                "join_quiet_search_s must cover join_quiet_min_ms + join_quiet_slack_ms, or a "
                "pause that contains src_in could be cut short by the search window"
            )
        return self


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class VenueOverride(_Settings):
    """A venue's change to the pause scan: loud rooms raise the silence threshold."""

    silence_db: float = Field(lt=0)


class FramingPreset(_Settings):
    """A shorts crop preset: a static zoom and frame-centre height for a whole segment.

    ``zoom`` 1.0 is the tallest crop the output aspect allows (full source height). ``y`` is the
    crop centre, 0-1 of the source height, and must keep the crop inside the picture.
    """

    name: str = Field(min_length=1)
    zoom: float = Field(ge=1.0)
    y: float = Field(default=0.5, ge=0, le=1)

    @model_validator(mode="after")
    def _inside(self) -> FramingPreset:
        half = 0.5 / self.zoom
        if not half - 1e-9 <= self.y <= 1 - half + 1e-9:
            raise ValueError(
                f"framing {self.name!r}: y {self.y:g} puts the crop outside the picture at zoom "
                f"{self.zoom:g}; it must lie between {half:g} and {1 - half:g}"
            )
        return self


class SteadySettings(_Settings):
    """The shorts camera path, ported from the supoclip planner's ``steady()``.

    Units: seconds, or fractions of the crop width (``dead_zone``) or of the source width
    (``safety_margin``, ``warn_margin``).
    """

    pre_smooth_s: float = Field(default=0.17, ge=0)
    start_median_s: float = Field(default=0.5, gt=0)
    dead_zone: float = Field(default=0.25, ge=0, lt=1)
    glide_sigma_s: float = Field(default=0.7, gt=0)
    safety_margin: float = Field(default=0.057, ge=0, lt=0.5)
    safety_window_s: float = Field(default=0.6, ge=0)
    safety_sigma_s: float = Field(default=0.3, gt=0)
    safety_passes: int = Field(default=4, ge=0)
    warn_margin: float = Field(default=0.047, ge=0, lt=0.5)
    key_step_s: float = Field(default=0.5, gt=0)
    # At most 0.95: keyframe x is rounded to 4 decimals, so at 1.0 a clamped path can land just
    # over no-whip-pan's limit and fail the producer's own lint gate.
    speed_headroom: float = Field(default=0.9, gt=0, le=0.95)


def _default_framing() -> list[FramingPreset]:
    return [FramingPreset(name="A", zoom=1.0, y=0.5), FramingPreset(name="B", zoom=1.22, y=0.45)]


class ProduceSettings(_Settings):
    """Parameters of the plan producers (``aar plan produce``); the plan linter ignores them.

    Every key has a default, so a profile without a ``[produce]`` block still loads. See
    docs/DESIGN.md "Plan producers" for what each one does.
    """

    default_camera: str = Field(default="A", min_length=1)
    min_pause_s: float = Field(default=0.9, gt=0)
    min_word_ms: float = Field(default=40.0, ge=0)
    beat_s: float = Field(default=0.45, ge=0)
    beat_lead_s: float = Field(default=0.25, ge=0)
    silence_db: float = Field(default=-42.0, lt=0)
    scan_window_ms: float = Field(default=10.0, gt=0)
    scan_rate_hz: int = Field(default=16000, ge=8000, le=192000)
    scan_chunk_s: float = Field(default=30.0, gt=0)
    join_nudge_ms: float = Field(default=150.0, ge=0)
    out_quiet_preroll_ms: float = Field(default=20.0, gt=0)
    min_shot_s: float = Field(default=1.0, gt=0)
    no_pause_warn_s: float = Field(default=60.0, gt=0)
    end_after_ms: float = Field(default=200.0, ge=0)
    crossfade_frames: int = Field(default=2, ge=1)
    punch_in_zoom: float = Field(default=1.25, gt=0)
    venues: dict[str, VenueOverride] = Field(default_factory=dict)
    steady: SteadySettings = Field(default_factory=SteadySettings)
    framing: list[FramingPreset] = Field(default_factory=_default_framing, min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ProduceSettings:
        if self.beat_s >= self.min_pause_s:
            raise ValueError("beat_s must be shorter than min_pause_s, or no pause is trimmed")
        if self.beat_lead_s > self.beat_s:
            raise ValueError("beat_lead_s is part of the beat, so it must not exceed beat_s")
        if self.punch_in_zoom == 1.0:
            raise ValueError("punch_in_zoom 1.0 is no punch-in; jump cuts need a zoom change")
        names = [preset.name for preset in self.framing]
        if len(set(names)) != len(names):
            raise ValueError(f"framing preset names must be unique, not {names}")
        return self

    def silence_for(self, venue: str | None) -> float:
        """The pause threshold for ``venue`` (the default when ``None``)."""
        if venue is None:
            return self.silence_db
        if venue not in self.venues:
            known = ", ".join(sorted(self.venues)) or "none"
            raise ProfileError(f"unknown venue {venue!r}; this profile's venues: {known}")
        return self.venues[venue].silence_db


class ExportSettings(_Model):
    """Settings for ``aar plan export`` (FCPXML). Optional, so an older profile still loads."""

    timeline_start: str = Field(
        default="01:00:00:00",
        pattern=r"^[0-9]{2}:[0-9]{2}:[0-9]{2}:[0-9]{2}$",
        description="Non-drop timecode the exported timeline starts at.",
    )


X264Preset = Literal[
    "ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"
]


class RenderSettings(_Settings):
    """Settings for ``aar plan render`` (an EditPlan to MP4 with ffmpeg on the hub, S18b).

    Optional, with defaults in code, so an older profile still loads. The loudness targets are
    the ``[qa]`` ones; these are how the render reaches them. See docs/DESIGN.md "Hub render".
    """

    font: str = Field(
        default="Montserrat",
        min_length=1,
        description="Title font family, looked up as a font file (AAR_FONTS_DIR, then the system "
        "font folders); DejaVu Sans is the documented fallback.",
    )
    master_codec: Literal["h264", "prores"] = Field(
        default="h264", description="The master's video codec; prores writes a .mov (HQ, PCM)."
    )
    review_crf: int = Field(default=20, ge=0, le=51)
    review_x264_preset: X264Preset = "medium"
    review_audio_kbps: int = Field(default=192, ge=64, le=512)
    master_crf: int = Field(default=14, ge=0, le=51)
    master_x264_preset: X264Preset = "slow"
    master_audio_kbps: int = Field(default=320, ge=64, le=512)
    longform_lra: float = Field(
        default=20.0,
        ge=1,
        le=50,
        description="loudnorm's LRA target for long-forms: high, so the linear (one gain) mode "
        "holds and the laugh range is kept.",
    )
    true_peak_margin_db: float = Field(
        default=0.5,
        ge=0,
        le=3,
        description="Headroom below [qa] true_peak_max_dbtp the audio is processed to, so the AAC "
        "encode stays under the maximum.",
    )
    limiter_attack_ms: float = Field(default=5.0, ge=0.1, le=80)
    limiter_release_ms: float = Field(default=50.0, ge=1, le=8000)
    short_gain_tolerance_lu: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="A short's gain is measured again after the limiter and corrected (up to "
        "twice) while it lands further than this from the target.",
    )
    fps_tolerance: float = Field(
        default=0.001,
        ge=0,
        lt=0.05,
        description="How far (as a share) a source's frame rate may sit from the plan's. Frames "
        "are taken by index and retimed to the plan's rate.",
    )
    timeout_base_s: float = Field(default=120.0, gt=0)
    timeout_per_media_s: float = Field(
        default=8.0, gt=0, description="Seconds of ffmpeg time allowed per second of timeline."
    )


class Profile(_Model):
    name: str
    description: str = ""
    qa: QATargets
    produce: ProduceSettings = Field(default_factory=ProduceSettings)
    rules: dict[str, RuleConfig]
    export: ExportSettings = Field(default_factory=ExportSettings)
    render: RenderSettings = Field(default_factory=RenderSettings)


def parse_profile(data: dict[str, Any], *, origin: str = "profile") -> Profile:
    """Build a ``Profile`` from parsed TOML, splitting each rule table into settings and params."""
    data = dict(data)
    rules: dict[str, Any] = {}
    for rule_id, table in (data.pop("rules", None) or {}).items():
        if not isinstance(table, dict):
            raise ProfileError(f"{origin}: [rules.{rule_id}] must be a table")
        table = dict(table)
        rules[rule_id] = {
            "enabled": table.pop("enabled", True),
            "severity": table.pop("severity", None),
            "params": table,
        }
    try:
        return Profile.model_validate({**data, "rules": rules})
    except ValidationError as exc:
        raise ProfileError(f"{origin}: {exc}") from exc


def load_profile(name_or_path: str | Path) -> Profile:
    """Load a built-in profile by name (``ccn``) or any profile TOML by path."""
    text = str(name_or_path)
    if _NAME.fullmatch(text):
        resource = resources.files("aar.plan").joinpath("profiles", f"{text}.toml")
        if not resource.is_file():
            raise ProfileError(f"no built-in rule profile named {text!r}")
        raw = resource.read_text(encoding="utf-8")
        origin = f"profile {text}"
    else:
        path = Path(text)
        if not path.is_file():
            raise ProfileError(f"profile file not found: {path}")
        raw = path.read_text(encoding="utf-8")
        origin = str(path)
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as exc:
        raise ProfileError(f"{origin}: invalid TOML: {exc}") from exc
    return parse_profile(data, origin=origin)
