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


class Profile(_Model):
    name: str
    description: str = ""
    qa: QATargets
    rules: dict[str, RuleConfig]


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
