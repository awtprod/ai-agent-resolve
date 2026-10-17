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

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
    """Targets checked after rendering by later slices; the plan linter does not use them."""

    longform_lufs: float
    short_lufs: float
    lufs_tolerance: float = Field(ge=0)
    review_mp4_max_mb: float = Field(gt=0)


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
