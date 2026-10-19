"""Rule engine: run a profile's rules over an EditPlan and return structured findings.

A rule is a function registered with :func:`rule`. It receives a :class:`LintContext` and
returns findings. Which rules run, and with what thresholds, comes from the profile.
Rules that need an optional input (transcript, laugh map) that was not supplied produce one
``info`` finding saying so; they never pass silently.

A rule's thresholds are either required (``params``: every profile that lists the rule must set
them) or optional (``defaults``: a profile may set them; the code default applies otherwise).
Optional thresholds let a rule grow without breaking profiles written before it grew. A rule
may also have a ``prepare`` hook that checks the thresholds' values and combinations; a bad
value is a :class:`ProfileError` before anything is linted.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aar.plan.inputs import LaughMap, Transcript
from aar.plan.model import EditPlan, Source
from aar.plan.profile import Profile, ProfileError, RuleConfig, Severity

Need = Literal["transcript", "laughs"]
Point = Literal["in", "out"]

_NEED_HELP: dict[str, tuple[str, str]] = {
    "transcript": ("a transcript", "--transcript FILE"),
    "laughs": ("a laugh map", "--laughs FILE"),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Ref(_Model):
    """A pointer into the plan: a segment index and/or record and source timecodes."""

    segment: int | None = None
    item: str | None = Field(default=None, description="Non-segment item, e.g. 'caption[2]'.")
    point: Point | None = Field(default=None, description="Which end of the segment.")
    source_id: str | None = None
    record_frame: int | None = None
    record_tc: str | None = None
    source_frame: int | None = None
    source_tc: str | None = None


class Finding(_Model):
    rule: str
    severity: Severity
    message: str
    fix_hint: str
    refs: list[Ref] = Field(default_factory=list)


class LintReport(_Model):
    profile: str
    ok: bool = Field(description="True when there are no error findings.")
    counts: dict[str, int]
    duration_frames: int
    duration_s: float
    findings: list[Finding]


Prepare = Callable[[Mapping[str, Any], dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class Rule:
    id: str
    severity: Severity  # the highest severity the rule can emit
    summary: str
    params: tuple[str, ...]  # required thresholds
    check: Callable[[LintContext], list[Finding]]
    needs: tuple[Need, ...] = ()
    defaults: Mapping[str, Any] = field(default_factory=dict)  # optional thresholds
    prepare: Prepare | None = None

    def allowed(self) -> set[str]:
        """Every threshold name a profile may set for this rule."""
        return set(self.params) | set(self.defaults)

    def resolve(self, given: Mapping[str, Any]) -> dict[str, Any]:
        """The thresholds the rule runs with: the profile's, then the code defaults.

        ``prepare`` (if any) sees what the profile set and the merged values, and may check,
        translate or reject them. A rejection is a ``ValueError``.
        """
        merged = {**self.defaults, **given}
        return merged if self.prepare is None else self.prepare(given, merged)


_REGISTRY: dict[str, Rule] = {}


def rule(
    rule_id: str,
    *,
    severity: Severity,
    summary: str,
    params: tuple[str, ...] = (),
    needs: tuple[Need, ...] = (),
    defaults: Mapping[str, Any] | None = None,
    prepare: Prepare | None = None,
) -> Callable[[Callable[[LintContext], list[Finding]]], Callable[[LintContext], list[Finding]]]:
    """Register a rule.

    ``params`` are the threshold names the profile must supply; ``defaults`` maps optional
    threshold names to the value used when the profile does not set them (``None`` means the
    rule's ``prepare`` decides). ``prepare`` checks and translates the merged thresholds.
    """

    def decorate(
        func: Callable[[LintContext], list[Finding]],
    ) -> Callable[[LintContext], list[Finding]]:
        if rule_id in _REGISTRY:
            raise ValueError(f"rule {rule_id!r} is registered twice")
        if clash := set(params) & set(defaults or {}):
            raise ValueError(f"rule {rule_id!r} lists {sorted(clash)} as required and optional")
        _REGISTRY[rule_id] = Rule(
            rule_id, severity, summary, params, func, needs, dict(defaults or {}), prepare
        )
        return func

    return decorate


def number(
    params: Mapping[str, Any],
    name: str,
    *,
    integer: bool = False,
    minimum: float | None = None,
    above: float | None = None,
) -> float:
    """Read one numeric threshold for a ``prepare`` hook, or raise a clear ``ValueError``.

    ``minimum`` is inclusive and ``above`` exclusive. Booleans are not numbers here.
    """
    value = params[name]
    kind = "a whole number" if integer else "a number"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be {kind}, not {value!r}")
    if integer and not (isinstance(value, int) or float(value).is_integer()):
        raise ValueError(f"{name} must be {kind}, not {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, not {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum:g}, not {value!r}")
    if above is not None and value <= above:
        raise ValueError(f"{name} must be more than {above:g}, not {value!r}")
    return value


def flag(params: Mapping[str, Any], name: str) -> bool:
    """Read one true/false threshold for a ``prepare`` hook, or raise a clear ``ValueError``."""
    value = params[name]
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, not {value!r}")
    return value


def all_rules() -> dict[str, Rule]:
    """Every registered rule by id (importing the rule modules on first use)."""
    from aar.plan import rules  # noqa: F401  (importing registers the rules)

    return dict(_REGISTRY)


class LintContext:
    """What a rule sees: the plan, its thresholds, optional inputs and layout helpers."""

    def __init__(
        self,
        plan: EditPlan,
        rule_id: str,
        params: dict[str, Any],
        transcript: Transcript | None,
        laughs: LaughMap | None,
    ) -> None:
        self.plan = plan
        self.rule_id = rule_id
        self.params = params
        self.transcript = transcript
        self.laughs = laughs
        self.sources = plan.source_map()
        self.ranges = plan.record_ranges()

    def source(self, index: int) -> Source | None:
        return self.sources.get(self.plan.segments[index].source)

    def usable(self, index: int) -> bool:
        """True if the segment names a known source and has a positive length."""
        return self.source(index) is not None and self.plan.segments[index].length > 0

    def recording(self, index: int) -> str | None:
        source = self.source(index)
        return None if source is None else self.plan.recording_key(source)

    def applies(self, recording: str | None, index: int) -> bool:
        """True if an input naming ``recording`` covers the segment's source.

        ``None`` covers everything; otherwise the name must match the source's recording id
        or, for a source without one, its id.
        """
        source = self.source(index)
        if recording is None:
            return True
        return source is not None and recording in (source.recording_id, source.id)

    def recording_range(self, index: int) -> tuple[int, int]:
        """Recording-time range of a segment; the source must be known."""
        source = self.source(index)
        assert source is not None, "recording_range needs a known source"
        seg = self.plan.segments[index]
        return source.recording_origin + seg.src_in, source.recording_origin + seg.src_out

    def is_continuous(self, outgoing: int, incoming: int, tolerance: int) -> bool:
        """True if the incoming segment resumes the same recording where the outgoing ended."""
        if self.recording(outgoing) != self.recording(incoming):
            return False
        gap = self.recording_range(incoming)[0] - self.recording_range(outgoing)[1]
        return abs(gap) <= tolerance

    def ref(self, index: int, point: Point | None = None) -> Ref:
        """A reference to a segment, optionally to its in or out point."""
        seg = self.plan.segments[index]
        fps = self.plan.fps
        ref = Ref(segment=index, source_id=seg.source, point=point)
        if point is not None:
            record = self.ranges[index][0 if point == "in" else 1]
            source = seg.src_in if point == "in" else seg.src_out
            ref.record_frame, ref.record_tc = record, fps.timecode(record)
            ref.source_frame, ref.source_tc = source, fps.timecode(source)
        return ref

    def ref_at(self, index: int, offset: int) -> Ref:
        """A reference to the frame ``offset`` frames into a segment."""
        seg = self.plan.segments[index]
        fps = self.plan.fps
        record = self.ranges[index][0] + offset
        source = seg.src_in + offset
        return Ref(
            segment=index,
            source_id=seg.source,
            record_frame=record,
            record_tc=fps.timecode(record),
            source_frame=source,
            source_tc=fps.timecode(source),
        )

    def time_jump(self, outgoing: int, incoming: int, tolerance: int) -> bool:
        """True if both segments are from one recording but the incoming one skips in time."""
        same = self.recording(outgoing) == self.recording(incoming)
        return same and not self.is_continuous(outgoing, incoming, tolerance)

    def record_ref(self, frame: int, item: str | None = None) -> Ref:
        """A reference to a record position that is not tied to one segment."""
        return Ref(item=item, record_frame=frame, record_tc=self.plan.fps.timecode(frame))

    def finding(
        self, severity: Severity, message: str, fix_hint: str, refs: list[Ref] | None = None
    ) -> Finding:
        return Finding(
            rule=self.rule_id,
            severity=severity,
            message=message,
            fix_hint=fix_hint,
            refs=refs or [],
        )


def validate_profile(profile: Profile) -> None:
    """Raise ``ProfileError`` if the profile names unknown rules or wrong thresholds."""
    known = all_rules()
    for rule_id, config in profile.rules.items():
        if rule_id not in known:
            raise ProfileError(
                f"profile {profile.name!r} lists unknown rule {rule_id!r}; "
                f"known rules: {', '.join(sorted(known))}"
            )
        rule_ = known[rule_id]
        given = set(config.params)
        if missing := set(rule_.params) - given:
            raise ProfileError(
                f"profile {profile.name!r} rule {rule_id!r} is missing {sorted(missing)}"
            )
        if extra := given - rule_.allowed():
            raise ProfileError(
                f"profile {profile.name!r} rule {rule_id!r} has unknown thresholds {sorted(extra)}"
            )
        _resolved(profile, rule_, config)


def _resolved(profile: Profile, rule_: Rule, config: RuleConfig) -> dict[str, Any]:
    """The thresholds a rule runs with; a value its ``prepare`` rejects is a ``ProfileError``."""
    try:
        return rule_.resolve(config.params)
    except ValueError as exc:
        raise ProfileError(f"profile {profile.name!r} rule {rule_.id!r}: {exc}") from exc


def _skipped(rule_: Rule, missing: list[Need]) -> Finding:
    wanted = " and ".join(_NEED_HELP[need][0] for need in missing)
    flags = " ".join(_NEED_HELP[need][1] for need in missing)
    return Finding(
        rule=rule_.id,
        severity="info",
        message=f"Skipped: this rule needs {wanted}, but none was supplied.",
        fix_hint=f"Supply it ({flags} on the CLI) to enable this check.",
    )


def _run_rule(rule_: Rule, config: RuleConfig, ctx: LintContext) -> list[Finding]:
    findings = rule_.check(ctx)
    for found in findings:
        if config.severity is not None and found.severity != "info":
            found.severity = config.severity
    return findings


def lint(
    plan: EditPlan,
    profile: Profile,
    *,
    transcript: Transcript | None = None,
    laughs: LaughMap | None = None,
) -> LintReport:
    """Run every enabled rule of the profile over the plan."""
    validate_profile(profile)
    known = all_rules()
    available: dict[str, Any] = {"transcript": transcript, "laughs": laughs}
    findings: list[Finding] = []
    for rule_id, config in profile.rules.items():
        if not config.enabled:
            continue
        rule_ = known[rule_id]
        if missing := [need for need in rule_.needs if available[need] is None]:
            findings.append(_skipped(rule_, missing))
            continue
        ctx = LintContext(plan, rule_id, _resolved(profile, rule_, config), transcript, laughs)
        findings.extend(_run_rule(rule_, config, ctx))
    counts = {level: sum(1 for f in findings if f.severity == level) for level in _LEVELS}
    total = plan.total_frames()
    return LintReport(
        profile=profile.name,
        ok=counts["error"] == 0,
        counts=counts,
        duration_frames=total,
        duration_s=round(plan.fps.seconds(total), 3),
        findings=findings,
    )


_LEVELS: tuple[Severity, ...] = ("error", "warn", "info")
