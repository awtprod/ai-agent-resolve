"""Fixtures for the ``aar.plan`` tests: the invented example plans and small lint helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

import pytest
from plan_support import FIXTURES, PlanData

from aar.plan.inputs import LaughMap, Transcript, load_laughs, load_transcript
from aar.plan.lint import LintReport, lint
from aar.plan.model import EditPlan
from aar.plan.profile import Profile, load_profile


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def clean_data() -> PlanData:
    """A fresh copy of the clean short, which passes every ccn rule."""
    data = json.loads((FIXTURES / "short_clean.plan.json").read_text(encoding="utf-8"))
    return copy.deepcopy(data)


@pytest.fixture
def transcript() -> Transcript:
    return load_transcript(FIXTURES / "short_clean.transcript.json")


@pytest.fixture
def laughs() -> LaughMap:
    return load_laughs(FIXTURES / "short_clean.laughs.json")


@pytest.fixture
def ccn() -> Profile:
    return load_profile("ccn")


@pytest.fixture
def run_lint(ccn: Profile) -> Callable[..., LintReport]:
    """Lint plan data (a dict) with the ccn profile unless another profile is given."""

    def run(
        data: PlanData,
        *,
        transcript: Transcript | None = None,
        laughs: LaughMap | None = None,
        profile: Profile | None = None,
    ) -> LintReport:
        plan = EditPlan.model_validate(data)
        return lint(plan, profile or ccn, transcript=transcript, laughs=laughs)

    return run
