"""The known-trap registry: stable ids, complete entries, cited by ops and modelled by the fake."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aar.node.ops import list_ops
from aar.node.traps import TRAPS, get_trap, trap_ids

LISTED = {
    "append-endframe-exclusive",
    "append-fps-short",
    "append-still-ignores-range",
    "append-ungraded",
    "av-append-video-only",
    "volume-key",
    "setproperty-after-color",
    "rcm-input-colorspace",
    "custom-res-resets-colour-science",
    "multicam-angles",
    "deliver-clobber",
    "render-uses-original-media",
    "pm-folder-scoped-load",
}


def test_every_known_trap_is_registered_with_all_fields() -> None:
    assert set(trap_ids()) >= LISTED
    for trap in TRAPS.values():
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", trap.id)
        for text in (trap.title, trap.symptom, trap.cause, trap.fix, trap.verified):
            assert text.strip()
        assert set(trap.to_dict()) == {"id", "title", "symptom", "cause", "fix", "verified"}


def test_trap_texts_are_generic() -> None:
    windows_path = re.compile(r"\b[A-Za-z]:[\\/]")
    for trap in TRAPS.values():
        text = " ".join(trap.to_dict().values())
        assert not windows_path.search(text), trap.id
        assert "\\" not in text, trap.id


def test_unknown_trap_id_raises() -> None:
    with pytest.raises(KeyError):
        get_trap("no-such-trap")


def test_ops_cite_only_registered_traps_and_every_listed_trap_is_cited_or_handled() -> None:
    cited = {t for spec in list_ops() for t in spec.traps}
    assert cited <= set(TRAPS)
    # deliver-clobber is handled by the broker; the rest are cited by the ops that handle them
    assert cited | {"deliver-clobber"} >= LISTED


def test_the_fake_models_every_trap() -> None:
    source = (Path(__file__).parent / "fake_resolve.py").read_text(encoding="utf-8")
    modelled = set(re.findall(r"# trap: ([a-z0-9-]+)", source))
    assert modelled == set(TRAPS)
