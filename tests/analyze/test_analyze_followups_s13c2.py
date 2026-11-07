"""S13c1 validation follow-ups.

L1: a model name looked up in the shared model directory must resolve inside it.
L2: the sync thresholds the validator found unpinned: the overlap minimum (5.0 s), the
consistency confirmation minimum (0.5) and the anchor eligibility bound.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from analyze_support import band_noise, tone, write_wav

from aar.analyze import AnalyzeError
from aar.analyze.audio import Mono
from aar.analyze.sync import SyncSettings, analyse_pair
from aar.analyze.whisper import resolve_model
from aar.cli import main

# ---------------------------------------------------------------- L1


def _model(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.bin").write_bytes(b"")
    return path


@pytest.fixture
def layout(tmp_path: Path) -> tuple[Path, Path]:
    """(the shared model dir, a model directory outside it)."""
    return tmp_path / "shared", _model(tmp_path / "elsewhere" / "outside-model")


def test_a_snapshot_symlink_out_of_the_model_dir_is_refused(layout):
    root, outside = layout
    snapshots = root / "models--Systran--faster-whisper-medium.en" / "snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "abc123").symlink_to(outside, target_is_directory=True)
    with pytest.raises(AnalyzeError) as error:
        resolve_model("medium.en", root)
    assert error.value.type == "model-outside-dir" and error.value.exit_code == 2
    assert str(outside.resolve()) in str(error.value)


def test_a_refs_main_climbing_out_of_the_model_dir_is_refused(layout):
    root, _ = layout
    repo = root / "models--Systran--faster-whisper-medium.en"
    (repo / "snapshots").mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("../../../elsewhere/outside-model\n", encoding="utf-8")
    with pytest.raises(AnalyzeError) as error:
        resolve_model("medium.en", root)
    assert error.value.type == "model-outside-dir"


@pytest.mark.parametrize("name", ["medium.en", "faster-whisper-medium.en"])
def test_a_named_model_symlinked_out_of_the_dir_is_refused(layout, name):
    root, outside = layout
    root.mkdir()
    (root / name).symlink_to(outside, target_is_directory=True)
    with pytest.raises(AnalyzeError) as error:
        resolve_model("medium.en", root)
    assert error.value.type == "model-outside-dir"


def test_a_model_bin_symlinked_out_of_the_dir_is_refused(layout):
    root, outside = layout
    (root / "medium.en").mkdir(parents=True)
    (root / "medium.en" / "model.bin").symlink_to(outside / "model.bin")
    with pytest.raises(AnalyzeError) as error:
        resolve_model("medium.en", root)
    assert error.value.type == "model-outside-dir"


def test_symlinks_that_stay_inside_the_model_dir_are_accepted(layout):
    root, _ = layout
    store = _model(root / "store" / "medium-v2")
    (root / "medium.en").symlink_to(store, target_is_directory=True)
    assert resolve_model("medium.en", root) == store.resolve()
    # The Hugging Face cache layout: model.bin in a snapshot is a symlink into the repo's blobs.
    repo = root / "models--Systran--faster-whisper-small.en"
    (repo / "blobs").mkdir(parents=True)
    (repo / "blobs" / "f00d").write_bytes(b"")
    snap = repo / "snapshots" / "rev1"
    snap.mkdir(parents=True)
    (snap / "model.bin").symlink_to(Path("../../blobs/f00d"))
    assert resolve_model("small.en", root) == snap.resolve()
    # The model directory itself may be a symlink.
    linked_root = root.parent / "models-link"
    linked_root.symlink_to(root, target_is_directory=True)
    assert resolve_model("small.en", linked_root) == snap.resolve()


def test_an_explicit_model_path_is_taken_as_given(layout):
    _, outside = layout
    assert resolve_model(str(outside)) == outside.resolve()


def test_the_cli_exits_2_for_a_model_outside_the_dir(layout, capsys, monkeypatch):
    import sys
    import types

    root, outside = layout
    root.mkdir()
    (root / "medium.en").symlink_to(outside, target_is_directory=True)
    monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
    media = write_wav(root.parent / "talk.wav", tone(1.0))
    out = root.parent / "t.json"
    code = main(["analyze", "transcript", str(media), "--out", str(out), "--model-dir", str(root)])
    body = json.loads(capsys.readouterr().out)
    assert code == 2 and body["error"]["type"] == "model-outside-dir"


# ---------------------------------------------------------------- L2

RATE = 8000
BAND = (150.0, 3500.0)


def noise(seconds: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return band_noise(round(seconds * RATE), RATE, rng, *BAND).astype(np.float32)


def pair(ref: np.ndarray, other: np.ndarray, **settings: Any) -> dict[str, Any]:
    return analyse_pair(
        np,
        Mono(ref, RATE, 0.0, 0),
        Mono(other, RATE, 0.0, 0),
        fps=Fraction(25),
        ref_origin=0,
        settings=SyncSettings(**settings),
    )


REF = noise(40.0, 11)


@pytest.mark.parametrize(("seconds", "no_overlap"), [(5.0, False), (4.999, True)])
def test_overlap_of_exactly_the_minimum_is_analysed(seconds, no_overlap):
    other = REF[8 * RATE : 8 * RATE + round(seconds * RATE)].copy()
    assert len(other) / RATE == seconds
    result = pair(REF, other)
    assert (result["status"] == "no-overlap") is no_overlap


def _alternating_blocks() -> np.ndarray:
    """The reference's even 2 s blocks inside unrelated noise, starting 0.5 s before it, so the
    overlap is exactly the reference's [0, 40] s and the offset is -0.5 s."""
    base = noise(40.0, 13)
    for block in range(0, 20, 2):
        span = slice(block * 2 * RATE, (block + 1) * 2 * RATE)
        base[span] = REF[span]
    return np.concatenate([noise(0.5, 14), base, noise(1.0, 15)])


@pytest.mark.parametrize(("min_confirm", "status"), [(0.5, "ok"), (0.5000001, "low-confidence")])
def test_confirmation_of_exactly_the_minimum_is_enough(min_confirm, status):
    # 20 consistency windows tile the overlap; the 10 over shared blocks confirm: exactly 0.5.
    result = pair(
        REF,
        _alternating_blocks(),
        anchors=20,
        anchor_s=2.0,
        min_agreement=0.5,
        window_s=2.0,
        step_s=2.0,
        min_confirm=min_confirm,
    )
    drift = result["drift"]
    assert (drift["windows"], drift["used"], drift["low_confidence"]) == (20, 10, 10)
    assert drift["confirmed"] == 0.5
    assert result["status"] == status
    if status == "ok":
        assert result["offset_s"] == pytest.approx(-0.5, abs=1e-5)
    else:
        assert "only 10 of 20 consistency windows confirm" in result["reason"]


@pytest.mark.parametrize(("other_s", "eligible"), [(25.5, 3), (25.0, 3), (24.95, 2)])
def test_eligible_windows_are_counted_exactly(other_s, eligible):
    # Five 10 s anchors start at 0, 7.5, 15, 22.5 and 30 s. The other file shares the first
    # 15 s with the reference, then unrelated noise. The anchor at 15 s cannot match, but at
    # offset 0 it lies inside a file of 25 s or more (ending exactly at 25.0 s counts), so it
    # is eligible and counts against the agreement; past a 24.95 s file it is not.
    other = REF[: round(other_s * RATE)].copy()
    other[15 * RATE :] = noise(other_s - 15, 12)
    result = pair(REF, other, anchors=5, anchor_s=10.0)
    coarse = result["coarse"]
    assert coarse["windows_agreeing"] == 2
    assert coarse["windows_eligible"] == eligible
    assert coarse["agreement"] == round(2 / eligible, 3)
    assert result["status"] == "ok"
