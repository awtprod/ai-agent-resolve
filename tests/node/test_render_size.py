"""S13i Part A: render output size and quality on ``render.build`` and the ``render`` job.

Every case runs the real ``render`` job over the broker against ``FakeResolve``, except the
parameter-bound and read-back unit tests, which exercise the models and ``_verify_size`` directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aar.node.jobs import JobState
from aar.node.jobs.render import RenderParams
from aar.node.ops.framework import OpError
from aar.node.ops.render import BuildParams, _verify_size

from .fake_resolve import DeliverState
from .jobs_support import AGENT_ID, JobWorld, MAIN_REF


def _params(world: JobWorld, **over: object) -> dict:
    params: dict = {
        "timeline": world.ref("Main"),
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
        "mark_in": 0,
        "mark_out": 99,
    }
    params.update(over)
    return params


def _calls(world: JobWorld, name: str) -> list:
    return [c for c in world.fake.calls if c[0] == name]


def _set_render_settings(world: JobWorld) -> list[dict]:
    return [c[1] for c in world.fake.calls if c[0] == "SetRenderSettings"]


def _run(world: JobWorld, **over: object) -> object:
    job_id = world.submit("render", _params(world, **over), world.locator(), agent_id=AGENT_ID)
    world.runner.run_until_settled()
    return world.runner.get(job_id)


# -- size and quality are applied --------------------------------------------------------------


def test_size_and_quality_applied_to_the_job(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    view = _run(
        world,
        cleanup=False,  # keep the job so its queue entry can be read
        width=1280,
        height=720,
        video_quality="High",
        export_audio=True,
        audio_codec="aac",
    )
    assert view is not None and view.state == JobState.SUCCEEDED.value, view and view.error
    assert len(world.project.render_jobs) == 1
    info = world.project.render_jobs[0].info
    assert info["FormatWidth"] == 1280
    assert info["FormatHeight"] == 720
    assert info["VideoQuality"] == "High"
    # The size/quality go through SetRenderSettings as the right Resolve keys.
    size_settings = [s for s in _set_render_settings(world) if "FormatWidth" in s]
    assert size_settings and size_settings[0] == {
        "FormatWidth": 1280,
        "FormatHeight": 720,
        "VideoQuality": "High",
        "ExportAudio": True,
        "AudioCodec": "aac",
    }


def test_bitrate_quality_is_passed_through(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    view = _run(world, cleanup=False, width=1920, height=1080, video_quality=4000)
    assert view is not None and view.state == JobState.SUCCEEDED.value
    assert world.project.render_jobs[0].info["VideoQuality"] == 4000


def test_no_size_given_is_unchanged(tmp_path: Path) -> None:
    """The old behaviour: with no size, no size keys are written and the job keeps the default."""
    world = JobWorld(tmp_path)
    view = _run(world, cleanup=False)
    assert view is not None and view.state == JobState.SUCCEEDED.value
    assert not [s for s in _set_render_settings(world) if "FormatWidth" in s]
    assert world.project.render_jobs[0].info["FormatWidth"] == 1920  # the fake's format default
    assert world.project.render_jobs[0].info["VideoQuality"] == 0  # 0 == automatic


# -- the size overrides the delivery preset (applied AFTER it loads) ---------------------------


def test_size_overrides_the_delivery_preset(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.fake.presets["delivery-4k"] = DeliverState(
        "mp4", "H264", 1, {"TargetDir": "D:/Out", "CustomName": "x", "FormatWidth": 3840,
                           "FormatHeight": 2160}
    )
    view = _run(world, cleanup=False, preset="delivery-4k", width=1280, height=720)
    assert view is not None and view.state == JobState.SUCCEEDED.value, view and view.error
    # The preset asked for 3840x2160; the request wins because it is set after the preset loads.
    assert world.project.render_jobs[0].info["FormatWidth"] == 1280
    assert world.project.render_jobs[0].info["FormatHeight"] == 720


# -- the human's Deliver size/quality are restored afterwards (DESIGN §5) ----------------------


def test_restore_brings_back_the_humans_size_and_quality(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    # The human left these on the Deliver page before the agent's render.
    world.project.deliver.settings.update(
        {"FormatWidth": 3840, "FormatHeight": 2160, "VideoQuality": "Best"}
    )
    original = world.project.deliver.copy()
    view = _run(world, width=1280, height=720, video_quality="Low")
    assert view is not None and view.state == JobState.SUCCEEDED.value, view and view.error
    # The snapshot preset carried the size/quality, and the restore loaded them back.
    assert world.project.deliver.settings["FormatWidth"] == 3840
    assert world.project.deliver.settings["FormatHeight"] == 2160
    assert world.project.deliver.settings["VideoQuality"] == "Best"
    assert world.project.deliver.settings == original.settings
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]  # no preset leak


# -- a size that Resolve ignores fails the build and cleans the job up -------------------------


def test_dropped_size_fails_with_cleanup(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.fake.faults.add("SetRenderSettings:drops-size")
    original = world.project.deliver.copy()
    view = _run(world, width=1280, height=720)
    assert view is not None and view.state == JobState.FAILED.value
    assert view.error["code"] == "render_size_ignored"
    # The just-added render job was discarded and the Deliver settings restored.
    assert world.project.render_jobs == []
    assert world.project.deliver.settings == original.settings
    assert not [p for p in world.fake.presets if p.startswith("aar-job-")]


# -- resume keeps working: the size is not applied twice, nor a different size -----------------


def test_resume_with_size_does_not_reapply(tmp_path: Path) -> None:
    world = JobWorld(tmp_path)
    world.project.render_complete_after = 5  # still rendering when we restart
    job_id = world.submit(
        "render", _params(world, width=1280, height=720), world.locator(), agent_id=AGENT_ID
    )
    for _ in range(30):
        world.runner.pump()
        if world.store.get(job_id).cursor.get("render_job_id"):
            break
    assert len(world.project.render_jobs) == 1
    assert world.project.render_jobs[0].info["FormatWidth"] == 1280

    runner = world.restart()  # clears world.fake.calls
    for _ in range(50):
        runner.pump()
        if runner.get(job_id).state == JobState.SUCCEEDED.value:
            break
    assert runner.get(job_id).state == JobState.SUCCEEDED.value
    # The resumed build adopted the recorded job: it never added a second one, and never
    # re-applied (or changed) the size.
    assert _calls(world, "AddRenderJob") == []
    assert not [s for s in _set_render_settings(world) if "FormatWidth" in s]


# -- parameter bounds (shared by RenderParams and BuildParams) ---------------------------------


def _render_params(**over: object) -> dict:
    base: dict = {
        "timeline": {"name": "Main", "unique_id": MAIN_REF["unique_id"]},
        "target_dir": "D:/Out",
        "custom_name": "show_v1",
    }
    base.update(over)
    return base


@pytest.mark.parametrize(
    "over",
    [
        {"width": 1281, "height": 720},  # odd width
        {"width": 1280, "height": 721},  # odd height
        {"width": 1280},  # height missing (both or neither)
        {"height": 720},  # width missing
        {"width": 10, "height": 10},  # below 16
        {"width": 9000, "height": 9000},  # above 8192
        {"video_quality": 0},  # 0 is refused
        {"video_quality": True},  # a bool is not a bit rate
        {"video_quality": 50},  # below 100 kbit/s
        {"video_quality": 500_000},  # above 400000 kbit/s
        {"video_quality": "Ultra"},  # not a named level
    ],
)
def test_bounds_rejected(over: dict) -> None:
    with pytest.raises(ValidationError):
        RenderParams.model_validate(_render_params(**over))


@pytest.mark.parametrize(
    "over",
    [
        {"width": 1280, "height": 720, "video_quality": 4000},
        {"width": 16, "height": 8192, "video_quality": "Best"},
        {"video_quality": 100},
        {"export_audio": False, "audio_codec": "aac"},
        {},  # no size at all
    ],
)
def test_bounds_accepted(over: dict) -> None:
    RenderParams.model_validate(_render_params(**over))


def test_build_params_share_the_bounds() -> None:
    with pytest.raises(ValidationError):
        BuildParams.model_validate(
            {
                "timeline": {"name": "Main", "unique_id": MAIN_REF["unique_id"]},
                "target_dir": "D:/Out",
                "custom_name": "x",
                "snapshot_preset": "aar-job-1-deliver",
                "width": 1280,  # height missing
            }
        )


# -- the read-back of the size off the queued job (``_verify_size``) ---------------------------


def _bp(**over: object) -> BuildParams:
    data: dict = {
        "timeline": {"name": "Main", "unique_id": MAIN_REF["unique_id"]},
        "target_dir": "D:/Out",
        "custom_name": "x",
        "snapshot_preset": "aar-job-1-deliver",
    }
    data.update(over)
    return BuildParams.model_validate(data)


def test_verify_size_none_when_no_size_requested() -> None:
    assert _verify_size({"FormatWidth": 1920, "FormatHeight": 1080}, _bp()) is None


def test_verify_size_true_on_match() -> None:
    params = _bp(width=1280, height=720)
    assert _verify_size({"FormatWidth": 1280, "FormatHeight": 720}, params) is True


def test_verify_size_false_when_no_keys() -> None:
    # The host did not echo the size keys: do not fail, just report it unverified.
    assert _verify_size({"OutputFilename": "x.mov"}, _bp(width=1280, height=720)) is False


def test_verify_size_raises_on_mismatch() -> None:
    params = _bp(width=1280, height=720)
    with pytest.raises(OpError) as exc:
        _verify_size({"FormatWidth": 1920, "FormatHeight": 1080}, params)
    assert exc.value.code == "render_size_ignored"
