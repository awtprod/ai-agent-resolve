"""Read-only cloud ops against the fake: cloud.projects, cloud.sync_status, cloud.freshness.

These drive the real broker + ScriptApiGateway + FakeResolve (no mocks of the ops). Every
test asserts the ops make no content writes, because they are read-only.
"""

from __future__ import annotations

from dataclasses import replace

from aar.node.ops import get_op
from aar.node.ops.framework import OpModel

from .fake_resolve import FakeResolve
from .ops_support import CONFIG, OpsWorld, Show, content_writes

# -- registry ---------------------------------------------------------------------------------


def test_cloud_ops_are_registered_read_only_with_their_traps() -> None:
    for name in ("cloud.projects", "cloud.sync_status", "cloud.freshness"):
        spec = get_op(name)
        assert spec.hot and not spec.mutates, name
    assert get_op("cloud.sync_status").traps == (
        "cloud-proxy-only-no-audio",
        "render-uses-original-media",
    )
    schema = get_op("cloud.sync_status").describe()["params_schema"]
    assert {"bin_path", "recursive", "cursor", "max_clips"} <= set(schema["properties"])
    assert "required" not in schema or schema["required"] == []


# -- cloud.projects ---------------------------------------------------------------------------


def test_cloud_projects_lists_the_folder_with_attributes_and_ids_only_the_open_one() -> None:
    show = Show()
    show.fake.project("Other").attributes["liveCollaborationMode"] = "multi_user"
    show.fake.project("Other").attributes["notes"] = "shared cut"
    # A project in another folder must not leak into the current folder's listing.
    show.fake.add_project("Archived", ["x"], ("Archive",))

    result = show.ok("cloud.projects", {})

    assert result.current_folder == "Shows"
    assert result.count == 2
    by_name = {p.name: p for p in result.projects}
    assert set(by_name) == {"Show", "Other"}
    assert by_name["Show"].is_open and by_name["Show"].unique_id == show.pid
    assert by_name["Other"].is_open is False and by_name["Other"].unique_id is None
    assert by_name["Other"].live_collaboration_mode == "multi_user"
    assert by_name["Other"].notes == "shared cut"
    assert by_name["Show"].creation_date == "2026-10-01T00:00:00+00:00"
    assert result.by_collaboration == {"single_user": 1, "multi_user": 1}
    assert result.open_project_id == show.pid
    assert result.database.db_type == "Cloud" and result.database.db_name == "acme cloud library"
    assert content_writes(show.fake) == []


def test_cloud_projects_flags_the_local_database() -> None:
    show = Show()
    show.fake.database = {"DbType": "Disk", "DbName": "Local Database"}
    result = show.ok("cloud.projects", {})
    assert [n.code for n in result.notes] == ["local_database"]


def test_cloud_projects_leaves_the_id_unset_when_the_open_name_is_ambiguous() -> None:
    show = Show()
    show.fake.add_project("Show", ["Main"], ("Shows",))  # a second "Show" in the same folder
    result = show.ok("cloud.projects", {})
    shows = [p for p in result.projects if p.name == "Show"]
    assert len(shows) == 2
    assert all(p.unique_id is None and not p.is_open for p in shows)
    assert [n.code for n in result.notes] == ["open_project_ambiguous"]
    assert result.notes[0].trap_id == "pm-folder-scoped-load"
    # The open project's id is still reported at the top level.
    assert result.open_project_id == show.pid


# -- cloud.sync_status ------------------------------------------------------------------------


def test_cloud_sync_status_summarises_per_clip_and_flags_proxy_only_media() -> None:
    show = Show()
    pool = show.project.media_pool
    # A proxy-only clip: original offline, a proxy present -> a build loses audio.
    pool.add_clip(
        ("Footage", "Day 1"),
        "cam proxy",
        online="Offline",
        proxy="1920x1080",
        cloud_sync=FakeResolve.CLOUD_SYNC_DOWNLOAD_NOT_FOUND,
    )
    # An offline clip with no proxy: a render fails on it entirely.
    pool.add_clip(
        ("Footage", "Day 1"),
        "cam gone",
        online="Offline",
        proxy="None",
        cloud_sync=FakeResolve.CLOUD_SYNC_DOWNLOAD_FAIL,
    )

    result = show.ok("cloud.sync_status", {})

    assert result.total_clips == 8 and result.scanned == 8 and result.done
    assert result.next_cursor is None
    by_name = {c.name: c for c in result.clips}
    assert by_name["cam proxy"].proxy_only and by_name["cam proxy"].has_proxy
    assert by_name["cam proxy"].cloud_sync == "CLOUD_SYNC_DOWNLOAD_NOT_FOUND"
    assert by_name["cam gone"].original_offline and not by_name["cam gone"].proxy_only
    assert by_name["cam A"].cloud_sync == "CLOUD_SYNC_DEFAULT" and not by_name["cam A"].proxy_only
    assert result.proxy_only == 1 and result.offline_no_proxy == 1
    assert result.status_counts["CLOUD_SYNC_DOWNLOAD_NOT_FOUND"] == 1
    assert result.status_counts["CLOUD_SYNC_DEFAULT"] == 6
    assert sum(result.status_counts.values()) == 8
    codes = {(n.code, n.trap_id) for n in result.notes}
    assert ("proxy_only", "cloud-proxy-only-no-audio") in codes
    assert ("media_offline", "render-uses-original-media") in codes
    assert content_writes(show.fake) == []


def test_cloud_sync_status_resumes_through_pages_with_a_cursor() -> None:
    show = Show()
    pool = show.project.media_pool
    for i in range(10):
        pool.add_clip(("Music",), f"stem {i}")
    total = show.ok("cloud.sync_status", {}).total_clips  # 6 base + 10 = 16

    seen: list[str] = []
    cursor = 0
    pages = 0
    while True:
        page = show.ok("cloud.sync_status", {"cursor": cursor, "max_clips": 5})
        pages += 1
        assert page.total_clips == total and page.cursor == cursor and page.scanned <= 5
        seen.extend(c.media_id for c in page.clips)
        if page.done:
            assert page.next_cursor is None
            break
        cursor = page.next_cursor
    assert pages == 4  # 5 + 5 + 5 + 1
    assert len(seen) == total and len(set(seen)) == total  # every clip once, no repeats


def test_cloud_sync_status_rejects_a_cursor_past_the_end() -> None:
    show = Show()
    error = show.fails("cloud.sync_status", {"cursor": 999})
    assert error.code == "cursor_past_end"
    assert error.detail["fix_hint"]


def test_cloud_sync_status_scopes_to_a_bin_without_recursing() -> None:
    show = Show()
    show.project.media_pool.add_clip(("Music",), "theme two", kind="audio")
    result = show.ok("cloud.sync_status", {"bin_path": ("Music",), "recursive": False})
    assert {c.name for c in result.clips} == {"theme", "theme two"}
    assert all(c.bin_path == ("Music",) for c in result.clips)


# -- cloud.freshness --------------------------------------------------------------------------


def test_cloud_freshness_reports_stale_folders_read_only() -> None:
    show = Show()
    pool = show.project.media_pool
    pool.folder("Footage").stale = True
    pool.folder("Footage", "Day 1").stale = True

    result = show.ok("cloud.freshness", {})

    stale = {f.bin_path for f in result.folders if f.stale}
    assert stale == {("Footage",), ("Footage", "Day 1")}
    assert result.stale_count == 2
    assert result.folder_count == 5 and not result.truncated
    assert [n.code for n in result.notes] == ["stale_folders"]
    assert content_writes(show.fake) == []


def test_cloud_freshness_without_stale_folders_has_no_note() -> None:
    show = Show()
    result = show.ok("cloud.freshness", {})
    assert result.stale_count == 0 and result.notes == []
    assert all(not f.stale for f in result.folders)


def test_cloud_freshness_truncates_at_max_folders() -> None:
    show = Show()
    result = show.ok("cloud.freshness", {"max_folders": 2})
    assert result.folder_count == 2 and result.truncated


def test_cloud_ops_target_a_folder_that_is_not_the_root() -> None:
    # The broker loads the target in its own folder; the op must read that folder, not the root.
    fake = FakeResolve({"Show": ["Main"]}, current="Show", folder=("Shows",))
    fake.add_project("Loose", ["m"], ())  # a project at the root must not appear
    world = OpsWorld(fake, replace(CONFIG, snapshot_project_manager_path=False))
    result = world.ok("cloud.projects", {})
    assert result.current_folder == "Shows"
    assert {p.name for p in result.projects} == {"Show"}
    assert isinstance(result, OpModel)
