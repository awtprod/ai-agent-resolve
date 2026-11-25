"""Helpers for the S25d tests: a synthetic library, seed and facts that fit the ccn recipe's
synthetic show (320x180 at 29.97, 01:00:00:00, 12 s), and the owner's edits of a placed
``Project.db`` made as row edits (as the S25a/S25c2 tests edit projects). Nothing here is
product code; every value is made up."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "drp"))

from libwrite_support import (
    facts_lines,
    linked_pair,
    make_library,
    mp4_facts,
    mp4_template,
    mp4_template_facts,
    write_seed,
)

CLIP = mp4_template()
RATE = 30000 / 1001
START = 108000  # 01:00:00:00 at 29.97 non-drop (the ccn profile's timeline start)
FRAMES = 360  # 12 s


def show_facts(pc_path: str, **change: Any) -> dict[str, Any]:
    """Facts of the synthetic show as the edit PC would measure its copy (an H.264 ``.mp4``
    with 16-bit PCM and a 01:00:00:00 timecode; outside the verified kind's envelope, so the
    media is unverified)."""
    data = mp4_facts(
        path=pc_path,
        video_width=320,
        video_height=180,
        video_coded_width=320,
        video_coded_height=180,
        video_r_frame_rate="30000/1001",
        video_avg_frame_rate="30000/1001",
        video_time_base="1/30000",
        video_duration_ts=FRAMES * 1001,
        video_nb_frames=str(FRAMES),
        video_display_aspect_ratio="16:9",
        audio_duration_ts=12 * 48000,
        data_time_base="1/30000",
        data_duration_ts=FRAMES * 1001,
        data_tags={"timecode": "01:00:00:00"},
    )
    data["ffprobe"]["format"]["duration"] = "12.012000"
    data.update(change)
    return data


def make_library_setup(base: Path, pc_paths: list[str], **lib: Any) -> dict[str, Any]:
    """A library, a seed and a facts file under ``base`` (inside the media root); the recipe's
    ``library`` params for them (both opt-ins on unless ``lib`` says otherwise)."""
    base.mkdir(parents=True, exist_ok=True)
    seeds = base / "seeds"
    seeds.mkdir(exist_ok=True)
    seed = write_seed(
        seeds / "Project.db",
        rate=RATE,
        start_frame=START,
        resolution=(320, 180),
        rate_text="29.97",
        clips=[CLIP],
        items=linked_pair(CLIP, 0xB000, start=86400, duration=48, src_in=0),
    )
    facts = facts_lines(
        base / "facts.jsonl", *[show_facts(p) for p in pc_paths], mp4_template_facts()
    )
    library = make_library(base / "Shared Library")
    params: dict[str, Any] = {
        "folder": str(library),
        "seed": str(seed),
        "facts": str(facts),
        "label": "Shared Library",
        "allow_unopened_operations": True,
        "allow_unverified_media": True,
    }
    params.update(lib)
    return params


# --------------------------------------------------------------------------- the owner's edits


def pictures_of(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """The picture items in record order, each with its linked sound item's id (the sound
    item at the same record position)."""
    cur = conn.execute(
        'SELECT Sm2TiItem_id, Start, Duration, "In", DbType FROM Sm2TiItem '
        "WHERE DbType IN ('Sm2TiVideoClip', 'Sm2TiAudioClip')"
    )
    rows = [dict(zip([d[0] for d in cur.description], r, strict=True)) for r in cur]
    sounds = {r["Start"]: r["Sm2TiItem_id"] for r in rows if r["DbType"] == "Sm2TiAudioClip"}
    out = [r for r in rows if r["DbType"] == "Sm2TiVideoClip"]
    out.sort(key=lambda r: int(r["Start"]))
    for row in out:
        row["sound"] = sounds[row["Start"]]
    return out


def pictures(db: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db)
    try:
        return pictures_of(conn)
    finally:
        conn.close()


def _reindex(conn: sqlite3.Connection) -> None:
    """Every track's ``Items`` vector dense from 0 in record order (as Resolve keeps it)."""
    links = conn.execute(
        "SELECT j.rowid, j.DbOwner, CAST(i.Start AS INTEGER) FROM Sm2TiItem_Sm2TiTrack j "
        "JOIN Sm2TiItem i ON i.Sm2TiItem_id = j.DbAssociate"
    ).fetchall()
    by_track: dict[str, list[tuple[int, int]]] = {}
    for rowid, owner, start in links:
        by_track.setdefault(owner, []).append((start, rowid))
    for entries in by_track.values():
        for index, (_, rowid) in enumerate(sorted(entries)):
            conn.execute(
                "UPDATE Sm2TiItem_Sm2TiTrack SET DbIndex = ? WHERE rowid = ?", (index, rowid)
            )


def save(db: Path, *edits: Any) -> None:
    """Run each ``edit(conn, pictures)`` on the placed project, in one transaction: a save."""
    conn = sqlite3.connect(db)
    try:
        for edit in edits:
            edit(conn, pictures_of(conn))
            _reindex(conn)
        conn.commit()
    finally:
        conn.close()
    assert not Path(str(db) + "-journal").exists()
    os.utime(db, (1_000_000_000, 1_000_000_000))  # settled at once (the sweep checks content)


def trim_tail(frames: int, index: int = -1) -> Any:
    def edit(conn: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
        item = items[index]
        length = str(int(item["Duration"]) - frames)
        for rid in (item["Sm2TiItem_id"], item["sound"]):
            conn.execute("UPDATE Sm2TiItem SET Duration = ? WHERE Sm2TiItem_id = ?", (length, rid))

    return edit


def delete(index: int) -> Any:
    """Ripple-delete the picture at ``index`` and its sound: later items move up."""

    def edit(conn: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
        gone = items[index]
        length = int(gone["Duration"])
        for rid in (gone["Sm2TiItem_id"], gone["sound"]):
            conn.execute("DELETE FROM Sm2TiItem_Sm2TiTrack WHERE DbAssociate = ?", (rid,))
            conn.execute("DELETE FROM Sm2TiItem WHERE Sm2TiItem_id = ?", (rid,))
        for later in items[index + 1 :]:
            start = str(int(later["Start"]) - length)
            for rid in (later["Sm2TiItem_id"], later["sound"]):
                conn.execute("UPDATE Sm2TiItem SET Start = ? WHERE Sm2TiItem_id = ?", (start, rid))

    return edit


def blade(index: int, at: int) -> Any:
    """Cut the picture at ``index`` (and its sound) in two, ``at`` frames into it, as the
    blade does: two items of one file, the second's in point ``at`` frames later. The new
    items are copies of the rows under fresh ids."""
    import uuid

    def edit(conn: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
        item = items[index]
        length = int(item["Duration"])
        assert 0 < at < length
        src_in = int(item["In"] or 0)
        for rid in (item["Sm2TiItem_id"], item["sound"]):
            row = conn.execute("SELECT * FROM Sm2TiItem WHERE Sm2TiItem_id = ?", (rid,)).fetchone()
            names = [d[0] for d in conn.execute("SELECT * FROM Sm2TiItem LIMIT 0").description]
            copy = dict(zip(names, row, strict=True))
            copy["Sm2TiItem_id"] = str(uuid.uuid4())
            copy["Start"] = str(int(item["Start"]) + at)
            copy["Duration"] = str(length - at)
            copy["In"] = str(src_in + at)
            conn.execute(
                f"INSERT INTO Sm2TiItem ({', '.join(chr(34) + n + chr(34) for n in copy)}) "
                f"VALUES ({', '.join('?' for _ in copy)})",
                list(copy.values()),
            )
            owner = conn.execute(
                "SELECT DbOwner FROM Sm2TiItem_Sm2TiTrack WHERE DbAssociate = ?", (rid,)
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO Sm2TiItem_Sm2TiTrack (DbOwner, DbAssociate, DbPropertyName, DbIndex) "
                "VALUES (?, ?, 'Items', 999)",
                (owner, copy["Sm2TiItem_id"]),
            )
            conn.execute("UPDATE Sm2TiItem SET Duration = ? WHERE Sm2TiItem_id = ?", (str(at), rid))

    return edit


def swap_first_two() -> Any:
    """Put the second picture (and its sound) first, the first after it."""

    def edit(conn: sqlite3.Connection, items: list[dict[str, Any]]) -> None:
        a, b = items[0], items[1]
        base = int(a["Start"])
        moves = (
            (b, base),
            (a, base + int(b["Duration"])),
        )
        for item, start in moves:
            for rid in (item["Sm2TiItem_id"], item["sound"]):
                conn.execute(
                    "UPDATE Sm2TiItem SET Start = ? WHERE Sm2TiItem_id = ?", (str(start), rid)
                )

    return edit
