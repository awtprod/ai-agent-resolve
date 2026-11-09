"""The title font for the render: the profile's family as a font file, else DejaVu Sans (S18b).

ffmpeg's ``drawtext`` needs a font file. The family (``[render] font``, Montserrat for CCN) is
looked up by file name in ``AAR_FONTS_DIR`` (``os.pathsep``-separated folders) and then the system
and user font folders; the static weights are preferred (SemiBold, Bold, Medium, Regular) over a
variable font. Montserrat is not installed on the hub by default, so the documented fallback is
DejaVu Sans (Bold), which every Ubuntu hub has. With neither, titles are not drawn and the report
says so. The walk is bounded, and nothing is read but directory listings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FONTS_DIR_ENV = "AAR_FONTS_DIR"
SYSTEM_FONT_DIRS = ("/usr/share/fonts", "/usr/local/share/fonts")
FALLBACK_FAMILY = "DejaVu Sans"
_FALLBACK_STEMS = ("dejavusans-bold", "dejavusans")
_WEIGHTS = ("semibold", "bold", "medium", "regular")
_SUFFIXES = (".ttf", ".otf")
#: At most this many directory entries are looked at in all.
MAX_ENTRIES = 50_000


@dataclass(frozen=True)
class FontChoice:
    family: str  # the family the profile asked for
    path: Path | None  # the file drawtext uses; None: titles are not drawn
    used: str | None  # the family of that file
    fallback: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "file": None if self.path is None else str(self.path),
            "used": self.used,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def font_dirs() -> list[Path]:
    """Where fonts are looked for, in order."""
    dirs = [Path(p) for p in os.environ.get(FONTS_DIR_ENV, "").split(os.pathsep) if p.strip()]
    dirs += [Path(p) for p in SYSTEM_FONT_DIRS]
    home = Path.home()
    dirs += [home / ".local" / "share" / "fonts", home / ".fonts"]
    return dirs


def _font_files(dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen = 0
    for root in dirs:
        if not root.is_dir():
            continue
        for current, subdirs, files in os.walk(root):
            subdirs.sort()
            for name in sorted(files):
                seen += 1
                if seen > MAX_ENTRIES:
                    return found
                if name.lower().endswith(_SUFFIXES):
                    found.append(Path(current) / name)
    return found


def _rank(path: Path) -> tuple[int, int, str]:
    stem = path.stem.lower()
    weight = next((i for i, w in enumerate(_WEIGHTS) if w in stem.replace("-", "")), len(_WEIGHTS))
    variable = 1 if "[" in stem or "variable" in stem else 0
    return variable, weight, path.name


def find_font(family: str) -> FontChoice:
    """The font file for ``family``, else the fallback (module docstring)."""
    files = _font_files(font_dirs())
    key = family.lower().replace(" ", "")
    matches = [p for p in files if p.stem.lower().replace(" ", "").startswith(key)]
    if matches:
        best = min(matches, key=_rank)
        return FontChoice(family, best, family, False, f"found {best.name}")
    for stem in _FALLBACK_STEMS:
        hit = next((p for p in files if p.stem.lower() == stem), None)
        if hit is not None:
            return FontChoice(
                family,
                hit,
                FALLBACK_FAMILY,
                True,
                f"no {family} font file in {FONTS_DIR_ENV} or the font folders; using "
                f"{FALLBACK_FAMILY} ({hit.name})",
            )
    return FontChoice(
        family,
        None,
        None,
        True,
        f"no {family} or {FALLBACK_FAMILY} font file found: titles are not drawn",
    )
