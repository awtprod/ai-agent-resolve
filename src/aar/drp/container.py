"""Open ``.drp`` / ``.drt`` / ``.drb`` files and read their members within fixed limits.

A project (``.drp``) and a timeline export (``.drt``) are zip archives of XML members. Each XML
member starts with ``<?xml ...?>`` and, on the next line, a version comment such as
``<!--DbAppVer="21.0.0.0048" DbPrjVer="17"-->``. The layout of a bin export (``.drb``) has not
been seen yet, so for it this module only lists what is there.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

from aar.drp.errors import ContainerError

__all__ = [
    "DEFAULT_MAX_MEMBERS",
    "DEFAULT_MAX_MEMBER_BYTES",
    "Container",
    "MemberInfo",
    "Versions",
    "open_container",
    "parse_version_header",
]

DEFAULT_MAX_MEMBER_BYTES = 512 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 100_000
_HEADER_SCAN_BYTES = 1024
_HEADER_RE = re.compile(rb"<!--\s*(DbAppVer\s*=\s*\"[^\"]*\"[^>]*?)\s*-->")
_ATTR_RE = re.compile(rb'(\w+)\s*=\s*"([^"]*)"')
_SUFFIX_KINDS = {".drp": "drp", ".drt": "drt", ".drb": "drb"}

# Leading bytes of things a .drb (or a damaged file) might turn out to be.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),
    (b"PK\x05\x06", "zip (empty)"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
    (b"\x1f\x8b", "gzip"),
    (b"<?xml", "xml"),
)


@dataclass(frozen=True, slots=True)
class Versions:
    """The version stamp from a member's header comment.

    ``offset``/``end`` locate the comment in the member, so a writer can restamp it.
    """

    app_version: str | None
    project_version: int | None
    raw: str
    member: str
    offset: int
    end: int


def parse_version_header(data: bytes, member: str = "") -> Versions | None:
    """Find the ``DbAppVer``/``DbPrjVer`` comment near the start of an XML member."""
    m = _HEADER_RE.search(data[:_HEADER_SCAN_BYTES])
    if m is None:
        return None
    attrs = {k.decode(): v.decode("utf-8", "replace") for k, v in _ATTR_RE.findall(m.group(1))}
    prj = attrs.get("DbPrjVer")
    return Versions(
        app_version=attrs.get("DbAppVer"),
        project_version=int(prj) if prj is not None and prj.isdigit() else None,
        raw=m.group(1).decode("utf-8", "replace"),
        member=member,
        offset=m.start(),
        end=m.end(),
    )


@dataclass(frozen=True, slots=True)
class MemberInfo:
    """One zip member as listed in the central directory."""

    name: str
    size: int
    compressed_size: int
    crc: int
    is_dir: bool


def _sniff_signature(head: bytes) -> str | None:
    for magic, name in _SIGNATURES:
        if head.startswith(magic):
            return name
    return None


def _sniff_kind(names: set[str]) -> str:
    has_seq = any(n.startswith("SeqContainer/") or "/SeqContainer" in n for n in names)
    if "project.xml" in names and "Gallery.xml" in names:
        return "drp"
    if "project.xml" in names and has_seq:
        return "drt"
    if any(n.startswith("MediaPool/") for n in names) and "project.xml" not in names:
        return "drb"
    return "unknown"


@dataclass(eq=False)
class Container:
    """An opened file. Use :func:`open_container`, and close it (or use ``with``)."""

    path: Path
    kind: str
    kind_source: str
    sniffed_kind: str
    size: int
    sha256: str
    signature: str | None
    head_hex: str
    members: list[MemberInfo]
    max_member_bytes: int
    warnings: list[str] = field(default_factory=list)
    _zip: zipfile.ZipFile | None = field(default=None, repr=False)

    @property
    def is_zip(self) -> bool:
        return self._zip is not None

    def names(self) -> list[str]:
        return [m.name for m in self.members if not m.is_dir]

    def xml_members(self) -> list[str]:
        return [n for n in self.names() if n.lower().endswith(".xml")]

    def read(self, name: str) -> bytes:
        """Read one member, refusing anything larger than ``max_member_bytes``."""
        if self._zip is None:
            raise ContainerError(f"{self.path.name} is not a zip archive; it has no members")
        try:
            info = self._zip.getinfo(name)
        except KeyError as exc:
            raise ContainerError(f"no member {name!r}") from exc
        if info.file_size > self.max_member_bytes:
            raise ContainerError(
                f"member {name!r} is {info.file_size} bytes, over the "
                f"{self.max_member_bytes}-byte limit"
            )
        try:
            with self._zip.open(info) as fh:
                data = fh.read(self.max_member_bytes + 1)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError) as exc:
            raise ContainerError(f"cannot read member {name!r}: {exc}") from exc
        if len(data) > self.max_member_bytes:
            raise ContainerError(f"member {name!r} inflates past the size limit")
        return data

    def versions(self) -> Versions | None:
        """The version stamp of ``project.xml``, or of the first XML member that has one."""
        names = self.xml_members()
        ordered = (["project.xml"] if "project.xml" in names else []) + [
            n for n in names if n != "project.xml"
        ]
        for name in ordered:
            found = parse_version_header(self.head(name), name)
            if found is not None:
                return found
        return None

    def head(self, name: str) -> bytes:
        """The first bytes of a member (enough for its version header)."""
        if self._zip is None:
            return b""
        try:
            with self._zip.open(name) as fh:
                return fh.read(_HEADER_SCAN_BYTES)
        except (zipfile.BadZipFile, OSError, EOFError, ValueError, KeyError) as exc:
            raise ContainerError(f"cannot read member {name!r}: {exc}") from exc

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None

    def __enter__(self) -> Container:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def open_container(
    path: str | Path,
    *,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> Container:
    """Open a file and work out what it is.

    ``kind`` comes from the suffix when it is ``.drp``/``.drt``/``.drb``; otherwise from the
    contents. When both are available and disagree, a warning says so.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
        digest = hashlib.sha256()
        with p.open("rb") as fh:
            head = fh.read(64)
            fh.seek(0)
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContainerError(f"cannot read {p}: {exc}") from exc

    warnings: list[str] = []
    signature = _sniff_signature(head)
    zf: zipfile.ZipFile | None = None
    members: list[MemberInfo] = []
    sniffed = "unknown"
    if zipfile.is_zipfile(p):
        try:
            zf = zipfile.ZipFile(p)
            infos = zf.infolist()
        except (zipfile.BadZipFile, OSError, ValueError) as exc:
            raise ContainerError(f"{p.name} looks like a zip but cannot be opened: {exc}") from exc
        if len(infos) > max_members:
            zf.close()
            raise ContainerError(f"{p.name} has {len(infos)} members, over the {max_members} limit")
        members = [
            MemberInfo(i.filename, i.file_size, i.compress_size, i.CRC, i.is_dir()) for i in infos
        ]
        sniffed = _sniff_kind({m.name for m in members if not m.is_dir})
    elif signature != "zip":
        warnings.append(
            f"not a zip archive (leading bytes look like {signature or 'nothing known'})"
        )

    suffix_kind = _SUFFIX_KINDS.get(p.suffix.lower())
    if suffix_kind is not None:
        kind, kind_source = suffix_kind, "suffix"
        if sniffed not in ("unknown", suffix_kind):
            warnings.append(f"suffix says {suffix_kind} but the contents look like {sniffed}")
    else:
        kind, kind_source = sniffed, "sniffed"
    return Container(
        path=p,
        kind=kind,
        kind_source=kind_source,
        sniffed_kind=sniffed,
        size=size,
        sha256=digest.hexdigest(),
        signature=signature,
        head_hex=head[:16].hex(),
        members=members,
        max_member_bytes=max_member_bytes,
        warnings=warnings,
        _zip=zf,
    )
