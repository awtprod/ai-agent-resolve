"""Edit-PC path <-> hub path, built on :class:`aar.drp.modify.paths.PathMapper`.

The edit PC is Windows; the hub is POSIX. ``PathMapper``'s "mac" side is just an absolute POSIX
prefix with no fixed root, so it maps edit-PC drive/UNC paths to and from hub paths directly. The
pairing table lives in the configuration (``path_map``); nothing here contacts a host.
"""

from __future__ import annotations

from aar.drp.modify.paths import MapResult, PathMapper, TableError
from aar.host.config import EditHostConfig
from aar.host.errors import ConfigError

__all__ = ["HostPathMap"]


class HostPathMap:
    """Maps between edit-PC paths and hub paths from a configured prefix table."""

    def __init__(self, table: tuple[tuple[str, str], ...]) -> None:
        if not table:
            raise ConfigError("no path map is configured (path_map is empty)")
        try:
            self._to_hub = PathMapper("windows_to_mac", list(table))
            self._to_edit = PathMapper("mac_to_windows", list(table))
        except TableError as exc:
            raise ConfigError(f"invalid path map: {exc}") from exc

    @classmethod
    def from_config(cls, config: EditHostConfig) -> HostPathMap:
        return cls(config.path_map)

    def edit_to_hub(self, path: str) -> MapResult:
        """Map an edit-PC (Windows) path to its hub (POSIX) path."""
        return self._to_hub.map(path)

    def hub_to_edit(self, path: str) -> MapResult:
        """Map a hub (POSIX) path to its edit-PC (Windows) path."""
        return self._to_edit.map(path)
