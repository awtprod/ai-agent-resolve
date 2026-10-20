"""Where a Resolve project lives: its Project Manager folder path, its name and its stable id.

``ProjectManager.LoadProject(name)`` only looks inside the *current* Project Manager folder,
and project names are only unique within one folder (trap ``pm-folder-scoped-load``). A name
alone is therefore not an address. ``ProjectLocator`` carries the full folder path from the
Project Manager root, so the gateway can navigate there before loading, and the project's
unique id, so the gateway can verify it loaded the right one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _escape(segment: str) -> str:
    return segment.replace("\\", "\\\\").replace("/", "\\/")


@dataclass(frozen=True)
class ProjectLocator:
    """A project's address in the current database.

    ``folder_path`` runs from the Project Manager root (``()`` is the root itself) down to the
    folder that holds the project. ``project_id`` is ``Project.GetUniqueId()``; when it is
    known, it is what identifies the project, and the name and path only say how to load it.
    """

    folder_path: tuple[str, ...]
    name: str
    project_id: str | None = None

    def __post_init__(self) -> None:
        path = tuple(self.folder_path)
        if not all(isinstance(seg, str) and seg for seg in path):
            raise ValueError(f"folder_path segments must be non-empty strings, got {path!r}")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("a project locator needs a non-empty name")
        if self.project_id is not None and (
            not isinstance(self.project_id, str) or not self.project_id
        ):
            raise ValueError("project_id must be a non-empty string or None")
        object.__setattr__(self, "folder_path", path)

    @property
    def key(self) -> str:
        """A stable string for queues and events: ``folder/sub/name``, with ``/`` escaped."""
        return "/".join(_escape(seg) for seg in (*self.folder_path, self.name))

    def matches(self, name: str | None, project_id: str | None) -> bool:
        """Is the project ``(name, project_id)`` this one?

        With a known ``project_id`` only the id counts (a renamed project is still the same
        project). Without one the name is all there is to compare.
        """
        if self.project_id is not None:
            return project_id == self.project_id
        return name == self.name

    def with_id(self, project_id: str) -> ProjectLocator:
        return ProjectLocator(self.folder_path, self.name, project_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder_path": list(self.folder_path),
            "name": self.name,
            "project_id": self.project_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProjectLocator:
        if not isinstance(data, Mapping):
            raise ValueError("project locator must be a JSON object")
        unknown = set(data) - {"folder_path", "name", "project_id"}
        if unknown:
            raise ValueError(f"unknown project locator keys: {sorted(unknown)}")
        path = data.get("folder_path", ())
        if not isinstance(path, list | tuple):
            raise ValueError("project locator folder_path must be an array of non-empty strings")
        if "name" not in data:
            raise ValueError("project locator needs a non-empty name")
        return cls(tuple(path), data["name"], data.get("project_id"))

    def __str__(self) -> str:
        suffix = f" [{self.project_id}]" if self.project_id else ""
        return f"{self.key}{suffix}"
