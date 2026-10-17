"""Exceptions raised by :mod:`aar.drp`."""

from __future__ import annotations

__all__ = ["BlobError", "ContainerError", "DrpError", "ZstdUnavailableError"]


class DrpError(Exception):
    """Base class for every error raised while reading a DRP/DRT/DRB file."""


class ContainerError(DrpError):
    """The file cannot be opened or a member cannot be read within the limits."""


class BlobError(DrpError):
    """A hex blob is malformed or does not match the layout it was decoded as."""


class ZstdUnavailableError(BlobError):
    """A zstd blob needs a real decompressor and none is installed."""
