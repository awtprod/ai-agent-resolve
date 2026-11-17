"""PC-side proof of a source/copy pair, carried beside the hub proxy.

Stdlib-only, like shrink: run from src on Windows. A receipt records a snapshot,
not a promise that a PC file can never be moved or edited afterwards.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

SUFFIX = ".aar-media.json"
MAX_BYTES = 64 * 1024
_SHA = re.compile(r"[0-9a-f]{64}")


class ReceiptError(ValueError):
    """Missing, malformed or mismatched media evidence."""


def receipt_path(copy: Path) -> Path:
    return copy.with_name(copy.name + SUFFIX)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.stat()
    if not stat.S_ISREG(value.st_mode):
        raise ReceiptError(f"not a regular media file: {path}")
    return _stat_identity(value)


@contextlib.contextmanager
def open_regular(path: Path) -> Iterator[BinaryIO]:
    """Open regular evidence without blocking on a FIFO or following its final symlink."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ReceiptError(f"evidence must be a regular file, never a symlink: {path}")
    fd = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ReceiptError(f"evidence must be a regular file: {path}")
        if _stat_identity(before) != _stat_identity(os.fstat(fd)):
            raise ReceiptError(f"evidence changed while opening: {path}")
        stream = os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise
    with stream:
        yield stream


def _hash(path: Path) -> str:
    before = _identity(path)
    with open_regular(path) as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if before != _identity(path):
        raise ReceiptError(f"media changed during verification: {path}")
    return digest


def _read(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ReceiptError(f"receipt cannot be a symlink: {path}")
    try:
        with open_regular(path) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ReceiptError("media receipt exceeds 64 KiB")
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("schema") != "aar.media-receipt/1":
            raise ValueError("wrong schema")
        for key in ("sha256", "copy_sha256"):
            if not isinstance(data.get(key), str) or not _SHA.fullmatch(data[key]):
                raise ValueError(f"invalid {key}")
        for key in ("bytes", "copy_bytes"):
            if type(data.get(key)) is not int or not 0 < data[key] < 2**53:
                raise ValueError(f"invalid {key}")
        path_value = data.get("source")
        if not isinstance(path_value, str) or not path_value or len(path_value) > 4096:
            raise ValueError("invalid source path")
        if any(ord(c) < 32 for c in path_value):
            raise ValueError("invalid source path")
        if data.get("verification") not in ("byte_identical", "verified_shrink"):
            raise ValueError("unknown verification")
        date = datetime.fromisoformat(data["captured_at"])
        if date.tzinfo is None:
            raise ValueError("capture date needs timezone")
        if data["verification"] == "byte_identical" and (
            data["sha256"] != data["copy_sha256"] or data["bytes"] != data["copy_bytes"]
        ):
            raise ValueError("byte-identical receipt disagrees")
        return data
    except FileNotFoundError:
        raise ReceiptError(f"media receipt missing: {path}") from None
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as exc:
        raise ReceiptError(f"invalid media receipt {path}: {exc}") from None


def write_receipt(source: Path, copy: Path) -> Path:
    """Verify a copy (identical or shrink-verified) and atomically record both identities."""
    source = source.resolve(strict=True)
    before, copied = _identity(source), _identity(copy)
    src_hash, dst_hash = _hash(source), _hash(copy)
    verification = "byte_identical"
    facts = None
    if src_hash != dst_hash:
        from aar.media.shrink import read_facts, verify

        original = read_facts(source)
        verify(original, copy)
        facts = original.summary()
        verification = "verified_shrink"
    # Network/Windows filesystems can have coarse change timestamps. Bind the receipt
    # to the bytes still present after the potentially long shrink verification as well.
    if (
        src_hash != _hash(source)
        or dst_hash != _hash(copy)
        or before != _identity(source)
        or copied != _identity(copy)
    ):
        raise ReceiptError("media changed during verification")
    data = {
        "schema": "aar.media-receipt/1",
        "source": str(source),
        "sha256": src_hash,
        "bytes": before[2],
        "copy_sha256": dst_hash,
        "copy_bytes": copied[2],
        "verification": verification,
        "facts": facts,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    target = receipt_path(copy)
    if target.exists() or target.is_symlink():
        _read(target)  # only replace our own well-formed receipt
    fd, temp = tempfile.mkstemp(prefix=".aar-receipt-", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, target)
    finally:
        Path(temp).unlink(missing_ok=True)
    return target


def check_receipt(copy: Path, windows_path: str) -> dict[str, Any]:
    """Verify the hub file's bytes and its mapped PC path against the recorded snapshot."""
    from aar.drp.modify.paths import ntfs_upper

    data = _read(receipt_path(copy))

    def normalize(value: str) -> str:
        return ntfs_upper(value.replace("/", "\\"))

    if normalize(data["source"]) != normalize(windows_path):
        raise ReceiptError(f"media receipt path does not match {windows_path!r}")
    if copy.stat().st_size != data["copy_bytes"] or _hash(copy) != data["copy_sha256"]:
        raise ReceiptError(f"hub media changed since its receipt: {copy}")
    return data


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("source", type=Path, help="Original proxy on the edit PC.")
    parser.add_argument(
        "--copy", type=Path, required=True, help="Its copy on the mapped hub share."
    )
    parser.set_defaults(func=run_command)


def run_command(args: argparse.Namespace) -> int:
    try:
        path = write_receipt(args.source, args.copy)
        print(json.dumps({"ok": True, "receipt": str(path)}))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run_command(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
