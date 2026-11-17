"""Media-copy evidence required before publishing a timeline to the PC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aar.media.receipt import ReceiptError, check_receipt, receipt_path
from aar.recipes.context import StepFailure


def media_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checked = []
    for row in rows:
        media = Path(row["plan_media"])
        if not receipt_path(media).exists():
            raise StepFailure(
                "media_receipt_missing",
                f"No PC media receipt for {media.name}",
                "On Windows run python -m aar.media.receipt SOURCE --copy SHARED_PROXY, "
                "or stage proxies with aar media shrink --out SHARE --receipt; retry export.",
            )
        try:
            evidence = check_receipt(media, row["windows_path"])
        except (ReceiptError, OSError) as exc:
            raise StepFailure(
                "media_receipt_invalid",
                str(exc),
                "Verify the PC source and hub proxy again; correct the media map if needed.",
            ) from None
        checked.append(
            {
                "path": row["windows_path"],
                "captured_at": evidence["captured_at"],
                "verification": evidence["verification"],
                "source_sha256": evidence["sha256"],
                "copy_sha256": evidence["copy_sha256"],
            }
        )
    if not checked:
        raise StepFailure("media_receipt_missing", "No media evidence to verify")
    return {"status": "verified_snapshot", "files": checked}
