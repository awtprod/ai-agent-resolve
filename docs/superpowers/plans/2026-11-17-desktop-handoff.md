# Desktop handoff implementation plan

Goal: deliver a usable timeline through the existing mapped share, verify the PC/hub media relationship, and bring returned Resolve timeline edits into the agent's recorded context.

Design: retain D8/D9 (originals on Windows, audio-preserving proxies on Linux) and the existing FCPXML delivery. Add media receipts created on the PC alongside proxy copies, bound to both files' hashes after existing shrink verification. Exports check supplied receipts and clearly distinguish checked snapshots from missing receipts. Publish a round-specific return path in the handoff; the existing cold worker imports a returned DRT into immutable plan/diff artifacts, and recipe status exposes them. Never automatically replace the active plan with a partial reconstruction. No new Resolve process or project switch.

Alternatives: live Resolve automation would require enabling the disabled node and new approval flows; duplicating all originals contradicts D8 and available hub storage. Neither is needed for this handoff.

Constraints: no production data changes/deployment; no silent overwrite of media or human edits; all file inputs bounded and validated; partial imports explicitly require attention; preserve existing approvals.

Review focus: stale/wrong receipt, changed source during hashing, partial DRT copy, duplicate worker/restart, returned edits with unsupported changes.

## Task 1 — Media receipts
- [x] Tests: matching and mismatched paths/hash, malformed/bounded receipts, missing receipts, actual audio-preserving copy.
- [x] Implement a stdlib PC receipt command and optional shrink integration; validate receipts in the real CCN export path; expose evidence in handoff notes.
- [x] Run focused media/CCN tests.

## Task 2 — Returned timeline pickup
- [x] Tests: round binding, real DRT parse, immutable artifacts, duplicate pickup, malformed/partial files, cancelled run, partial import warning.
- [x] Publish the exact return path with each export. Process settled returned files using the existing importer in the cold worker. Surface plan/diff artifact IDs and attention status through recipe status and the agent guide.
- [x] Run focused recipe and importer tests.

## Task 3 — Verification and delivery
- [x] Run make check, independent review, fix substantive findings, record evidence and live limitations.
- [x] Update PLAN.md and workflow docs. Provide reviewable branch/PR; deployment remains human-gated.
