# PLAN: slice board

This board is the source of truth for status, not chat. Design: [`docs/DESIGN.md`](docs/DESIGN.md).

Each slice is built by one implementer in its own worktree, `../aar-impl-<slice>`, on branch `impl/<slice>`, cut from `dev`. A separate agent validates it. Slices merge into `dev` in the order of their dependencies.

Status values: `todo` · `in-progress` · `review` · `merged` · `blocked(<reason>)`.

## Definition of Done (every slice)

- `make check` is green: `ruff check`, `ruff format --check` and `pytest`. The report includes the real output.
- New behaviour has tests that fail without the change. No tautological tests and no mocks of the thing under test.
- No test calls Resolve unless it is marked `live` and gated on `AAR_LIVE=1`. Live tests are read-only unless DESIGN §7 grants approval.
- Nothing is committed that should not be: no real project files, no secrets, no IPs or hostnames, no private paths. Real samples go in `corpus/` (git-ignored) or `$AAR_DRP_CORPUS`.
- Public functions have type hints. Models are dataclasses or pydantic. Errors are never swallowed.
- The implementer's report lists files changed, test output, what is verified and what is not, and any follow-ups.

## Slices

| id  | slice | depends on | status | notes |
| --- | ----- | ---------- | ------ | ----- |
| S0  | Scaffold: `pyproject` (src layout, package `aar`), ruff + pytest, `make check`, the `aar` CLI entry point, `.gitignore` with `corpus/` | none | merged | |
| S1  | `aar.drp` reader: container (drp/drt/drb), version header, tolerant scanner, typed model, blob codecs (fixed doubles, keyed dict, protobuf wire, zstd detect), `aar drp inspect --json` / `describe` | S0 | merged | all 64 corpus project files read; 4/4 decoder mutations caught. Not decoded yet: protobuf field meanings, enabled and drop-frame flags, `.drb` |
| S1h | `aar.drp` hostile-input hardening: zip error classes, a non-backtracking comment scan, guarded `int()`, CLI encoding and broken-pipe handling, codec mix-up without zstd, issue dedupe | S1 | in-progress | from the S1 review. Review of 6f291cd: MERGE-WITH-FOLLOWUPS; all 10 fixes held, no tracebacks in about 100 hostile cases plus 6000 fuzz runs. Round 2: a clip's path can still come from the audio stream (duplicate DbId, stream-less or lost-video clip); `describe` quadratic in member count; invalid JSON under cp1252; memory budget (140x amplification); public `join_clip_path` for S2 |
| S2  | `aar.drp` semantic diff: `aar drp diff a b` | S1 | in-progress | match objects by DbId; compare raw bytes for anything not decoded. Round 2 after review of 2feb5df: reported IDENTICAL for 12 single edits; `--only` hides changes silently; nesting and order of units not compared; duplicate zip members; CLI crashes with exit 1. Adds `--gate` mode for S8. Clip-path rule matches S1h |
| S3  | `aar.node` broker core, with no Resolve: `ResolveGateway` protocol plus a fake, context snapshot and restore, leases, per-project hot-op queue with switch batching, presence guard | S0 | merged | live read-only smoke passed on Resolve 21.1. Gap: the Project Manager folder is captured and diffed but never restored, and `load_project` loads by name inside the current PM folder (moved to S6) |
| S4  | `aar.hub` core: SQLite event log, project registry with stable ids, work graph, MCP server with read tools | S0 | merged | 3 review rounds; races clean; 7fd1b12 |
| S4f | `aar.hub` follow-ups: `match_or_conflict` compares canonical JSON and checks project scope, one test per field, read-only open without a race, clean CLI errors | S4 | review | round 2 at 9ba2e73 (single commit on 7fd1b12): race recovery re-checks the winner (fork race: 1 winner, 5 conflicts), registry conflict tests, `OSError` mapping, safe path encoding with a NUL guard; 371 passed. Keeps `mode=rw` plus `query_only` rather than `mode=ro`, because the hub's live database is in WAL mode. Re-validating |
| S5  | `aar.plan`: EditPlan schema, the `ccn` rule profile, `plan.lint` | S0 | merged | 133 tests; follow-up: confirm `end-on-laugh` (ends ≤500 ms after laugh) against real CCN cuts |
| S6  | Node op library: verified hot ops with the known traps built in (project assert, OpenPage bounce, AudioVolume, exclusive endFrame, CopyGrades, Deliver restore, never BRAW) | S3 | review | round 2 at 19c690a (rebased; round-1 commit is 4d63f28): all majors F1–F5 and minors F6–F14 reported fixed. Leases are keyed by `project_id` and the timeline unique id; the human's project is located only by identity; 494 passed; 99/99 mutations caught (implementer's run). Re-validating. F15 (D3 gate) goes to S7. Live writes stay sandbox-only (D3) |
| S7  | Node job runner: checkpointed and resumable, watchdog, render submit and poll | S3, S6 | todo | |
| S8  | `aar.drp` modify: whitelisted edits plus sandbox round-trip validation | S1, S6 | blocked(D3 sandbox approval) | offline part can start |
| S9  | `aar.drp` create: build and `ExportProject`, plus version-matched templates | S8 | blocked(D3) | |
| S10 | Cloud ops via the node: list, load, create, import, restore, sync status, freshness | S6 | blocked(D3) | read-only listing can start |
| S11 | Collaboration: presence, agent branch timelines, review threads mirrored to markers | S4, S6 | todo | |
| S12 | Human console (web): review queue, approve or reject, lease board, emergency stop | S4 | review | 79b2cab: 494 passed; 35 guard mutations (all caught except two redundant consume layers); screenshots in `/tmp/aar-s12-shots` checked. Rebase onto S4f before merging |
| S13 | CCN recipes: port the long-form, shorts, captions, thumbpipe and publish steps onto the work graph | S5, S7, S11 | todo | |
| S14 | Spike: second Resolve instance, `-nogui`, `-rr` beside the GUI on the edit PC | none | blocked(D4 approval + idle window) | |

## Open decisions (DESIGN §7)

| id | decision | default taken |
| -- | -------- | ------------- |
| D1 | Standalone app vs Command Center module | standalone; Command Center talks to it over MCP |
| D2 | Hosting and who collaborates | tailnet-only hub for v1 |
| D3 | Use the edit PC's Local Database as the sandbox | blocked; awaiting Andrew |
| D4 | Multi-instance spike on the edit PC | blocked; awaiting Andrew |
| D5 | GitHub remote and Prism runner | none yet; slices merge locally into `dev` after independent review |

## Log

- 2026-10-17: Research done, design v1 written, board created.
- 2026-10-17: S0 merged into `dev` (e5404e8). Wave 1 (S1, S3, S4, S5) dispatched in parallel worktrees; real DRP/DRT samples are being copied read-only into `../aar-corpus` (git-ignored, outside the repo).
- 2026-10-17: S5 merged (95440b4). Corpus ready: 73 files, all hashes verified, DbPrjVer 14/15/17.
- 2026-10-17: S3 merged. The live `get_context` smoke on the edit PC was read-only and matched every assumed return shape: an empty codec and the root PM folder both read as `""`, which becomes `None`. `dev` `make check`: 202 passed, 1 skipped (the live test).
- 2026-10-17: S1 merged (`dev` `make check`: 267 passed, 2 skipped). The S4 review sent it back for fixes: a real hostname in its tests, artifact ids that can point outside the store, and work-graph idempotency and staleness bugs. S1h and S2 dispatched.
- 2026-10-17: S6 rebased onto `dev` (18d527c, 357 passed, 6 skipped) and sent to independent review. Its live probe made 21 read-only calls; I audited them and found no setters. S1h fixes are in e1cde85. Stills now show "codec unknown" instead of borrowing the audio codec.
- 2026-10-17: S4 merged (7fd1b12; `dev` `make check`: 323 passed, 2 skipped). S4f and S12 dispatched. S2 sent to independent review.
- 2026-10-17: S6 review: MERGE-WITH-FOLLOWUPS (38 of 53 mutations caught; API calls verified against the 21.1 stubs; private data clean). I sent it back for round 2 anyway, because S7 and S11 build on the broker's lease and restore paths.
- 2026-10-17: S12 built and sent to independent review. S4f review found an append race-recovery bug that was already in dev; it goes back for round 2.
- 2026-10-17: S2 review: MERGE-WITH-FOLLOWUPS as a general diff, REWORK as S8's post-write gate (15 of 28 mutations survived, 12 of them reporting IDENTICAL). Symmetry was clean on 109 corpus pairs, and a trial merge with S1h was clean. Sent back for round 2.
- 2026-10-17: S6 round 2 at 19c690a sent back to the same validator. Its mutation harness can reuse a stale `.pyc` and give false results; re-running with `PYTHONDONTWRITEBYTECODE=1`.
- 2026-10-17: S4f round 2 (9ba2e73) and S12's d8c492c (unusable `--db` and `--artifacts` exit 2; foreign SQLite files never switched to WAL) sent to their validators.
- 2026-10-17: S1h review: MERGE-WITH-FOLLOWUPS; sent back for a short round 2 (fixes listed in the S1h row) before merging. Corpus marker pins move out of the repo into the corpus dir.
