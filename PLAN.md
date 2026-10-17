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
| S1h | `aar.drp` hostile-input hardening: zip error classes, a non-backtracking comment scan, guarded `int()`, CLI encoding and broken-pipe handling, codec mix-up without zstd, issue dedupe | S1 | review | round 2 at 282d9fd: clip path comes from the element type (4 repros fixed); `describe` linear; JSON valid in any encoding; budgets of 6M elements, 600 MiB XML and 400 MiB decompressed per file (about 20x the corpus max); exit 74 on output errors; public `join_clip_path`; pins moved into the corpus dir. 344 passed; 0 of 2,554 corpus clips changed. Re-validating. **Squash-merge**: the corpus hash and frame pins exist only in e1cde85 and 6f291cd |
| S2  | `aar.drp` semantic diff: `aar drp diff a b` | S1 | review | match objects by DbId; compare raw bytes for anything not decoded. Round 2 at 84c7eed: 356 passed, corpus 131 passed, 34/34 mutations caught; `--gate` built; nesting and order checks; `--only` counts hidden changes; exit codes 0/1/2/141. Follow-up commit: DbId children of `*Set` unordered across tags (Resolve rewrites them in hash order). Re-validating; rebase onto S1h before merging |
| S3  | `aar.node` broker core, with no Resolve: `ResolveGateway` protocol plus a fake, context snapshot and restore, leases, per-project hot-op queue with switch batching, presence guard | S0 | merged | live read-only smoke passed on Resolve 21.1. Gap: the Project Manager folder is captured and diffed but never restored, and `load_project` loads by name inside the current PM folder (moved to S6) |
| S4  | `aar.hub` core: SQLite event log, project registry with stable ids, work graph, MCP server with read tools | S0 | merged | 3 review rounds; races clean; 7fd1b12 |
| S4f | `aar.hub` follow-ups: `match_or_conflict` compares canonical JSON and checks project scope, one test per field, read-only open without a race, clean CLI errors | S4 | in-progress | 9ba2e73 review: MERGE-WITH-FOLLOWUPS; race fix verified (6-12 forked processes: 0 wrong results, was 90). Round 3 (short): `abspath` collapses `..` through symlinks and can open the wrong database; WAL switched on before the schema check; Windows-safe URI; legacy NaN in `match_or_conflict`; tests for 6 surviving mutants |
| S5  | `aar.plan`: EditPlan schema, the `ccn` rule profile, `plan.lint` | S0 | merged | 133 tests; follow-up: confirm `end-on-laugh` (ends ≤500 ms after laugh) against real CCN cuts |
| S6  | Node op library: verified hot ops with the known traps built in (project assert, OpenPage bounce, AudioVolume, exclusive endFrame, CopyGrades, Deliver restore, never BRAW) | S3 | review | round 2 at 19c690a (rebased; round-1 commit is 4d63f28): all majors F1–F5 and minors F6–F14 reported fixed. Leases are keyed by `project_id` and the timeline unique id; the human's project is located only by identity; 494 passed; 99/99 mutations caught (implementer's run). Re-validating. F15 (D3 gate) goes to S7. Live writes stay sandbox-only (D3) |
| S7  | Node job runner: checkpointed and resumable, watchdog, render submit and poll | S3, S6 | todo | |
| S8  | `aar.drp` modify: whitelisted edits plus sandbox round-trip validation | S1, S6 | blocked(D3 sandbox approval) | offline part can start |
| S9  | `aar.drp` create: build and `ExportProject`, plus version-matched templates | S8 | blocked(D3) | |
| S10 | Cloud ops via the node: list, load, create, import, restore, sync status, freshness | S6 | blocked(D3) | read-only listing can start |
| S11 | Collaboration: presence, agent branch timelines, review threads mirrored to markers | S4, S6 | todo | |
| S12 | Human console (web): review queue, approve or reject, lease board, emergency stop | S4 | in-progress | review of 79b2cab+d8c492c: MERGE-WITH-FOLLOWUPS (31/38 mutations caught; XSS, headers, CSRF, artifact route and approval races clean). Round 2 queued: 3 malformed events permanently break read pages including `/control`; `set_status` bypasses the review gate; header and body-cap tests. Identity = reachability (D2, follow-up). Rebase onto S4f round 3 before merging |
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
- 2026-10-17: Five agents stopped on a 429 rate limit (both Opus 5.5 and Sonnet 5.5 credentials cooling down); resumed in stages. S12 review: MERGE-WITH-FOLLOWUPS; malformed events break pages, so round 2. S4f re-validation: MERGE-WITH-FOLLOWUPS; short round 3 on Opus 4.8.
- 2026-10-17: S2 round 2 (84c7eed) sent to the same validator. Decision: DbId children inside `*Set` are unordered across tags, because a real Resolve export reorders them; otherwise the gate would fail on every round trip.
- 2026-10-17: S1h round 2 (282d9fd) sent to the same validator. S1h will be squash-merged so the corpus-derived pins never reach `dev` history (they were added in e1cde85 and removed in 282d9fd). Element budget kept at 6M (about 20x the corpus max), so large real projects aren't read partially.
