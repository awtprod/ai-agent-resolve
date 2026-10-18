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
| S1h | `aar.drp` hostile-input hardening: zip error classes, a non-backtracking comment scan, guarded `int()`, CLI encoding and broken-pipe handling, codec mix-up without zstd, issue dedupe | S1 | merged | squash-merged 8d20797 (dev `make check` 400 passed, 3 skipped; corpus `tests/drp` 207 passed). Round 2 review: MERGE-WITH-FOLLOWUPS; M1–M3 fixed; all three budgets trip at their defaults; 53 of 81 mutants killed, survivors are test gaps. Follow-ups → S1h2 |
| S1h2 | `aar.drp` hardening follow-ups: tests for cumulative element and XML charging, the qCompress and nested keyed-dict budgets and the default budget values; `OverflowError` on a huge finite FrameRate × MediaExtents (`reader.py:1019`); non-finite retime `speed` gives invalid JSON (`blobs.py:799-812`); one issue per budget trip; the 6M element default really costs ~305 B per element and ~1.8 GB for a 29 KB bomb (fix the comment, make the budget configurable); a timing guard against uncounted work; stronger marker pins. From S2 round 3: tests for X12 (prolog `.strip()`) and X15 (member-only version-stamp change); compare the DbPrjVer and DbAppVer spelling (`17` versus `017` is silent today); exit 120 when stderr is broken (`_err` and `_silence_stdout` should cover stderr); the DESIGN.md gate paragraph should list every hidden class and say that a project DbId change passes the gate; `short_id` should show the differing suffix; a test that pins the gate on `zstd-unavailable` on its own, by monkeypatching `blobs.zstd_backend` to None (mutant N11 survives) | S1h | review | 88003d6 (rebased onto 6a339ae); 1064 passed, corpus 406 passed; bomb RSS 1818 MB to 921 MB at the 3M default. Validator af923324d85734d17 |
| S2  | `aar.drp` semantic diff: `aar drp diff a b` | S1 | merged | 7320a5e; 586 passed / 6 skipped, with the corpus 717 passed. Follow-ups are in S1h2 |
| S3  | `aar.node` broker core, with no Resolve: `ResolveGateway` protocol plus a fake, context snapshot and restore, leases, per-project hot-op queue with switch batching, presence guard | S0 | merged | live read-only smoke passed on Resolve 21.1. Gap: the Project Manager folder is captured and diffed but never restored, and `load_project` loads by name inside the current PM folder (moved to S6) |
| S4  | `aar.hub` core: SQLite event log, project registry with stable ids, work graph, MCP server with read tools | S0 | merged | 3 review rounds; races clean; 7fd1b12 |
| S4f | `aar.hub` follow-ups: `match_or_conflict` compares canonical JSON and checks project scope, one test per field, read-only open without a race, clean CLI errors | S4 | merged | merged after 4 rounds (dev `make check` 476 passed, 3 skipped). Race fix verified (0 wrong results, was 90); foreign SQLite files stay byte-identical; `..` and symlinks resolved by the OS; actor part of idempotency identity; Windows-safe `_sqlite_uri` (UNC refused); strict NaN on the request side. Round 4 validated by the coordinator: diff review, URI probe, dev `make check` |
| S5  | `aar.plan`: EditPlan schema, the `ccn` rule profile, `plan.lint` | S0 | merged | 133 tests; follow-up: confirm `end-on-laugh` (ends ≤500 ms after laugh) against real CCN cuts |
| S6  | Node op library: verified hot ops with the known traps built in (project assert, OpenPage bounce, AudioVolume, exclusive endFrame, CopyGrades, Deliver restore, never BRAW) | S3 | merged | 24c296c; 793 passed / 10 skipped. Round 3b (708d790) verified by me: mutants P39, P40, P42 and N1 are caught |
| S7  | Node job runner: checkpointed and resumable, watchdog, render submit and poll | S3, S6 | review | rebased onto dev at 8fbedcf; make check 1062 passed / 10 skipped. Validator a75c017c25f27d557. Top risk to probe: render.add_job reuses any queued job with the same OutputFilename and later deletes it, so a human job could be reused and deleted |
| S8  | `aar.drp` modify: whitelisted edits plus sandbox round-trip validation | S1, S6 | blocked(D3 sandbox approval) | offline part can start. Gate allow-list needed: a Resolve round trip (002→003) still differs on 11 opaque lines (DbSavedTime, ProjectAgeInMs, regenerated nested `SmPTZRPreset` DbIds, CurPlayheadPosition, the settings blob); S8 adds a documented list of fields Resolve rewrites on every export. Also: 16 of 64 corpus files read with `blob-partial`, so the gate can never pass for them (4 identical pairs fail); the allow-list or the reader must handle it |
| S9  | `aar.drp` create: build and `ExportProject`, plus version-matched templates | S8 | blocked(D3) | |
| S10 | Cloud ops via the node, read-only: cloud.projects, cloud.sync_status, cloud.freshness | S6 | merged | 378eb2e; 1066 passed / 11 skipped. Validator MERGE-WITH-FOLLOWUPS (43 mutants); the follow-ups were pinned in round 3 (01e3658). Live read-only check: proxy_only 238 / fully_offline 233 / proxy_offline 105 of 710 clips, matching the classifier exactly |
| S10w | Cloud writes via the node: load, create, import, restore, RefreshFolders. Needs the cloud.create, cloud.restore, project.load and db.switch approval scopes | S10, S7 | blocked(D3) | needs live-write approval and a sandbox library; switching the DB closes the open cloud project |
| S11 | Collaboration: presence, agent branch timelines, review threads mirrored to markers | S4, S6 | in-progress (round 2) | b83364b validated CHANGES-REQUIRED by a02b64ac778ff95b5 (78 mutants, 43 survived). HIGH: a failed feed delivery crashes the broker and leaks the lease (I reproduced p01); the guard is fail-open (require_branch=False) and checks the name, not the owner. MED: clip writes unguarded, mirror unbounded and not atomic, newline in note ids, node_id whitespace, weak hub records, unbounded MCP output. Round 2 sent: an outbox, a fail-closed guard backed by a branch registry keyed by unique_id, a pre-flighted mirror. Whichever of S7 and S11 lands second does the runner wiring |
| S12 | Human console (web): review queue, approve or reject, lease board, emergency stop | S4 | merged | e71f980; 1041 passed / 10 skipped. Validator: 086515e MERGE; round 2 MERGE-WITH-FOLLOWUPS, which are in S12f |
| S12f | Console and hub follow-ups. Tests for the 22 real behaviours no test pins: forged node and edge shapes G09/G10/G12/G13; withdrawal rules R02/R04–R07; the `add_note` validator R11; a non-object control payload C04; the lease TTL upper bound L03; a duplicate grant resurrecting a consumed token, double consume and revoke after use A04–A06; the skip cap, bool, empty text and ts caps S02–S05; the notice cap W11; the chunked body cap W07; CSP `form-action` W13; NotStopped W16. Pin, or move to `actor=`, the approvals grant/revoke retry by a different human (X13/X14). The `control.resume()` retry crashes with a 500 on a non-object stored payload (`control.py:310`). Truncate skip reasons. Run the fold validators in the S4 writers (graph `add_node`/`add_edge`, registry `create`/`rename`/`alias`). A `review.withdrawn` or `review.decided` appended straight to the log, bypassing the writers, can still mislabel who acted (D2 class). Nesting deeper than 252 levels breaks MCP `work_graph` and `events_tail` | S12 | review | b39090f (on 01def85); 1118 passed / 11 skipped; 25/25 listed mutants pinned (the implementer mapped the IDs themselves); public API unchanged (105 names). Validator a213c4637f907648f is re-checking with its original mutations and probing whether a concurrent writer can break the decision/withdrawal adjacency rule |
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
- 2026-10-17: S2 follow-up e9a1423 sent to the S2 validator. S12 round 2 dispatched. Volatile export fields go on S8 as a gate allow-list, not into the general diff.
- 2026-10-17: S4f round 3 (7d1b7d7) sent to the same validator. Open questions for it: lenient NaN matching on the request side, and whether putting actor in the idempotency identity breaks any S12 retry path.
- 2026-10-17: S6 round 2 review: MERGE-WITH-FOLLOWUPS. Short round 3: mutating ops must name the timeline by unique id (name-only refs caused a cross-holder block), and the validator's 11 survivor probes become tests.
- 2026-10-17: S2 round 2 review: MERGE-WITH-FOLLOWUPS, but a unit root's DbId can change without failing the gate (H3). Short round 3 before S2 merges.
- 2026-10-17: S6 round 3 (82a53b1) sent to the same validator.
- 2026-10-17: S2 round 3 (d5e0b18) sent to the same validator.
- 2026-10-17: S4f round 3 review: MERGE-WITH-FOLLOWUPS. Tiny round 4 for a Windows path regression and test gaps.
- 2026-10-17: S12 round 2 (2936ca3) sent to the same validator.
- 2026-10-17: S1h squash-merged (8d20797). S1h2 opened for the review follow-ups. S2 rebasing onto dev.
- 2026-10-17: S4f merged into dev. S12 rebasing onto it.
- 2026-10-17: S2 round 3 approved with follow-ups (moved to S1h2); waiting for the rebased tip.
- 2026-10-17: S6 round 3 approved with follow-ups; round 3b (schema text and 3 tests) sent before merge.
- 2026-10-17: S2 merged into dev (7320a5e). S1h2 dispatched.
- 2026-10-17: S6 merged into dev (24c296c). Dispatched S7 and S11 (stacked on S12 before it merges) and the read-only part of S10.
- 2026-10-17: S10 read-only part at 92a5398. My live probe found 2 classification bugs (round 2 sent); validator dispatched.
- 2026-10-17: S12 merged into dev (e71f980). S12f opened for its follow-ups.
- 2026-10-18: the 429 wave stopped S7, S11, S12f and the S10 validator, and S1h2 stalled. S11 had committed, so I rebased it onto dev (b83364b, green) and sent it to validation. S7 and S1h2 resumed; S12f is queued to stay under the rate limit.
- 2026-10-18: S7 and S1h2 finished, rebased and green, and sent to validators. S10 validated MERGE-WITH-FOLLOWUPS; round 3 pins those follow-ups before merge. S12f resumed.
- 2026-10-18: S10 read-only merged (378eb2e). The cloud writes moved to S10w, blocked(D3).
- 2026-10-18: S11 validated CHANGES-REQUIRED; round 2 sent to a4aca6262d7cc79ed.
- 2026-10-18: S12f finished and sent to the S12 validator.
