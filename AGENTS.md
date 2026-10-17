# AGENTS.md

AI-Agent-Resolve (`aar`) is a companion app for DaVinci Resolve Studio, built mainly for AI agents to use. Design: [`docs/DESIGN.md`](docs/DESIGN.md).

- **Board:** [`PLAN.md`](PLAN.md) is the source of truth for slice status. The Definition of Done is in it; every slice must meet it.
- **Setup and check:** `make venv && make check`. `make check` runs `ruff check`, `ruff format --check` and `pytest`; `make fmt` fixes formatting.
- **Worktrees and branches:** each slice is built in its own worktree `../aar-impl-<slice>` on branch `impl/<slice>`, cut from `dev`. Slices merge into `dev` after independent review. `main` gets only the reviewed `dev` -> `main` PR.
- **Live tests:** tests that need a running Resolve are marked `live` and only run with `AAR_LIVE=1`. They are read-only unless DESIGN section 7 grants approval.

## Safety rules (enforced in code, not just written here)

- Before every hot op, the node checks that the expected project is the one open. It refuses to switch project or database while a human is active, unless a human forces it from the console.
- Switching databases, creating or restoring cloud projects, and writing to a production project all need a human approval token, which is recorded in the event log.
- Deliver settings are recorded before a render preset loads and restored afterwards.
- Never change Input Color Space on BRAW clips, and never sweep a colour space across a whole bin.
- Never connect to the cloud library's database directly.
- DRP writes always go to new files. Originals are never overwritten.
- Real project files used for testing stay in a git-ignored corpus (`corpus/`, or `$AAR_DRP_CORPUS`) and are never committed.
