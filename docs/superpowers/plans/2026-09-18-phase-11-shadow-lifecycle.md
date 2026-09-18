# Phase 11 Shadow Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a bounded EURUSD shadow lifecycle wrapper and machine-readable Phase 10 eligibility audit without changing trading, Phase 7, or training semantics.

**Architecture:** Reuse the existing `forex-watch once`, Phase 5 evaluator, Phase 8 importer/rebuilder, and Phase 10 factory through a small injectable orchestration module plus a CLI script. Read-only source adapters and the existing eligibility classifier remain authoritative; the new audit only adds bounded field/status evidence.

**Tech Stack:** Python 3, SQLite read-only adapters, existing CLI subprocess boundaries, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-18-phase-11-shadow-lifecycle-design.md`

## Global Constraints

- EURUSD only for the first real lifecycle; no uncontrolled loop.
- MT5 access remains read-only and serialized by the existing watcher.
- Phase 7 sources/artifacts are read-only and frozen.
- Phase 10 eligibility is unchanged and fail-closed.
- No order execution, Ollama/LLM training, GPU/cloud work, or Phase 12.
- Fresh non-empty Phase 10 output roots are rejected; no destructive cleanup.
- Reports never contain prompts, completions, reasoning, credentials, or rendered evidence text.

### Task 1: Add the bounded audit contract (RED → GREEN)

**Files:**
- Create: `tradingagents/finetuning/shadow_audit.py`
- Test: `tests/test_phase11_shadow_audit.py`

**Interfaces:**
- `build_exclusion_audit(report, phase56, phase8, phase9) -> tuple[dict[str, object], ...]`
- `summarize_report(report, audit) -> dict[str, object]`

- [ ] Write tests proving the six authoritative reasons map to the exact source fields/statuses, sensitive text is excluded, ordering is deterministic, and eligible rows produce an empty audit.
- [ ] Run the focused test and confirm it fails because the module is absent.
- [ ] Implement read-only source lookups and bounded scalar evidence while reusing the existing `DatasetExclusion` reason values; never reimplement eligibility.
- [ ] Run the focused test until green.

### Task 2: Add lifecycle orchestration (RED → GREEN)

**Files:**
- Create: `tradingagents/finetuning/shadow_cycle.py`
- Test: `tests/test_phase11_shadow_cycle.py`

**Interfaces:**
- `ShadowCycleConfig` frozen contract for database, Phase 8 root, Phase 10 output root, Phase 9 audit path, horizon, and command settings.
- `run_shadow_cycle(config, *, command_runner, now=None) -> dict[str, object]`.

- [ ] Write tests proving collect invokes `forex-watch once` first, then Phase 8 import/rebuild; evaluate invokes Phase 5 only after collection and never starts before the approved horizon; command failures stop without later stages; and all output roots are fresh/non-destructive.
- [ ] Run the focused test and confirm the expected missing-module failure.
- [ ] Implement the smallest subprocess-injection seam using bounded argument lists and environment-only runtime overrides; do not import or duplicate the graph.
- [ ] Run the focused test until green.

### Task 3: Add the operator script and CLI contract (RED → GREEN)

**Files:**
- Create: `scripts/phase11_shadow_cycle.py`
- Test: `tests/test_phase11_shadow_cycle_cli.py`

- [ ] Write tests for `--help`, EURUSD defaults, machine-readable output, explicit shadow banner, missing-path failure, and rejection of a non-empty Phase 10 root.
- [ ] Run the focused CLI tests and confirm failure before implementation.
- [ ] Implement argument parsing, environment/config forwarding for the frozen Phase 7 and evidence roots, bounded JSON output, and exit codes.
- [ ] Run the focused CLI tests until green.

### Task 4: Documentation and targeted verification

**Files:**
- Modify: `docs/phase11-finetuning.md`
- Test: existing affected Phase 5/8/9/10 test modules (read-only verification only)

- [ ] Add the operator command examples, one-lifecycle procedure, fresh-root rules, and exact no-training/no-execution boundary.
- [ ] Run the new tests plus focused forex, evaluation, experience, and dataset tests.
- [ ] Run Ruff on changed Python files, compileall on changed packages, and `git diff --check`.

### Task 5: One real lifecycle and handoff

- [ ] Verify dedicated DB, Phase 7 root, embedding model, and Phase 8/9 paths before execution.
- [ ] Run exactly one EURUSD shadow lifecycle; do not retry an actual graph/LLM failure.
- [ ] Run Phase 5 evaluation only for the shortest approved horizon when mature; report `PENDING` otherwise.
- [ ] Build Phase 10 into a new output root and emit the exclusion audit.
- [ ] Stop if eligible count is zero; do not start training or bulk collection.
- [ ] Review worktree and commit only the wrapper, tests, and documentation.
