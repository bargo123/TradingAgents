# Phase 9 Task 15 Verification Report

Date: 2026-09-12
Branch: `codex/phase-9-evidence-reasoning`
Scope base: `70bf6796ab74df99eb47e9e67083b6e2d1678c79...HEAD`

## Verdict

**PHASE 9 NOT COMPLETE**

The mandatory real local Qwen/Ollama evidence-enabled smoke was not run because
no accepted Phase 8 root was available. The required gate result is:

`PHASE 9 REAL SMOKE PREREQUISITE FAILED`

The source Phase 5/6 database, Phase 7 catalog, and local embedding-model path
exist. Ollama is installed and `ollama list` reports local `qwen3.5:2b` and
`qwen3.5:4b` models. Every discovered Phase 8 catalog candidate failed the
read-only preflight: candidates either had no published active generation or
had missing `trust_policy_version` metadata. No real smoke command was
substituted, no hosted endpoint was used, and no smoke output/artifact was
created.

## Required focused RED command list

- `python -m pytest -q tests/test_forex_phase9_contracts.py ... tests/test_phase9_smoke_harness.py` — **162 passed** in 28.97s.
- Literal `python -m pytest -q tests/test_knowledge_*.py tests/test_experience_*.py` — **command failed before collection** because PowerShell passed the literal wildcard path (`file or directory not found`). PowerShell-expanded equivalent — **308 passed** in 65.20s.
- Literal `python -m pytest -q tests/test_forex_shadow*.py ...` — **command failed before collection** for the same PowerShell wildcard behavior. PowerShell-expanded equivalent — **123 passed, 2 skipped** in 7.67s.
- `python -m pytest -q tests/test_checkpoint_lifecycle.py tests/test_checkpoint_resume.py` — **15 passed** in 3.03s.

The wildcard failures were recorded as invoked; no production code or tests
were edited to hide them.

## Required GREEN verification

- `python -m pytest -q` — **1416 passed, 6 skipped, 18 warnings, 71 subtests passed** in 172.96s.
- `python -m ruff check tradingagents cli scripts tests` — **passed** (`All checks passed!`).
- `python -m compileall -q tradingagents cli scripts` — **passed**.
- `git diff --check` — **passed**.

## Whole-branch scope review

The implementation scope contains 43 changed files (8,104 insertions and 179
deletions), limited to the Phase 9 forex evidence path, its CLI wiring, smoke /
replay scripts, and tests. The review explicitly found:

- No order or execution API was added; MT5 references in replay are read-only
  snapshot/clock types, and mutation names occur only in defensive forbidden-
  surface guards/tests.
- No Phase 7/8 writer, import, rebuild, ingestion, model download, or
  maintenance call was added. Runtime uses read-only catalog adapters; the
  only evidence retrieval call is the runner-owned integration service's one
  `EvidenceOrchestrator.query` call.
- No trust-policy weakening, per-agent retrieval, model-generated query,
  cloud evidence payload, fine-tuning, or training path was added.
- No Phase 10 code or stock evidence switch was added. Evidence CLI/config
  additions are forex-specific; stock graph/CLI signatures remain covered by
  regression tests.
- No credential, complete prompt, completion, or private-reasoning leakage
  was found in the production diff; privacy guards intentionally contain
  forbidden-token lists.

## Working-tree / artifact state

The worktree was clean before this review and has no generated smoke report,
database, model, prompt, completion, or reasoning artifact. The only changes
from this task are this verification report and the Task 15 ledger status.

Because the mandatory real smoke did not meet its prerequisite gate, this report
does not claim Phase 9 acceptance or permit Phase 10 work.
