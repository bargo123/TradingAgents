# Phase 10 Dataset Factory

Phase 10 builds a deterministic, local-only training/evaluation dataset from
already-approved Phase 5/6 shadow records, Phase 8 experience records, and
Phase 9 evidence-audit records. It is an offline data-quality boundary; it does
not make trading decisions or execute orders.

## Scope and safety

The factory reads source SQLite databases through read-only adapters. It never
opens MT5, calls Ollama or another LLM, invokes tools, makes network requests,
or writes to Phase 5/6/7/8/9 artifacts. The stock `tradingagents` CLI and
`forex-watch` are unchanged. Generated files belong only below the configured
dataset output root. Every manifest records safety counters, which must remain:

```json
{"network_attempts":0,"llm_calls":0,"tool_calls":0,"mt5_calls":0}
```

An empty eligible set is a valid, explicit result. No label is fabricated from
an incomplete or unavailable decision.

## Inputs and provenance

`DatasetConfig` names one or more Phase 5/6 source databases, a Phase 8 root,
an optional Phase 9 audit database, and a separate output root. Adapters use
SQLite read-only mode and bind source identities to file/WAL marks and row
fingerprints. The source files are never modified. The factory joins by the
stable decision/source-run identities and preserves phase-labelled fingerprints
for `phase56`, `phase8`, and `phase9`.

Each canonical example carries bounded, JSON-safe provenance including source
paths/IDs, source hashes and snapshot fingerprints, decision and evidence
fingerprints, schema/policy versions, and the analysis timestamp. Sensitive
fields and values (prompts, completions, reasoning/CoT, credentials, and token
material) are rejected at contract boundaries rather than redacted into an
ambiguous record.

## Canonical example and eligibility

The published row schema is `phase10.dataset.example.v1`, with these top-level
sections:

```text
example_id, decision, market, research, outcome, trust, provenance,
schema_version
```

The `decision` section retains the normalized action and bounded decision
metadata. `market` retains the analyzed quote/snapshot fields. `research`
retains only bounded Phase 9 evidence/context metadata and recomputed eligibility
facts (for example status, bundle/integration, selected counts, audit references,
context hash, and gate outcomes). It does not copy visible agent report prose or
raw research payloads. Prompts, completions, reasoning/CoT, and token material
are never stored.
`outcome` retains evaluation basis/horizon, analysis and decision-reference
timestamps/quotes, statuses, directional BUY and SELL counterfactuals, HOLD
opportunity-cost semantics, and cost-aware MFE/MAE when supplied by Phase 5.
`trust` records the recomputed trust/context/normalization/citation gates.

Only trusted Tier A, complete, normalized, temporally valid records with a
complete eligible outcome and valid evidence/provenance become examples. The
closed exclusion vocabulary is:

```text
UNTRUSTED_TIER, DECISION_CONTEXT_INCOMPLETE, TEMPORAL_INVALID,
OUTCOME_INELIGIBLE, OUTCOME_UNAVAILABLE, NORMALIZATION_FAILED,
CITATION_INVALID, EVIDENCE_INCOMPLETE, SCHEMA_UNSUPPORTED,
PROVENANCE_INCOMPLETE, DUPLICATE, SOURCE_INTEGRITY_FAILED
```

Excluded candidates are written to `excluded.jsonl` with their reason(s) and
bounded diagnostics. Tier B/C and any malformed or unverifiable record remain
excluded; they are not silently promoted for balancing.

## Deterministic splits

Eligible examples are sorted by UTC analysis timestamp, decision ID,
evaluation basis, horizon, and example ID. Group identity is the source run
when present, otherwise the decision identity, so related rows cannot cross
partitions. The stable chronological allocation targets 70% train, 15%
validation, and 15% test using largest-remainder allocation. Small populations
are reported as `INSUFFICIENT_DATA`, never padded or randomly reshuffled.

The writer emits identical rows in `train.jsonl`, `validation.jsonl`, and
`test.jsonl` according to the manifest assignment. Validation rejects duplicate
IDs, cross-split groups, non-chronological partitions, unknown rows, changed
rows, and inconsistent counts.

## Generation layout and validation

Each published generation is a content-derived, immutable directory below the
output root:

```text
<output-root>/<generation-id>/
  manifest.json
  examples.jsonl
  excluded.jsonl
  train.jsonl
  validation.jsonl
  test.jsonl
```

`manifest.json` contains schema, canonicalization, eligibility, split, and
source-adapter versions; source fingerprints; policy/config metadata; counts,
reason counts, class/profile/symbol/evaluation summaries; split status; file
hash/size/newline metadata; generation identity; and safety counters.
`validate_generation` verifies all hashes, rows, schemas, IDs, provenance,
source identity, split assignments, summaries, and generation identity without
writing. A generation is published atomically and an existing generation is
never overwritten.

## CLI usage

The standalone CLI is `python -m tradingagents.datasets.cli` (a project entry
point may also expose it as `dataset-factory`):

```powershell
python -m tradingagents.datasets.cli build `
  --source-db C:\path\phase6.db `
  --phase8-root C:\path\phase8 `
  --phase9-audit C:\path\evidence_audit.sqlite3 `
  --output-root C:\path\phase10-output

python -m tradingagents.datasets.cli validate `
  --generation C:\path\phase10-output\<generation-id>
python -m tradingagents.datasets.cli status --output-root C:\path\phase10-output
```

`build` supports repeatable `--source-db`, `--evaluation-basis`,
`--horizon-seconds`, and `--allow-empty/--no-allow-empty`; `--json` emits a
bounded machine-readable report. `status` only reads manifests and
`validate` only reads one generation. The acceptance harness
`scripts/phase10_dataset_smoke.py` requires existing inputs and a fresh,
non-populated output root; it refuses to delete or overwrite an existing user
artifact root and reports fingerprints and safety counters.

## Incremental and recovery behavior

Builds are idempotent for unchanged source bytes/WAL state and deterministic
for the same source snapshots and policy versions. New or changed source rows
are incorporated on the next build; removed rows/sources are represented in
inventory diagnostics and are absent from the newly published generation.
Exact duplicate source files and duplicate joined identities are excluded.
Broken optional Phase 8/9 sources are isolated as visible source-integrity
errors; one bad source does not corrupt already-published generations. Explicit
rebuild means selecting a new output root or changing a version/configuration;
the factory does not mutate a prior generation.

## Verification contract

Task-level tests use tiny SQLite/JSON fixtures and fake observations only. CI
does not require MT5, Ollama, network, model downloads, CUDA, or the real
library. Acceptance verification runs focused Phase 10 tests, affected Phase
5–9 tests, Ruff, compileall, and `git diff --check`. One real smoke may read
the approved Phase 5/6, Phase 8, and Phase 9 artifacts into a new output root;
it must report the exact output path, source fingerprints before/after, and
zero safety counters. A zero-example real result is valid when the source
quality gates exclude all candidates.

## Phase 11 handoff

Phase 10 ends at deterministic, auditable JSONL plus manifest validation. It
does not train, fine-tune, evaluate profitability, create labels beyond
preserved Phase 5 outcomes, or feed data into TradingAgents. Phase 11 may
consume only validated generations and must independently define dataset
selection, leakage controls, model/training policy, and approval gates. Any
future Experience Memory remains separate from published Knowledge RAG and
must not be merged into this dataset without an explicitly versioned contract.
