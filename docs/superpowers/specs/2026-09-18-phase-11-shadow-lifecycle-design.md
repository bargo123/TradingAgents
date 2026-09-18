# Phase 11 Shadow Lifecycle and Eligibility Audit Design

## Scope

This bounded Phase 11 extension provides a repeatable, one-symbol shadow-data
collection lifecycle and a machine-readable explanation of Phase 10
eligibility. It reuses the existing `forex-watch once`, Phase 5 evaluator,
Phase 8 importer/rebuilder, Phase 9 audit, and Phase 10 Dataset Factory
boundaries.

Phase 7 remains read-only, pinned, and frozen. The workflow never sends an
order, calls an MT5 mutation API, changes the stock CLI, weakens eligibility,
starts training, or starts Phase 12.

## Current evidence

The authoritative real Phase 10 row was excluded for six independent reasons:

| Reason | Source evidence |
| --- | --- |
| `UNTRUSTED_TIER` | Phase 8 record was `TIER_C_DIAGNOSTIC_ONLY`. |
| `DECISION_CONTEXT_INCOMPLETE` | `shadow_decisions.decision_context_status=INCOMPLETE`; Phase 9 node context hashes were absent. |
| `TEMPORAL_INVALID` | `decision_reference_status=INVALID_TEMPORAL`; the reference quote preceded decision completion. |
| `NORMALIZATION_FAILED` | `action` was null and `normalization_status=FAILED`. |
| `CITATION_INVALID` | Phase 9 audit was `INVALID_REFERENCE`; available `K1,K2` were neither used nor rejected. |
| `OUTCOME_INELIGIBLE` | Phase 5 evaluations were `INELIGIBLE` with `source_context_eligible=0`. |

The audit is diagnostic only; it cannot promote a row.

## Recommended architecture

`scripts/phase11_shadow_cycle.py` is a bounded operator wrapper. It invokes
the existing `forex-watch once` command for EURUSD, then explicitly runs the
existing Phase 8 import/rebuild and Phase 10 build commands against dedicated
operator-supplied roots. The wrapper does not construct a second graph or MT5
provider. Evidence roots and the frozen Phase 7 generation are supplied via
the existing environment/configuration seam.

The wrapper supports two safe stages:

1. `collect`: one read-only shadow decision and Phase 8 import/rebuild.
2. `evaluate`: the existing Phase 5 pending evaluator, followed by a fresh
   Phase 10 build and an eligibility audit. Evaluation is attempted only when
   the approved horizon is mature; otherwise the result remains `PENDING`.

The default target is EURUSD and the shortest already-approved horizon is
300 seconds. No timestamps are fabricated and no horizon policy is changed.

## Audit contract

The wrapper emits one bounded JSON object containing decision identity, source
paths/fingerprints, candidate/eligible counts, reason counts, and an
`exclusion_audit` list. Each audit item has `decision_id`, `reason`, and a
bounded `upstream` mapping with only source field names and scalar/status
values. Prompts, completions, reasoning, secrets, and rendered evidence text
are never emitted.

The operator may provide a dedicated runtime cache directory. When supplied,
the runner's existing Phase 9 audit default resolves beneath that directory,
so a new lifecycle does not append to an acceptance audit database.

The mapping is generated from the existing Phase 10 `DatasetExclusion`
records and read-only source rows; it does not duplicate or alter the
eligibility predicate. Missing fields are represented as `null` and remain an
explicit failure.

## Safety and immutability

- `forex-watch once` retains its lease and serialized MT5-operation guards.
- Phase 5 uses its existing read-only historical quote API and reports
  `llm_calls=0`.
- Phase 8 imports append-only into a dedicated root; existing acceptance roots
  are rejected if they already contain a published generation unless the
  operator intentionally selects that root for normal append-only use.
- Each Phase 10 build uses a fresh output root; an existing non-empty root is
  rejected rather than deleted or overwritten.
- All output reports include `executed=False`, mutation/network counters, and
  source/artifact fingerprints.

## Failure behavior

Any subprocess error, missing path, stale lease, invalid source, or non-zero
collector result stops the cycle with a typed status and bounded error type.
The wrapper never retries a graph/LLM failure and never converts an incomplete
decision into a training example.

## Verification boundary

Deterministic tests cover command ordering, fresh-root checks, exact audit
reason mapping, pending-versus-complete evaluation, no mutation/network
construction, and no stock-CLI changes. One real EURUSD lifecycle is run only
after those tests pass. If the resulting Phase 10 eligible count is zero, the
workflow stops and reports the exact remaining upstream reason(s).
