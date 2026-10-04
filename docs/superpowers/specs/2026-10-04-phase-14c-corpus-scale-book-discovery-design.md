# Phase 14C — Corpus-Scale Book Strategy Discovery

**Status:** Design submitted for review; implementation is not authorized until this spec and a subsequent implementation plan are approved.

**Prepared against:** `shadow-pm-reliability` at `418dd8c3fd29197f2edfe5d83681d7248dd72d1a`.

**Phase 14B prerequisite:** committed at `418dd8c` (`fix: bound Phase 14B Qwen evidence extraction`).

**Frozen Phase 7 source:** `C:\p7fast`; active generation `gen_607de64268a04a6ab09ffa1e160fc280`; generation fingerprint `428929b8060d96dd59be8e10ca338ded9534c12265a657aff5fad5cc97ffe3d1`; population hash `sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17`.

## 1. Purpose and boundaries

Phase 14C broadens discovery coverage over the already-published Phase 7 corpus. It addresses the known Phase 14B limitation—20 actionable-presence groups selected from one narrow query pass—without changing the Qwen model, prompt meaning, response schemas, bounded token settings, deterministic evidence grammar, replay behavior, or promotion gates.

The current Phase 7 catalog has 101 non-Linux source resources: 93 active indexed unique documents, four `NEEDS_OCR` resources, and four exact duplicate aliases. The expected unique-source denominator is 97; the searchable denominator is 93; the active generation contains 32,087 chunks. The run must re-read and report its own inventory rather than trusting these design-time counts.

Phase 14C is an explicit, offline research operation. It does not alter Phase 7 or the live Forex/HFT path. It does not generate orders, change risk, replace an incumbent, use MT5, alter Phase 11A, mix experience into published-book retrieval, train a model, or send knowledge to a hosted service. Automatic candidate status remains capped at `SHADOW_CHALLENGER`.

## 2. Goals and non-goals

Goals:

- Search the full validated Phase 7 index through a deterministic, versioned family query bank with multiple formulations for important concepts.
- Measure retrieval and actionable-evidence coverage by unique source document, while distinguishing indexed books from duplicate aliases and `NEEDS_OCR` sources.
- Diversify by family, book, and section; deduplicate repeated retrievals without erasing independent provenance.
- Reuse the existing compact, local Qwen presence/concept/atomic extraction contracts and validated Phase 14B cache entries.
- Assemble source-grounded concepts and complete specs conservatively, track source conflicts, and distinguish book facts from any research parameter requirement.
- Recheck the current `range_rejection` and `momentum_continuation` implementations component by component.
- Evaluate only complete, HFT-suitable, deterministically implemented candidates using the existing causal replay and Phase 14 gate.
- Preserve resumability, auditable provenance, safe telemetry, and immutable Phase 7/source inputs.

Non-goals:

- Proving profitability or forcing a candidate count.
- Treating semantic retrieval rank or a model classification as evidence that a rule is supported.
- Parsing/rebuilding Phase 7, OCR, changing the embedding model, or changing the Phase 7 query/index contract.
- Weakening the rule grammar, schema, HFT suitability, replay preflight, sample minima, cost checks, or promotion gate.
- Generating or executing arbitrary model-written code, adding an LLM to any runtime tick path, or changing Phase 12D/live behavior.
- Training/fine-tuning, MT5 access, order execution, or automatic promotion beyond `SHADOW_CHALLENGER`.

## 3. Alternatives and recommendation

1. **Increase one global query's `top_k`.** Rejected. It still favors the same high-ranking books and generic passages, and provides no family-level coverage accounting.
2. **Run a separate full model extraction over every chunk or book.** Rejected. This repeats expensive semantic work, turns retrieval into unbounded prompt construction, and increases opportunities for unsupported composition.
3. **Versioned multi-query retrieval, deterministic book/family diversity, then staged local classification and exact-source assembly. Recommended.** It reuses Phase 7 hybrid retrieval, the existing compact Phase 14B model boundary/cache and strict evidence grammar. It makes retrieval breadth measurable before spending model calls, bounds classifier work, and preserves every selected source reference.

## 4. Architecture

```text
Pinned, read-only Phase 7 generation
        ↓
Unique-source inventory + versioned family/query bank
        ↓
Hybrid retrieval fan-out (semantic + lexical + RRF)
        ↓
Exact dedup + score/provenance retention
        ↓
Family/book/section-diverse bounded evidence selection
        ↓
Existing tiny local-Qwen presence classifier and cache
        ↓
Existing compact concept grouping + atomic evidence-index extraction
        ↓
Exact-source rule grammar and provenance validation
        ↓
Canonical concept clustering / conflict-aware multi-source assembly
        ↓
StrategySpec completeness + data/HFT suitability
        ↓
Finite deterministic strategy registry
        ↓
Existing causal replay, costs, and unchanged Phase 14 gate
        ↓
At most SHADOW_CHALLENGER
```

New discovery code is isolated under `tradingagents/self_enhancement/` and an explicit `cli/phase14c.py`. The existing Phase 14B pipeline remains usable with its present defaults. Phase 14C can add public bounded/batch seams around its existing presence and extraction stages, but must preserve the same schemas, prompt intent, output limits, loopback-only provider, safe telemetry, and cache identity rules. It must not replace the Phase 14B pipeline with a new model workflow.

The Phase 14C CLI opens the knowledge generation and indexes read-only/offline. It does not import or construct MT5, a watcher, a supervisor, or a broker gateway. The command is never called from runtime HFT modules.

## 5. Source inventory, pinning, and identity

Before work, validate the active generation ID, full generation fingerprint, population hash, `VALIDATED` status, and matched vector/lexical readiness against explicit expected values. Recheck them after discovery and evaluation. Any mismatch aborts with a typed pinned-generation error and no model calls.

The existing `KnowledgeCatalog` has document lookup and per-document chunks but no public all-document listing. To avoid changing frozen Phase 7 code, Phase 14C will use a small read-only inventory adapter against the catalog SQLite file with `mode=ro` and `PRAGMA query_only=ON`. It validates the expected catalog schema and reads resources, aliases, documents, and active chunk counts only. It performs no writes, ingestion, schema initialization, or checkpointing. Unknown schema/state values fail closed.

Inventory identities are deduplicated by Phase 7's source-hash/document identity policy. Exact duplicate aliases count once as a document and remain reported as aliases. `NEEDS_OCR` resources count in the 97-source inventory but not among searchable/indexed documents. They are reported as unavailable to text retrieval; no OCR or candidate is synthesized for them. The report separately gives total unique sources, indexed/queryable documents, aliases, OCR-only sources, and any other non-searchable states.

Every hit is checked against the pinned generation and catalog document/chunk/source hash, exact stored text, active/readiness flags, and projection population identity before entering a cache key or model request. A mismatch aborts or quarantines that hit; it is never repaired from another chunk.

## 6. Deterministic discovery query bank

Add a closed, versioned query bank. Each entry has a stable family code, formulation ID, query text, and optional deterministic content-type restriction. Query text is a search request only; its presence is not a claim that any source supports the family.

The initial bank covers applicable families from the request: scalping, momentum, momentum continuation, breakout, failed/false breakout, range rejection, mean reversion, pullback, trend continuation/reversal, support/resistance, volatility expansion/contraction, channel breakout, price action, candle reversal, session open/breakout, London/New York/overlap sessions, time-based entry/exit, trailing stop, breakeven/profit protection, failed-move exit, momentum-reversal/no-progress/volatility exits, risk/reward, position sizing, expected move, short-term/intraday FX, tick scalping, and microstructure. Unsupported corpus/tooling categories may be omitted only with a recorded rationale and bank-version change.

Important families have multiple deterministic phrasings (e.g. failed-breakout reversal, return-inside-range, false-breakout entry/exit). Phrase variants are siblings of one family; query count is not evidence count. Bank ordering, formulation ordering, query hashes, and retrieval parameters are stable and included in the run identity.

Each query uses the existing local hybrid `KnowledgeQueryService`; dense and lexical ranks are fused by its existing RRF and reranking implementation. A bounded `top_k` is used per formulation rather than one global corpus query. Query-family, formulation, rank, scores, content type, and full source provenance are attached to the retrieval record. No internet search or external vector service is used.

## 7. Diversity, caps, and deduplication

First deduplicate repeated hits by `(document_id, chunk_id, source_hash)`, preserving the complete set of query/family/formulation references and score observations. Exact repeated text and near-duplicate text may be clustered for diversity, but all provenance records remain attached. Near-duplicate clustering is a deterministic selection aid, not a destructive merge.

Use deterministic diversity ordering across family, document, section, and retrieval score. Proposed V1 bounds, subject to plan-level fixture validation:

- at most two distinct chunks from one document within one query family after sibling formulations are merged;
- at most two chunks from a section within a family;
- at most eight three-sentence presence groups per document and forty per family;
- a hard run limit of 750 presence groups, with groups selected by deterministic round-robin over family → document → section.

These caps target hundreds of groups rather than the prior 20 while bounding local-model work. A document/family that has no qualifying evidence cues is not forced to produce a group or candidate. If a cap leaves eligible groups unclassified, the report records the exact deferred count and returns an incomplete/budget-limited status; it must not call that exhaustive classifier coverage. Limits are versioned and never silently expanded. The final implementation plan must include measured retrieval-pool sizes and a conservative runtime estimate from a small dry-run before the real model sweep.

For exact byte aliases, count only one Phase 7 document. For exact/near-identical text across distinct document IDs, preserve all source records, flag the duplicate cluster, and do not claim independent multi-book support solely from repeated text. Independent-source counts use unique, non-alias document IDs and exclude sources identified as duplicate copies for the specific supporting passage.

## 8. Evidence preparation and local-model work

Deterministic code segments each selected chunk into stable source-addressed sentences/spans using the existing Phase 14B segmentation/ID rules. It ranks candidates with the existing rule-cue score and deterministic retrieval/diversity tie-breakers. The broad screen does not ask the model to author rules.

The presence stage reuses the current closed output (`ACTIONABLE`, `NONE`, `AMBIGUOUS`), compact evidence indexes, temperature-zero local Ollama boundary, same token limit, and same strict response schema. New public batch methods may allow more selected sentences than Phase 14B's current 60-sentence cap; the Phase 14B default remains unchanged. Model calls stay sequential and bounded, with cache reuse and safe per-call latency/token/finish/failure metadata. At most the existing bounded split retry behavior applies; no retry loop is added.

Only evidence classified actionable proceeds to concept grouping and atomic rule extraction. Model responses can select closed labels and positions in the exact supplied batch. They cannot create a quote, offset, rule value, document identity, source hash, confidence, numeric parameter, or missing rule. Exact values and spans are resolved from the indexed source by deterministic code and checked by the existing finite grammar/unit/provenance validators. `AMBIGUOUS`, malformed, unsupported, or out-of-range output is not promoted and is counted by its failure class.

Cache keys continue to bind generation ID/fingerprint, model and resolved model version, prompt/schema/segmentation versions, task, and ordered stable evidence IDs. Phase 14B compact-cache entries may be reused only on exact key match. Cache records contain validated structured output and safe scalar telemetry only, never prompts, completions, or reasoning. Cache writes are outside Phase 7 and source databases.

## 9. Canonical concepts and multi-source assembly

Each accepted atomic rule is represented by its parsed semantic tuple (stage, feature/operator/direction/value/unit/condition/horizon as applicable) and one or more exact source spans. Model-assigned family labels are proposals, not evidence. A canonical concept requires a closed family plus deterministic rule-level anchor compatibility; lexical/name similarity alone cannot merge concepts.

For cross-document assembly, sources may contribute complementary stages only when they share at least one matching, source-supported entry/confirmation/invalidation anchor and have no incompatible same-stage rule. Every assembled rule keeps its own `EvidenceSpan`; model-selected evidence cannot be relabeled as support for a different rule. Different entry anchors form separate concepts. Contradictory same-stage rules become explicit variants or `SOURCE_CONFLICT`; the assembler never picks the more convenient source. A conflict or missing required executable stage prevents an executable spec.

The Phase 14C result envelope records one of:

- `SINGLE_SOURCE_COMPLETE`
- `MULTI_SOURCE_CONSISTENT`
- `COMPOSITE_RESEARCH_HYPOTHESIS`

`COMPOSITE_RESEARCH_HYPOTHESIS` explicitly means a cross-source combination not stated as a complete strategy in any one book. It is non-executable in V1 and cannot be described as a book-authored strategy. The existing strict `StrategySpec` remains unchanged; Phase 14C stores this classification in a separate typed assembly record that references the spec. Only complete source-supported records with no unresolved conflict can proceed to suitability/implementation.

Multi-book counts use independent unique sources, not chunks, pages, aliases, or repeated quotations. All supporting and contradicting source references remain in the record.

## 10. Research parameters and unspecified values

No missing value is invented to finish a StrategySpec. When a concept is source-supported but a required numeric parameter is not stated, record `RESEARCH_PARAMETER_REQUIRED` in a separate typed parameter request and leave the executable rule absent. Do not attribute a later selected research value to a book.

V1 does not automatically choose a numerical grid unless a predeclared deterministic bound is already justified by a named feature/unit contract and explicitly marked research-derived. A research parameter must be separate from source claims, reproducible, bounded before evaluation, and run through existing Phase 14 validation. If no such approved range exists, the spec remains partial and no replay occurs. This preserves room for later bounded research without making discovery manufacture numbers.

## 11. HFT/data suitability and current-strategy mapping

Run every assembled record through the existing `classify_suitability`, `StrategySpec.is_executable`, and finite `BookStrategyRegistry`; add no capability merely because a book mentions it. True L2/queue position, cross-venue latency, institutional order flow, or other unavailable input is `UNAVAILABLE_DATA`. Missing stages, conflicts, unsupported grammar, or unimplemented deterministic operators remain non-executable with explicit reason codes.

Only the offline Phase 14 deterministic registry may gain a reviewed primitive if the corpus contains a fully specified supported concept and current finite primitives cannot express it. No generated Python, `exec`, dynamic imports, runtime model access, or changes to live `forex/hft` strategies/engines are allowed. The first diverse candidate batch is capped at 5–10 canonical candidates and may be smaller or empty.

Recheck `range_rejection` and `momentum_continuation` against the current source implementation, not their names. The comparison uses a versioned implementation-contract snapshot derived from actual strategy defaults and exits. For each strategy report `entry`, `confirmation`, `expected_move`, `exit`, `risk`, and `holding_horizon` as `MATCHED`, `PARTIALLY_MATCHED`, or `NO_DIRECT_MATCH`, with exact book evidence only for matched components. Missing, unrepresentable, or contradictory code components remain unmatched/conflicted. The currently observed code includes momentum thresholds of 2 points, 0.6 direction persistence, and 3 ticks; range thresholds of 0.2 edge fraction, 2 points minimum range, and 4 ticks; family-specific dynamic exit profiles are in `forex/hft/engines.py`. Implementation snapshot tests must detect drift before claiming a mapping. Risk controls are contextual constraints and are not inferred from book passages.

## 12. Deterministic implementation and causal evaluation

Only complete, provenance-valid, HFT-suitable specs supported by the finite offline registry are candidates. Candidate selection is based on evidence completeness, data compatibility, source clarity, concept diversity, and implementation feasibility—not historical profitability.

Replay reuses the current Phase 14 preflight and `BookCandidateExperiment` path. At execution time it must verify read-only source fingerprints and candidate readiness; prior counts are not authorization. The expected existing clean preflight to recheck is 118 verified experiences, three quarantined/excluded, 21,131 causal ticks, and 18 segments. Replay remains causal and segment-isolated with development, validation, walk-forward, and unseen partitions. Commission stays `UNKNOWN`; use existing cost-sensitivity cases. No gate, minimum, split, exit, risk, fill, or metric is weakened. No complete HFT-suitable candidates means no replay; replay blockers are reported exactly.

Only the unchanged Phase 14 promotion gate can accept a candidate. Insufficient samples remain insufficient; all candidates may fail. Highest automatic state is `SHADOW_CHALLENGER`, which is observational only. No incumbent replacement, live HFT integration, MT5, DEMO execution, or real-money path is introduced.

## 13. CLI, artifacts, and resumability

Add an explicit `phase14c` CLI with metadata-only `status` and explicit `discover-and-evaluate` (or equivalently separated `discover` and `evaluate`) operations. Status must not instantiate the embedder/model or read market/trading databases. Discovery requires explicit Phase 7 root, expected generation and fingerprint, dedicated new Phase 14C artifact root, and local model options only when model classification is explicitly enabled. No supervisor/watcher command invokes this CLI.

Proposed fresh artifact layout:

```text
<phase14c-root>/
  run-manifest.json
  corpus-inventory.json
  query-bank.json
  retrieval/evidence.jsonl
  selection/coverage.json
  classification/validated-results.jsonl
  concepts/concepts.jsonl
  concepts/conflicts.jsonl
  specs/strategy-specs.jsonl
  specs/assembly-records.jsonl
  mappings/current-strategy-mapping.json
  evaluation/candidates/<candidate-id>/...
  phase14c-report.json
```

Each run artifact root is created only after preflight and must be absent; an existing root is never deleted or overwritten. An interrupted run may resume only through an explicit resume mode that verifies exact generation, query-bank/config fingerprint, model version, cache identity, and existing result schemas before extending the run. A mismatch fails closed and preserves both roots. The separate compact atomic cache may be reused or extended only outside Phase 7 and Phase 14A/HFT/DEMO source databases.

The manifest records schema/query-bank/selection/assembly/mapping versions; Phase 7 identity; source inventory; model/version/settings; cache path; source DB fingerprints; feature/data contract; replay/gate versions; and safe counts/telemetry. It does not record prompts, completions, reasoning, credentials, or model-authored prose not needed as a validated source-free summary.

## 14. Statuses and metrics

Valid final run states include:

- `DISCOVERY_COMPLETE_NO_COMPLETE_SPEC`
- `DISCOVERY_COMPLETE_NO_HFT_SUITABLE_SPEC`
- `CANDIDATES_EVALUATED_NONE_PASSED`
- `SHADOW_CHALLENGER_CREATED`
- explicit non-success states for pinned-generation mismatch, invalid provenance, provider/schema failure, incomplete coverage/budget, replay preflight failure, or source mutation.

Never report a candidate evaluation state if no candidate replay occurred. Discovery can be complete with zero candidates.

The report includes:

- source universe counts: unique total, indexed/searchable, aliases, `NEEDS_OCR`, other excluded/unavailable;
- retrieved/selected/classified coverage: books represented, books never retrieved, books with actionable evidence, family coverage, section coverage, raw query hits, unique chunks, duplicate clusters, deferred-by-cap counts;
- presence groups (`ACTIONABLE`, `NONE`, `AMBIGUOUS`, failures), unique actionable evidence, concepts, independent multi-source concepts, rule accept/reject counts, source conflicts, assembly modes, complete/partial specs, and suitability outcomes;
- per-call and aggregate model/cache/token/runtime/failure/retry statistics, with no raw text;
- existing-strategy component mapping;
- candidates implemented/evaluated, trade counts, stage metrics, cost sensitivity, gate decisions/rejections, and challengers;
- Phase 7 generation/fingerprint/inventory and Phase 14A/HFT/DEMO source fingerprints before and after, plus all test/tool results.

If any selection cap prevents all planned evidence groups from being classified, show the residual count and do not claim full classifier exhaustion. Retrieval coverage does not imply that every book contains a strategy; only source-validated rules count as actionable.

## 15. Verification and acceptance

Fixture tests use tiny synthetic catalogs and fake query/model boundaries; they never call Ollama, MT5, internet, or download models. They cover:

- query-bank determinism, family/formulation coverage, version/hash changes, and no unsupported claim semantics;
- full-corpus query fan-out orchestration, multi-query retrieval, RRF rank metadata, pinned generation checks, per-document/per-family/per-section caps, exact chunk dedup, near-duplicate clustering without provenance loss, and book coverage accounting including OCR-only/aliases;
- deterministic sentence IDs, group ordering, hundreds-scale selection bounds, budget-deferred status, cache reuse, interrupt/resume identity checks, and no prompts/completions/reasoning persistence;
- strict presence classification and existing tokenizer/schema/output behavior unchanged; actionable-only downstream work;
- concept clustering, incompatible anchor separation, same-stage conflicts/variants, independent-source counts, compatible multi-source assembly, explicit composite-research labeling, and unsupported cross-source joins;
- every executable claim exactly traceable to generation/document/hash/chunk/span/quote and parser grammar; research parameters never attributed to a book; no invented numeric values;
- HFT suitability, unavailable feature rejection, finite registry allowlist, candidate cap, component-wise existing-strategy mapping/drift detection, and no LLM/runtime import on the HFT tick path;
- causal replay fixture parity, segment isolation, no metric/gate weakening, no replay of incomplete/non-HFT specs, no promotion above `SHADOW_CHALLENGER`, and no broker API access.

Real acceptance runs the broad frozen-corpus query bank and the configured bounded classifier through completion (or reports a precise budget-limited incomplete state), resumes exact cache hits, and continues through extraction/assembly/suitability/evaluation only when eligible candidates exist. Verify no model output or cache operation writes under the Phase 7 root. Compare active Phase 7 identity/inventory and all replay source fingerprints before and after. Never claim end-to-end evaluation if there is no complete HFT-suitable candidate. Full suite/Ruff/compileall/diff checks are required after implementation approval and execution.

## 16. Safety invariants

- Phase 7 is frozen: no parse, source, catalog, embedding, vector, lexical, generation, or manifest writes.
- Only preinstalled local FastEmbed and the explicit loopback-only Qwen 2B adapter are used; no network/cloud provider.
- Phase 11A, Phase 14A, and Phase 12D sources are read-only; published knowledge and trading experience remain separate.
- No model call or retrieval occurs on the live HFT tick path.
- Unsupported, ambiguous, incomplete, conflicting, out-of-domain, or unavailable-data rules fail closed.
- Book-sourced rules and research parameters remain distinct and separately labeled.
- Existing live strategies, risk settings, watcher, broker gateway, and execution code are untouched.
- Candidate outcomes cannot exceed `SHADOW_CHALLENGER`; real-money execution remains impossible.

## 17. Phase handoff

Phase 14C ends after the offline corpus discovery, eligible candidate replay, and final report. A later phase would be required for any runtime observation integration, further learning from trading experience, strategy promotion, or execution. This design does not authorize those steps.

## Self-review

- The design uses a broad versioned query bank, not a global top-20 sample.
- Phase 7 has no write path; the inventory adapter is explicitly read-only because the frozen catalog lacks a public document-list method.
- The existing Phase 14B local model/schema/token/caching boundary is preserved; only bounded batch access is proposed.
- Sources/aliases/OCR-only items are distinct in denominators and coverage.
- Every selected model reference resolves to indexed source text and complete provenance; model labels and similarity do not validate a rule.
- Composite research hypotheses and numerical research parameters cannot become book facts or executable strategies.
- Phase 12D/live strategy behavior, risk, MT5, execution, training, and Phase 11A are outside the change boundary.
- Replay and promotion reuse the current existing gates; no success, sample, or profitability threshold is weakened.
