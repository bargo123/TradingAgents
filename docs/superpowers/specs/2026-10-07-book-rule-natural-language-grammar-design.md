# Source-grounded natural-language book rules

Date: 2026-10-07

Status: draft for user review. Approval to write this document is recorded;
implementation and deployment are not approved by that decision.

## 1. Purpose and boundaries

Make the offline book-discovery pipeline capable of recognizing explicit rules
written in ordinary book language. Keep the local model involved in proposing
concepts and rule stages; deterministic code must establish what the source
actually supports before a rule can become executable.

The intended path remains:

```text
pinned Phase 7 chunks -> offline local-model proposals
-> deterministic source/meaning validation -> HFT suitability
-> reviewed deterministic primitives -> causal, costed replay
```

This change does not promise profitable trades, zero losses, or 10% daily returns.
It does not increase position concurrency, alter the incumbent strategies,
change risk limits, restart a supervisor, or enable live-money execution. The
separate multi-position proposal is outside this design.

No model enters the tick path. No new book candidate is automatically promoted
into the DEMO runtime. Existing execution authority and reconciliation gates
remain in force.

## 2. Evidence for the change

The completed run at `C:\phase14c\runs\run-20261006-topk30` reached 83 of 93
queryable books, screened 417 groups, and produced eight actionable labels.
Four extracted concepts produced five unsupported rules and no complete spec.
Its outcome was INCOMPLETE / DISCOVERY_INCOMPLETE_BUDGET_LIMITED, not success.

A separate read-only audit applied the production sentence segmentation and
parser to all 32,087 active chunks in the pinned generation: 576,134 source spans,
zero supported spans. The parser requires verbatim canonical sentences with
uppercase directions, feature identifiers, comparators, units and explicit
condition forms. The local model currently selects indexes and stage labels; it
cannot rewrite those source sentences. Increasing retrieval alone cannot fix
this mismatch on the audited corpus.

A lexical review scan found 405 spans in 24 documents containing a numeric
pips/points/ticks/seconds unit and a trading-action term. These are review
material, not 405 valid rules. They include possible examples, criticism,
discretionary advice, non-FX rules and incomplete statements.

The approved Phase 14B/14C designs intentionally restrict model authority and
exact-source parsing. This document extends that interface explicitly; it does
not reinterpret the prior design as permission to infer missing semantics.

## 3. Chosen approach

Use a finite, reviewed natural-language normalizer with field-level source
proofs. The local model selects supplied evidence and proposes a closed concept
family/stage. It does not choose thresholds, produce replacement quotes, or
decide that ambiguous prose is executable.

Keeping the canonical grammar alone preserves the current zero-match limitation.
Accepting arbitrary model-authored rule objects would make exact quotation checks
insufficient: a real quote can still be assigned an unsupported meaning. Neither
alternative is suitable for this change.

The normalizer must recognize a reviewed source construction, bind its operands,
and return either a complete supported interpretation or a typed rejection.
Pattern coverage can grow through separately reviewed source cases. Broad fuzzy
matching, embedding similarity and model confidence are not acceptance criteria.

## 4. Source support and executability are separate

A source-supported rule may still require unavailable features or a new primitive.
For example, an explicit moving-average crossover is not automatically the
runtime's momentum feature; a pip-denominated stop distance is not automatically
a condition on momentum. Neither may be relabelled to fit the existing engine.

The result states are:

- REJECTED: provenance, interpretation or consistency failed.
- SUPPORTED_NONEXECUTABLE: meaning is explicit, but data, primitive, parameters,
  context, suitability or required strategy stages are unavailable.
- EXECUTABLE_ELIGIBLE: all source, feature, primitive, completeness and HFT checks
  passed. This permits offline replay only, not broker execution or promotion.

Missing values produce parameter requests/non-executable records, not defaults.
Discretionary words such as sensible, realistic, next move or sufficiently strong
do not become numeric thresholds. A model selecting a stage does not establish
that the source describes that stage.

## 5. Normalization proof contract

Add an offline normalized-rule record rather than overwriting source quotations.
It contains:

- a stable rule ID, stage and proposed concept family;
- exact evidence addresses: generation, document, source hash, chunk, offsets
  and original quote for each supporting span;
- the parsed operation and typed operands, direction, comparator, exact Decimal
  values, source units, conditions and any explicitly stated horizon;
- per-field evidence bindings and a reviewed grammar-pattern ID;
- grammar, evidence-schema and feature-binding versions;
- a semantic fingerprint and typed support/executability result.

The deterministic resolver must be able to recompute every accepted field from
the evidence and reviewed pattern. It must not trust a serialized model claim.
An omitted field, invented value, reversed comparison, wrong stage or unsupported
negation fails validation.

Context may be used only as explicitly addressed supporting spans, with source
relationships checked. Start with the same document and contiguous context in
the same chunk; cross-chunk definitions remain non-executable in this first
implementation. A bundle is bounded to eight spans from one chunk, each at most
600 characters. Exceeding the bound is reported, not silently truncated into a
different rule. Unreferenced definitions and unrelated passages are not context.

The new versioned model-response schema may select indexes from these enumerated
same-chunk spans and propose a stage/family. The resolver, not the model, binds
the fields and checks that the selected context actually defines the rule.

The original raw text remains unchanged. Case/whitespace/token normalization is
performed on a separate view with a character-offset map to the source. Damaged
PDF glyphs are not repaired by guesswork. No canonical sentence is fabricated
and attributed to a book.

## 6. Reviewed rule coverage

Initial patterns cover explicit directional actions with literal numeric
comparisons, explicit numeric targets/limits, and explicitly stated holding-time
limits. A pattern requires a source case from this corpus and counterexamples
before being enabled. It must distinguish a recommendation from a hypothetical
example, a quoted rule being criticized, a prohibited action, and a statement
about exchange regulation.

Feature aliases are a closed reviewed mapping. A successful word match is not
sufficient: the source's feature definition, lookback, price basis and unit must
match the causal feature contract. Undefined lookbacks, daily/minute/tick unit
confusion, missing anchors and chart-only dependencies remain non-executable.

Do not silently convert pips to broker points. Store source units and values
exactly; this first implementation rejects executable unit mismatches. A later
instrument-specific conversion needs separately reviewed metadata and a recorded
derivation. Likewise, an explicit minute horizon cannot become seconds with a
shorter magnitude to make a rule HFT suitable.

Completeness still requires confirmation, entry, exit, expected move, horizon,
invalidation, profit protection and stop behavior. A missing stage is not filled
from a generic risk policy. Multi-book assembly must preserve independently
supported compatible semantics, rather than stitch unrelated advice together.

New features, strategy families and execution primitives are not part of this
grammar change. Supported rules requiring them are reported for later work.

## 7. Component integration

The shared resolver becomes the single meaning-validation boundary for new
normalized records. Integrate it with:

1. `book_atomic_extraction.py`: bounded offline evidence selection and proposed
   stages; deterministic normalization and typed unresolved results.
2. `book_drafter.py` / `book_pipeline.py`: draft assembly and source-proof
   validation, with no fallback to unproved model prose.
3. `phase14c_assembly.py`: completeness, conflicts, independent-source checks
   and non-executable parameter requests.
4. `book_strategies.py`: instantiate only approved primitives after independently
   revalidating the normalized record. Preserve its condition restrictions;
   unsupported conditions cannot be accepted by the parser and ignored at runtime.
5. `phase14c_mapping.py`: attribution from verified normalized semantics, not
   lexical overlap or a forced match to momentum/range.
6. Reports and artifacts: show stage-by-stage rejection reasons and separate
   grammar-supported, primitive-supported and replay-eligible counts.

Legacy canonical records keep their existing parser and field comparisons.
Natural-language proof records use an explicitly versioned schema and resolver.
Do not weaken the legacy StrategySpec validator to accept unproved records.
Use a V2 offline StrategySpec envelope containing structured strategy fields,
the original evidence and a normalization proof for every claim. Each V2 consumer
recomputes the proof through the resolver before using those fields. It never
passes a natural-language quote into a legacy claim and expects the canonical
parser to accept it. The research registry accepts a validated V2 candidate only
when feature bindings, primitives and condition restrictions match exactly;
otherwise it remains supported/non-executable. The incumbent live registry path
does not receive V2 candidates or change in this work.

The implementation plan must enumerate every consumer of the parser and both
schema paths. No new natural-language record is usable by one component while
another silently reparses it under the legacy grammar.

## 8. Identity, caching and reproducibility

Bind grammar version, normalization schema, feature-binding contract, prompt
schema and evidence-bundle identity into new cache and artifact identities.
Keep Phase 7 generation/document/chunk/source checks and before/after source
fingerprints. A grammar change must not reuse a prior unsupported result as a
current validation result, nor reinterpret an old accepted result without replay.

Model requests remain native loopback-only Ollama calls, offline, temperature 0,
with finite context/output/timeout bounds and existing bounded retries. Store
validated structured results and safe telemetry only; never store credentials,
model reasoning or unrestricted completions. Source text is data, not an
instruction that can change model authority or validation rules.

Frozen indexes, source databases and historical artifacts are read-only. New
results go to a distinct research artifact root. Existing FAILED/INCOMPLETE runs
are not edited to look complete or resumed outside their supported lifecycle.

## 9. Verification and acceptance

Use genuine source addresses for positive integration cases and test-only
strings solely for parser edge cases. Synthetic fixtures must never appear in
corpus discovery counts, training experience, replay promotion or DEMO orders.

Required checks:

- Exact span/hash/generation failures, changed quotes and incorrect offsets reject.
- Each admitted natural-language pattern has a genuine source case and negative
  cases for direction, comparator, number, unit, stage, negation and context.
- A model citing a real but irrelevant sentence still fails meaning validation.
- General advice, criticism, regulation and hypothetical examples do not qualify
  as executable rules merely because they contain numeric/action terms.
- PDF corruption, implicit feature windows, unsupported conditions, missing
  stages and unavailable features produce explicit non-executable/rejected results.
- Legacy canonical parsing and acceptance remain unchanged. Unknown schema,
  grammar or feature-contract versions fail closed.
- Resolver output, draft validation, assembly, registry and mapping agree on the
  same rule semantics. No component bypasses field/proof comparison.
- Model responses stay compact and source-addressed; missing evidence does not
  trigger semantic repair, an internet-model fallback or manual-only extraction.
- A bounded genuine-corpus run exercises the new path and records all residual
  coverage limits, rejected interpretations, available primitives and replay gates.
- Any eligible complete candidate undergoes chronological causal replay with
  verified costs, segment boundaries and unchanged promotion criteria. Unresolved
  commission, latency, quote-quality or data gaps cannot be labelled solved by
  better grammar.
- The offline command does not construct MT5, a gateway, supervisor or watcher.
  Incumbent strategy files, execution configuration, risk and concurrency remain
  untouched by this implementation.

Success for this engineering change is genuine source-supported normalization
with consistent fail-closed validation across consumers. If the corpus still
cannot supply a complete HFT spec, report the exact residual missing fields or
primitive/data requirements. Do not claim the broader trading goal achieved.
The prior run's unretrieved books and deferred groups are separate coverage
limits; extending the grammar does not establish that corpus discovery is complete.

## 10. Delivery and review gates

Write and review an implementation plan only after this written spec is approved.
The plan must select real source cases, detail schema compatibility and consumer
changes, and define bounded model-run verification before product edits begin.
Execution method is chosen at that plan review; no subagent work is assumed.

After implementation and verification, hand over offline artifacts and remaining
blockers. Deployment, live DEMO promotion and multi-position runtime changes
require their own reviewed decisions and verification.
