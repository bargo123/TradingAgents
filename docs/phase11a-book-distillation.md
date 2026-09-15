# Phase 11A book-knowledge distillation

Phase 11A creates a separate, grounded knowledge dataset from the validated
Phase 7 catalog.  It is a dataset-construction boundary for a later Phase 11
knowledge-training stage; it is not a trading strategy, decision, RAG, or
execution feature.

## Boundaries

* `new books` and the Phase 7 catalog are read only.  Phase 11A never rewrites,
  reparses, renames, or deletes a source resource.
* Only Phase 7 canonical chunks and their provenance are accepted as source
  evidence.  The factory does not independently parse books.
* PDF/EPUB parsing, OCR, MT5, TradingAgents decisions, Phase 8/9 evidence,
  Phase 10 eligibility, execution, model training, and Phase 12 evaluation
  remain outside this phase.
* Lessons are `BOOK_KNOWLEDGE`.  They cannot be coerced into
  `ForexPortfolioDecision` or mixed with Phase 10 `TRADING_EXPERIENCE` rows.
* Teacher output is bounded and optional.  Prompts, completions, private
  reasoning, credentials, and network responses are never persisted.

## Commands

The standalone entry point is `knowledge-distill`; it does not alter the
existing `tradingagents`, `knowledge`, `forex-shadow`, or `forex-watch` CLIs.

```powershell
# Read-only scan and deterministic bounded plan; writes the plan outside Phase 7.
knowledge-distill plan --phase7-root C:\path\to\phase7 --output C:\path\to\plan.json

# Distill only when an explicit teacher adapter is configured.
knowledge-distill distill --plan C:\path\to\plan.json --output-root C:\path\to\phase11a

# Validate hashes, schema, provenance, and immutable generation files.
knowledge-distill validate --generation C:\path\to\phase11a\generation-...

# Inspect bounded metadata without constructing a parser, embedder, or teacher.
knowledge-distill inspect --generation C:\path\to\phase11a\generation-...
```

`plan` performs no teacher calls.  `validate` and `inspect` perform no source
  or model construction.  `distill` returns
  `DISTILLATION_TEACHER_NOT_CONFIGURED` when no explicit adapter is available;
  it never fabricates a lesson to make a run appear successful.  Long runs are
  always explicit commands.

## Real Phase 7 acceptance

Run the bounded, planning-only harness after a validated Phase 7 generation
exists:

```powershell
python scripts/phase11a_real_acceptance.py `
  --phase7-root C:\path\to\phase7 `
  --report C:\path\to\reports\phase11a-real-acceptance.json
```

The report contains generation/document/chunk/packet counts, plan and source
fingerprints, bounded diagnostics, and zero `network_attempts`, `llm_calls`,
and `mt5_calls`.  It does not contain source text or model output.  The report
path must be outside the Phase 7 root, and a changed fingerprint returns
`SOURCE_MUTATED` rather than being ignored.  If no teacher is configured, the
generation status is `DISTILLATION_TEACHER_NOT_CONFIGURED`; the deterministic
fixture suite is the positive end-to-end generation proof.

## Artifacts and recovery

Each successful factory run creates a new directory below the chosen output
root:

```text
generation-<content-hash>/
  manifest.json
  examples.jsonl
  excluded.jsonl
  train.jsonl
  validation.jsonl
  test.jsonl
  source_index.json
```

Publication is staged and create-only.  Existing generations are never
overwritten.  A failed packet becomes a bounded `excluded.jsonl` record, and
an interrupted staging directory is left for diagnosis; rerunning with the
same destination cannot corrupt or replace a published generation.  Validate
the generation before handing its train/validation rows to Phase 11.  The
knowledge test split is reserved for the later evaluation phase and is not
exposed by `KnowledgeDatasetBinding`.

## Lesson and safety policy

The closed lesson types are definitions, explanations, mechanisms,
comparisons, scenarios, assumptions/limitations, equation interpretations,
failure modes, and multi-source syntheses.  Every factual claim has exact
Phase 7 document/hash/page/section/chunk provenance.  Unsupported claims,
missing refs, schema errors, excessive verbatim overlap, hidden reasoning,
future outcomes, and fabricated BUY/SELL/HOLD targets are explicit
exclusions, never heuristic repairs.

## Later handoff

Phase 11A publishes only the `BOOK_KNOWLEDGE` stream.  A later Phase 11
curriculum may consume knowledge train/validation first and a separate Phase
10 experience stream second.  An empty Phase 10 stream remains
`EMPTY_ELIGIBLE_SET`.  Phase 7 RAG remains a runtime knowledge source; this
phase does not inject lessons into any TradingAgents graph.  Phase 12 owns
held-out knowledge evaluation and candidate comparison.
