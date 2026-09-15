# Phase 11A — HFT/FX Book Knowledge Distillation Design

**Status:** approved implementation design
**Date:** 2026-09-15
**Baseline:** `fd80d4cc67a03558d30bc4837253de2932d24476`
**Branch:** `codex/phase-11a-book-distillation`

## 1. Goals and non-goals

Phase 11A turns the validated, local Phase 7 knowledge view into a grounded,
copyright-conscious lesson dataset that a later Phase 11 cloud training run
can consume. It teaches concepts and reasoning; it does not create trading
signals or outcomes.

It does not run cloud training, choose a model, change Phase 7 parsing or
indexes, change Phase 10 eligibility, alter Phase 9 decisions, call MT5,
execute orders, use experience memory as knowledge, persist hidden reasoning,
or begin Phase 12.

## 2. Alternatives and recommendation

1. **Export canonical Phase 7 chunks directly.** This is simple, but creates
   raw-book SFT data, teaches reproduction rather than reasoning, and loses
   grounding/quality decisions.
2. **A local deterministic factory with an injectable structured teacher.** A
   read-only Phase 7 adapter creates bounded source packets; a teacher proposes
   lessons; independent deterministic validation enforces grounding, safety,
   deduplication, and grouped splits; immutable generations publish accepted
   and excluded records. This is inspectable and works offline with a fake
   teacher, while leaving a stronger teacher configurable for production.
3. **A retrieval-time teacher service.** This reduces local storage, but makes
   training-data reproducibility, offline testing, and auditability dependent
   on a live service.

Option 2 is selected. It preserves Phase 7 authority, keeps teacher choice
model-agnostic, and makes every acceptance/rejection auditable without
coupling the existing TradingAgents or fine-tuning paths to an LLM.

## 3. Architecture and boundaries

```text
Phase 7 catalog + active generation (read only)
        ↓
Phase 7 source adapter
        ↓
deterministic bounded source packets / plan
        ↓
injectable structured teacher (optional)
        ↓
schema + grounding + safety + duplicate validators
        ↓
immutable Phase 11A generation
        ↓
explicit knowledge curriculum source for Phase 11
```

The implementation lives in `tradingagents.distillation`. It never writes
inside the Phase 7 artifact root or source books. The existing
`tradingagents.finetuning` package receives only a separate, explicit
knowledge-source adapter; its Phase 10 decision adapter and trainer behavior
remain unchanged.

## 4. Package/module boundaries

* `tradingagents/distillation/models.py` — frozen versioned lesson, source
  reference, packet, plan, manifest, split, and report contracts.
* `errors.py` — typed bounded failures and closed exclusion reasons.
* `phase7.py` — read-only catalog/generation/chunk adapter and provenance
  extraction; no parser, ingestor, or source-root writes.
* `planning.py` — deterministic topic/block selection and bounded packets.
* `teacher.py` — teacher protocol, configuration, structured result seam, and
  optional configured-provider boundary; no credentials in artifacts.
* `grounding.py` — claim/reference validation, unsupported-claim and
  contradiction checks, and no-action/no-future policy.
* `quality.py` — schema, length, copyright-overlap, and exclusion policy.
* `dedup.py` — canonical IDs and exact/near-duplicate guards.
* `splits.py` — deterministic source-group assignments and leakage checks.
* `writer.py` — atomic immutable generation publication, hashes, and validation.
* `factory.py` — plan/distill/validate orchestration with counters and safety
  telemetry.
* `curriculum.py` — explicit knowledge-stage binding for Phase 11; it never
  merges knowledge rows with Phase 10 decisions or changes eligibility.
* `cli.py` — standalone `knowledge-distill plan|distill|validate|inspect`.

## 5. Versioned contracts

The V1 identifiers are:

* `phase11a-knowledge-example.v1`
* `phase11a-distillation-policy.v1`
* `phase11a-grounding-policy.v1`
* `phase11a-split-policy.v1`
* `phase11a-manifest.v1`

Every persisted record carries the relevant version. Unknown versions fail
closed. A `SourceRef` carries document ID, filename, source hash, page/range,
chapter/section, content type, chunk/block ID, and Phase 7 generation ID.
`GroundingClaim` maps one factual claim to one or more exact source refs.
`KnowledgeExample` carries a closed lesson type, topic, difficulty, concise
system/user/assistant lesson text, source refs, claims, policies, quality
status, and source fingerprints. No random IDs are used: IDs hash canonical
source refs, normalized instructions, lesson type, and policy/schema versions.

## 6. Source authority and read-only behavior

`Phase7KnowledgeSource` opens an existing `catalog.sqlite3`, verifies an
active validated matched generation, and reads current documents/chunks and
their stored provenance. It does not instantiate Docling/FastEmbed, parse
PDF/EPUB, call the query writer, or modify any Phase 7 file. Missing or
incomplete artifacts produce typed errors. The plan and manifest record the
Phase 7 generation ID and immutable catalog/projection fingerprints.

## 7. Source packets and planning

The planner groups nearby blocks by concept and durable source boundary:
document, chapter/section, equation plus definitions, or table plus caption
and explanation. It selects deterministically by normalized topic-term score,
reading order, and chunk ID; ties are lexical. Packets have explicit maximum
blocks, characters, and estimated tokens. Whole books are never sent to a
teacher. A plan records only bounded packet evidence, refs, policy versions,
and a request fingerprint; it never records teacher prompts or hidden output.

## 8. Teacher and two-stage generation

`TeacherConfig` records provider/model/version, temperature, output limit,
schema/policy versions, and timeout. `Teacher` is an injectable protocol, so
repository tests use a deterministic fake and do not call Ollama or cloud
providers. A configured production teacher may produce a structured candidate
with claims and refs; a second optional judge can reject it. The factory
always runs deterministic validation independently of either teacher. Teacher
responses are not claimed byte-deterministic; the request fingerprint and
actual generation ID are distinct.

## 9. Grounding, copyright, and safety

Every factual claim must reference a block in the packet and every ref must be
present in Phase 7. Missing refs, unsupported claims, contradictions, and
incomplete targets are exclusions, never repaired heuristically. Lessons use
bounded paraphrase; a deterministic normalized-token overlap guard rejects
excessive verbatim copying. Equations, variables, definitions, tables, and
technical terms may be preserved when needed for accuracy, with their
content-type metadata intact.

Scenario lessons discuss mechanisms and risks without assigning BUY/SELL/HOLD
or fabricating an entry, exit, PnL, reward, or future outcome. Recursive
sensitive-field checks reject prompts, completions, reasoning/CoT, secrets,
and future/outcome fields. Hidden teacher/judge reasoning is never persisted.

## 10. Lesson types and difficulty

The closed V1 lesson types are `DEFINITION`, `CONCEPT_EXPLANATION`,
`MECHANISM`, `COMPARISON`, `SCENARIO_APPLICATION`,
`ASSUMPTION_AND_LIMITATION`, `EQUATION_INTERPRETATION`, and `FAILURE_MODE`,
plus `MULTI_SOURCE_SYNTHESIS`. Difficulty is `FOUNDATIONAL`, `INTERMEDIATE`,
or `ADVANCED`, chosen from validated source/lesson criteria or a constrained
teacher field. Topic and lesson distributions are manifest metadata.

## 11. Quality and exclusion policy

The closed exclusion set includes `SOURCE_PROVENANCE_INCOMPLETE`,
`GROUNDING_FAILED`, `UNSUPPORTED_CLAIM`, `CONTRADICTION`,
`TARGET_INCOMPLETE`, `SCHEMA_INVALID`, `DUPLICATE`, `NEAR_DUPLICATE`,
`SOURCE_PACKET_TOO_LARGE`, `LESSON_TOO_LONG`, `VERBATIM_OVERLAP_EXCESSIVE`,
`UNSAFE_FUTURE_OUTCOME_INFERENCE`, `TEACHER_FAILED`, and `JUDGE_REJECTED`.
Every excluded candidate is written to `excluded.jsonl` with bounded reason
codes and diagnostics. Failed candidates are not silently dropped.

## 12. Grouped split policy

Accepted examples are grouped by the closest durable Phase 7 boundary,
preferably `document_id + chapter + section path`. A stable hash/order assigns
whole groups to approximately 70/15/15 train/validation/test buckets. No
related group crosses splits. With too few groups for a meaningful partition,
the manifest reports `INSUFFICIENT_DATA` instead of fabricating balance. The
test split is held out for Phase 12 knowledge evaluation and is never read by
the Phase 11 trainer.

## 13. Immutable storage and manifest

Each run publishes create-only under `<phase11a-root>/<generation-id>/`:

```text
manifest.json
examples.jsonl
excluded.jsonl
train.jsonl
validation.jsonl
test.jsonl
source_index.json
```

Files are canonical JSONL with stable ordering and hashes. The manifest records
all policy/schema versions, Phase 7 generation/fingerprints, teacher/judge
identity, counts/distributions, split status/counts, exclusion reasons,
source coverage, file hashes, request fingerprint, generation identity,
`llm_calls`, `network_attempts`, and `mt5_calls`. Publication stages in a
temporary directory and atomically creates a destination; existing generations
are never overwritten. Validation checks every hash and provenance binding.

## 14. Curriculum handoff

`curriculum.py` exposes a typed `KnowledgeDatasetBinding` and stage manifest
that loads only Phase 11A train/validation records. Knowledge examples retain
`source_type=BOOK_KNOWLEDGE`; Phase 10 records retain their existing
`source_type=TRADING_EXPERIENCE`. A caller must choose an explicit stage or
ordered curriculum. No book example is coerced into `ForexPortfolioDecision`,
and an empty Phase 10 generation remains `EMPTY_ELIGIBLE_SET`.

## 15. CLI and failure behavior

The standalone `knowledge-distill` command has:

* `plan --phase7-root PATH --output PATH` — read-only source scan/packet plan;
* `distill --plan PATH --output-root PATH` — configured teacher generation;
* `validate --generation PATH` — no teacher calls, hash/provenance/safety audit;
* `inspect --generation PATH` — bounded counts and distributions only.

If no teacher is configured, `distill` returns
`DISTILLATION_TEACHER_NOT_CONFIGURED` without fabricating lessons. Typed
statuses include `SOURCE_INVALID`, `PLAN_INVALID`, `PARSE_FAILED`,
`TEACHER_FAILED`, `VALIDATION_FAILED`, `GENERATION_EXISTS`, and
`INSUFFICIENT_DATA`. One bad packet does not corrupt prior generations.

## 16. Reproducibility and privacy

The request fingerprint includes Phase 7 generation/fingerprints, packet
selection settings, teacher identity/parameters, and all policy versions.
Teacher output may vary, so the generation ID is based on accepted content;
canonicalization, splitting, and writing are deterministic once outputs are
accepted. No credentials, prompts, completions, reasoning, or source-root
mutations are stored. The core path is offline and has zero MT5/network calls;
an optional teacher boundary is explicit and opt-in.

## 17. Testing and acceptance

Fixture tests use tiny structured blocks and a deterministic fake teacher. They
cover read-only source access, deterministic plans/IDs, grounding acceptance
and rejection, schema and sensitive-field failures, no-action/future leakage,
copyright overlap, duplicate protection, grouped split leakage, test
isolation, manifest/file hashes, interrupted publication, and Phase 11
curriculum separation. HFT-oriented fixtures cover order-flow imbalance,
liquidity, market impact, inventory risk, volatility, equations, and tables,
with lesson types across definition, mechanism, scenario, comparison, equation,
and failure mode. No CI test parses real books, downloads models, calls Ollama,
calls MT5, or requires CUDA.

The bounded real acceptance opens the existing Phase 7 artifact read-only,
reports discovered documents/chunks/planned packets and unchanged source
fingerprints, and runs generation only when a teacher is explicitly
configured. An unconfigured teacher is reported as
`DISTILLATION_TEACHER_NOT_CONFIGURED`, not treated as a fabricated success.

## 18. Performance and future seams

The default planner uses one process, bounded packets, and streaming catalog
reads suitable for Windows 10, Ryzen 5 7430U, 16 GB RAM, and integrated
graphics. No CUDA or cloud embedding is assumed. Teacher, judge, near-duplicate
detector, packet scorer, split policy, and curriculum stage are interfaces for
future stronger implementations. Phase 7 RAG remains in the eventual runtime;
Phase 11A does not inject lessons into TradingAgents decisions.

## 19. Explicit Phase 8/12 handoff boundary

Phase 11A consumes published Phase 7 knowledge only. Phase 10 verified
experience remains a separate source for later curriculum stages. Phase 12
owns held-out knowledge evaluation, candidate comparison, model promotion, and
any future runtime integration. No Phase 8 memory, MT5, execution, strategy
generation, or trading outcome is introduced here.
