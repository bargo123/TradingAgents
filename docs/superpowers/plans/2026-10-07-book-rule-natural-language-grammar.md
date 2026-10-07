# Source-grounded Natural-language Book Rules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recognize explicit natural-language book rules offline, with recomputable evidence proofs and honest reporting of unsupported execution requirements.

**Architecture:** Add a versioned normalization envelope alongside the unchanged canonical path. One resolver verifies source identity and reconstructs semantics; extraction, assembly, the research registry, attribution and artifact reload all use that resolver. Only existing deterministic primitives can become replay eligible.

**Tech Stack:** Existing Python, dataclasses, Decimal, Pydantic, pytest, read-only SQLite and native loopback Ollama; no new dependency.

**Spec:** `docs/superpowers/specs/2026-10-07-book-rule-natural-language-grammar-design.md` (commit `46fd7fe`).

## Global Constraints

- No model enters the tick path.
- No new book candidate is automatically promoted into the DEMO runtime.
- No changes to incumbent strategies, risk, concurrency, supervisor lifecycle or broker state.
- Same-chunk context only: at most eight spans, each at most 600 characters; no guessed PDF repair.
- Original quotes, offsets, generation and source hashes remain immutable.
- Missing values and unavailable primitives remain non-executable; no defaults or pip/point conversions.
- Required stages remain confirmation, entry, exit, expected move, horizon, invalidation, profit protection and stop behavior.
- Frozen indexes, source databases and historical artifacts are read-only.
- Synthetic parser fixtures never count as genuine corpus evidence or replay promotion.
- Implement only after human plan review and execution-method selection.

## Review Focus

1. Overlapping or reordered context spans must not change field attribution (Task 2).
2. Negation, criticism and hypothetical examples must not become instructions merely because the model selected them (Task 3).
3. Serialized field/proof tampering must fail again at artifact reload, not just initial extraction (Task 6).
4. A valid stop distance must not be mapped into an unrelated momentum threshold (Tasks 3 and 5).
5. A real quote with unavailable feature definitions must remain non-executable across every consumer (Tasks 4–7).

## File Map and Shared Interfaces

New files under `tradingagents/self_enhancement/`:

- `book_normalization_models.py`: strict V2 evidence/proof/envelope contracts and deterministic identities.
- `book_natural_language.py`: reviewed patterns and the sole source/meaning resolver.
- `book_v2_pipeline.py`: assembly, completeness and suitability for V2 envelopes.
- `book_v2_registry.py`: research-only primitive binding and attribution; no live import path.

Modify existing extraction/drafter, Phase 14C runner/models and CLI for explicit V2 dispatch. Do not rewrite the legacy parser or loosen `validate_strategy_draft`, `BookStrategyRegistry.create`, or legacy artifact loading.

Contracts to define in Task 2:

```python
EvidenceBundle(spans: tuple[EvidenceSpan, ...])
FieldProof(field: str, evidence_index: int, start_offset: int, end_offset: int)
NormalizedRuleV2(rule_id: str, stage: RuleStage, family: str,
    operation: str, operands: tuple[str, ...], direction: RuleDirection,
    operator: RuleOperator, value: Decimal | None, unit: str | None,
    condition: str | None, horizon_seconds: int | None,
    evidence: EvidenceBundle, field_proofs: tuple[FieldProof, ...],
    pattern_id: str, schema_version: str, grammar_version: str,
    feature_contract_version: str, semantic_fingerprint: str)
NormalizationResult(status: NormalizationStatus,
    rule: NormalizedRuleV2 | None, reason_codes: tuple[str, ...])
StrategyEnvelopeV2(strategy_id: str, family: str,
    rules: tuple[NormalizedRuleV2, ...], schema_version: str)
CandidateValidationV2(envelope: StrategyEnvelopeV2,
    status: NormalizationStatus, reason_codes: tuple[str, ...],
    missing_stages: tuple[RuleStage, ...])
```

`NormalizationStatus` has exactly REJECTED, SUPPORTED_NONEXECUTABLE and EXECUTABLE_ELIGIBLE. Serialization uses Decimal strings, strict keys and finite values; unknown versions fail closed. Context offsets refer to original chunk text, not normalized token positions.

Resolver interface (Task 3):

```python
normalize_rule(bundle: EvidenceBundle, *, stage: RuleStage, family: str,
    source_lookup: Callable[[EvidenceSpan], str]) -> NormalizationResult
verify_rule(rule: NormalizedRuleV2, *,
    source_lookup: Callable[[EvidenceSpan], str]) -> NormalizationResult
```

`source_lookup` must return verified pinned chunk text or raise a provenance error; file/catalog lookup is injected, never performed by model output. `verify_rule` recomputes rather than trusting claimed fields or hashes.

### Task 1: Genuine source-case contract

**Files:** Create `tests/fixtures/book_natural_language_source_cases.json`; create `tests/test_book_natural_language_sources.py`.

**Interfaces:** Produce a strict case list containing case ID, generation, document/chunk/source hash, offsets, exact spans, proposed stage/family, expected operation and expected support/execution blockers.

- [ ] Read the spec and all existing parser consumers, including runner artifact reload and candidate evaluation. Record the actual call sites in the implementation report.
- [ ] Select genuine cases from pinned `C:/p7fast/catalog.sqlite3` using SQLite URI `mode=ro` and `PRAGMA query_only=ON`. Use generation `gen_607de64268a04a6ab09ffa1e160fc280`; obtain identities from table columns, not guessed filenames.
- [ ] Cover literal numeric comparison, numeric target/stop and holding-time language where genuinely present. Include the observed “lower than 34 ticks, but never more” passage as a context-dependent counterexample, and the bar-high stop-entry passage as unavailable chart-feature/instrument evidence. Exact source addresses must be captured before adding cases.
- [ ] Write `test_source_cases_have_exact_offsets_and_hashes`: assert each case's quote equals pinned text slice and all case IDs are unique. Add `test_source_cases_separate_support_from_execution` asserting chart-feature cases cannot be EXECUTABLE_ELIGIBLE.
- [ ] Run `python -m pytest tests/test_book_natural_language_sources.py -q`; first confirm expected missing loader/contract failures, then implement the fixture loader in that test module and obtain passing tests. Mark catalog-dependent checks with explicit skip when the pinned catalog is absent; qualification requires no such skip.
- [ ] Do not enable a pattern without a genuine, unambiguous positive source case. If a proposed family has no such case, record PATTERN_NO_GENUINE_CASE and leave it disabled; do not manufacture an acceptance case.
- [ ] Commit only these fixture/test files: `test: pin genuine natural-language rule evidence`.

### Task 2: V2 proof and envelope contracts

**Files:** Create `book_normalization_models.py` and `tests/test_book_normalization_models.py`.

**Interfaces:** Produce the contracts listed above, `to_dict()` / strict `from_dict()` for each, and `normalization_identity(rule: NormalizedRuleV2) -> str`.

- [ ] Write tests asserting bundle sizes 1 and 8 pass; 0 and 9 fail; 600 characters pass and 601 fail. Reject different chunk/document/generation/source identities, duplicate/overlapping spans, reversed source order and out-of-range field proofs.
- [ ] Add round-trip tests preserving Decimal `"0.10"`, original quote characters and offsets. Assert unknown keys, bool-as-int, NaN/Infinity and unsupported schema versions reject.
- [ ] Run `python -m pytest tests/test_book_normalization_models.py -q`; verify failures before implementation.
- [ ] Implement immutable contracts. Version constants: `book-normalization-v2`, `book-natural-language-v1`, `book-feature-binding-v1`. Fingerprints include every semantic field, pattern, evidence address and version using canonical JSON; identity never replaces source verification.
- [ ] Run the contract tests plus `tests/test_phase14b_book_pipeline.py`; require passes and unchanged legacy serialization.
- [ ] Commit: `feat: add strict offline book normalization contracts`.

### Task 3: Finite natural-language resolver

**Files:** Create `book_natural_language.py` and `tests/test_book_natural_language.py`.

**Interfaces:** Produce `normalize_rule` and `verify_rule` as specified; a closed pattern registry with IDs `numeric-comparison-v1`, `numeric-distance-v1`, `holding-limit-v1`, enabled only by Task 1's genuine cases.

- [ ] Write source-case tests pinning exact parsed operation, Decimal, direction/operator, original units and field spans. Assert unsupported lookback, price basis, instrument, chart bars or missing anchors remain non-executable. Numeric distance uses operation `price_distance`, never a momentum feature.
- [ ] Add test-only minimal pairs for changed direction, comparator, number, units, stage, negation, criticism, hypothetical language and damaged glyphs. Explicit ambiguous language must reject, not infer missing operands.
- [ ] Run `python -m pytest tests/test_book_natural_language.py -q`; verify failures.
- [ ] Implement source verification followed by a separate offset-mapped token view. Accept only whole reviewed constructions with checked same-chunk definitions. Reject unconsumed semantic clauses rather than parse an attractive substring.
- [ ] Implement closed feature bindings only where definition, lookback, price basis and unit equal the existing causal contract. Preserve source condition text; unavailable condition bindings cannot become generic `after entry` or confirmation defaults.
- [ ] Implement `verify_rule` by rerunning normalization and comparing every field/proof/version/fingerprint. A genuine supported but unmappable rule returns SUPPORTED_NONEXECUTABLE; invalid provenance or contradictory meaning returns REJECTED.
- [ ] Run resolver, contract and existing atomic-extraction tests; require passes.
- [ ] Commit: `feat: normalize reviewed source rules with field proofs`.

### Task 4: Model-assisted V2 extraction and cache separation

**Files:** Modify `book_atomic_extraction.py`, `book_drafter.py`; create `tests/test_book_v2_extraction.py`.

**Interfaces:** Add `AtomicStrategyExtractor.extract_actionable_v2(sentences: Sequence[EvidenceSentence]) -> tuple[NormalizationResult, ...]`. Keep existing public V1 methods and result contracts unchanged.

- [ ] Write mocked-teacher tests: the response may contain only proposed stage/family and 1–8 enumerated same-chunk evidence indexes. Arbitrary numeric fields, rewritten quotes, unknown indexes and cross-chunk bundles reject.
- [ ] Test that the local model is actually called on a cache miss; this is not a manual-only extraction path. Test schema, prompt, grammar, feature-contract and bundle changes invalidate V2 cache identity.
- [ ] Run `python -m pytest tests/test_book_v2_extraction.py -q`; verify failures.
- [ ] Add strict V2 response models and bounded bundle selection, then call Task 3's resolver. Expose typed unresolved reasons rather than dropping rejected selections. Drafter dispatches V2 proposals into the V2 assembly path without legacy claim conversion.
- [ ] Preserve native loopback-only requests, temperature 0, context/output/timeout bounds and one split-retry level. Store only validated selection/results and safe telemetry; no reasoning/completion persistence.
- [ ] Run V2 extraction and existing drafter/atomic tests. Assert legacy cache records cannot be loaded as V2 normalization results.
- [ ] Commit: `feat: propose source-addressed V2 book rules offline`.

### Task 5: Consistent assembly, research binding and attribution

**Files:** Create `book_v2_pipeline.py`, `book_v2_registry.py`, `tests/test_book_v2_pipeline.py`, `tests/test_book_v2_registry.py`; modify `phase14c_assembly.py` and `phase14c_mapping.py` with explicit V2 dispatch entry points.

**Interfaces:** Produce `validate_envelope(envelope: StrategyEnvelopeV2, *, source_lookup: Callable[[EvidenceSpan], str]) -> CandidateValidationV2`; `assemble_v2(results: Sequence[NormalizationResult], *, source_lookup: Callable[[EvidenceSpan], str]) -> tuple[CandidateValidationV2, ...]`; `create_research_strategy(candidate: CandidateValidationV2, *, source_lookup: Callable[[EvidenceSpan], str]) -> DeterministicBookStrategy`; `map_v2(candidate: CandidateValidationV2, *, snapshot: CurrentStrategyContractSnapshot, source_lookup: Callable[[EvidenceSpan], str]) -> tuple[CurrentStrategyMapping, ...]` using existing Phase 14C mapping contracts and their drift check.

- [ ] Write completeness tests requiring all eight stages; conflicting directions, incompatible source definitions and unsupported stages never produce eligibility. Validate independent-source constraints using existing assembly policy, not a new relaxed policy.
- [ ] Write binding tests asserting the current `_CONDITIONS`, family permissions, numeric domains and exit dependencies are preserved. Unavailable features, pip distances and conditions reject registry construction even with a serialized eligible status.
- [ ] Test V2 mapping uses verified signatures; literal “momentum” or “range” mentions alone cannot attribute an incumbent strategy. Test every consumer agrees on unsupported feature status.
- [ ] Run both new test modules; verify failures.
- [ ] Implement explicit V2 flow through the resolver. Reuse existing deterministic primitives only after proof/binding checks; do not fabricate canonical source sentences or route V2 evidence through `BookStrategyRegistry.create`.
- [ ] Run V2 pipeline/registry tests and legacy assembly, mapping, suitability and strategy tests; require passes.
- [ ] Commit: `feat: validate and bind V2 research candidates consistently`.

### Task 6: Versioned discovery artifacts and replay reload

**Files:** Modify `phase14c_models.py`, `phase14c_runner.py`, `cli/phase14c.py`; create `tests/test_book_v2_artifacts.py`; extend `tests/test_phase14c_cli.py`.

**Interfaces:** Add keyword `normalization_schema: str = "canonical-v1"` to discovery dispatch and CLI `discover --normalization-schema {canonical-v1,book-normalization-v2}`. Produce `load_v2_discovery_artifacts(artifact_root: Path, *, source_lookup: Callable[[EvidenceSpan], str]) -> tuple[CandidateValidationV2, ...]`. V2 artifact metadata identifies all normalization versions; legacy loader does not accept it.

- [ ] Write tests for fresh-root enforcement, source overlap rejection, unknown version, mismatched manifest identity, hash changes and field/proof tampering. Tampering must reject even if an attacker recomputes artifact hashes.
- [ ] Test old artifacts remain byte-unchanged and canonical default behavior remains unchanged. Status output contains separate supported/non-executable/eligible counts plus typed reasons, without raw teacher output.
- [ ] Run artifact and CLI tests; verify failures.
- [ ] Implement separate V2 artifact serialization, dispatch, reload and manifest/cache identities. Do not resume a V1 run as V2. Revalidate every rule on reload and immediately before research strategy construction.
- [ ] Extend candidate evaluation to dispatch validated V2 candidates through `create_research_strategy`; retain existing chronological replay, source qualification, cost and promotion gates. No eligible candidate means NOT_RUN with exact reasons, not successful replay.
- [ ] Run V2 artifact tests and legacy resume, CLI, integration and replay suites; require passes.
- [ ] Commit: `feat: persist and reload proof-verified V2 discovery artifacts`.

### Task 7: Offline end-to-end qualification

**Files:** Create `tests/test_book_v2_integration.py`; create `docs/reports/book-natural-language-qualification-20261007.md`.

**Interfaces:** Consume Tasks 1–6; produce a qualification report and a distinct research artifact root. This task introduces no new product API.

- [ ] Write integration tests with broker/supervisor constructors patched to raise if invoked. A compact mocked-teacher response must reach normalization and assembly; incomplete genuine rules must stay non-executable through reload and mapping.
- [ ] Run `python -m pytest tests/test_book_v2_integration.py -q`; verify failures before adding the integration wiring needed to pass.
- [ ] Run the regression command below; require zero failures. Report catalog-dependent skips explicitly.

  ```powershell
  $bookPlanTests = Get-ChildItem -LiteralPath tests -File | Where-Object {
      $_.Name -match '^test_(book_v2.*|book_normalization_models|book_natural_language.*|phase14b_book.*|phase14c.*)\.py$'
  } | ForEach-Object { $_.FullName }
  python -m pytest @bookPlanTests -q
  ```
- [ ] Run one genuine bounded V2 extraction batch of at most eight addressed source spans from Task 1 with local `qwen3.5:2b`, loopback `http://127.0.0.1:11434`, context 8192, output 512, timeout 300 seconds, temperature 0 and existing bounded retries. Verify installed model digest against the run identity first; no remote fallback or model download.
- [ ] Write artifacts under a new root such as `C:/phase14c/runs/run-20261007-natural-language-v2`; fail if it exists rather than overwrite. Record selected addresses, safe call telemetry, all gate counts/reasons and before/after pinned-source fingerprints.
- [ ] If and only if a complete eligible candidate emerges, use frozen qualification sources with existing costed chronological replay. Unknown commission/latency or quote-quality gaps remain explicit blockers. Never place a broker order as qualification.
- [ ] Record residual pattern, retrieval and primitive coverage, no profit guarantee, and proof that live configuration/strategies were untouched (`git diff` scoped to runtime files). A grammar success is not full corpus discovery or profitable HFT proof.
- [ ] Commit tests and the report only: `test: qualify source-grounded V2 discovery offline`.

## Plan Self-review and Handoff

All spec requirements map to Tasks 1–7: source/proofs 1–3; model authority/cache 4; six consumer paths 4–6; versioned artifact reload/replay 6; genuine offline verification and unchanged runtime 7. New signatures are defined above or in their owning task. Legacy behavior has regression coverage at every integration boundary.

No product code or model run is authorized by this plan document alone. After human review, choose subagent-driven or native execution. Native is recommended here because the tasks share strict proof contracts and one resolver; implement sequentially and use a fresh whole-branch reviewer after tests pass. Deployment and multi-position changes remain separate work.
