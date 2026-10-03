# Phase 14B Book-Derived Strategy and Market Autonomy Implementation Plan

> **For agentic workers:** This plan is executed inline by the current agent as explicitly requested. Work task-by-task in RED→GREEN order and keep a progress ledger under the plan's `.superpowers/sdd` workspace.

**Goal:** Connect frozen Phase 7 knowledge to strictly provenance-validated deterministic HFT strategy research, and keep the existing single-owner DEMO runtime healthy while market closure is calendar-confirmed and reopen is revalidated.

**Architecture:** Use a loopback-only local Ollama structured draft adapter, an exact-span/controlled-grammar evidence validator, a finite deterministic strategy registry, and the current Phase 14 replay and promotion gate. Separately, add a broker-specific calendar state machine to the existing DEMO HFT owner; closure idles in place and fresh ticks must pass the full safety sequence before incumbent processing resumes. Book candidates can only replay or be observed by a gateway-free shadow observer.

**Tech Stack:** Python 3.11+, dataclasses, standard-library `urllib`/`zoneinfo`/SQLite, existing Phase 7 `KnowledgeQueryService`, Ollama native `/api/chat` JSON Schema, existing HFT feature/strategy/fill/risk primitives, pytest/Ruff.

**Spec:** `docs/superpowers/specs/2026-10-03-phase-14b-book-derived-autonomy-design.md`

## Global Constraints

- Phase 7 root `C:\p7fast`, generation `gen_607de64268a04a6ab09ffa1e160fc280`; read-only, no ingestion/rebuild/re-embedding, and verify fingerprint before/after.
- Phase 11A artifacts are not read or modified; no training/fine-tuning.
- Phase 14A HFT/DEMO SQLite inputs are read-only; Phase 14B writes only to a dedicated artifact root.
- Preserve all 118 verified executions and exclude the 3 quarantined executions from validated experience.
- Use the canonical 21,131 ticks / 18 segments without crossing segment boundaries or shuffling.
- Keep commission `UNKNOWN`; never report it as zero/known.
- Keep the existing `CandidatePromotionGate`, minimum sample requirements, cost cases, incumbent and risk limits unchanged.
- Local model is explicit offline drafting only at loopback; no cloud/network fallback, model download, prompt/completion/reasoning persistence, or tick-path model/RAG access.
- Missing/invalid/ambiguous calendar means `MARKET_UNKNOWN`; no generic weekend schedule; calendar time alone never authorizes trading.
- Market closure/reopen changes preserve the single owner, heartbeat, lease and existing incumbent DEMO gateway checks.
- New book candidates have no DEMO gateway and maximum automatic status `SHADOW_CHALLENGER`.
- Do not force trades, weaken thresholds/gates, alter Phase 7 sources, or permit real-money execution.
- Preserve the pre-existing untracked `watch_dashboard.py`; never stage it.

## Review Focus

- A valid-looking quote cached across a closed session must not be mistaken for resumed activity: test repeated raw tick identity while calendar is closed and require no opening transition.
- A valid calendar with missing/expired broker connectivity must produce data error/unknown, not healthy closure: test terminal/account failures.
- A fall-back DST ambiguous broker timestamp must not be normalized by guessing: test `MARKET_UNKNOWN`/clock failure.
- A model may cite a real but irrelevant quote: test that the finite source-rule grammar rejects mismatched operator, side, value and condition even with an exact span.
- A shadow challenger must not mutate the incumbent or access the DEMO gateway: test with a gateway that raises on any attribute access and verify the incumbent path is unchanged.

---

### Task 1: Broker Calendar and Market-State Contracts

**Files:**
- Create: `tradingagents/forex/hft/market_calendar.py`
- Create: `tradingagents/forex/hft/market_lifecycle.py`
- Test: `tests/test_phase14b_market_calendar.py`

**Interfaces:**
- `BrokerSessionCalendar.from_json(path) -> BrokerSessionCalendar`
- `BrokerSessionCalendar.status_at(at_utc, *, broker_server, symbol) -> CalendarStatus`
- `MarketLifecycleController.observe(calendar_status, *, terminal_healthy, account_healthy, tick_fresh, error_reason=None) -> MarketObservation`
- Calendar status is `OPEN`, `CLOSED`, or `UNKNOWN`; lifecycle states are the five spec states. Error reason is separately one of stale/disconnected/clock/mismatch/data.

- [ ] **Step 1: Write failing tests** for exact broker/symbol matching, valid coverage, weekly sessions, full-date holidays, date overrides/early close, timezone/DST conversion, fall-back ambiguity, malformed/overlapping sessions, and absent/expired calendars returning UNKNOWN.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_market_calendar.py`; expected failures name the missing types/loader/transition behavior.
- [ ] **Step 3: Implement immutable calendar parsing and state transitions** using `zoneinfo`; reject unsupported/ambiguous records rather than guessing.
- [ ] **Step 4: Verify GREEN** with the focused test file.
- [ ] **Step 5: Commit** `feat: model broker market sessions for phase 14b`.

### Task 2: Read-Only Tick Probe and Same-Owner Idle/Reopen Lifecycle

**Files:**
- Modify: `tradingagents/dataflows/mt5/provider.py`
- Modify: `tradingagents/dataflows/mt5/models.py` and `errors.py` only if the typed observation requires it
- Modify: `tradingagents/forex/hft/demo_runtime.py`
- Modify: `tradingagents/forex/supervisor.py`
- Modify: `cli/forex_supervisor.py`
- Test: `tests/test_phase14b_market_lifecycle.py`
- Test: `tests/test_mt5_provider.py`
- Test: `tests/test_forex_supervisor.py`

**Interfaces:**
- Add `MT5Provider.probe_tick(symbol) -> Mt5TickProbe`: connectivity/symbol validation and raw timestamp identity only; it never returns a tradable quote or bypasses clock validation for a decision.
- Add `DemoRuntimeConfig.market_calendar_path: Path | None` and `market_closed_poll_interval_seconds: float` (bounded positive default).
- Add `DemoHftRuntime.market_status` and `.market_reason`; `run_once()` returns an idle status without raising for a healthy, calendar-confirmed closure.
- Plumb an explicit `--market-session-calendar` option only through `forex_supervisor run`; no stock CLI changes.

- [ ] **Step 1: Write failing lifecycle tests**: closed without needing a fresh tick; stale cached probe does not count as resumed activity; open calendar plus no fresh quote is not closed; unhealthy terminal/account is not closed; closure keeps owner/lease and no order/reconnect call occurs; fresh tick causes `MARKET_OPENING_VALIDATION` then validates clock→DEMO→reconcile→rate circuit→risk/state before `MARKET_OPEN`; every validation failure remains no-trade.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_market_lifecycle.py tests/test_mt5_provider.py`.
- [ ] **Step 3: Implement the non-trading tick-availability probe and in-place lifecycle**. Catch only typed no-tick/availability cases. Keep existing provider and leases; do not initialize/restart MT5 in the idle loop. The availability probe is never passed into strategy/risk/fill code.
- [ ] **Step 4: Verify GREEN**, then run `python -m pytest -q tests/test_forex_supervisor.py tests/test_phase12d_hft_first.py tests/test_phase12d_hft_first_persistence.py` to prove current supervised DEMO wiring and default strategy behavior remain unchanged.
- [ ] **Step 5: Commit** `feat: idle and resume demo hft around broker sessions`.

### Task 3: StrategySpec and Atomic Rule Evidence Contracts

**Files:**
- Create: `tradingagents/self_enhancement/strategy_specs.py`
- Test: `tests/test_phase14b_strategy_specs.py`

**Interfaces:**
- `StrategyRuleClaim`: closed typed fields for stage, operator, direction, optional value/unit/condition/horizon, and `EvidenceSpan`.
- `EvidenceSpan`: generation/document/chunk/source hash plus exact start/end offsets and quote.
- `StrategySpec`: required provenance/model/schema/hash fields and all rule groups from the spec; missing rules remain `UNSPECIFIED`.
- `RuleValidationResult`: `SUPPORTED`, `UNSUPPORTED`, `INSUFFICIENT_SPECIFICATION`, or `UNAVAILABLE_DATA`, plus safe reason codes.

- [ ] **Step 1: Write failing contract tests** for strict enums/types, deterministic serialization/hash, unknown-field rejection, source-supported vs research-hypothesis separation, and UNSPECIFIED values not being coerced or filled.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_strategy_specs.py`.
- [ ] **Step 3: Implement frozen typed contracts** with canonical serialization and content hash.
- [ ] **Step 4: Verify GREEN** with the focused test file.
- [ ] **Step 5: Commit** `feat: define provenance-bound phase 14b strategy specs`.

### Task 4: Local-Only Structured Strategy Drafter

**Files:**
- Create: `tradingagents/self_enhancement/book_drafter.py`
- Test: `tests/test_phase14b_book_drafter.py`

**Interfaces:**
- `OllamaStrategyDrafter(endpoint, model, *, timeout_seconds, max_output_tokens, transport) -> draft(hits, max_specs) -> DraftResult`.
- Only `http://127.0.0.1`/`http://localhost` Ollama native `/api/chat`; request uses temperature 0, `think: false`, strict JSON Schema, bounded `num_predict`, no streaming, and no retry loop.
- Persist-safe result contains parsed schema, model/config metadata, token/runtime metadata, and digest only; no prompt/completion/reasoning field.

- [ ] **Step 1: Write failing fake-transport tests** for loopback acceptance/non-loopback rejection, strict schema request flags, timeout/output bound, malformed JSON/schema failure, provider failure, and absence of prompt/completion/reasoning in result serialization.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_book_drafter.py`.
- [ ] **Step 3: Implement the loopback-only adapter** with an injected HTTP transport and strict `StrategyDraftBatch` validation.
- [ ] **Step 4: Verify GREEN** and confirm no online provider is imported or used.
- [ ] **Step 5: Commit** `feat: add local offline phase 14b strategy drafter`.

### Task 5: Phase 7 Retrieval, Deterministic Provenance Validator, Suitability, Dedup and Mapping

**Files:**
- Create: `tradingagents/self_enhancement/book_pipeline.py`
- Modify: `tradingagents/self_enhancement/book_factory.py` only to route new validated specs without removing legacy behavior
- Test: `tests/test_phase14b_book_pipeline.py`
- Test: `tests/test_phase14_candidates.py`

**Interfaces:**
- `BookStrategyPipeline(query_service, drafter, *, pinned_generation, pinned_fingerprint).extract(queries, *, max_specs=10) -> StrategyBatchReport`.
- Each draft span is resolved only from returned hits; validator checks pinned generation, catalog document/chunk/hash, exact source substring and offsets, then parses a finite reviewed source-rule grammar and compares operator/direction/value/unit/condition exactly.
- `classify_suitability(spec, available_features) -> SuitabilityResult`; `deduplicate_specs(specs) -> tuple[StrategySpec, ...]` preserves independent document provenance; `map_existing_strategy(strategy_id, hits) -> MappingReport` returns per-component MATCHED/PARTIALLY_MATCHED/NO_DIRECT_MATCH.

- [ ] **Step 1: Write failing tests** where exact valid support passes; wrong chunk/hash/span, unrelated exact quote, wrong operator, wrong side, invented value, condition mismatch, unavailable L2, incomplete exit/horizon, duplicate chunks from one book, and equivalent rules across books are handled conservatively; current strategy components are not given fake citations.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_book_pipeline.py tests/test_phase14_candidates.py`.
- [ ] **Step 3: Implement only finite grammar mappings**. Unknown/ambiguous wording is rejected or left non-executable; no broad semantic similarity acceptance and no research parameter is attributed to a source.
- [ ] **Step 4: Verify GREEN** with focused tests.
- [ ] **Step 5: Commit** `feat: validate book-derived strategy provenance`.

### Task 6: Deterministic Candidate Registry and Complete Entry/Exit Concepts

**Files:**
- Create: `tradingagents/forex/hft/book_strategies.py`
- Modify: `tradingagents/forex/hft/strategies.py` only for a shared protocol/type if needed; default incumbent set/order must remain byte-for-byte behaviorally equivalent.
- Test: `tests/test_phase14b_book_strategies.py`
- Test: `tests/test_phase12d_hft_first.py`

**Interfaces:**
- `BookStrategyRegistry.create(spec: StrategySpec) -> DeterministicStrategy`; only a finite allowlist of supported rule operators is constructible.
- Every deterministic candidate returns the existing `StrategySignal` shape from causal `TickFeatures`; IDs are namespaced and map to an existing explicit regime-family permission rather than bypassing `StrategicRegimeState`.
- Candidate exit contract is validated with the entry spec before construction; incomplete source exit/maximum horizon cannot instantiate.

- [ ] **Step 1: Write failing tests** for allowlisted strategies, rejected unknown operators, exact directional/regime/spread gating, state reset, invalid/incomplete entry/exit coherence, and unchanged default Momentum/Range outputs.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_book_strategies.py tests/test_phase12d_hft_first.py`.
- [ ] **Step 3: Implement registered deterministic primitives only**; no `exec`, import-from-model, or generated Python.
- [ ] **Step 4: Verify GREEN**.
- [ ] **Step 5: Commit** `feat: implement validated book strategy primitives`.

### Task 7: Strategy Replay, Metrics and Existing Promotion Gate

**Files:**
- Modify: `tradingagents/self_enhancement/models.py`
- Modify: `tradingagents/self_enhancement/store.py` to include serialized StrategySpec in candidate payload without schema migration
- Modify: `tradingagents/self_enhancement/replay.py`
- Modify: `tradingagents/self_enhancement/evaluation.py` only for candidate metrics if absent
- Modify: `tradingagents/self_enhancement/orchestrator.py` via a separate book-candidate entrypoint; preserve existing `run_once()` behavior
- Test: `tests/test_phase14b_book_replay.py`
- Test: `tests/test_phase14_promotion.py`

**Interfaces:**
- `CandidateSpec.strategy_spec: Mapping[str, Any] | None = None` remains absent for legacy candidates.
- `ReplayEvaluator.evaluate()` uses existing path for legacy candidates and the registry for validated book candidates.
- `BookCandidateExperiment.run(...)` reads Phase 14A/HFT/DEMO inputs read-only, copies verified experience into a new Phase 14B catalog, runs each segment independently, and returns stage/cost/gate reports.

- [ ] **Step 1: Write failing tests** proving legacy replay is unchanged; strategy state never crosses 18 segments; each segment has chronological development/validation/unseen; commission remains unknown; trade metrics/exit reasons are complete; source DBs are opened read-only; existing CandidatePromotionGate/minima unchanged; rejected/insufficient candidates cannot create promotions; qualified candidates can create only SHADOW_CHALLENGER.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_book_replay.py tests/test_phase14_promotion.py`.
- [ ] **Step 3: Implement candidate-aware replay and separate experiment orchestration** without changing Phase 14A source rows or gate constants.
- [ ] **Step 4: Verify GREEN** with focused tests and `tests/test_phase14_replay.py tests/test_phase14_evaluation.py tests/test_phase14_orchestrator.py`.
- [ ] **Step 5: Commit** `feat: replay book candidates through phase 14 gates`.

### Task 8: Gateway-Free Shadow Challenger Observer

**Files:**
- Create: `tradingagents/self_enhancement/challenger.py`
- Modify: `tradingagents/forex/hft/demo_runtime.py` for optional read-only observation hook, default disabled
- Test: `tests/test_phase14b_challenger.py`
- Test: `tests/test_phase12d_hft_first.py`

**Interfaces:**
- `ShadowChallengerObserver.on_tick(features, regime, risk_context) -> tuple[ChallengerObservation, ...]`.
- Observer is constructed only from a validated `SHADOW_CHALLENGER` registry and has no broker provider/order gateway reference. It uses shadow fill/position accounting only and writes to a separate Phase 14B observer store.

- [ ] **Step 1: Write failing tests** proving identical causal inputs, hypothetical signals/fills/exits/cost/MFE/MAE/duration are captured; invalid/stale regime suppresses candidate observation; observer cannot access gateway; incumbent decision and order result are identical with observer disabled/enabled; no observer candidate emits order intent.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_challenger.py tests/test_phase12d_hft_first.py`.
- [ ] **Step 3: Implement the isolated shadow observer** with explicit candidate allowlist and fail-closed loading.
- [ ] **Step 4: Verify GREEN**.
- [ ] **Step 5: Commit** `feat: observe qualified phase 14b shadow challengers`.

### Task 9: Explicit Phase 14B CLI and Offline Candidate Run

**Files:**
- Create: `cli/phase14b.py`
- Create: `tests/test_phase14b_cli.py`
- Modify: `pyproject.toml` only if a console script entry is required by repository conventions
- Artifacts: new dedicated path `data_cache/phase14b-book-strategies-20261003/` only after existence/source-fingerprint checks

**Interfaces:**
- `phase14b status --artifact-root ...`
- `phase14b draft-and-evaluate --knowledge-root C:\p7fast --expected-generation gen_607de64268a04a6ab09ffa1e160fc280 --hft-db ... --demo-db ... --phase14-artifact-root ... --model qwen3.5:2b`.
- Extraction is explicit; status/runtime commands never instantiate the drafter. Root must be new or the command must use a fresh run ID without overwriting prior data.

- [ ] **Step 1: Write failing CLI tests** for no implicit model calls, pinned Phase 7 generation/fingerprint mismatch refusal, explicit loopback model use, read-only source DBs, fresh output root, and safe report serialization.
- [ ] **Step 2: Verify RED** with `python -m pytest -q tests/test_phase14b_cli.py`.
- [ ] **Step 3: Implement commands and run one bounded offline batch** against targeted strategy queries with existing Phase 7 assets and qwen3.5:2b; no Ollama pull, MT5, full Phase 11A, or live supervisor. Persist only specs/provenance/diagnostics/metrics, not prompts/completions/reasoning.
- [ ] **Step 4: Verify GREEN** for CLI fixtures; then inspect the actual report, Phase 7 status/fingerprint and Phase 14A source fingerprints before/after.
- [ ] **Step 5: Commit** `feat: add explicit offline phase 14b research command`.

### Task 10: Final Phase 14B Verification and Report

**Files:**
- Modify: `docs/superpowers/specs/2026-10-03-phase-14b-book-derived-autonomy-design.md` only if implementation evidence requires a clarified contract
- Modify: `docs/superpowers/plans/2026-10-03-phase-14b-book-derived-autonomy.md` only for logged, justified rulings
- Tests: focused Phase 14B, Phase 14, Phase 12D, then full suite

- [ ] **Step 1: Run focused tests** for every Task 1–9 test file.
- [ ] **Step 2: Run broader suites**: all `tests/test_phase14*.py`, `tests/test_phase12*.py`, `tests/test_forex_supervisor.py`, and MT5 provider tests.
- [ ] **Step 3: Run full suite** with `python -m pytest -q`; record all failures exactly.
- [ ] **Step 4: Run Ruff** on `tradingagents cli scripts tests`, `python -m compileall -q tradingagents cli scripts`, and `git diff --check`.
- [ ] **Step 5: Verify safety/fingerprints**: Phase 7 active generation+fingerprint unchanged; Phase 14A/HFT/DEMO inputs unchanged; no MT5 call; no runtime/model import from HFT hot path; real-money execution impossible; `watch_dashboard.py` remains untouched/untracked.
- [ ] **Step 6: Final branch review** against the spec, then report the actual model result, every extraction/suitability/replay/gate count, current calendar availability/state, tests, commits and final HEAD. Do not claim market-closed runtime activation or a shadow challenger unless the explicit evidence exists.

## Plan Self-Review

- Spec coverage: calendar/schema/state/idle/reopen in Tasks 1–2; local model and evidence validation in Tasks 3–5; deterministic implementation and replay in Tasks 6–7; observer in Task 8; offline actual run in Task 9; all gates in Task 10.
- Interface consistency: market config/controller/probe are produced in Tasks 1–2; StrategySpec and drafter in Tasks 3–4; validated specs feed registry and CandidateSpec in Tasks 5–7; promoted shadow candidates feed Task 8; Task 9 uses all services without coupling to watcher startup.
- No source or incumbent behavior is modified by design; tests explicitly pin compatibility.
- Review focus has corresponding tests in Tasks 1, 2, 5 and 8.
- Output root absence is verified before creation; no cleanup or overwrite of existing artifacts is part of this plan.
