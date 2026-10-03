# Phase 14B — Book-Derived Self-Enhancement and Market-Closed Autonomy

**Status:** Approved for implementation by the user on 2026-10-03.  
**Baseline:** `c66f6e234cbd155841b448927d3306b1c417f8da`  
**Scope:** TradingAgents Forex Phase 14B only.

## 1. Goals and non-goals

Connect the frozen Phase 7 knowledge generation to the Phase 14 research plane so a local model can draft auditable strategy specifications, a deterministic validator can reject unsupported rules, and supported/data-compatible concepts can be implemented and causally evaluated. Add a market lifecycle that treats a calendar-confirmed closure as healthy idle state and autonomously revalidates before resuming HFT when broker ticks return.

This phase does not rebuild or write Phase 7, regenerate Phase 11A, train models, loosen Phase 14 gates, change the incumbent, force trades, create any book-derived order path, or permit real-money execution. Model use is an explicit offline drafting operation only. No books, RAG, models, network retrieval, or dynamic code generation are permitted on the HFT tick path.

## 2. Approved architecture and alternatives

Three approaches were considered:

1. Add generated strategies directly to the live incumbent engine. Rejected: it would couple unvalidated research to DEMO execution.
2. Keep book extraction/replay isolated, add only a read-only shadow-challenger observer to the existing tick loop, and make market idle/reopen a supervisor-owned lifecycle. **Selected:** shares established features, regime, risk and fill primitives while leaving incumbent decisions and gateway untouched.
3. Generate executable Python from model output. Rejected: arbitrary generated code is not an acceptable execution boundary.

The offline pipeline is `Phase 7 retrieval → loopback-only local model StrategySpec draft → deterministic source/rule validator → completeness gate → HFT suitability → deterministic registered implementation → segmented replay/walk-forward/cost gates → existing Phase 14 promotion gate → at most SHADOW_CHALLENGER`.

The runtime lifecycle is `session calendar + terminal/account evidence → explicit market state`. A calendar predicts whether a session should be open; it never grants permission to trade. A resumed broker tick triggers opening validation before any strategy evaluation.

## 3. Package boundaries

- `tradingagents/forex/hft/market_calendar.py`: immutable, versioned broker/symbol calendar and strict JSON loader.
- `tradingagents/forex/hft/market_lifecycle.py`: `MARKET_CLOSED`, `MARKET_OPENING_VALIDATION`, `MARKET_OPEN`, `MARKET_UNKNOWN`, `MARKET_DATA_ERROR` state transitions and safe status/reason metadata.
- `tradingagents/dataflows/mt5/provider.py`: read-only tick-availability observation that does not require an existing fresh broker-clock calibration. It may trigger validation, but its result is never a trading tick.
- `tradingagents/self_enhancement/strategy_specs.py`: immutable StrategySpec, atomic source-rule claims, source spans, validation and suitability contracts.
- `tradingagents/self_enhancement/book_drafter.py`: explicit local-only Ollama structured-JSON draft client; no import or invocation from runtime modules.
- `tradingagents/self_enhancement/book_pipeline.py`: Phase 7 query adapter, deterministic evidence validation, rule-grammar mapping, completeness, suitability and semantic deduplication.
- `tradingagents/forex/hft/book_strategies.py`: finite registry of reviewed deterministic primitives implementing the existing strategy signal contract; no model-generated code.
- `tradingagents/self_enhancement/replay.py` and `evaluation.py`: candidate entry-strategy support using current causal features, fill, risk, exit, split and metric components. Existing candidate behavior remains the default when no book StrategySpec is attached.
- `tradingagents/self_enhancement/challenger.py`: non-executing observer for already-qualified shadow challengers; it has no DEMO gateway reference.
- `cli/phase14b.py`: explicit offline extraction/evaluation/status command. It opens Phase 7 projections for read only and writes only below a dedicated Phase 14B artifact root.

## 4. Market-state contract and calendar

The states are `MARKET_CLOSED`, `MARKET_OPENING_VALIDATION`, `MARKET_OPEN`, `MARKET_UNKNOWN`, and `MARKET_DATA_ERROR`. A separate reason distinguishes `STALE_DATA`, `BROKER_DISCONNECTED`, `BROKER_CLOCK_ERROR`, `CALENDAR_BROKER_MISMATCH`, and `DATA_FAILURE`.

Calendar JSON must identify broker server, exact resolved symbol, IANA time zone, version, validity coverage, weekly sessions, full-date holiday closures, and date-specific early/special-session overrides. Local `zoneinfo` supplies DST conversion. Overlapping or malformed sessions, missing coverage, an expired version, broker mismatch, or ambiguous current time produce `MARKET_UNKNOWN`; there is no generic Friday/Sunday fallback.

`MARKET_CLOSED` requires a valid matching calendar to say closed and healthy terminal/account connectivity. It does not require a tick to prove closure. A fresh broker tick that contradicts a closed calendar produces `MARKET_OPENING_VALIDATION` with `CALENDAR_BROKER_MISMATCH`, never silent suppression. Calendar-open plus missing/stale ticks is not `MARKET_CLOSED`.

The runtime remains the single MT5 owner and retains its leases while idle. During `MARKET_CLOSED`, it does no reconnect/reinitialize loop, order attempt, synthetic tick, or recovery-budget consumption. It performs bounded read-only health/tick observations and continues heartbeats. Missing/invalid calendar or data is fail-closed and cannot generate an order.

After a genuinely resumed tick, the same owner transitions to `MARKET_OPENING_VALIDATION`: calibrate/validate broker time and tick freshness; verify connected terminal and authoritative `ACCOUNT_TRADE_MODE_DEMO`; reconcile owned positions/orders with the broker; verify order-rate and DEMO risk circuits; validate strategic state. Only if all checks pass does state become `MARKET_OPEN`/`RUNNING` and the tick reach strategy evaluation. Failed validation remains no-trade and does not bypass reconciliation. Calendar time alone never starts trading.

## 5. Local model draft and deterministic evidence validation

Drafting is explicit opt-in and loopback-only (`127.0.0.1`); the adapter refuses non-loopback endpoints. It uses a configured locally installed Ollama model, temperature 0, native JSON-schema output, thinking disabled, finite context/output/timeout bounds, and no automatic model download. The draft schema is closed/strict. The adapter records only provider/model/version/configuration, prompt/schema versions, latency/token metadata where available, draft hash, and safe validation diagnostics. It never persists prompts, completion text, or reasoning.

Each atomic rule (direction, entry, confirmation, invalidation, expected move, exit, profit protection, stop behavior, horizon, session/volatility/spread filter) contains a source reference and exact text span. The runtime binds generation, document, chunk, source hash and text from the retrieved Phase 7 hit; the model cannot supply or override those identities. Validation verifies the active generation/fingerprint, document/chunk/hash membership, exact span offsets and exact quote, then parses the quote through a finite reviewed source-rule grammar. The parsed operator, direction, value/unit and condition must equal the proposed atomic rule. Unknown/ambiguous grammar, mismatched operator/direction/value/condition, or absent evidence is rejected; lexical similarity alone is not support.

Unspecified source facts remain `UNSPECIFIED`. Missing required rule fields yield `INSUFFICIENT_SPECIFICATION`; unavailable data yields `UNAVAILABLE_DATA`; rules needing unsupported deterministic primitives yield `UNIMPLEMENTABLE_AUTONOMOUSLY`. A later research parameter is represented separately as `RESEARCH_HYPOTHESIS_PARAMETER`, never as book evidence. It is not automatically supplied in this phase. Only fully supported, complete and data-compatible specifications enter executable HFT research.

## 6. StrategySpec, suitability and deduplication

Each immutable StrategySpec contains ID/name/family; source books and exact chunks; required data; atomic entry/confirmation/direction/invalidation/expected-move/exit/profit-protection/risk/horizon/session/volatility/spread rules; explicit unspecified fields; implementation confidence; suitability; validation results; Phase 7 generation/fingerprint; model and prompt/schema versions; creation timestamp; and deterministic content hash.

Suitability is one of `HFT_SUITABLE`, `SHORT_TERM_SUITABLE`, `INTRADAY_ONLY`, `SWING_ONLY`, `UNIMPLEMENTABLE`, `UNIMPLEMENTABLE_AUTONOMOUSLY`, `UNAVAILABLE_DATA`, or `INSUFFICIENT_SPECIFICATION`. Only `HFT_SUITABLE` plus complete verified source rules and supported data/primitives can be implemented.

Deduplication uses a normalized rule/parameter fingerprint; independent source books remain attached to the canonical family. Multiple chunks from one document count as one book. Candidate selection is by clarity, data availability, implementation confidence, short-horizon fit and independent source support, never retrospective profitability. The initial target is 5–10 candidates, but the pipeline must return fewer or none rather than invent or pad.

Existing `range_rejection` and `momentum_continuation` are audited field-by-field (`entry`, `filter`, `expected move`, `exit`, `risk`, `horizon`) and classified `MATCHED`, `PARTIALLY_MATCHED`, or `NO_DIRECT_MATCH`. No synthetic provenance or automatic incumbent modification is allowed.

## 7. Deterministic implementations and shadow boundary

New strategies are selected from a finite registry of deterministic primitives implementing the existing signal contract. Runtime inputs are causal `TickFeatures`, current regime/direction permission, current spread, per-strategy state and risk context. The strategy does not receive retrieved text or invoke a model. Every implemented candidate has a coherent source-supported entry/confirmation/invalidation/expected-move/exit/protection/horizon concept; otherwise it remains non-executable research.

The existing incumbent strategy and gateway behavior remain unchanged. A candidate can be inserted only into causal replay and, after passing every current Phase 14 gate, a read-only shadow challenger observer. That observer records candidate signal/hypothetical fill/exit/cost/MFE/MAE/holding-time data and cannot send orders or alter incumbent state. Maximum automatic status is `SHADOW_CHALLENGER`; no DEMO challenger, promotion to incumbent, risk change, or MT5 mutation occurs in Phase 14B.

## 8. Experience and replay protocol

Use the 118 Phase 14A `VERIFIED_EXECUTION` records as observational evidence only; retain and exclude the three quarantined executions. Book evidence creates a supported hypothesis, not a causal conclusion. Phase 14A SQLite and HFT/DEMO source databases are read-only inputs. Phase 14B writes a separate, dedicated artifact root.

Replay is over the canonical 21,131 ticks and 18 causal segments; no state crosses segment boundaries. Each segment is split chronologically into development, validation and unseen holdout with existing guard gaps; rolling walk-forward is chronological; unseen parameters/specs are frozen before evaluation. Cost scenarios use observed bid/ask/spread/slippage plus the existing predefined additional-cost cases. Commission remains `UNKNOWN` and cannot be reported as zero or known. Candidate reporting includes trade count, trades/hour, side counts, win/loss, expectancy, profit factor, drawdown, MFE/MAE/capture, profit-to-loss flips, holding duration, spread/slippage sensitivity, session/regime distribution and exit reasons.

The existing `CandidatePromotionGate`, its sample minima and reasons are authoritative and unchanged. Insufficient samples remain `INSUFFICIENT_EVIDENCE`; all candidates may be rejected. Passing candidates may advance only to `SHADOW_CHALLENGER`.

## 9. CLI, storage and reproducibility

`phase14b status`, `phase14b extract`, and `phase14b evaluate` are explicit offline operations; starting a watcher/supervisor never invokes extraction. The extraction command refuses a missing/unvalidated Phase 7 generation or a generation/fingerprint differing from the pinned values. It never calls ingestion, parser, embedding or index writers. The model command is explicitly opt-in, bounded and loopback-only.

Validated specs, source spans, failures and run reports live under a dedicated Phase 14B artifact root, separate from Phase 7, Phase 11A and Phase 14A source artifacts. Existing Phase 14A tables/data are not rewritten; replay inputs are opened read-only. Each result records dataset/spec/generation/model/grammar/strategy-code fingerprints and `commission_known=false`.

## 10. Tests and acceptance

Offline fixture tests cover calendar validity/time zones/DST/holidays/overrides/ambiguity; healthy closed idle and lease heartbeats; open-calendar stale data; disconnection/clock/data failures; calendar/broker mismatch; reopen validation ordering and fail-closed behavior; no broker calls from calendar-only transitions; loopback/model-output bounds; strict JSON/schema; exact provenance/span/grammar checks for operator/direction/value/condition; unspecified and unavailable data rejection; dedup retaining independent book provenance; current-strategy mapping honesty; deterministic registry allowlist; coherent entry/exit; segment isolation; chronological walk-forward/unseen freeze; costs/UNKNOWN commission; sample gate and no promotion; shadow observer has no gateway; real-money remains impossible; runtime import graph has no Phase 7/model access on the tick path.

The actual offline run verifies before/after Phase 7 active generation, population/fingerprint and source files; it uses only the existing local model and local indexes. It reports every draft, rejection, supported family, candidate metrics, gate decision and whether a shadow challenger was recorded. No live market or order is required for acceptance.

## 11. Phase boundaries

Phase 7 remains frozen/read-only; Phase 11A artifacts are untouched. Phase 14A verified executions are observational only. No training, Phase 15+, RAG/model call on the live tick path, incumbent change, DEMO order from candidates, or real-money execution is in scope.
