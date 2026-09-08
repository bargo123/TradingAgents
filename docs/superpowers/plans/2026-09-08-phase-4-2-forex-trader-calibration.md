# Phase 4.2 Forex Trader Calibration Implementation Plan

**Implementation status:** Completed and validated 2026-09-08. The checklist
below records the original TDD execution plan; final evidence is in the
validation report.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calibrate the standalone forex shadow workflow for intraday decisions using sufficient deterministic MT5 history, strict validity, forex-safe prompts/news, and per-agent performance evidence.

**Architecture:** Add a small forex profile/feature layer consumed by the existing one-snapshot runner and graph state. Use a forex-only Portfolio Manager schema and persistence fields, wrap shared forex graph nodes for profile-aware prompts and callback labels, and keep stock branches/defaults untouched. Add a forex-safe global-news wrapper and a metrics-only callback extension; no execution surface is introduced.

**Tech Stack:** Python 3.10+, dataclasses/Pydantic, LangGraph/LangChain callbacks, existing MT5 provider and LLM registry, SQLite, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-08-phase-4-2-forex-trader-calibration-design.md`

## Global Constraints

- Keep `forex-shadow` separate from `tradingagents` and leave `cli/main.py` stock behavior unchanged.
- Keep MT5 read-only, one snapshot per run, and `executed=False`; never add `order_send`, buy/sell, close/modify, pending-order, SL/TP, or live paths.
- Keep Market/News, Bull/Bear, Research Manager, Trader, Aggressive/Conservative/Neutral Risk, and Portfolio Manager stages enabled.
- Default forex profile is `INTRADAY`; default history is 100 bars per M1/M5/M15/H1; prompt context is deterministic and does not dump raw bars.
- Missing macro/calendar/event data must be rendered exactly as `MACRO/EVENT DATA UNAVAILABLE`; never infer that no event occurred.
- Forex normalization is structured and fail-closed; no BUY/SELL/HOLD guessing from prose, and invalid month/year horizons remain explicit failures.
- Read provider/model/API-key values through the existing environment/config system; do not add or print credentials. Preserve Ollama/local support and allow configured hosted providers.
- Telemetry stores numeric metrics only, including per-agent name/model/input/output/reasoning tokens and elapsed time; never store chain-of-thought text.

---

### Task 1: Add the explicit intraday profile and deterministic timeframe features

**Files:**
- Create: `tradingagents/forex/profile.py`
- Modify: `tradingagents/forex/context.py`
- Modify: `tradingagents/forex/__init__.py`
- Modify: `tradingagents/graph/propagation.py`
- Modify: `tradingagents/agents/utils/agent_states.py`
- Test: `tests/test_forex_phase42_context.py`

**Interfaces:**
- Produces `ForexAnalysisProfile`, `INTRADAY_PROFILE`, `resolve_forex_profile(name)`, `build_forex_profile_context(profile)`, and `calculate_timeframe_features(bars)`.
- `Propagator.create_initial_state(..., forex_analysis_profile="INTRADAY")` carries the profile for forex runs while leaving stock call sites valid.

- [ ] Write failing tests for profile resolution, exact profile context, deterministic multi-bar features, and `INSUFFICIENT_DATA` for one candle.
- [ ] Run `.venv\Scripts\python.exe -m pytest tests/test_forex_phase42_context.py -q` and confirm the missing-module/API failures.
- [ ] Implement the profile dataclass and feature calculation: first-open-to-last-close return; max high/min low/range; multi-bar direction; true-range average; range percentage; close position; and explicit insufficient-data values.
- [ ] Update `build_forex_market_context` to include profile, horizon, validity, macro status, feature summaries, and bounded serialized feature payloads without raw-bar prompt text.
- [ ] Add the optional profile field to state initialization and exports without changing stock defaults or stock prompt text.
- [ ] Re-run the focused tests and existing context/graph-mode tests; commit `feat: add intraday forex profile and market features`.

### Task 2: Add forex structured validity and backward-compatible persistence

**Files:**
- Modify: `tradingagents/agents/schemas.py`
- Modify: `tradingagents/forex/shadow.py`
- Modify: `tradingagents/forex/runner.py`
- Modify: `tests/test_forex_shadow_contract.py`
- Modify: `tests/test_forex_shadow_runner.py`

**Interfaces:**
- Produces `ForexPortfolioDecision`, `render_forex_pm_decision`, and optional `forex_profile` validation in `normalize_portfolio_manager_result`.
- `ShadowTradeDecision` gains backward-compatible `analysis_profile`, `valid_for_seconds`, and `valid_until`; SQLite migrates missing columns and round-trips them.
- `ForexShadowRunner.run(..., analysis_profile="INTRADAY")` persists profile/validity and passes it into initial state/context while keeping one snapshot.

- [ ] Write failing tests for rejecting month/year horizons and mismatched/invalid profile validity, deriving the declared profile interval when the field is omitted, and SQLite round-trip/migration defaults.
- [ ] Run the focused tests and confirm the schema/fields/migration failures.
- [ ] Implement the forex-only Pydantic subclass with exact profile/horizon/validity descriptions and validators; keep `PortfolioDecision` unchanged.
- [ ] Extend normalization with optional forex checks, preserving raw result and null action on failure.
- [ ] Extend the frozen decision/store with UTC validity fields, safe defaults, migration, insert, and row parsing; enforce positive bounded validity and `valid_until == snapshot + validity` when both are present.
- [ ] Update runner extraction/persistence to use the structured field when valid or the selected profile's declared interval when omitted; never parse prose; set the multi-timeframe analysis label.
- [ ] Re-run all shadow contract/runner tests and commit `feat: persist intraday forex decision validity`.

### Task 3: Make forex news and all downstream prompts intraday-safe

**Files:**
- Create: `tradingagents/forex/news.py`
- Modify: `tradingagents/dataflows/yfinance_news.py`
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/agents/analysts/news_analyst.py`
- Modify: `tradingagents/agents/researchers/bull_researcher.py`
- Modify: `tradingagents/agents/researchers/bear_researcher.py`
- Modify: `tradingagents/agents/managers/research_manager.py`
- Modify: `tradingagents/agents/trader/trader.py`
- Modify: `tradingagents/agents/risk_mgmt/aggressive_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/conservative_debator.py`
- Modify: `tradingagents/agents/risk_mgmt/neutral_debator.py`
- Modify: `tradingagents/agents/managers/portfolio_manager.py`
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/agents/utils/agent_utils.py`
- Test: `tests/test_forex_phase42_prompts.py`

**Interfaces:**
- `tradingagents.forex.news.get_forex_global_news` is a tool named `get_global_news`, uses forex-only broad queries, and returns the exact unavailable sentinel for no-data/error results.
- `get_forex_profile_context(state)` is appended only for forex state; stock prompt branches remain unchanged.

- [ ] Write failing tests for forex-safe query selection/tool binding, exact unavailable output, intraday/no-month-year prompt language across all forex downstream agents, and stock prompt regression.
- [ ] Run the focused prompt tests and confirm the missing wrapper/profile language failures.
- [ ] Implement the query-profile context and forex news wrapper; keep the tool's public name `get_global_news` and preserve stock news behavior.
- [ ] Add profile context to forex branches and use `ForexPortfolioDecision`/renderer in the forex PM; explicitly prohibit issuer valuation, equity allocation, and long horizons.
- [ ] Wire the wrapper into forex GraphSetup only; leave stock `get_global_news` tools and CLI unchanged.
- [ ] Re-run prompt, graph-mode, and structured-agent tests; commit `feat: calibrate forex prompts and macro uncertainty`.

### Task 4: Record per-agent metrics and configurable quick/deep thinking

**Files:**
- Create: `tradingagents/forex/telemetry.py`
- Modify: `cli/stats_handler.py`
- Modify: `tradingagents/graph/setup.py`
- Modify: `tradingagents/graph/trading_graph.py`
- Modify: `tradingagents/llm_clients/openai_client.py`
- Modify: `tradingagents/default_config.py`
- Modify: `tradingagents/forex/runner.py`
- Modify: `cli/forex_shadow.py`
- Test: `tests/test_forex_phase42_telemetry.py`

**Interfaces:**
- `agent_context(name, model)` labels a synchronous graph node through a context variable.
- `StatsCallbackHandler.get_stats()` retains existing counters and adds `agents: {name: {model, calls, tokens_in, tokens_out, reasoning_tokens, elapsed_seconds}}`.
- Optional forex quick/deep reasoning/thinking config is passed only to the selected LLM role/provider; the forex Trader, Research Manager, and Portfolio Manager retain deep reasoning while stock assignments remain unchanged.

- [ ] Write failing tests for agent context labels, usage extraction (including reasoning details), elapsed metrics, absence of prompt/chain-of-thought storage, and quick/deep config separation.
- [ ] Run the focused telemetry tests and confirm missing metrics/config failures.
- [ ] Implement context labeling and GraphSetup wrappers for every LLM-bearing node; wrap only node execution, never tool/clear nodes.
- [ ] Extend the callback handler with run-id-safe timing and numeric usage extraction from standard and provider-specific metadata; preserve stock display compatibility.
- [ ] Add opt-in `forex_quick_reasoning_effort`, `forex_deep_reasoning_effort`, and Ollama `think` extra-body forwarding without hard-coded credentials or global reasoning disablement.
- [ ] Merge per-agent metrics into runner results and print a compact metrics-only summary in the forex CLI.
- [ ] Re-run telemetry, stock CLI, provider-client, and full shadow tests; commit `feat: instrument forex agent performance`.

### Task 5: CLI, documentation, and real Phase 4.2 validation

**Files:**
- Modify: `cli/forex_shadow.py`
- Modify: `docs/forex-shadow.md`
- Create: `docs/superpowers/reports/2026-09-08-phase-4-2-forex-trader-calibration.md`
- Test: `tests/test_forex_shadow_cli.py`
- Test: `tests/test_forex_shadow_validation.py`

**Interfaces:**
- CLI adds `--analysis-profile` (default `INTRADAY`) and keeps existing stock command untouched; output includes profile, bars/features, macro status, validity, and per-agent numeric metrics.
- Validation report records a genuine completed decision or an exact provider-unavailable failure, compared with Phase 4.1's 2595.45 seconds.

- [ ] Write failing CLI/report tests for profile/validity/macro/features/per-agent evidence and `EXECUTED: FALSE`.
- [ ] Run focused CLI/validation tests and confirm missing evidence fields.
- [ ] Implement parser/config wiring and evidence rendering without secrets or execution options.
- [ ] Run the real opt-in MT5 read-only preflight, capture positions/orders before and after, and run `forex-shadow --symbol EURUSD --count 100` through the complete graph with the configured provider; use hosted configuration if already available, otherwise retain Ollama/local config with explicit bounded options.
- [ ] Query the persisted row and capture snapshot features, profile/validity, raw PM result, normalization status, callback totals/per-agent timings, runtime, and unchanged MT5 state; if the provider fails, record the exact non-zero error and mark the end-to-end criterion unverified.
- [ ] Update docs/report with exact evidence and Phase 4.1 runtime comparison; run full pytest, Ruff, compileall, `git diff --check`, and source safety scans; commit `test: validate phase 4.2 forex calibration`.

## Verification Gate

Before reporting completion, run:

```text
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\ruff.exe check .
.venv\Scripts\python.exe -m compileall -q tradingagents cli
git diff --check
git status --short
```

Also run the opt-in MT5 integration test and a read-only before/after positions/orders capture when the local terminal is available. Do not claim a completed decision without a persisted row and raw Portfolio Manager evidence.
