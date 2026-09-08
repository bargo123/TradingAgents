# Phase 4.2 Forex Trader Calibration Design

**Status:** Approved for implementation from the Phase 4.2 request.

**Goal:** Make the standalone `forex-shadow` run represent a bounded intraday forex decision with sufficient deterministic MT5 history, explicit macro-data uncertainty, strict decision validity, and evidence-only runtime telemetry.

## Scope and safety

Phase 4.2 changes only the forex shadow boundary and the shared agent nodes' forex branches. The existing stock CLI, stock prompts, stock graph defaults, MT5 provider, and provider read-only contract remain unchanged. The graph still runs Market/News, Bull/Bear, Research Manager, Trader, all three Risk Analysts, and Portfolio Manager. No order, position, pending-order, SL/TP, execution, Phase 5 evaluation, RAG, or training path is added.

Every run still owns one `MT5Provider`, fetches one `ForexMarketSnapshot`, injects one deterministic context, reuses the cached adapter, and persists a `ShadowTradeDecision` with `executed=False`.

## Forex analysis profile

`INTRADAY` is the default and currently supported profile. The profile is represented explicitly in runner arguments, graph state, prompts, persisted decisions, and CLI evidence. It defines:

- decision horizon: minutes to hours, never months or years;
- validity: a bounded deterministic interval (one hour) from the MT5 snapshot timestamp;
- default history: 100 bars for each M1, M5, M15, and H1 series.

The profile resolver is fail-closed for unknown profile names so a future profile can be added without changing stock behavior.

The forex Portfolio Manager uses a forex-only structured schema that retains the shared rating fields and adds `analysis_profile` and `valid_for_seconds`. Its time-horizon validator rejects month/year/long-term equity language. If a provider omits the optional validity field, the runner uses the selected profile's declared interval; it never guesses from prose. A present invalid profile, horizon, or validity value causes normalization to fail with a null action.

## Deterministic market context

The snapshot serializer continues to bound persisted bar payloads. Prompt context never includes raw bar arrays. Instead, each required timeframe has a deterministic feature summary containing candle count, latest OHLC, multi-bar return (first open to latest close), recent high/low/range, direction, average true range, range percentage, and latest-close position within the recent range. With fewer than two candles, direction and multi-bar return are explicitly `INSUFFICIENT_DATA`; a single candle is never called a trend.

The context begins with the live MT5 source label, profile/horizon/validity lines, quote metadata, and the exact macro status `MACRO/EVENT DATA UNAVAILABLE`. Phase 4.2 does not infer the absence of CPI, NFP, central-bank, geopolitical, or other scheduled events from an empty provider result.

## Forex-safe news and prompts

The forex News analyst binds one tool named `get_global_news`, implemented through a forex-safe wrapper that selects only broad/global/macro queries and converts an unavailable/error/no-data result to `MACRO/EVENT DATA UNAVAILABLE`. It never binds ticker news, fundamentals, insider data, StockTwits, Reddit, or company queries. All forex branches of Bull/Bear, Research Manager, Trader, Risk Analysts, and Portfolio Manager state the intraday horizon, use pair price-action/spread/volatility evidence, and forbid issuer valuation, equity allocation, dividends, earnings, and month/year holding periods. Stock branches are left as-is.

## Runtime telemetry

The existing callback statistics handler gains an optional per-agent map. Graph nodes are wrapped with a context label and model identifier before invoking the existing LLMs. Callback events record call count, model, input tokens, output tokens, reasoning tokens when a provider reports them, and elapsed seconds. Only numeric metrics are retained; prompt text and chain-of-thought are never stored. Provider-specific limits remain configurable: existing OpenAI/Google/Anthropic knobs continue to apply, with optional forex quick/deep overrides and an Ollama `think` extra-body flag when explicitly configured. Deep reasoning remains available for Research Manager and Portfolio Manager.

## Persistence and compatibility

`ShadowTradeDecision` gains `analysis_profile`, `valid_for_seconds`, and `valid_until` with backward-compatible defaults. `ShadowDecisionStore.initialize()` adds missing columns to existing SQLite databases with safe defaults. New rows persist the profile/validity alongside the raw structured Portfolio Manager result, normalized action/status, symbols, snapshot quote/time, provider/model identifiers, feature-bearing snapshot JSON, and `executed=0`.

## Validation

Tests cover stock-mode regression, profile/horizon validation, single-candle handling, deterministic multi-bar features, exact macro-unavailable semantics, forex-safe tool binding/prompts, one-snapshot caching, persistence/validity, per-agent metrics without text, and the mutation-free surface. A real opt-in EURUSD run uses the configured provider (hosted when available, otherwise Ollama), the production default history, and the complete graph. The report records provider/models, bars/features, calls/tokens/per-agent timings, raw and normalized PM evidence, validity/profile, macro status, shadow ID/database, and unchanged MT5 positions/orders. A failed provider run is reported as unavailable, never fabricated.
