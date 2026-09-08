# Phase 4 Design: MT5 Forex Shadow Mode

**Status:** Design approved in discussion; implementation has not started  
**Date:** 2026-09-08  
**Repository:** C:\AITrading\TradingAgents  
**Phase 4 baseline:** d4f4a49aab5958c11f9b4f693f66477e19f1d194

## 1. Purpose

Phase 4 adds a standalone, read-only forex analysis entry point named
'forex-shadow'. It obtains one normalized market snapshot from a local
MetaTrader 5 terminal, supplies that context to the existing TradingAgents
graph, converts the resulting portfolio-manager recommendation to
BUY/SELL/HOLD, and stores a shadow decision with 'executed = false'.

The existing stock CLI and its normal data path remain unchanged. This phase
does not expose MT5 to LangGraph as a general-purpose tool registry and does
not create any execution path. The integration is deliberately shaped so a
later forex command can reuse the runner and storage interfaces without
changing the stock command or weakening the shadow-only boundary.

## 2. Scope and non-goals

### In scope

- A separate 'tradingagents.forex' package for forex context, MT5 tool
  adaptation, shadow-decision normalization, persistence, and orchestration.
- Optional forex mode parameters on the reusable TradingAgents graph while
  preserving stock defaults.
- A noninteractive 'forex-shadow' console script.
- A configurable, forex-safe analyst allow-list whose default is 'market,news'.
- Unit, compatibility, persistence, safety, and guarded integration tests.
- Documentation and one guarded real-terminal smoke run when the local MT5
  demo terminal and local Qwen runtime are available.

### Explicitly out of scope

- Any call to 'order_send', position open/close, buy/sell, modify-position,
  pending-order, stop-loss, take-profit, or other broker mutation API.
- A 'forex-demo' or 'forex-live' command, live-money support, execution
  credentials, or execution configuration.
- TradingAgents/LangGraph tool registration for Phase 4 beyond the graph-local
  injected adapter used by this runner.
- Changes to 'cli/main.py', the existing 'tradingagents' stock console entry
  point, or the stock analyst selection UX.
- Company fundamentals, P/E, EPS, revenue, dividends, StockTwits, Reddit, or
  other stock/social concepts in the default forex analysis.
- A full forex macro layer (economic calendar, central-bank, rates, inflation,
  employment, or equivalent provider). News is intentionally lightweight and
  global in this phase.
- RiskGovernor execution, RAG, training, ONNX, optimization, or automated
  trading.

## 3. Selected architecture

The implementation uses a thin forex-specific orchestration layer around the
existing graph/components:

~~~
forex-shadow
  -> validate CLI and MT5 configuration
  -> create one MT5Provider session
  -> resolve the requested broker symbol (fail closed)
  -> fetch one ForexMarketSnapshot
  -> build a compact normalized market context
  -> inject the context and MT5 adapter into TradingAgentsGraph
  -> invoke existing forex-compatible analysts and debate/risk/PM nodes
  -> normalize the PM rating to BUY/SELL/HOLD
  -> persist ShadowTradeDecision (executed is always false)
  -> print the shadow-only result and metrics
  -> shut down the provider in a finally block
~~~

The runner, rather than the provider or graph, owns provider initialization and
shutdown. The adapter receives an already-connected provider and cannot
initialize or close it. There is one snapshot fetch per run; the adapter
serves the cached snapshot to the market analyst and any later read-only
inspection in that run.

The normal stock CLI keeps its current graph construction, tools, prompts,
configuration, and interaction flow. New graph parameters default to the
current stock behavior, so a stock caller that does not opt into forex mode
sees no semantic change.

## 4. New package and responsibilities

The following package is added:

~~~
tradingagents/forex/
  __init__.py
  context.py
  tools.py
  shadow.py
  runner.py
~~~

### context.py

'build_forex_market_context(snapshot)' produces a deterministic, compact
string suitable for a graph state field and prompt interpolation. It begins
with:

~~~
SOURCE: LIVE MT5 BROKER DATA (read-only; not Yahoo Finance)
~~~

It identifies the requested and resolved symbols, UTC snapshot time, bid, ask,
mid, spread, spread points, digits, and point. For each available M1, M5, M15,
and H1 series it includes only the latest/basic derived values: OHLC, bar
direction, return, high, low, range, and count. It never embeds a provider
object representation, arbitrary broker payload, or an unbounded bar dump.
Missing optional series are represented explicitly rather than fabricated.

### tools.py

'MT5ToolAdapter' is a read-only facade over an injected Phase 3
'MT5Provider'. It exposes only these graph-local operations:

- 'get_mt5_market_snapshot(symbol)'
- 'get_mt5_tick(symbol)'
- 'get_mt5_bars(symbol,timeframe,count)'
- 'get_mt5_account_context()'
- 'get_mt5_positions(symbol=None)'
- 'get_mt5_spread(symbol)'

The adapter returns normalized strings or primitive mappings, never raw MT5
objects. 'get_mt5_market_snapshot' returns the cached normalized snapshot when
the runner has already fetched it; it must not trigger a second snapshot
request. The adapter does not call 'initialize', 'shutdown', or any mutation
method and contains no order API name.

### shadow.py

'ShadowTradeDecision' is a frozen value object representing one completed
analysis. It contains a UUID decision id, UTC creation time, analysis date,
requested and resolved symbols, canonical action, optional confidence,
reference bid/ask/mid/spread/spread-points, analysis timeframe, compact
context, trader/portfolio-manager summaries, optional bull/bear summaries,
provider/model metadata, an explicit serialized snapshot, future-evaluation
status, optional outcome/reflection fields, a source run id, and
'executed=False'.

Its post-initialization validation rejects any non-false executed value,
invalid action, naive timestamp, or non-UTC timestamp. Action normalization
uses the existing rating extractor and fails with an explicit review error
when the portfolio-manager output has no recognized rating:

- 'Buy' and 'Overweight' become 'BUY'.
- 'Hold' becomes 'HOLD'.
- 'Underweight' and 'Sell' become 'SELL'.

'ShadowDecisionStore' is a small stdlib-SQLite repository. It creates the
database on first use, uses parameterized transactions, and makes repeated
writes of the same decision idempotent. It has operations to record, retrieve,
list pending decisions, and update a future outcome/reflection without ever
changing the executed flag.

### runner.py

'ForexShadowRunner' is the only Phase 4 component that coordinates provider
lifecycle, graph invocation, decision normalization, persistence, and
metrics. It:

1. Validates the requested analyst list and rejects 'social' and
   'fundamentals' in forex mode.
2. Initializes an injected or lazily imported Phase 3 'MT5Provider'.
3. Resolves the requested pair using exact case-insensitive matching first,
   then controlled normalized broker variants. Prefixes and suffixes such as
   'mEURUSD', 'EURUSDm', and 'EURUSD#' are allowed only when they reduce to the
   requested six-letter base; if more than one plausible candidate remains,
   'Mt5SymbolAmbiguousError' is raised.
4. Fetches exactly one 'ForexMarketSnapshot'.
5. Builds the compact context and initial graph state.
6. Invokes the existing compiled graph directly with that state. It does not
   use the stock 'propagate' path because that path resolves Yahoo identity and
   writes the stock Markdown memory log.
7. Extracts the trader and portfolio-manager reports, normalizes the PM
   recommendation, stores a 'ShadowTradeDecision', and returns the decision
   plus run metrics.
8. Shuts down the provider in 'finally', including on validation, graph, or
   persistence failure.

The runner may use existing graph report rendering for diagnostics, but forex
shadow decisions are stored only in the dedicated SQLite store and are not
mixed into the stock memory log.

## 5. Graph integration

'TradingAgentsGraph' gains optional constructor parameters:

- 'market_data_mode="stock"' (the existing default)
- 'mt5_tools=None'

The graph state gains optional fields:

- 'market_data_mode'
- 'market_context'

'Propagator.create_initial_state' accepts and carries those fields while
retaining its current defaults for stock callers. The forex runner builds
'instrument_context' with the existing helper using the pair and
'asset_type="forex"'; it does not call the Yahoo identity resolver.

'get_instrument_context_from_state' appends the provided compact
'market_context' to the shared context used by prompts. This gives the
research manager, bull/bear researchers, trader, risk debators, and portfolio
manager the same normalized source without duplicating prompt plumbing.
Stock state with no market context renders exactly as before.

When 'market_data_mode="forex_mt5"':

- The market ToolNode contains only the injected MT5 snapshot adapter.
- The news ToolNode is restricted to lightweight global-news context.
- Graph setup passes mode and adapter references to analyst factories.
- 'social' and 'fundamentals' are rejected before graph construction.
- The run/checkpoint signature includes the market-data mode to prevent
  accidental cross-mode checkpoint reuse.

When the mode is the default 'stock', existing ToolNodes, selected analysts,
prompts, and graph topology remain unchanged. No new global TradingAgents
tool is registered.

## 6. Forex-safe analyst policy

The command accepts a comma-separated analyst list for experimentation, but
the Phase 4 validator is fail closed. The default is exactly:

~~~
market,news
~~~

The existing downstream nodes remain enabled by the shared graph topology:
Bull Researcher, Bear Researcher, Research Manager, Trader, all three risk
debators, and Portfolio Manager.

Forex prompt branches describe currency-pair price action, bid/ask/spread,
volatility, and broad market/news context. They explicitly state that company
fundamentals are unavailable and must not be inferred. The bull/bear and risk
prompts must not introduce stock fundamentals terminology. The social
sentiment analyst and company fundamentals analyst cannot be selected in
forex mode. Future forex-safe analysts can be added to the allow-list without
altering the stock CLI.

## 7. Shadow decision persistence

The default database is
'data_cache_dir/shadow_decisions.db', independently of LangGraph
checkpoints. A caller may supply '--db-path' for tests or a separate
deployment. The initial schema is:

~~~
shadow_decisions(
  decision_id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  analysis_date TEXT NOT NULL,
  requested_symbol TEXT NOT NULL,
  resolved_symbol TEXT NOT NULL,
  action TEXT NOT NULL CHECK (action IN ('BUY','SELL','HOLD')),
  confidence REAL,
  reference_bid REAL NOT NULL,
  reference_ask REAL NOT NULL,
  reference_mid REAL NOT NULL,
  spread REAL NOT NULL,
  spread_points REAL NOT NULL,
  analysis_timeframe TEXT NOT NULL,
  trader_summary TEXT NOT NULL,
  portfolio_manager_summary TEXT NOT NULL,
  bull_summary TEXT,
  bear_summary TEXT,
  llm_provider TEXT,
  quick_model TEXT,
  deep_model TEXT,
  snapshot_json TEXT NOT NULL,
  executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0),
  future_evaluation_status TEXT NOT NULL DEFAULT 'PENDING',
  outcome_raw REAL,
  outcome_alpha REAL,
  outcome_resolved_at TEXT,
  reflection TEXT,
  source_run_id TEXT
)
~~~

An index on '(resolved_symbol, analysis_date)' supports later evaluation.
Snapshot JSON is an explicit normalized serialization of the Phase 3 model,
not 'repr()' or a broker object dump. There is no store method that can mark a
decision executed.

## 8. CLI contract

'pyproject.toml' adds the console entry:

~~~
forex-shadow = "cli.forex_shadow:main"
~~~

'cli/forex_shadow.py' uses noninteractive argument parsing. The initial
options are:

- '--symbol' (default 'EURUSD')
- '--count' (positive integer, default chosen by the runner)
- '--analysis-date' (UTC ISO date, default current UTC date)
- '--terminal-path' (optional MT5 terminal path)
- '--db-path' (optional shadow SQLite path)
- '--analysts' (default 'market,news')
- optional provider/model/backend overrides already supported by the
  repository configuration

Credentials are not accepted on the command line. The command prints these
lines prominently before any analysis:

~~~
MT5 FOREX — SHADOW MODE
NO ORDER WILL BE SENT
~~~

On successful persistence it prints:

~~~
SHADOW DECISION RECORDED
EXECUTED: FALSE
~~~

Connection, symbol, graph, rating, and persistence failures return a
non-zero exit status with an actionable error and never fall back to Yahoo or
another market-data vendor.

## 9. Safety invariants

The safety boundary is structural, not a caller convention:

- No Phase 4 source file imports or references 'order_send', buy/sell
  execution, close/modify-position, pending-order, or SL/TP mutation APIs.
- 'MT5ToolAdapter' has only read methods; it cannot own provider lifecycle.
- 'ShadowTradeDecision' can only be constructed with 'executed=False'.
- The SQLite schema has 'CHECK (executed = 0)' and the store never updates it.
- The CLI has no execution mode or execution credentials.
- Explicit MT5 mode has no Yahoo fallback.
- The existing stock command is not modified to expose this mode.

Tests include a production-source scan and explicit reflection checks proving
that 'MT5Provider' exposes no mutation/execution API such as 'order_send',
buy/sell, close-position, or modify-position methods.

## 10. Verification and test plan

### Unit tests

- deterministic context formatting, UTC timestamps, bounded output, and no
  raw object repr;
- adapter delegation, normalized primitive output, cache behavior, and no
  raw MT5 calls;
- symbol resolution exact-match precedence, controlled prefixes/suffixes,
  ambiguity, and fail-closed errors;
- graph mode defaults and stock compatibility;
- forex analyst allow-list rejection of social/fundamentals;
- PM rating extraction and BUY/SELL/HOLD mapping, including missing-rating
  failure;
- frozen decision validation and normalized snapshot serialization;
- SQLite schema, restart persistence, idempotent duplicate recording, pending
  listing, outcome updates, and immutable executed false.

### Integration and compatibility tests

- existing stock CLI and market ToolNode tests remain unchanged and passing;
- one-snapshot runner test with fake provider and fake graph;
- graph-state/context propagation test proving all shared downstream prompts
  receive the same forex context;
- source scan for forbidden mutation symbols.

### Guarded real-terminal smoke test

An opt-in test ('RUN_MT5_INTEGRATION=1') connects to the local MT5 demo
terminal, resolves EURUSD (including broker prefix/suffix handling), obtains a
snapshot through the adapter, and verifies UTC normalization and non-empty
bid/ask/bar data. It does not invoke the LLM or mutate the account.

After implementation, run one real 'forex-shadow --symbol EURUSD' with the
local Qwen backend when available. Record provider connection status, resolved
symbol, snapshot timestamp, spread, selected analysts, LLM/runtime metrics,
decision id/action, database path, and 'executed=false'. Capture positions and
orders before and after and verify they are byte-for-byte unchanged. If the
terminal or Qwen runtime is unavailable, report that verification as
unverified rather than fabricating a result.

The complete verification gate is the existing test suite, the new tests,
Ruff/lint checks used by the repository, the guarded provider integration
test, and the documented one-run smoke result.

## 11. Performance and observability

The runner records elapsed runtime and counts provider snapshot/tool calls and
LLM calls. Snapshot and context payloads are bounded to keep prompts and
SQLite rows small. Provider shutdown is timed only for diagnostics and never
skipped to improve latency. Errors include the requested/resolved symbol and
mode where available, without leaking credentials.

## 12. Future extension point

The forex runner and decision store form the seam for a later
'forex-demo' or 'forex-live' command. Such a command would need a separately
reviewed execution subsystem and additional authorization; it must not alter
'cli/main.py', the stock entry point, or the current shadow decision schema
in a way that permits execution. Phase 4 deliberately leaves that work
unimplemented.

## 13. Acceptance criteria

Phase 4 is complete only when all of the following are true:

1. 'forex-shadow' is a separate command and the existing stock CLI diff is
   empty.
2. A real or fake MT5 session produces one normalized
   'ForexMarketSnapshot', with exact-first fail-closed symbol resolution and
   UTC tick timestamps preferring 'time_msc'.
3. The default forex analyst set is 'market,news'; social and fundamentals are
   rejected; the shared downstream agents run with forex-safe prompts.
4. The existing graph/components are reused with injected context; no
   duplicate agent framework is introduced.
5. A completed run persists a normalized 'ShadowTradeDecision' whose action is
   BUY, SELL, or HOLD and whose executed value is structurally false.
6. Safety tests prove no MT5 mutation/execution API is exposed and no order is
   sent.
7. Unit/integration tests and repository lint checks pass, and the guarded
   real-terminal smoke verification is either recorded as passing or clearly
   marked unavailable/unverified.
8. No Phase 5 execution, macro expansion, training, ONNX, RAG, or workflow
   registration work is included.
