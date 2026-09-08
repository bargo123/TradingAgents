# Phase 3 Read-Only MT5 Provider Design

**Date:** 2026-09-08  
**Repository:** `C:\AITrading\TradingAgents`  
**Scope:** Standalone MetaTrader 5 data access for a locally installed MT5 demo terminal.

## Goal

Add a clean, testable Python boundary between TradingAgents and the locally installed MetaTrader 5 terminal. Phase 3 provides normalized read-only terminal, account, symbol, tick, bar, position, order, spread, and market-snapshot data. It does not change the TradingAgents agent graph or register LangGraph tools.

## Constraints

- The MT5 path is read-only. The provider will not expose or wrap any order placement, position opening/closing, pending-order, or stop-loss/take-profit mutation API.
- No credentials, login values, broker server names, or terminal paths are hard-coded. An optional terminal path may be supplied by the caller or CLI.
- MT5 is optional at package-install time because the official package is Windows-specific. The Windows development environment will install the `mt5` extra for live smoke verification.
- Existing TradingAgents dataflow routing and LangGraph registration remain unchanged in this phase.
- Missing dependency, unavailable/closed terminal, failed initialization, disconnected account, missing symbol, ambiguous symbol, and missing market data produce typed, actionable provider errors.

## Alternatives considered

### Isolated MT5 package (chosen)

Create `tradingagents/dataflows/mt5/` with separate models, errors, timeframe mapping, and provider modules. A lazy import and injectable API object keep normal package imports and unit tests independent of a Windows terminal. This gives Phase 4 a stable boundary without coupling broker state to the existing stock-vendor router.

### Existing vendor-router integration (not chosen)

Adding MT5 to `tradingagents.dataflows.interface` would make existing stock tools select a broker session and could change upstream analysis behavior. It is outside the Phase 3 scope and conflicts with the explicit no-agent-architecture-change constraint.

### Monolithic provider/CLI module (not chosen)

Keeping API calls, normalization, symbol matching, errors, snapshot assembly, and command-line formatting in one file would be harder to test and extend. Focused modules provide clearer ownership and a smaller future LangGraph adapter surface.

## Components

### `tradingagents/dataflows/mt5/models.py`

Frozen dataclasses represent normalized data and contain no raw MetaTrader5 objects:

- `Mt5TerminalInfo`
- `Mt5AccountInfo`
- `Mt5SymbolInfo`
- `Mt5Tick`
- `Mt5Bar`
- `Mt5Position`
- `Mt5Order`
- `Mt5Spread`
- `ForexMarketSnapshot`

Timestamps are timezone-aware UTC `datetime` values. Bars and positions in a snapshot are immutable tuples. The snapshot has explicit `m1_candles`, `m5_candles`, `m15_candles`, and `h1_candles` fields so future feature computation can be added without changing transport calls.

### `tradingagents/dataflows/mt5/errors.py`

The provider error hierarchy distinguishes dependency, lifecycle/connection, timeframe, symbol, account, and data failures. Messages include the requested symbol/timeframe and MT5's last-error detail when available, but never credentials.

### `tradingagents/dataflows/mt5/timeframes.py`

`SUPPORTED_TIMEFRAMES` and `TIMEFRAME_ATTRIBUTES` define the seven supported string names (`M1`, `M5`, `M15`, `M30`, `H1`, `H4`, `D1`). `resolve_timeframe()` validates the name and resolves the corresponding attribute on the injected MT5 API module, allowing unit tests to use a small fake module.

### `tradingagents/dataflows/mt5/provider.py`

`MT5Provider` (also exported as `Mt5Provider`) owns the MT5 module and lifecycle. It lazy-loads the dependency unless an API object is injected, initializes without login/password arguments, checks terminal/account connectivity, and converts every response into the models above. Public methods are:

```text
initialize() -> bool
shutdown() -> None
is_connected() -> bool
get_terminal_info() -> Mt5TerminalInfo
get_account_info() -> Mt5AccountInfo
get_symbols() -> tuple[Mt5SymbolInfo, ...]
find_symbol(query: str) -> str
ensure_symbol(symbol: str) -> str
get_tick(symbol: str) -> Mt5Tick
get_bars(symbol: str, timeframe: str, count: int) -> tuple[Mt5Bar, ...]
get_positions(symbol: str | None = None) -> tuple[Mt5Position, ...]
get_orders(symbol: str | None = None) -> tuple[Mt5Order, ...]
get_spread(symbol: str) -> Mt5Spread
get_market_snapshot(symbol: str, count: int = 100) -> ForexMarketSnapshot
```

`find_symbol()` first selects an exact case-insensitive name. If no exact match exists, it normalizes the query and considers broker variants that preserve the same forex base, including separator and non-separator suffixes beyond a fixed allow-list. The normalization strips harmless case/whitespace and common broker punctuation but never changes the six-letter currency base or treats an unrelated instrument as a match. Zero candidates raises a not-found error; more than one plausible candidate raises an ambiguity error listing the candidates. `ensure_symbol()` calls `symbol_select()` for the resolved name and fails if the broker cannot make it visible.

Ticks and bars are normalized from namedtuples, mappings, or NumPy structured rows using field access that does not leak the original object. Tick timestamps prefer MT5 `time_msc` (millisecond precision) and fall back to `time`, and both become timezone-aware UTC datetimes. `copy_rates_from_pos()` supplies bars; positions and orders are read through `positions_get()` and `orders_get()`. `Mt5SymbolInfo` always carries broker `digits` and `point`; a spread contains both price difference and point difference, using that point size.

`get_market_snapshot()` resolves/selects the symbol once, reads the tick, four required candle series, account, and symbol-relevant positions, and stores the tick timestamp as the snapshot timestamp. Empty/invalid required data raises a typed error rather than producing a partial or fabricated snapshot.

### `scripts/test_mt5_connection.py`

The CLI accepts `--symbol` (default `EURUSD`), optional `--count`, and optional `--terminal-path`. It prints a stable, secret-free report containing connection status, terminal summary, account server/balance/equity/free margin, resolved symbol, bid/ask/spread, the latest M5 candle, and open positions. Provider errors are rendered as one actionable line and produce a non-zero exit code. The CLI always shuts the provider down in `finally`.

### Documentation

`docs/mt5-provider.md` explains installation of the optional extra, the smoke command, supported timeframes, symbol-resolution behavior, common initialization problems, and the explicit Phase 3 read-only boundary.

## Testing strategy

`tests/test_mt5_provider.py` uses a deterministic fake MT5 module and covers initialization success/failure, lifecycle/connection state, timeframe mapping, exact/normalized-variant/ambiguous/missing symbol resolution, tick `time_msc`/`time` normalization, bar/account normalization, spread calculation using `digits`/`point`, snapshot assembly, and explicit absence of mutation/execution methods (`order_send`, buy/sell, close-position, and modify-position variants). A guarded `tests/test_mt5_integration.py` test is marked `integration` and runs only when `RUN_MT5_INTEGRATION=1`; it connects to an already logged-in terminal, reads EURUSD (or `MT5_TEST_SYMBOL`), and never sends an order. Existing tests are run unchanged after the new tests.

## Future extension point

Phase 4 can add LangChain wrappers or a LangGraph `ToolNode` using the provider's typed methods. That adapter will be a separate layer; the MT5 transport and normalization modules do not need to change.
