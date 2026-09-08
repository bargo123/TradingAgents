# Phase 3 Read-Only MT5 Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone, typed, read-only MetaTrader 5 provider with broker-symbol resolution, normalized forex snapshots, a secret-free smoke CLI, tests, and documentation without changing TradingAgents or LangGraph behavior.

**Architecture:** Keep all MT5 code under tradingagents/dataflows/mt5/. A lazy-loaded, injectable API adapter owns terminal lifecycle and converts MT5 namedtuples/structured rows into frozen dataclasses. The existing tradingagents.dataflows.interface router, agent tools, graph setup, and agent workflow remain untouched; Phase 4 can add a separate adapter later.

**Tech Stack:** Python 3.10+, official MetaTrader5 package as an optional mt5 extra, standard-library dataclasses/argparse/logging, pytest, and the repository's existing Ruff configuration.

**Spec:** docs/superpowers/specs/2026-09-08-mt5-read-only-provider-design.md

## Global Constraints

- The MT5 provider is read-only and contains no mutation/execution methods or wrappers.
- No login, password, server, or terminal path is hard-coded; only an optional caller/CLI terminal path is accepted.
- Supported timeframes are exactly M1, M5, M15, M30, H1, H4, and D1.
- Exact symbol matches win; normalized broker variants are considered only when they preserve the requested forex base; ambiguity and missing symbols are explicit errors.
- Tick timestamps prefer time_msc, fall back to time, and are timezone-aware UTC datetimes.
- Mt5SymbolInfo includes digits and point.
- Existing TradingAgents dataflow routing, LangGraph tool registration, and agent workflow are not modified.
- Automated tests never require a terminal by default; the live integration test is opt-in and never sends an order.

---

### Task 1: Declare the optional MT5 dependency

**Files:**
- Modify: pyproject.toml

**Interfaces:**
- Produces the install extra tradingagents[mt5], containing MetaTrader5>=5.0.45.

- [ ] **Step 1: Add the optional dependency without changing core dependencies**

Add this entry beside the existing bedrock extra:

~~~toml
mt5 = [
    "MetaTrader5>=5.0.45",
]
~~~

- [ ] **Step 2: Install the extra in the repository virtual environment**

Run from C:\AITrading\TradingAgents:

~~~powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[mt5]'
~~~

Expected: pip exits 0 and installs/imports the official MetaTrader5 distribution on this Windows machine. If the package is unavailable for this interpreter/platform, retain the optional extra and record the exact pip error for the final report; do not make it a mandatory dependency.

- [ ] **Step 3: Verify the dependency is importable**

~~~powershell
& '.\.venv\Scripts\python.exe' -c "import MetaTrader5 as mt5; print(mt5.__version__ if hasattr(mt5, '__version__') else 'MetaTrader5 import OK')"
~~~

Expected: a version or MetaTrader5 import OK.

- [ ] **Step 4: Commit**

~~~powershell
git add pyproject.toml
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "build: add optional MetaTrader5 dependency"
~~~

---

### Task 2: Write model and timeframe tests first (RED)

**Files:**
- Create: tests/test_mt5_models.py

**Interfaces:**
- Consumes the public names that the implementation must provide from tradingagents.dataflows.mt5.models and tradingagents.dataflows.mt5.timeframes.
- Produces failing tests for the normalized dataclass shape, UTC timestamp contract, and seven-name timeframe contract.

- [ ] **Step 1: Write the failing tests**

~~~python
from datetime import datetime, timezone

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar, Mt5SymbolInfo, Mt5Tick
from tradingagents.dataflows.mt5.timeframes import SUPPORTED_TIMEFRAMES, TIMEFRAME_ATTRIBUTES


@pytest.mark.unit
def test_supported_timeframes_have_mt5_constant_names():
    assert SUPPORTED_TIMEFRAMES == ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
    assert TIMEFRAME_ATTRIBUTES["M5"] == "TIMEFRAME_M5"
    assert TIMEFRAME_ATTRIBUTES["D1"] == "TIMEFRAME_D1"


@pytest.mark.unit
def test_symbol_info_carries_digits_and_point():
    info = Mt5SymbolInfo(name="EURUSD.a", digits=5, point=0.00001)
    assert info.name == "EURUSD.a"
    assert info.digits == 5
    assert info.point == 0.00001


@pytest.mark.unit
def test_snapshot_uses_immutable_candle_tuples_and_utc_timestamp():
    timestamp = datetime(2026, 9, 8, 10, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(timestamp, 1.0, 1.1, 0.9, 1.05, 10, 2, 10)
    tick = Mt5Tick("EURUSD.a", timestamp, 1.04999, 1.05001)
    snapshot = ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSD.a",
        bid=tick.bid,
        ask=tick.ask,
        spread=0.00002,
        spread_points=2.0,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=None,
        positions=(),
    )
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert isinstance(snapshot.m5_candles, tuple)
    assert snapshot.m5_candles[0].close == 1.05
~~~

- [ ] **Step 2: Run the tests and verify the expected failure**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_models.py -q
~~~

Expected: collection fails because tradingagents.dataflows.mt5 does not exist yet. This is the required RED state, not a test typo.

- [ ] **Step 3: Commit the RED tests**

~~~powershell
git add tests/test_mt5_models.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "test: specify MT5 normalized models and timeframes"
~~~

---

### Task 3: Implement models, errors, timeframe mapping, and package exports (GREEN)

**Files:**
- Create: tradingagents/dataflows/mt5/__init__.py
- Create: tradingagents/dataflows/mt5/models.py
- Create: tradingagents/dataflows/mt5/errors.py
- Create: tradingagents/dataflows/mt5/timeframes.py
- Test: tests/test_mt5_models.py

**Interfaces:**
- Produces frozen dataclasses Mt5TerminalInfo, Mt5AccountInfo, Mt5SymbolInfo, Mt5Tick, Mt5Bar, Mt5Position, Mt5Order, Mt5Spread, and ForexMarketSnapshot.
- Produces errors Mt5ProviderError, Mt5DependencyError, Mt5InitializationError, Mt5NotConnectedError, Mt5TimeframeError, Mt5SymbolError, Mt5SymbolNotFoundError, Mt5SymbolAmbiguousError, Mt5AccountError, Mt5AccountDisconnectedError, and Mt5DataError.
- Produces SUPPORTED_TIMEFRAMES, TIMEFRAME_ATTRIBUTES, normalize_timeframe(), and resolve_timeframe().

- [ ] **Step 1: Implement the immutable model fields**

Use @dataclass(frozen=True, slots=True). At minimum, use these constructor fields (optional metadata may default to None):

~~~python
class Mt5SymbolInfo:
    name: str
    description: str | None = None
    digits: int | None = None
    point: float | None = None
    visible: bool | None = None
    trade_mode: int | None = None
    currency_base: str | None = None
    currency_profit: str | None = None


class Mt5Tick:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: int | None = None
    volume_real: float | None = None


class Mt5Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int | None = None
    real_volume: int | None = None
~~~

Mt5TerminalInfo, Mt5AccountInfo, Mt5Position, Mt5Order, Mt5Spread, and ForexMarketSnapshot use the fields defined in the approved spec; ForexMarketSnapshot has explicit m1_candles, m5_candles, m15_candles, h1_candles, account, and positions tuple fields.

- [ ] **Step 2: Implement typed provider errors and timeframe resolution**

~~~python
SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
TIMEFRAME_ATTRIBUTES = {name: f"TIMEFRAME_{name}" for name in SUPPORTED_TIMEFRAMES}


def normalize_timeframe(value: str) -> str:
    key = str(value).strip().upper()
    if key not in TIMEFRAME_ATTRIBUTES:
        raise Mt5TimeframeError(
            f"Unsupported MT5 timeframe {value!r}; supported: {', '.join(SUPPORTED_TIMEFRAMES)}"
        )
    return key


def resolve_timeframe(value: str, api) -> int:
    key = normalize_timeframe(value)
    try:
        return getattr(api, TIMEFRAME_ATTRIBUTES[key])
    except AttributeError as exc:
        raise Mt5DependencyError(
            f"MT5 API does not expose {TIMEFRAME_ATTRIBUTES[key]}"
        ) from exc
~~~

- [ ] **Step 3: Re-export the public model/error/timeframe names**

tradingagents/dataflows/mt5/__init__.py must export all model classes, errors, SUPPORTED_TIMEFRAMES, TIMEFRAME_ATTRIBUTES, normalize_timeframe, and resolve_timeframe; it must not import MetaTrader5 eagerly.

- [ ] **Step 4: Run the model tests and verify GREEN**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_models.py -q
~~~

Expected: all model/timeframe tests pass.

- [ ] **Step 5: Run Ruff on the new package**

~~~powershell
& '.\.venv\Scripts\python.exe' -m ruff check tradingagents/dataflows/mt5 tests/test_mt5_models.py
~~~

Expected: exit 0.

- [ ] **Step 6: Commit**

~~~powershell
git add tradingagents/dataflows/mt5 tests/test_mt5_models.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "feat: add normalized MT5 models and timeframe mapping"
~~~

---

### Task 4: Write provider behavior and safety tests first (RED)

**Files:**
- Create: tests/test_mt5_provider.py

**Interfaces:**
- Consumes the public models/timeframe/error names from Task 3.
- Produces a deterministic fake MT5 module and failing tests for lifecycle, connection errors, flexible symbol matching, tick/bar/account/spread normalization, and the explicit read-only surface.

- [ ] **Step 1: Add a fake MT5 API fixture**

The fake exposes timeframe constants and only read methods (initialize, shutdown, terminal_info, account_info, symbols_get, symbol_select, symbol_info, symbol_info_tick, copy_rates_from_pos, positions_get, orders_get, last_error). Include symbols EURUSD.a, EURUSDm, USDJPY, and a symbol with a separator suffix such as EURUSD.raw; return deterministic records with digits=5 and point=0.00001.

- [ ] **Step 2: Write the failing provider tests**

~~~python
@pytest.mark.unit
def test_initialize_success_and_shutdown(fake_api):
    provider = MT5Provider(api=fake_api)
    assert provider.initialize() is True
    assert provider.is_connected() is True
    provider.shutdown()
    assert provider.is_connected() is False


@pytest.mark.unit
def test_initialize_failure_surfaces_last_error(fake_api):
    fake_api.initialize_result = False
    fake_api.error = (10004, "terminal unavailable")
    with pytest.raises(Mt5InitializationError, match="terminal unavailable"):
        MT5Provider(api=fake_api).initialize()


@pytest.mark.unit
def test_exact_symbol_match_wins_over_variants(fake_api):
    provider = initialized_provider(fake_api)
    assert provider.find_symbol("eurusd.a") == "EURUSD.a"


@pytest.mark.unit
def test_normalized_variant_is_resolved_without_fixed_suffix_allowlist(fake_api):
    fake_api.symbol_records = [fake_api.symbol("EURUSD.raw")]
    provider = initialized_provider(fake_api)
    assert provider.find_symbol(" EURUSD ") == "EURUSD.raw"


@pytest.mark.unit
def test_multiple_plausible_variants_raise_ambiguity(fake_api):
    provider = initialized_provider(fake_api)
    with pytest.raises(Mt5SymbolAmbiguousError, match="EURUSD"):
        provider.find_symbol("EURUSD")


@pytest.mark.unit
def test_missing_symbol_raises_not_found(fake_api):
    provider = initialized_provider(fake_api)
    with pytest.raises(Mt5SymbolNotFoundError, match="GBPUSD"):
        provider.find_symbol("GBPUSD")


@pytest.mark.unit
def test_tick_prefers_millisecond_timestamp(fake_api):
    provider = initialized_provider(fake_api)
    tick = provider.get_tick("USDJPY")
    assert tick.timestamp == datetime.fromtimestamp(1700000000.123, tz=timezone.utc)
    assert tick.bid == 150.123


@pytest.mark.unit
def test_tick_falls_back_to_second_timestamp(fake_api):
    fake_api.tick_time_msc = None
    provider = initialized_provider(fake_api)
    assert provider.get_tick("USDJPY").timestamp == datetime.fromtimestamp(
        1700000000, tz=timezone.utc
    )


@pytest.mark.unit
def test_bars_account_and_spread_are_normalized(fake_api):
    provider = initialized_provider(fake_api)
    bars = provider.get_bars("USDJPY", "M5", 2)
    account = provider.get_account_info()
    spread = provider.get_spread("USDJPY")
    assert len(bars) == 2 and bars[-1].close == 150.125
    assert account.balance == 10_000.0 and account.free_margin == 9_500.0
    assert spread.price == pytest.approx(0.003)
    assert spread.points == pytest.approx(30.0)


@pytest.mark.unit
def test_provider_exposes_no_mutation_or_execution_api(fake_api):
    provider = MT5Provider(api=fake_api)
    forbidden = {
        "order_send", "buy", "sell", "open_position", "close_position",
        "modify_position", "modify_order", "place_order", "cancel_order",
    }
    assert forbidden.isdisjoint(dir(provider))
~~~

The safety assertion is intentionally explicit and must remain even if future methods are added.

- [ ] **Step 3: Run the provider tests and verify the expected RED state**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_provider.py -q
~~~

Expected: collection or test failures because provider.py and MT5Provider do not exist yet.

- [ ] **Step 4: Commit the RED tests**

~~~powershell
git add tests/test_mt5_provider.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "test: define read-only MT5 provider behavior"
~~~

---

### Task 5: Implement the lifecycle, symbol resolution, and read-only provider (GREEN)

**Files:**
- Create: tradingagents/dataflows/mt5/provider.py
- Modify: tradingagents/dataflows/mt5/__init__.py
- Test: tests/test_mt5_provider.py

**Interfaces:**
- Produces MT5Provider and the compatibility alias Mt5Provider with the public methods from the approved spec.
- Consumes the model/error/timeframe modules from Task 3 and an optional injected API object.

- [ ] **Step 1: Implement lazy dependency loading and lifecycle guards**

Use importlib.import_module("MetaTrader5") only when initialize() is called. On ModuleNotFoundError, raise Mt5DependencyError with the install command. initialize() calls api.initialize() (or api.initialize(path=terminal_path) when a path was provided), sets internal state only after success, checks terminal/account connectivity, and raises Mt5InitializationError with last_error() detail on failure. shutdown() is idempotent; is_connected() returns False when uninitialized, terminal info is absent/disconnected, or account info is absent.

- [ ] **Step 2: Implement safe raw-field and UTC normalization helpers**

~~~python
def _field(raw, name: str, default=None):
    if raw is None:
        return default
    if isinstance(raw, Mapping):
        return raw.get(name, default)
    try:
        return getattr(raw, name)
    except AttributeError:
        try:
            return raw[name]
        except (KeyError, IndexError, TypeError):
            return default


def _utc_timestamp(raw, *, prefer_msc: bool = False) -> datetime:
    value = _field(raw, "time_msc") if prefer_msc else None
    if value in (None, 0):
        value = _field(raw, "time")
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
~~~

The tick normalizer always calls _utc_timestamp(raw, prefer_msc=True); bars and position/order timestamps use second-based time fields.

- [ ] **Step 3: Implement flexible but fail-closed symbol resolution**

Normalize input by trimming and uppercasing. First return the original broker name for an exact case-insensitive match. Otherwise derive a six-letter forex base only when the first six characters are alphabetic; accept a candidate when it has the same base and its remainder is either a single alphanumeric marker or a separator-led broker suffix made of alphanumerics/separators. Reject multi-letter suffixes with no separator (for example, EURUSDJPY) so an unrelated instrument cannot be guessed. Return one candidate, raise Mt5SymbolAmbiguousError with all candidates for more than one, and raise Mt5SymbolNotFoundError for none. ensure_symbol() calls symbol_select(resolved, True) and raises Mt5SymbolError if selection fails.

- [ ] **Step 4: Implement read-only fetch and normalization methods**

Implement get_terminal_info, get_account_info, get_symbols, find_symbol, ensure_symbol, get_tick, get_bars, get_positions, get_orders, and get_spread. get_bars validates count > 0, maps the timeframe through resolve_timeframe, calls copy_rates_from_pos, and raises Mt5DataError for no/invalid rows. get_spread reads the symbol's normalized point and returns Mt5Spread(symbol, price=ask-bid, points=(ask-bid)/point). Positions/orders return empty tuples when there are no open records and filter by the resolved symbol when requested.

- [ ] **Step 5: Export the provider without importing it into LangGraph**

Add only MT5Provider and Mt5Provider to tradingagents.dataflows.mt5.__init__. Do not edit tradingagents/dataflows/interface.py, tradingagents/agents/utils/agent_utils.py, tradingagents/graph/trading_graph.py, or any graph setup file.

- [ ] **Step 6: Run provider tests and verify GREEN**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_models.py tests/test_mt5_provider.py -q
~~~

Expected: all model and provider tests pass, including the explicit mutation-surface test.

- [ ] **Step 7: Commit**

~~~powershell
git add tradingagents/dataflows/mt5 tests/test_mt5_provider.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "feat: add read-only MT5 provider"
~~~

---

### Task 6: Write snapshot and CLI rendering tests first (RED)

**Files:**
- Modify: tests/test_mt5_provider.py
- Create: tests/test_mt5_cli.py

**Interfaces:**
- Consumes MT5Provider.get_market_snapshot() and the future CLI render_report() function.
- Produces failing tests for required M1/M5/M15/H1 series, symbol-relevant positions, and secret-free output.

- [ ] **Step 1: Add the failing snapshot test**

~~~python
@pytest.mark.unit
def test_market_snapshot_contains_required_series_and_symbol_positions(fake_api):
    provider = initialized_provider(fake_api)
    snapshot = provider.get_market_snapshot("USDJPY", count=2)
    assert snapshot.symbol == "USDJPY"
    assert len(snapshot.m1_candles) == 2
    assert len(snapshot.m5_candles) == 2
    assert len(snapshot.m15_candles) == 2
    assert len(snapshot.h1_candles) == 2
    assert snapshot.account.balance == 10_000.0
    assert all(position.symbol == "USDJPY" for position in snapshot.positions)
    assert snapshot.ask >= snapshot.bid
~~~

- [ ] **Step 2: Add the failing CLI rendering test**

Create a deterministic ForexMarketSnapshot fixture and assert:

~~~python
from scripts.test_mt5_connection import render_report


@pytest.mark.unit
def test_render_report_is_secret_free(snapshot_fixture):
    output = render_report(snapshot_fixture, terminal_name="MetaTrader 5")
    assert "MT5 CONNECTED" in output
    assert "Resolved symbol:" in output
    assert "M5:" in output and "Last candle:" in output
    assert "Balance:" in output and "Free margin:" in output
    assert "password" not in output.lower()
    assert "order_send" not in output
~~~

- [ ] **Step 3: Run the tests and verify the expected RED state**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_provider.py tests/test_mt5_cli.py -q
~~~

Expected: the snapshot method and CLI module are missing, so the tests fail for the intended feature gaps.

- [ ] **Step 4: Commit the RED tests**

~~~powershell
git add tests/test_mt5_provider.py tests/test_mt5_cli.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "test: specify MT5 market snapshot and smoke output"
~~~

---

### Task 7: Implement snapshot assembly, smoke CLI, and documentation (GREEN)

**Files:**
- Modify: tradingagents/dataflows/mt5/provider.py
- Create: scripts/test_mt5_connection.py
- Create: docs/mt5-provider.md
- Test: tests/test_mt5_provider.py
- Test: tests/test_mt5_cli.py

**Interfaces:**
- MT5Provider.get_market_snapshot(symbol: str, count: int = 100) -> ForexMarketSnapshot
- render_report(snapshot: ForexMarketSnapshot, terminal_name: str | None = None) -> str
- CLI command: python scripts/test_mt5_connection.py --symbol EURUSD [--count N] [--terminal-path PATH]

- [ ] **Step 1: Implement get_market_snapshot() with one resolved symbol**

Resolve/select the symbol once, read its tick and symbol info, fetch M1, M5, M15, and H1 bars using the private resolved-symbol helper, read account information, filter positions to the resolved symbol, compute spread from the tick and point, and return a ForexMarketSnapshot timestamped with the tick's UTC timestamp. Raise a typed error for missing required bars/account/tick data instead of returning a partial snapshot.

- [ ] **Step 2: Implement secret-free report rendering**

render_report() must emit these sections and values only: MT5 CONNECTED, terminal name/connection state, account server/currency/balance/equity/free margin, requested/resolved symbol, bid/ask/spread/spread points, the latest M5 OHLCV candle, and open positions (or None). It must not print login passwords, environment variables, raw MT5 objects, or mutation method names.

- [ ] **Step 3: Implement the argparse CLI with graceful failure**

~~~python
parser.add_argument("--symbol", default="EURUSD")
parser.add_argument("--count", type=int, default=100)
parser.add_argument("--terminal-path", default=None)
~~~

Instantiate MT5Provider(terminal_path=args.terminal_path), initialize it, build the snapshot, fetch terminal info, print render_report(), catch Mt5ProviderError and print the provider error message prefixed with MT5 ERROR: to stderr with exit code 2, and always call shutdown() in finally. Reject non-positive --count through argparse/provider validation. No credentials are accepted or passed.

- [ ] **Step 4: Write the provider usage documentation**

docs/mt5-provider.md must explain:

~~~markdown
pip install -e ".[mt5]"
python scripts/test_mt5_connection.py --symbol EURUSD
~~~

Include the supported timeframe table, exact-first/flexible-suffix symbol resolution and ambiguity behavior, terminal/account initialization failure messages, the optional --terminal-path, the opt-in integration test command, and a bold statement that Phase 3 is read-only and does not register LangGraph tools.

- [ ] **Step 5: Run snapshot and CLI tests and verify GREEN**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_models.py tests/test_mt5_provider.py tests/test_mt5_cli.py -q
~~~

Expected: all focused tests pass.

- [ ] **Step 6: Run Ruff on all changed Python files**

~~~powershell
& '.\.venv\Scripts\python.exe' -m ruff check tradingagents/dataflows/mt5 scripts/test_mt5_connection.py tests/test_mt5_models.py tests/test_mt5_provider.py tests/test_mt5_cli.py
~~~

Expected: exit 0.

- [ ] **Step 7: Commit**

~~~powershell
git add tradingagents/dataflows/mt5/provider.py scripts/test_mt5_connection.py docs/mt5-provider.md tests/test_mt5_provider.py tests/test_mt5_cli.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "feat: add MT5 market snapshot smoke CLI and docs"
~~~

---

### Task 8: Add guarded integration coverage and perform full verification

**Files:**
- Create: tests/test_mt5_integration.py
- Verify: pyproject.toml, tradingagents/dataflows/mt5/, scripts/test_mt5_connection.py, docs/mt5-provider.md

**Interfaces:**
- Produces an opt-in integration test controlled by RUN_MT5_INTEGRATION=1 and MT5_TEST_SYMBOL (default EURUSD).
- Verifies no existing agent/LangGraph files were modified and no mutation API is present.

- [ ] **Step 1: Write the guarded integration test**

~~~python
@pytest.mark.integration
def test_live_mt5_read_only_snapshot():
    if os.getenv("RUN_MT5_INTEGRATION") != "1":
        pytest.skip("set RUN_MT5_INTEGRATION=1 to use the local terminal")
    provider = MT5Provider()
    try:
        provider.initialize()
        snapshot = provider.get_market_snapshot(os.getenv("MT5_TEST_SYMBOL", "EURUSD"), count=2)
        assert snapshot.bid > 0
        assert snapshot.ask >= snapshot.bid
        assert snapshot.m5_candles
        assert snapshot.account.balance >= 0
    finally:
        provider.shutdown()
~~~

The test contains no call or reference to any mutation method and assumes the terminal is already open and logged into the demo account.

- [ ] **Step 2: Run the complete existing and new unit suite**

~~~powershell
& '.\.venv\Scripts\python.exe' -m pytest -m "not integration" -q
~~~

Expected: exit 0 with zero failures and no new warnings/errors.

- [ ] **Step 3: Run the guarded integration test when the terminal is accessible**

~~~powershell
$env:RUN_MT5_INTEGRATION = '1'
& '.\.venv\Scripts\python.exe' -m pytest tests/test_mt5_integration.py -m integration -q
Remove-Item Env:RUN_MT5_INTEGRATION
~~~

Expected when MT5 is open and connected: PASS. If the terminal is unavailable, capture the typed failure and use the smoke CLI to report the same limitation; never bypass the error or send an order.

- [ ] **Step 4: Run the real smoke command**

~~~powershell
& '.\.venv\Scripts\python.exe' scripts/test_mt5_connection.py --symbol EURUSD
~~~

Record the actual connection result, resolved broker symbol, bid/ask, latest M5 candle, account values, and positions for the final report. Do not print secrets and do not alter terminal/account state.

- [ ] **Step 5: Verify the read-only and scope boundaries**

~~~powershell
rg -n "order_send|open_position|close_position|modify_position|place_order|cancel_order" tradingagents/dataflows/mt5 scripts/test_mt5_connection.py
git diff --name-only HEAD~8..HEAD
git status --short
~~~

Expected: the provider source has no mutation wrappers, the only mentions are safety tests/docs if present, no LangGraph/agent registration files appear in the implementation commits, and the worktree is clean except for intentionally user-owned files.

- [ ] **Step 6: Commit the integration test**

~~~powershell
git add tests/test_mt5_integration.py
git -c user.name='Codex' -c user.email='codex@localhost' commit -m "test: add opt-in MT5 integration smoke"
~~~
