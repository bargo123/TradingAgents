from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from cli.forex_supervisor import build_parser
from tradingagents.dataflows.mt5.errors import Mt5BrokerClockError
from tradingagents.dataflows.mt5.models import Mt5Tick
from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
from tradingagents.forex.hft.demo_runtime import DemoHftRuntime, DemoRuntimeConfig
from tradingagents.forex.hft.demo_store import DemoExecutionStore
from tradingagents.forex.hft.market_lifecycle import MarketReason, MarketState
from tradingagents.forex.hft.plan_store import AtomicPlanStore
from tradingagents.forex.hft.regime_store import AtomicRegimeStore
from tradingagents.forex.watcher import SerializedMt5OperationGate

UTC_NOW = datetime(2026, 6, 1, 10, tzinfo=UTC)


def _calendar_file(path: Path, *, session: tuple[str, str] = ("09:00", "17:00")) -> Path:
    data = {
        "schema_version": 1,
        "calendar_version": "fixture-1",
        "broker_server": "Broker-Demo",
        "symbol": "EURUSD",
        "timezone": "UTC",
        "valid_from": "2026-01-01",
        "valid_through": "2026-12-31",
        "weekly_sessions": {"mon": [{"open": session[0], "close": session[1]}]},
        "holidays": [],
        "date_overrides": {},
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class _Provider:
    def __init__(self, identities: tuple[tuple[int, int] | None, ...], *, fail_clock: bool = False):
        self.identities = list(identities)
        self.fail_clock = fail_clock
        self.calls: list[str] = []
        self.trade_mode = 0
        self.connected = True
        self.clock_config = SimpleNamespace(max_tick_age_seconds=120, max_future_skew_seconds=2)

    def is_connected(self):
        self.calls.append("connected")
        return self.connected

    def get_account_info(self):
        self.calls.append("account")
        return SimpleNamespace(
            login=123,
            server="Broker-Demo",
            currency="USD",
            trade_mode=self.trade_mode,
            balance=10000.0,
            equity=10000.0,
            margin=0.0,
            margin_free=10000.0,
            leverage=100,
        )

    def get_terminal_info(self):
        self.calls.append("terminal")
        return SimpleNamespace(company="Fixture", connected=self.connected)

    def probe_tick(self, symbol):
        self.calls.append("probe")
        identity = self.identities.pop(0) if len(self.identities) > 1 else self.identities[0]
        if identity is None:
            return SimpleNamespace(symbol=symbol, available=False, identity=None)
        seconds, milliseconds = identity
        return SimpleNamespace(
            symbol=symbol,
            available=True,
            identity=(seconds, milliseconds),
        )

    def calibrate_broker_clock(self, _symbol):
        self.calls.append("calibrate")
        if self.fail_clock:
            raise Mt5BrokerClockError("clock calibration failed")
        return SimpleNamespace(is_calibrated=True)

    def get_tick(self, symbol):
        self.calls.append("tick")
        return Mt5Tick(symbol, UTC_NOW, 1.1, 1.1001)

    def get_positions(self, _symbol=None):
        self.calls.append("positions")
        return ()

    def get_orders(self, _symbol=None):
        self.calls.append("orders")
        return ()

    def initialize(self):
        self.calls.append("initialize")
        return True

    def reinitialize(self):
        self.calls.append("reinitialize")
        return True


class _BrokerApi:
    ACCOUNT_TRADE_MODE_DEMO = 0

    def __init__(self):
        self.order_calls = 0

    def order_send(self, _request):
        self.order_calls += 1
        raise AssertionError("market lifecycle must not send orders")


def _runtime(tmp_path: Path, provider: _Provider, *, calendar_path: Path) -> tuple[DemoHftRuntime, _BrokerApi]:
    store = DemoExecutionStore(tmp_path / "demo.sqlite3")
    api = _BrokerApi()
    gateway = VerifiedDemoExecutionGateway(
        provider,
        api,
        store,
        run_id="market-test",
        demo_trade_mode=0,
        magic=12012012,
        gate=SerializedMt5OperationGate(),
    )
    runtime = DemoHftRuntime(
        provider,
        AtomicPlanStore(),
        gateway=gateway,
        store=store,
        regime_store=AtomicRegimeStore(),
        config=DemoRuntimeConfig(
            artifact_path=tmp_path / "demo.sqlite3",
            symbol="EURUSD",
            hft_first=True,
            market_calendar_path=calendar_path,
        ),
        clock=lambda: UTC_NOW,
    )
    return runtime, api


def test_market_closed_runtime_idles_without_tick_or_reinitialization(tmp_path: Path) -> None:
    calendar = _calendar_file(tmp_path / "calendar.json", session=("11:00", "17:00"))
    provider = _Provider(((1_780_312_800, 1_780_312_800_000),))
    runtime, api = _runtime(tmp_path, provider, calendar_path=calendar)

    initial = runtime.run_once()
    result = runtime.run_once()

    assert initial["status"] == "MARKET_UNKNOWN"
    assert result["status"] == "MARKET_CLOSED"
    assert result["action"] == "NO_ACTION"
    assert runtime.market_status is MarketState.MARKET_CLOSED
    assert "tick" not in provider.calls
    assert "initialize" not in provider.calls
    assert "reinitialize" not in provider.calls
    assert api.order_calls == 0


def test_stale_probe_identity_does_not_trigger_reopen_validation(tmp_path: Path) -> None:
    calendar = _calendar_file(tmp_path / "calendar.json")
    stale = (1_780_312_800, 1_780_312_800_000)
    provider = _Provider((stale, stale))
    runtime, api = _runtime(tmp_path, provider, calendar_path=calendar)

    first = runtime.run_once()
    second = runtime.run_once()

    assert first["status"] == second["status"] == "MARKET_UNKNOWN"
    assert runtime.market_reason is MarketReason.STALE_DATA
    assert "calibrate" not in provider.calls
    assert "tick" not in provider.calls
    assert api.order_calls == 0


def test_open_session_stale_probe_uses_hft_poll_interval(tmp_path: Path) -> None:
    calendar = _calendar_file(tmp_path / "calendar.json")
    provider = _Provider(((100, 100_000),))
    runtime, _api = _runtime(tmp_path, provider, calendar_path=calendar)
    runtime.config = DemoRuntimeConfig(
        artifact_path=tmp_path / "demo.sqlite3",
        symbol="EURUSD",
        hft_first=True,
        poll_interval_seconds=0.05,
        market_closed_poll_interval_seconds=2.0,
        market_calendar_path=calendar,
    )

    assert runtime._poll_delay_seconds(
        {"status": "MARKET_UNKNOWN", "market_reason": MarketReason.STALE_DATA.value}
    ) == 0.05


def test_genuinely_changed_probe_requires_full_reopen_validation(tmp_path: Path) -> None:
    calendar = _calendar_file(tmp_path / "calendar.json")
    provider = _Provider(((100, 100_000), (100, 100_000), (101, 101_000)))
    runtime, api = _runtime(tmp_path, provider, calendar_path=calendar)

    assert runtime.run_once()["status"] == "MARKET_UNKNOWN"
    assert runtime.run_once()["status"] == "MARKET_UNKNOWN"
    result = runtime.run_once()

    assert runtime.market_status is MarketState.MARKET_OPEN
    assert result["status"] == "PROCESSED"
    assert provider.calls.index("calibrate") < provider.calls.index("tick")
    assert provider.calls.index("tick") < provider.calls.index("positions")
    assert provider.calls.index("positions") < provider.calls.index("orders")
    assert "initialize" not in provider.calls
    assert "reinitialize" not in provider.calls
    assert api.order_calls == 0


def test_reopen_validation_failure_stays_fail_closed(tmp_path: Path) -> None:
    calendar = _calendar_file(tmp_path / "calendar.json")
    provider = _Provider(((100, 100_000), (101, 101_000)), fail_clock=True)
    runtime, api = _runtime(tmp_path, provider, calendar_path=calendar)

    runtime.run_once()
    result = runtime.run_once()

    assert result["status"] == "MARKET_DATA_ERROR"
    assert runtime.market_status is MarketState.MARKET_DATA_ERROR
    assert runtime.market_reason is MarketReason.BROKER_CLOCK_ERROR
    assert "tick" not in provider.calls
    assert api.order_calls == 0


def test_market_closed_poll_interval_is_bounded_and_positive(tmp_path: Path) -> None:
    from tradingagents.forex.hft.demo_runtime import DemoRuntimeConfig

    assert DemoRuntimeConfig(market_closed_poll_interval_seconds=2).market_closed_poll_interval_seconds == 2
    for invalid in (0, -1, 61, float("inf")):
        with pytest.raises(ValueError, match="market_closed_poll_interval_seconds"):
            DemoRuntimeConfig(market_closed_poll_interval_seconds=invalid)


def test_cli_accepts_explicit_market_calendar_only_for_supervisor_run() -> None:
    args = build_parser().parse_args(["run", "--market-session-calendar", "broker.json"])

    assert args.market_session_calendar == "broker.json"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["status", "--market-session-calendar", "broker.json"])
