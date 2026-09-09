"""RED tests defining the read-only MT5 provider contract."""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tradingagents.dataflows.mt5.errors import (
    Mt5BrokerClockError,
    Mt5DataError,
    Mt5InitializationError,
    Mt5SymbolAmbiguousError,
    Mt5SymbolNotFoundError,
)
from tradingagents.dataflows.mt5.clock import Mt5BrokerClock
from tradingagents.dataflows.mt5.provider import MT5Provider


class FakeMT5:
    COPY_TICKS_ALL = -1
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_M30 = 30
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self):
        self.initialize_result = True
        self.error = (0, "")
        self.connected = False
        self.shutdown_called = False
        self.tick_time_msc = 1_700_000_000_123
        self.rate_timeframes = []
        self.tick_ranges = []
        self.tick_range_result = ()
        self.symbol_records = [
            self.symbol("EURUSD.a"),
            self.symbol("EURUSDm"),
            self.symbol("mEURUSD"),
            self.symbol("USDJPY"),
            self.symbol("EURUSD.raw"),
        ]

    @staticmethod
    def symbol(name):
        is_jpy = name.upper().endswith("USDJPY")
        return SimpleNamespace(
            name=name,
            description=f"{name} fake symbol",
            digits=3 if is_jpy else 5,
            point=0.001 if is_jpy else 0.00001,
            visible=True,
            trade_mode=0,
            currency_base="USD",
            currency_profit="JPY" if is_jpy else "USD",
        )

    def initialize(self):
        self.connected = self.initialize_result
        return self.initialize_result

    def shutdown(self):
        self.shutdown_called = True
        self.connected = False

    def terminal_info(self):
        return SimpleNamespace(
            name="MetaTrader 5",
            company="Fake Broker",
            version="5.00",
            build=4000,
            connected=self.connected,
        )

    def account_info(self):
        return SimpleNamespace(
            login=123456,
            server="Fake-Demo",
            currency="USD",
            balance=10_000.0,
            equity=9_900.0,
            profit=-100.0,
            margin=400.0,
            margin_free=9_500.0,
            leverage=100,
        )

    def symbols_get(self):
        return tuple(self.symbol_records)

    def symbol_select(self, name, enable=True):
        return any(record.name == name for record in self.symbol_records)

    def symbol_info(self, name):
        return next((record for record in self.symbol_records if record.name == name), None)

    def symbol_info_tick(self, name):
        is_jpy = name.upper().endswith("USDJPY")
        bid, ask = (150.123, 150.126) if is_jpy else (1.10000, 1.10020)
        return SimpleNamespace(
            time=1_700_000_000,
            time_msc=self.tick_time_msc,
            bid=bid,
            ask=ask,
            last=ask,
            volume=12,
            volume_real=12.0,
        )

    def copy_rates_from_pos(self, name, timeframe, start_pos, count):
        self.rate_timeframes.append(timeframe)
        is_jpy = name.upper().endswith("USDJPY")
        base = 150.123 if is_jpy else 1.10000
        step = 0.001 if is_jpy else 0.00001
        return tuple(
            {
                "time": 1_700_000_000 + index * 300,
                "open": base + index * step,
                "high": base + (index + 2) * step,
                "low": base - index * step,
                "close": base + (index + 1) * step,
                "tick_volume": 100 + index,
                "spread": 3,
                "real_volume": 100 + index,
            }
            for index in range(start_pos, start_pos + count)
        )

    def copy_ticks_range(self, name, start, end, flags):
        self.tick_ranges.append((name, start, end, flags))
        return self.tick_range_result

    def positions_get(self, **kwargs):
        return (
            SimpleNamespace(
                ticket=101,
                symbol="USDJPY",
                type=0,
                volume=0.10,
                price_open=150.100,
                price_current=150.123,
                profit=2.30,
                time=1_700_000_000,
            ),
            SimpleNamespace(
                ticket=202,
                symbol="EURUSD.a",
                type=1,
                volume=0.20,
                price_open=1.10020,
                price_current=1.10000,
                profit=-4.00,
                time=1_700_000_000,
            ),
        )

    def orders_get(self, **kwargs):
        return (
            SimpleNamespace(
                ticket=303,
                symbol="USDJPY",
                type=2,
                volume_current=0.15,
                price_open=150.120,
                price_current=150.123,
                sl=150.000,
                tp=150.500,
                time_setup=1_700_000_000,
            ),
        )

    def last_error(self):
        return self.error


@pytest.fixture
def fake_api():
    return FakeMT5()


def _zero_clock(symbol="EURUSD"):
    return Mt5BrokerClock(
        offset_seconds=0,
        status="CALIBRATED",
        calibrated_at_utc=datetime.now(timezone.utc),
        server="Fake-Demo",
        symbol=symbol,
        sample_count=1,
        max_residual_seconds=0.0,
        source="TEST",
    )


def initialized_provider(fake_api, *, broker_clock=None):
    provider = MT5Provider(api=fake_api, broker_clock=broker_clock or _zero_clock())
    provider.initialize()
    return provider


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
def test_initialize_post_check_failure_is_typed_and_shuts_down(fake_api):
    def terminal_info_failure():
        raise RuntimeError("inspection failed")

    fake_api.terminal_info = terminal_info_failure
    with pytest.raises(Mt5InitializationError, match="post-initialization"):
        MT5Provider(api=fake_api).initialize()
    assert fake_api.shutdown_called is True


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
def test_normalized_prefix_variant_is_resolved(fake_api):
    fake_api.symbol_records = [fake_api.symbol("mEURUSD")]
    provider = initialized_provider(fake_api)
    assert provider.find_symbol("EURUSD") == "mEURUSD"


@pytest.mark.unit
def test_hash_broker_decoration_is_resolved(fake_api):
    fake_api.symbol_records = [fake_api.symbol("EURUSD#")]
    provider = initialized_provider(fake_api)
    assert provider.find_symbol("EURUSD") == "EURUSD#"


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
def test_repeated_forex_core_is_not_a_valid_broker_variant(fake_api):
    fake_api.symbol_records = [fake_api.symbol("EURUSD.EURUSD")]
    provider = initialized_provider(fake_api)
    with pytest.raises(Mt5SymbolNotFoundError):
        provider.find_symbol("EURUSD")


@pytest.mark.unit
def test_tick_prefers_millisecond_timestamp(fake_api):
    provider = initialized_provider(fake_api)
    tick = provider.get_tick("USDJPY")
    assert tick.timestamp == datetime.fromtimestamp(1_700_000_000.123, tz=timezone.utc)
    assert tick.bid == 150.123


@pytest.mark.unit
def test_tick_falls_back_to_second_timestamp(fake_api):
    fake_api.tick_time_msc = None
    provider = initialized_provider(fake_api)
    assert provider.get_tick("USDJPY").timestamp == datetime.fromtimestamp(
        1_700_000_000, tz=timezone.utc
    )


def test_provider_requires_calibrated_broker_clock_before_exposing_tick(fake_api):
    provider = MT5Provider(api=fake_api)
    provider.initialize()

    with pytest.raises(Mt5BrokerClockError, match="broker clock calibration"):
        provider.get_tick("USDJPY")


def test_provider_rejects_stale_injected_broker_clock(fake_api):
    stale_clock = Mt5BrokerClock(
        offset_seconds=0,
        status="CALIBRATED",
        calibrated_at_utc=datetime.now(timezone.utc) - timedelta(hours=2),
        server="Fake-Demo",
        symbol="USDJPY",
        sample_count=1,
        max_residual_seconds=0.0,
        source="TEST",
    )
    provider = initialized_provider(fake_api, broker_clock=stale_clock)

    with pytest.raises(Mt5BrokerClockError, match="stale"):
        provider.get_tick("USDJPY")


def test_provider_normalizes_each_timestamped_surface_once(fake_api):
    clock = Mt5BrokerClock(
        offset_seconds=3 * 3600,
        status="CALIBRATED",
        calibrated_at_utc=datetime.now(timezone.utc),
        server="Fake-Demo",
        symbol="USDJPY",
        sample_count=1,
        max_residual_seconds=0.0,
        source="TEST",
    )
    provider = initialized_provider(fake_api, broker_clock=clock)
    tick_expected = datetime.fromtimestamp(1_700_000_000.123, tz=timezone.utc) - timedelta(hours=3)
    bar_expected = datetime.fromtimestamp(1_700_000_000, tz=timezone.utc) - timedelta(hours=3)

    tick = provider.get_tick("USDJPY")
    spread = provider.get_spread("USDJPY")
    bars = provider.get_bars("USDJPY", "M5", 1)
    snapshot = provider.get_market_snapshot("USDJPY", count=1)

    assert tick.timestamp == tick_expected
    assert spread.timestamp == tick_expected
    assert bars[0].timestamp == bar_expected
    assert snapshot.timestamp == tick_expected
    assert snapshot.m1_candles[0].timestamp == bar_expected
    assert snapshot.broker_clock == clock


def test_provider_calibrates_from_fresh_live_tick_samples(fake_api):
    offset = 2 * 3600
    true_now = datetime(2026, 9, 9, 12, 0, tzinfo=timezone.utc)
    observation_times = iter(
        [
            true_now - timedelta(milliseconds=100),
            true_now + timedelta(milliseconds=100),
            true_now - timedelta(milliseconds=100),
            true_now + timedelta(milliseconds=100),
            true_now - timedelta(milliseconds=100),
            true_now + timedelta(milliseconds=100),
            true_now + timedelta(seconds=1),
        ]
    )
    fake_api.symbol_records = [fake_api.symbol("EURUSD")]
    fake_api.tick_time_msc = int((true_now + timedelta(seconds=offset - 0.5)).timestamp() * 1000)
    provider = MT5Provider(
        api=fake_api,
        clock_now=lambda: next(observation_times),
    )
    provider.initialize()

    tick = provider.get_tick("EURUSD")

    assert tick.timestamp == datetime(2026, 9, 9, 11, 59, 59, 500_000, tzinfo=timezone.utc)
    assert provider.broker_clock is not None
    assert provider.broker_clock.offset_seconds == offset
    assert provider.broker_clock.sample_count == 3


@pytest.mark.unit
def test_historical_ticks_prefer_milliseconds_and_normalize_utc(fake_api):
    fake_api.symbol_records = [fake_api.symbol("mEURUSD")]
    fake_api.tick_range_result = (
        {
            "time": 1_700_000_000,
            "time_msc": 1_700_000_000_123,
            "bid": 1.1000,
            "ask": 1.1002,
            "last": 1.1001,
            "volume": 3,
        },
        {
            "time": 1_700_000_001,
            "time_msc": None,
            "bid": 1.1001,
            "ask": 1.1003,
            "last": 1.1002,
            "volume": 4,
        },
    )
    provider = initialized_provider(fake_api)
    start = datetime(2023, 11, 14, 2, 0, tzinfo=timezone.utc)
    end = datetime(2023, 11, 14, 2, 1, tzinfo=timezone.utc)

    ticks = provider.get_ticks_range("EURUSD", start, end)

    assert len(ticks) == 2
    assert ticks[0].symbol == "mEURUSD"
    assert ticks[0].timestamp == datetime.fromtimestamp(
        1_700_000_000.123, tz=timezone.utc
    )
    assert ticks[1].timestamp == datetime.fromtimestamp(1_700_000_001, tz=timezone.utc)
    assert fake_api.tick_ranges == [("mEURUSD", start, end, fake_api.COPY_TICKS_ALL)]


@pytest.mark.unit
def test_historical_ticks_normalize_non_utc_inputs_and_use_flags(fake_api):
    fake_api.symbol_records = [fake_api.symbol("EURUSD.raw")]
    fake_api.tick_range_result = ()
    provider = initialized_provider(fake_api)
    start = datetime(2023, 11, 14, 4, 0, tzinfo=timezone(timedelta(hours=2)))
    end = datetime(2023, 11, 14, 4, 1, tzinfo=timezone(timedelta(hours=2)))

    assert provider.get_ticks_range("EURUSD", start, end, flags=7) == ()
    name, actual_start, actual_end, flags = fake_api.tick_ranges[-1]
    assert name == "EURUSD.raw"
    assert actual_start == datetime(2023, 11, 14, 2, 0, tzinfo=timezone.utc)
    assert actual_end == datetime(2023, 11, 14, 2, 1, tzinfo=timezone.utc)
    assert flags == 7


@pytest.mark.unit
def test_historical_ticks_reject_invalid_ranges_and_bad_rows(fake_api):
    fake_api.symbol_records = [fake_api.symbol("EURUSD")]
    provider = initialized_provider(fake_api)
    aware = datetime(2023, 11, 14, 2, 0, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        provider.get_ticks_range("EURUSD", datetime(2023, 11, 14, 2, 0), aware)
    with pytest.raises(ValueError, match="end"):
        provider.get_ticks_range("EURUSD", aware, aware - timedelta(seconds=1))

    fake_api.tick_range_result = None
    with pytest.raises(Mt5DataError, match="copy_ticks_range"):
        provider.get_ticks_range("EURUSD", aware, aware)
    fake_api.tick_range_result = ({"time": None, "time_msc": None, "bid": 1.1, "ask": 1.2},)
    with pytest.raises(Mt5DataError, match="tick range"):
        provider.get_ticks_range("EURUSD", aware, aware)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("timeframe_name", "expected_constant"),
    [
        ("M1", FakeMT5.TIMEFRAME_M1),
        ("M5", FakeMT5.TIMEFRAME_M5),
        ("M15", FakeMT5.TIMEFRAME_M15),
        ("M30", FakeMT5.TIMEFRAME_M30),
        ("H1", FakeMT5.TIMEFRAME_H1),
        ("H4", FakeMT5.TIMEFRAME_H4),
        ("D1", FakeMT5.TIMEFRAME_D1),
    ],
)
def test_get_bars_resolves_each_supported_timeframe(
    fake_api, timeframe_name, expected_constant
):
    provider = initialized_provider(fake_api)
    provider.get_bars("USDJPY", timeframe_name, 1)
    assert fake_api.rate_timeframes[-1] == expected_constant


@pytest.mark.unit
def test_bars_account_and_spread_are_normalized(fake_api):
    provider = initialized_provider(fake_api)
    bars = provider.get_bars("USDJPY", "M5", 2)
    account = provider.get_account_info()
    spread = provider.get_spread("USDJPY")
    assert len(bars) == 2 and bars[-1].close == 150.125
    assert account.balance == 10_000.0 and account.free_margin == 9_500.0
    assert spread.price == pytest.approx(0.003)
    assert spread.points == pytest.approx(3.0)


@pytest.mark.unit
def test_active_orders_use_mt5_time_setup_and_volume_current(fake_api):
    provider = initialized_provider(fake_api)
    orders = provider.get_orders("USDJPY")
    assert len(orders) == 1
    assert orders[0].volume == pytest.approx(0.15)
    assert orders[0].time == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


@pytest.mark.unit
def test_market_snapshot_contains_required_series_and_symbol_positions(fake_api):
    provider = initialized_provider(fake_api)
    snapshot = provider.get_market_snapshot("USDJPY", count=2)

    assert snapshot.symbol == "USDJPY"
    assert snapshot.timestamp == datetime.fromtimestamp(
        1_700_000_000.123, tz=timezone.utc
    )
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert snapshot.bid == pytest.approx(150.123)
    assert snapshot.ask == pytest.approx(150.126)
    assert snapshot.spread == pytest.approx(0.003)
    assert snapshot.spread_points == pytest.approx(3.0)
    assert len(snapshot.m1_candles) == 2
    assert len(snapshot.m5_candles) == 2
    assert len(snapshot.m15_candles) == 2
    assert len(snapshot.h1_candles) == 2
    assert all(
        candle.timestamp.tzinfo == timezone.utc
        for candles in (
            snapshot.m1_candles,
            snapshot.m5_candles,
            snapshot.m15_candles,
            snapshot.h1_candles,
        )
        for candle in candles
    )
    assert snapshot.account.login == 123456
    assert snapshot.account.server == "Fake-Demo"
    assert snapshot.account.currency == "USD"
    assert snapshot.account.balance == pytest.approx(10_000.0)
    assert snapshot.account.equity == pytest.approx(9_900.0)
    assert snapshot.account.profit == pytest.approx(-100.0)
    assert snapshot.account.margin == pytest.approx(400.0)
    assert snapshot.account.free_margin == pytest.approx(9_500.0)
    assert snapshot.account.leverage == 100
    assert len(snapshot.positions) == 1
    assert all(position.symbol == "USDJPY" for position in snapshot.positions)
    assert snapshot.positions[0].ticket == 101


@pytest.mark.unit
def test_provider_exposes_no_mutation_or_execution_api(fake_api):
    provider = MT5Provider(api=fake_api)
    forbidden = {
        "order_send",
        "buy",
        "sell",
        "open_position",
        "close_position",
        "modify_position",
        "modify_order",
        "place_order",
        "cancel_order",
    }
    assert forbidden.isdisjoint(dir(provider))


@pytest.mark.unit
@pytest.mark.parametrize("field", ["time", "bid", "ask"])
def test_malformed_tick_data_raises_typed_error(fake_api, field):
    provider = initialized_provider(fake_api)
    tick = fake_api.symbol_info_tick("USDJPY")
    setattr(tick, field, None)
    if field == "time":
        tick.time_msc = None
    fake_api.symbol_info_tick = lambda _name: tick
    with pytest.raises(Mt5DataError, match="Invalid tick data"):
        provider.get_tick("USDJPY")


@pytest.mark.unit
@pytest.mark.parametrize(("method", "record"), [
    ("get_positions", SimpleNamespace(ticket=None, symbol="USDJPY", time=1_700_000_000)),
    ("get_orders", SimpleNamespace(ticket=1, symbol="USDJPY", time=None)),
])
def test_malformed_position_or_order_data_raises_typed_error(fake_api, method, record):
    provider = initialized_provider(fake_api)
    setattr(fake_api, f"{method[4:]}_get", lambda: (record,))
    with pytest.raises(Mt5DataError, match="Invalid MT5"):
        getattr(provider, method)()


@pytest.mark.unit
def test_malformed_spread_data_raises_typed_error(fake_api):
    provider = initialized_provider(fake_api)
    fake_api.symbol_info_tick = lambda _name: SimpleNamespace(time=None, time_msc=None, bid=None, ask=1.1)
    with pytest.raises(Mt5DataError, match="Invalid spread data"):
        provider.get_spread("USDJPY")


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method", "api_method"),
    [
        ("get_symbols", "symbols_get"),
        ("get_positions", "positions_get"),
        ("get_orders", "orders_get"),
    ],
)
def test_none_mt5_collection_response_raises_typed_data_error(fake_api, method, api_method):
    provider = initialized_provider(fake_api)
    fake_api.error = (10006, "transport unavailable")
    setattr(fake_api, api_method, lambda: None)
    with pytest.raises(Mt5DataError, match="transport unavailable"):
        getattr(provider, method)()
