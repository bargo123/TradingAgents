"""RED tests defining the read-only MT5 provider contract."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from tradingagents.dataflows.mt5.provider import MT5Provider

from tradingagents.dataflows.mt5.errors import (
    Mt5InitializationError,
    Mt5SymbolAmbiguousError,
    Mt5SymbolNotFoundError,
)


class FakeMT5:
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
        self.tick_time_msc = 1_700_000_000_123
        self.rate_timeframes = []
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
            free_margin=9_500.0,
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

    def positions_get(self, **kwargs):
        return ()

    def orders_get(self, **kwargs):
        return ()

    def last_error(self):
        return self.error


@pytest.fixture
def fake_api():
    return FakeMT5()


def initialized_provider(fake_api):
    provider = MT5Provider(api=fake_api)
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
    assert tick.timestamp == datetime.fromtimestamp(1_700_000_000.123, tz=timezone.utc)
    assert tick.bid == 150.123


@pytest.mark.unit
def test_tick_falls_back_to_second_timestamp(fake_api):
    fake_api.tick_time_msc = None
    provider = initialized_provider(fake_api)
    assert provider.get_tick("USDJPY").timestamp == datetime.fromtimestamp(
        1_700_000_000, tz=timezone.utc
    )


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
