import math
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from typing import Any

from .clock import Mt5BrokerClock


def _utc_timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timezone.utc.utcoffset(value):
        raise ValueError(f"{name} must be UTC")
    return value.astimezone(timezone.utc)


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite numeric value")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite numeric value") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}numeric value")
    return result


def _nonnegative_integral(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _optional_finite(value: Any, name: str, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, name)
    if nonnegative and result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class Mt5TerminalInfo:
    name: str | None = None
    company: str | None = None
    version: str | None = None
    build: int | None = None
    connected: bool | None = None

    def __post_init__(self) -> None:
        for name in ("name", "company", "version"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))
        if self.build is not None:
            object.__setattr__(self, "build", _nonnegative_integral(self.build, "build"))
        if self.connected is not None and not isinstance(self.connected, bool):
            raise ValueError("connected must be a bool when provided")

@dataclass(frozen=True, slots=True)
class Mt5AccountInfo:
    login: int | None = None
    server: str | None = None
    currency: str | None = None
    trade_mode: int | None = None
    balance: float | None = None
    equity: float | None = None
    profit: float | None = None
    margin: float | None = None
    free_margin: float | None = None
    leverage: int | None = None

    def __post_init__(self) -> None:
        if self.login is not None:
            object.__setattr__(self, "login", _nonnegative_integral(self.login, "login"))
        for name in ("server", "currency"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))
        if self.trade_mode is not None:
            object.__setattr__(self, "trade_mode", _nonnegative_integral(self.trade_mode, "trade_mode"))
        for name in ("balance", "equity", "profit", "margin", "free_margin"):
            value = _optional_finite(getattr(self, name), name)
            if value is not None:
                object.__setattr__(self, name, value)
        if self.leverage is not None:
            object.__setattr__(self, "leverage", _nonnegative_integral(self.leverage, "leverage"))

@dataclass(frozen=True, slots=True)
class Mt5SymbolInfo:
    name: str
    description: str | None = None
    digits: int | None = None
    point: float | None = None
    visible: bool | None = None
    trade_mode: int | None = None
    currency_base: str | None = None
    currency_profit: str | None = None
    volume_min: float | None = None
    volume_max: float | None = None
    volume_step: float | None = None
    trade_stops_level: int | None = None
    trade_freeze_level: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_text(self.name, "symbol name"))
        if self.digits is not None:
            object.__setattr__(self, "digits", _nonnegative_integral(self.digits, "digits"))
        if self.point is not None:
            object.__setattr__(self, "point", _finite_number(self.point, "point", positive=True))
        if self.visible is not None and not isinstance(self.visible, bool):
            raise ValueError("visible must be a bool when provided")
        if self.trade_mode is not None:
            object.__setattr__(self, "trade_mode", _nonnegative_integral(self.trade_mode, "trade_mode"))
        for name in ("currency_base", "currency_profit"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonempty_text(value, name))
        for name in ("volume_min", "volume_max", "volume_step"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite_number(value, name, positive=True))
        for name in ("trade_stops_level", "trade_freeze_level"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _nonnegative_integral(value, name))

@dataclass(frozen=True, slots=True)
class Mt5Tick:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: int | None = None
    volume_real: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _nonempty_text(self.symbol, "tick symbol"))
        object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp, "tick timestamp"))
        bid = _finite_number(self.bid, "bid", positive=True)
        ask = _finite_number(self.ask, "ask", positive=True)
        if ask < bid:
            raise ValueError("tick ask must be greater than or equal to bid")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        if self.last is not None:
            object.__setattr__(self, "last", _finite_number(self.last, "last"))
        if self.volume is not None:
            object.__setattr__(self, "volume", _nonnegative_integral(self.volume, "volume"))
        if self.volume_real is not None:
            object.__setattr__(self, "volume_real", _finite_number(self.volume_real, "volume_real"))
            if self.volume_real < 0:
                raise ValueError("volume_real must be non-negative")

@dataclass(frozen=True, slots=True)
class Mt5Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: int
    spread: int | None = None
    real_volume: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp, "bar timestamp"))
        values = {
            name: _finite_number(getattr(self, name), name, positive=True)
            for name in ("open", "high", "low", "close")
        }
        if values["high"] < max(values["open"], values["close"], values["low"]):
            raise ValueError("bar OHLC values are inconsistent")
        if values["low"] > min(values["open"], values["close"], values["high"]):
            raise ValueError("bar OHLC values are inconsistent")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "tick_volume", _nonnegative_integral(self.tick_volume, "tick_volume"))
        if self.spread is not None:
            object.__setattr__(self, "spread", _nonnegative_integral(self.spread, "spread"))
        if self.real_volume is not None:
            object.__setattr__(self, "real_volume", _nonnegative_integral(self.real_volume, "real_volume"))

@dataclass(frozen=True, slots=True)
class Mt5Position:
    ticket: int
    symbol: str
    type: int | None = None
    volume: float | None = None
    price_open: float | None = None
    price_current: float | None = None
    profit: float | None = None
    time: datetime | None = None
    magic: int | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticket", _nonnegative_integral(self.ticket, "position ticket"))
        object.__setattr__(self, "symbol", _nonempty_text(self.symbol, "position symbol"))
        if self.type is not None:
            object.__setattr__(self, "type", _nonnegative_integral(self.type, "position type"))
        for name in ("volume", "price_open", "price_current"):
            value = _optional_finite(getattr(self, name), name, nonnegative=True)
            if value is not None:
                object.__setattr__(self, name, value)
        value = _optional_finite(self.profit, "profit")
        if value is not None:
            object.__setattr__(self, "profit", value)
        if self.time is not None:
            object.__setattr__(self, "time", _utc_timestamp(self.time, "position time"))
        if self.magic is not None:
            object.__setattr__(self, "magic", _nonnegative_integral(self.magic, "position magic"))
        if self.comment is not None:
            object.__setattr__(self, "comment", _nonempty_text(self.comment, "position comment"))

@dataclass(frozen=True, slots=True)
class Mt5Order:
    ticket: int
    symbol: str
    type: int | None = None
    volume: float | None = None
    price_open: float | None = None
    price_current: float | None = None
    sl: float | None = None
    tp: float | None = None
    time: datetime | None = None
    magic: int | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "ticket", _nonnegative_integral(self.ticket, "order ticket"))
        object.__setattr__(self, "symbol", _nonempty_text(self.symbol, "order symbol"))
        if self.type is not None:
            object.__setattr__(self, "type", _nonnegative_integral(self.type, "order type"))
        for name in ("volume", "price_open", "price_current", "sl", "tp"):
            value = _optional_finite(getattr(self, name), name, nonnegative=True)
            if value is not None:
                object.__setattr__(self, name, value)
        if self.time is not None:
            object.__setattr__(self, "time", _utc_timestamp(self.time, "order time"))
        if self.magic is not None:
            object.__setattr__(self, "magic", _nonnegative_integral(self.magic, "order magic"))
        if self.comment is not None:
            object.__setattr__(self, "comment", _nonempty_text(self.comment, "order comment"))

@dataclass(frozen=True, slots=True)
class Mt5Spread:
    symbol: str
    bid: float
    ask: float
    price: float
    points: float
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _nonempty_text(self.symbol, "spread symbol"))
        bid = _finite_number(self.bid, "bid", positive=True)
        ask = _finite_number(self.ask, "ask", positive=True)
        if ask < bid:
            raise ValueError("spread ask must be greater than or equal to bid")
        price = _finite_number(self.price, "spread price")
        points = _finite_number(self.points, "spread points")
        if price < 0 or points < 0:
            raise ValueError("spread values must be non-negative")
        if not math.isclose(price, ask - bid, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("spread price must equal ask minus bid")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "points", points)
        if self.timestamp is not None:
            object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp, "spread timestamp"))

@dataclass(frozen=True, slots=True)
class ForexMarketSnapshot:
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    spread: float
    spread_points: float
    m1_candles: tuple[Mt5Bar, ...]
    m5_candles: tuple[Mt5Bar, ...]
    m15_candles: tuple[Mt5Bar, ...]
    h1_candles: tuple[Mt5Bar, ...]
    account: Mt5AccountInfo | None
    positions: tuple[Mt5Position, ...]
    symbol_info: Mt5SymbolInfo | None = None
    broker_clock: Mt5BrokerClock | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _utc_timestamp(self.timestamp, "snapshot timestamp"))
        object.__setattr__(self, "symbol", _nonempty_text(self.symbol, "snapshot symbol"))
        bid = _finite_number(self.bid, "bid", positive=True)
        ask = _finite_number(self.ask, "ask", positive=True)
        if ask < bid:
            raise ValueError("snapshot ask must be greater than or equal to bid")
        spread = _finite_number(self.spread, "spread")
        spread_points = _finite_number(self.spread_points, "spread points")
        if spread < 0 or spread_points < 0:
            raise ValueError("snapshot spread values must be non-negative")
        if not math.isclose(spread, ask - bid, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("snapshot spread must equal ask minus bid")
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(self, "spread", spread)
        object.__setattr__(self, "spread_points", spread_points)
        for name in ("m1_candles", "m5_candles", "m15_candles", "h1_candles"):
            candles = tuple(getattr(self, name))
            if any(not isinstance(item, Mt5Bar) for item in candles):
                raise ValueError(f"{name} must contain Mt5Bar values")
            object.__setattr__(self, name, candles)
        positions = tuple(self.positions)
        if any(not isinstance(item, Mt5Position) for item in positions):
            raise ValueError("positions must contain Mt5Position values")
        object.__setattr__(self, "positions", positions)
        if self.account is not None and not isinstance(self.account, Mt5AccountInfo):
            raise ValueError("account must be Mt5AccountInfo or None")
        if self.symbol_info is not None and not isinstance(self.symbol_info, Mt5SymbolInfo):
            raise ValueError("symbol_info must be Mt5SymbolInfo or None")
        if self.broker_clock is not None and not isinstance(self.broker_clock, Mt5BrokerClock):
            raise ValueError("broker_clock must be Mt5BrokerClock or None")
        if self.symbol_info is not None and self.symbol_info.point is not None:
            expected_points = spread / self.symbol_info.point
            if not math.isclose(spread_points, expected_points, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("snapshot spread_points do not match symbol point")
