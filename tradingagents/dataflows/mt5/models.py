from dataclasses import dataclass
from datetime import datetime

from .clock import Mt5BrokerClock


@dataclass(frozen=True, slots=True)
class Mt5TerminalInfo:
    name: str | None = None
    company: str | None = None
    version: str | None = None
    build: int | None = None
    connected: bool | None = None

@dataclass(frozen=True, slots=True)
class Mt5AccountInfo:
    login: int | None = None
    server: str | None = None
    currency: str | None = None
    balance: float | None = None
    equity: float | None = None
    profit: float | None = None
    margin: float | None = None
    free_margin: float | None = None
    leverage: int | None = None

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

@dataclass(frozen=True, slots=True)
class Mt5Tick:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float | None = None
    volume: int | None = None
    volume_real: float | None = None

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

@dataclass(frozen=True, slots=True)
class Mt5Spread:
    symbol: str
    bid: float
    ask: float
    price: float
    points: float
    timestamp: datetime | None = None

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
