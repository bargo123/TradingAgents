"""Read-only, normalized access to a MetaTrader 5 terminal."""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from contextlib import suppress
from datetime import datetime, timezone
from typing import Any

from .errors import (
    Mt5AccountDisconnectedError,
    Mt5DataError,
    Mt5DependencyError,
    Mt5InitializationError,
    Mt5NotConnectedError,
    Mt5SymbolAmbiguousError,
    Mt5SymbolError,
    Mt5SymbolNotFoundError,
)
from .models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Order,
    Mt5Position,
    Mt5Spread,
    Mt5SymbolInfo,
    Mt5TerminalInfo,
    Mt5Tick,
)
from .timeframes import resolve_timeframe

_CURRENCY_CODES = frozenset(
    ["AED", "ARS", "AUD", "BGN", "BHD", "BRL", "CAD", "CHF", "CLP", "CNY", "COP", "CZK", "DKK", "EGP", "EUR", "GBP", "HKD", "HUF", "IDR", "ILS", "INR", "JPY", "KRW", "KWD", "MXN", "MYR", "NOK", "NZD", "OMR", "PHP", "PLN", "QAR", "RON", "RUB", "SAR", "SEK", "SGD", "THB", "TRY", "TWD", "USD", "XAG", "XAU", "ZAR"]
)
_SEPARATOR = re.compile(r"[._#-]")


def _field(raw: Any, name: str, default: Any = None) -> Any:
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


def _first_field(raw: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = _field(raw, name)
        if value is not None:
            return value
    return default


def _utc_timestamp(raw: Any, *, prefer_msc: bool = False) -> datetime:
    value = _field(raw, "time_msc") if prefer_msc else None
    if value in (None, 0):
        value = _field(raw, "time")
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)


def _pair_candidates(name: str) -> list[str]:
    """Return valid currency-pair cores in a controlled broker symbol form."""
    pairs: list[str] = []
    for start in range(max(0, len(name) - 5)):
        core = name[start : start + 6]
        if len(core) != 6 or core[:3] not in _CURRENCY_CODES or core[3:] not in _CURRENCY_CODES:
            continue
        prefix, suffix = name[:start], name[start + 6 :]
        prefix_ok = (
            not prefix
            or (prefix.isalnum() and len(prefix) <= 2)
            or (prefix.endswith((".", "_", "#", "-")) and bool(prefix[:-1] or prefix == "#"))
        )
        suffix_ok = (
            not suffix
            or (suffix.isalnum() and len(suffix) <= 2)
            or (suffix[0] in "._#-" and bool(_SEPARATOR.sub("", suffix) or suffix == "#"))
        )
        if prefix_ok and suffix_ok:
            pairs.append(core)
    return pairs


def _normalize_symbol_name(symbol: str) -> str:
    normalized = str(symbol).strip().upper()
    if not normalized:
        return normalized
    pairs = _pair_candidates(normalized)
    if len(pairs) != 1:
        return ""
    return next(iter(pairs))


class MT5Provider:
    """A deliberately read-only MT5 provider with optional API injection."""

    def __init__(self, *, api: Any | None = None, terminal_path: str | None = None) -> None:
        self._api = api
        self._terminal_path = terminal_path
        self._initialized = False

    def _load_api(self) -> Any:
        if self._api is None:
            try:
                self._api = importlib.import_module("MetaTrader5")
            except ModuleNotFoundError as exc:
                raise Mt5DependencyError(
                    "MetaTrader5 is required; install it with: pip install MetaTrader5"
                ) from exc
        return self._api

    def _last_error(self) -> Any:
        try:
            return self._api.last_error()
        except Exception:
            return None

    def _connection_error(self, message: str) -> str:
        detail = self._last_error()
        return f"{message}: {detail!r}" if detail not in (None, (0, "")) else message

    def _failed_collection(self, operation: str) -> Mt5DataError:
        return Mt5DataError(f"MT5 {operation} returned no data: {self._last_error()!r}")

    def _symbols(self) -> Any:
        raw_symbols = self._api.symbols_get()
        if raw_symbols is None:
            raise self._failed_collection("symbols_get")
        return raw_symbols

    def _shutdown_after_failed_initialize(self) -> None:
        with suppress(Exception):
            self._api.shutdown()

    def initialize(self) -> bool:
        api = self._load_api()
        try:
            success = api.initialize(path=self._terminal_path) if self._terminal_path else api.initialize()
        except Exception as exc:
            raise Mt5InitializationError(f"MT5 initialization failed: {self._last_error()!r}") from exc
        if not success:
            raise Mt5InitializationError(f"MT5 initialization failed: {self._last_error()!r}")
        try:
            terminal, account = api.terminal_info(), api.account_info()
        except Exception as exc:
            self._shutdown_after_failed_initialize()
            raise Mt5InitializationError(
                self._connection_error("MT5 post-initialization inspection failed")
            ) from exc
        if terminal is None or not _field(terminal, "connected", False):
            self._shutdown_after_failed_initialize()
            raise Mt5InitializationError(self._connection_error("MT5 terminal is not connected"))
        if account is None:
            self._shutdown_after_failed_initialize()
            raise Mt5AccountDisconnectedError(self._connection_error("MT5 account is not connected"))
        self._initialized = True
        return True

    def shutdown(self) -> None:
        if self._initialized and self._api is not None:
            self._api.shutdown()
        self._initialized = False

    def is_connected(self) -> bool:
        if not self._initialized or self._api is None:
            return False
        try:
            terminal, account = self._api.terminal_info(), self._api.account_info()
        except Exception:
            return False
        return bool(terminal and _field(terminal, "connected", False) and account)

    def _require_connected(self) -> None:
        if not self._initialized:
            raise Mt5NotConnectedError("MT5 provider has not been initialized")
        if not self.is_connected():
            raise Mt5AccountDisconnectedError("MT5 terminal or account is disconnected")

    def get_terminal_info(self) -> Mt5TerminalInfo:
        self._require_connected()
        raw = self._api.terminal_info()
        return Mt5TerminalInfo(**{key: _field(raw, key) for key in Mt5TerminalInfo.__dataclass_fields__})

    def get_account_info(self) -> Mt5AccountInfo:
        self._require_connected()
        raw = self._api.account_info()
        if raw is None:
            raise Mt5AccountDisconnectedError("MT5 account is not connected")
        values = {key: _field(raw, key) for key in Mt5AccountInfo.__dataclass_fields__}
        if values["free_margin"] is None:
            values["free_margin"] = _field(raw, "margin_free")
        return Mt5AccountInfo(**values)

    def get_symbols(self) -> tuple[Mt5SymbolInfo, ...]:
        self._require_connected()
        raw_symbols = self._symbols()
        fields = Mt5SymbolInfo.__dataclass_fields__
        return tuple(Mt5SymbolInfo(**{key: _field(raw, key) for key in fields}) for raw in raw_symbols)

    def find_symbol(self, symbol: str) -> str:
        self._require_connected()
        requested = str(symbol).strip().upper()
        records = self._symbols()
        exact = [str(_field(raw, "name")) for raw in records if str(_field(raw, "name", "")).upper() == requested]
        if exact:
            return exact[0]
        base = _normalize_symbol_name(requested)
        candidates = [
            str(_field(raw, "name"))
            for raw in records
            if _normalize_symbol_name(str(_field(raw, "name", "")).upper()) == base and base
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise Mt5SymbolAmbiguousError(f"Ambiguous MT5 symbol {symbol!r}: {', '.join(candidates)}")
        raise Mt5SymbolNotFoundError(f"MT5 symbol not found: {symbol!r}")

    def ensure_symbol(self, symbol: str) -> str:
        resolved = self.find_symbol(symbol)
        if not self._api.symbol_select(resolved, True):
            raise Mt5SymbolError(f"Unable to select MT5 symbol {resolved!r}")
        return resolved

    def get_tick(self, symbol: str) -> Mt5Tick:
        resolved = self.ensure_symbol(symbol)
        return self._get_tick_resolved(resolved)

    def _get_tick_resolved(self, resolved: str) -> Mt5Tick:
        raw = self._api.symbol_info_tick(resolved)
        if raw is None:
            raise Mt5DataError(f"No tick available for {resolved!r}")
        try:
            return Mt5Tick(
                symbol=resolved,
                timestamp=_utc_timestamp(raw, prefer_msc=True),
                bid=float(_field(raw, "bid")),
                ask=float(_field(raw, "ask")),
                last=_field(raw, "last"),
                volume=_field(raw, "volume"),
                volume_real=_field(raw, "volume_real"),
            )
        except (TypeError, ValueError, OSError) as exc:
            raise Mt5DataError(f"Invalid tick data for {resolved!r}") from exc

    def get_bars(self, symbol: str, timeframe: str, count: int, start_pos: int = 0) -> tuple[Mt5Bar, ...]:
        if count <= 0:
            raise Mt5DataError("Bar count must be greater than zero")
        resolved = self.ensure_symbol(symbol)
        return self._get_bars_resolved(resolved, timeframe, count, start_pos)

    def _get_bars_resolved(self, resolved: str, timeframe: str, count: int, start_pos: int = 0) -> tuple[Mt5Bar, ...]:
        if count <= 0:
            raise Mt5DataError("Bar count must be greater than zero")
        rows = self._api.copy_rates_from_pos(resolved, resolve_timeframe(timeframe, self._api), start_pos, count)
        if rows is None:
            raise Mt5DataError(f"No bars available for {resolved!r}")
        try:
            bars = tuple(Mt5Bar(timestamp=_utc_timestamp(row), open=float(_field(row, "open")), high=float(_field(row, "high")), low=float(_field(row, "low")), close=float(_field(row, "close")), tick_volume=int(_field(row, "tick_volume")), spread=_field(row, "spread"), real_volume=_field(row, "real_volume")) for row in rows)
        except (TypeError, ValueError, OSError) as exc:
            raise Mt5DataError(f"Invalid bar data for {resolved!r}") from exc
        if not bars:
            raise Mt5DataError(f"No bars available for {resolved!r}")
        return bars

    def get_positions(self, symbol: str | None = None) -> tuple[Mt5Position, ...]:
        self._require_connected()
        resolved = self.find_symbol(symbol) if symbol is not None else None
        return self._get_positions_resolved(resolved)

    def _get_positions_resolved(self, resolved: str | None) -> tuple[Mt5Position, ...]:
        raw_positions = self._api.positions_get()
        if raw_positions is None:
            raise self._failed_collection("positions_get")
        try:
            return tuple(
                Mt5Position(ticket=int(_field(raw, "ticket")), symbol=str(_field(raw, "symbol")), type=_field(raw, "type"), volume=_field(raw, "volume"), price_open=_field(raw, "price_open"), price_current=_field(raw, "price_current"), profit=_field(raw, "profit"), time=_utc_timestamp(raw))
                for raw in raw_positions if resolved is None or str(_field(raw, "symbol")) == resolved
            )
        except (TypeError, ValueError, OSError) as exc:
            raise Mt5DataError("Invalid MT5 position data") from exc

    def get_orders(self, symbol: str | None = None) -> tuple[Mt5Order, ...]:
        self._require_connected()
        resolved = self.find_symbol(symbol) if symbol is not None else None
        raw_orders = self._api.orders_get()
        if raw_orders is None:
            raise self._failed_collection("orders_get")
        try:
            return tuple(
                Mt5Order(ticket=int(_field(raw, "ticket")), symbol=str(_field(raw, "symbol")), type=_field(raw, "type"), volume=_first_field(raw, "volume_current", "volume"), price_open=_field(raw, "price_open"), price_current=_field(raw, "price_current"), sl=_field(raw, "sl"), tp=_field(raw, "tp"), time=_utc_timestamp({"time": _first_field(raw, "time_setup", "time")}))
                for raw in raw_orders if resolved is None or str(_field(raw, "symbol")) == resolved
            )
        except (TypeError, ValueError, OSError) as exc:
            raise Mt5DataError("Invalid MT5 order data") from exc

    def get_spread(self, symbol: str) -> Mt5Spread:
        resolved = self.ensure_symbol(symbol)
        info, tick = self._api.symbol_info(resolved), self._api.symbol_info_tick(resolved)
        point = _field(info, "point")
        if info is None or point in (None, 0) or tick is None:
            raise Mt5DataError(f"Cannot calculate spread for {resolved!r}")
        try:
            bid, ask = float(_field(tick, "bid")), float(_field(tick, "ask"))
            price = ask - bid
            return Mt5Spread(resolved, bid, ask, price, price / float(point), _utc_timestamp(tick, prefer_msc=True))
        except (TypeError, ValueError, OSError, ZeroDivisionError) as exc:
            raise Mt5DataError(f"Invalid spread data for {resolved!r}") from exc

    def get_ticks_range(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        flags: int | None = None,
    ) -> tuple[Mt5Tick, ...]:
        """Return normalized historical ticks without mutating the terminal."""
        self._require_connected()
        if not isinstance(start, datetime) or start.tzinfo is None:
            raise ValueError("start must be timezone-aware")
        if not isinstance(end, datetime) or end.tzinfo is None:
            raise ValueError("end must be timezone-aware")
        start = start.astimezone(timezone.utc)
        end = end.astimezone(timezone.utc)
        if end < start:
            raise ValueError("end must be greater than or equal to start")

        resolved = self.ensure_symbol(symbol)
        api_flags = (
            flags
            if flags is not None
            else getattr(self._api, "COPY_TICKS_ALL", -1)
        )
        copy_ticks_range = getattr(self._api, "copy_ticks_range", None)
        if not callable(copy_ticks_range):
            raise Mt5DataError("MT5 copy_ticks_range is unavailable")
        try:
            rows = copy_ticks_range(resolved, start, end, api_flags)
        except Exception as exc:
            raise Mt5DataError(
                f"MT5 copy_ticks_range failed for {resolved!r}: {self._last_error()!r}"
            ) from exc
        if rows is None:
            raise self._failed_collection("copy_ticks_range")
        try:
            return tuple(
                Mt5Tick(
                    symbol=resolved,
                    timestamp=_utc_timestamp(row, prefer_msc=True),
                    bid=float(_field(row, "bid")),
                    ask=float(_field(row, "ask")),
                    last=_field(row, "last"),
                    volume=_field(row, "volume"),
                    volume_real=_field(row, "volume_real"),
                )
                for row in rows
            )
        except (TypeError, ValueError, OSError) as exc:
            raise Mt5DataError(f"Invalid tick range data for {resolved!r}") from exc

    def get_market_snapshot(self, symbol: str, count: int = 100) -> ForexMarketSnapshot:
        """Capture a normalized, read-only multi-timeframe market snapshot."""
        self._require_connected()
        if count <= 0:
            raise Mt5DataError("Bar count must be greater than zero")
        resolved = self.ensure_symbol(symbol)
        info = self._api.symbol_info(resolved)
        if info is None or _field(info, "point") in (None, 0):
            raise Mt5DataError(f"Cannot calculate spread for {resolved!r}")
        tick = self._get_tick_resolved(resolved)
        try:
            point = float(_field(info, "point"))
            spread = tick.ask - tick.bid
            spread_points = spread / point
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise Mt5DataError(f"Invalid spread data for {resolved!r}") from exc
        candles = {
            timeframe: self._get_bars_resolved(resolved, timeframe, count)
            for timeframe in ("M1", "M5", "M15", "H1")
        }
        account = self.get_account_info()
        positions = self._get_positions_resolved(resolved)
        return ForexMarketSnapshot(
            timestamp=tick.timestamp,
            symbol=resolved,
            bid=tick.bid,
            ask=tick.ask,
            spread=spread,
            spread_points=spread_points,
            symbol_info=Mt5SymbolInfo(**{key: _field(info, key) for key in Mt5SymbolInfo.__dataclass_fields__}),
            m1_candles=candles["M1"],
            m5_candles=candles["M5"],
            m15_candles=candles["M15"],
            h1_candles=candles["H1"],
            account=account,
            positions=positions,
        )


Mt5Provider = MT5Provider
