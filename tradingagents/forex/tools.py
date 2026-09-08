from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.tools import StructuredTool

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot
from tradingagents.dataflows.mt5.provider import MT5Provider

from .context import snapshot_to_dict


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_value(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {
            name: _normalize_value(getattr(value, name))
            for name in value.__dataclass_fields__
        }
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return {key: _normalize_value(item) for key, item in vars(value).items()}
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    if isinstance(value, datetime):
        return _utc_iso(value)
    return value


class MT5ToolAdapter:
    """Read-only graph adapter around an already-connected MT5 provider."""

    def __init__(
        self,
        provider: MT5Provider,
        snapshot: ForexMarketSnapshot | None = None,
    ) -> None:
        self._provider = provider
        self._cached_snapshot = snapshot

    def cache_snapshot(self, snapshot: ForexMarketSnapshot) -> None:
        self._cached_snapshot = snapshot

    def get_mt5_market_snapshot(self, symbol: str) -> dict[str, Any]:
        if self._cached_snapshot is None:
            raise RuntimeError(
                "MT5ToolAdapter requires a cached snapshot; call cache_snapshot() first"
            )
        if symbol.casefold() != self._cached_snapshot.symbol.casefold():
            raise RuntimeError(
                "MT5ToolAdapter cached snapshot symbol does not match requested symbol"
            )
        return snapshot_to_dict(self._cached_snapshot)

    def get_mt5_tick(self, symbol: str) -> dict[str, Any]:
        return _normalize_value(self._provider.get_tick(symbol))

    def get_mt5_bars(self, symbol: str, timeframe: str, count: int) -> list[dict[str, Any]]:
        return _normalize_value(self._provider.get_bars(symbol, timeframe, count))

    def get_mt5_account_context(self) -> dict[str, Any]:
        return _normalize_value(self._provider.get_account_info())

    def get_mt5_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return _normalize_value(self._provider.get_positions(symbol))

    def get_mt5_spread(self, symbol: str) -> dict[str, Any]:
        return _normalize_value(self._provider.get_spread(symbol))

    def as_tools(self) -> list[Any]:
        return [
            StructuredTool.from_function(
                func=self.get_mt5_market_snapshot,
                name="get_mt5_market_snapshot",
                description="Return the cached normalized MT5 forex market snapshot.",
            ),
            StructuredTool.from_function(
                func=self.get_mt5_tick,
                name="get_mt5_tick",
                description="Return the current MT5 tick as JSON-safe primitives.",
            ),
            StructuredTool.from_function(
                func=self.get_mt5_bars,
                name="get_mt5_bars",
                description="Return MT5 bars as JSON-safe primitives.",
            ),
            StructuredTool.from_function(
                func=self.get_mt5_account_context,
                name="get_mt5_account_context",
                description="Return the read-only MT5 account context.",
            ),
            StructuredTool.from_function(
                func=self.get_mt5_positions,
                name="get_mt5_positions",
                description="Return the current MT5 positions as JSON-safe primitives.",
            ),
            StructuredTool.from_function(
                func=self.get_mt5_spread,
                name="get_mt5_spread",
                description="Return the current MT5 spread as JSON-safe primitives.",
            ),
        ]
