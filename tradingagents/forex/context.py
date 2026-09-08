from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar

_SOURCE_LABEL = "SOURCE: LIVE MT5 BROKER DATA (read-only; not Yahoo Finance)"
_MAX_SERIALIZED_BARS = 100


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware UTC values")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: Any) -> Any:
    if isinstance(value, float):
        return str(value)
    return value


def _bar_to_dict(bar: Mt5Bar) -> dict[str, Any]:
    return {
        "timestamp": _utc_iso(bar.timestamp),
        "open": bar.open,
        "high": bar.high,
        "low": bar.low,
        "close": bar.close,
        "tick_volume": bar.tick_volume,
        "spread": bar.spread,
        "real_volume": bar.real_volume,
    }


def _validate_json_safe(payload: Any) -> Any:
    """Ensure a serialized payload contains no NaN/Infinity or opaque values."""
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("serialized MT5 context must contain finite JSON-safe values") from exc
    return payload


def _coerce_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _coerce_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, list):
        return [_coerce_json_value(item) for item in value]
    if isinstance(value, datetime):
        return _utc_iso(value)
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            extracted = value.item()
        except Exception:
            extracted = value
        else:
            if extracted is not value:
                return _coerce_json_value(extracted)
    return value


def _symbol_info_to_dict(snapshot: ForexMarketSnapshot) -> dict[str, Any]:
    info = snapshot.symbol_info
    if info is None:
        return {}
    return {
        "name": info.name,
        "description": info.description,
        "digits": info.digits,
        "point": info.point,
        "visible": info.visible,
        "trade_mode": info.trade_mode,
        "currency_base": info.currency_base,
        "currency_profit": info.currency_profit,
    }


def _account_to_dict(account: Any | None) -> dict[str, Any] | None:
    if account is None:
        return None
    fields = ("login", "server", "currency", "balance", "equity", "profit", "margin", "free_margin", "leverage")
    return {field: getattr(account, field, None) for field in fields}


def _position_to_dict(position: Any) -> dict[str, Any]:
    opened_at = getattr(position, "time", None)
    return {
        "ticket": getattr(position, "ticket", None),
        "symbol": getattr(position, "symbol", None),
        "type": getattr(position, "type", None),
        "volume": getattr(position, "volume", None),
        "price_open": getattr(position, "price_open", None),
        "price_current": getattr(position, "price_current", None),
        "profit": getattr(position, "profit", None),
        "time": _utc_iso(opened_at) if opened_at else None,
    }


def snapshot_to_dict(snapshot: ForexMarketSnapshot) -> dict[str, Any]:
    """Serialize a normalized MT5 snapshot into a JSON-safe mapping."""
    payload = _coerce_json_value(
        {
        "timestamp": _utc_iso(snapshot.timestamp),
        "symbol": snapshot.symbol,
        "quote": {
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "spread": snapshot.spread,
            "spread_points": snapshot.spread_points,
        },
        "symbol_metadata": _symbol_info_to_dict(snapshot),
        "account": _account_to_dict(snapshot.account),
        "positions": [_position_to_dict(position) for position in snapshot.positions],
        "candles": {
            "M1": [_bar_to_dict(bar) for bar in snapshot.m1_candles[-_MAX_SERIALIZED_BARS:]],
            "M5": [_bar_to_dict(bar) for bar in snapshot.m5_candles[-_MAX_SERIALIZED_BARS:]],
            "M15": [_bar_to_dict(bar) for bar in snapshot.m15_candles[-_MAX_SERIALIZED_BARS:]],
            "H1": [_bar_to_dict(bar) for bar in snapshot.h1_candles[-_MAX_SERIALIZED_BARS:]],
        },
        }
    )
    return _validate_json_safe(payload)


def _summarize_series(label: str, bars: tuple[Mt5Bar, ...]) -> str:
    if not bars:
        return f"{label}: unavailable"

    latest = bars[-1]
    direction = "UP" if latest.close > latest.open else "DOWN" if latest.close < latest.open else "FLAT"
    high = max(bar.high for bar in bars)
    low = min(bar.low for bar in bars)
    span = high - low
    base = latest.open or latest.close or 1.0
    change = (latest.close - latest.open) / base if base else 0.0
    return (
        f"{label}: count={len(bars)} latest="
        f"O={_number(latest.open)} H={_number(latest.high)} "
        f"L={_number(latest.low)} C={_number(latest.close)} "
        f"direction={direction} return={_number(change)} "
        f"high={_number(high)} low={_number(low)} range={_number(span)}"
    )


def build_forex_market_context(snapshot: ForexMarketSnapshot) -> str:
    """Build a compact, deterministic prompt context for forex shadow mode."""
    payload = snapshot_to_dict(snapshot)
    quote = payload["quote"]
    metadata = payload["symbol_metadata"]
    account = payload["account"] or {}
    lines = [
        _SOURCE_LABEL,
        f"Requested symbol: {payload['symbol']}",
        f"Resolved symbol: {payload['symbol']}",
        f"Snapshot UTC: {payload['timestamp']}",
        (
            "Quote: "
            f"bid={quote['bid']} ask={quote['ask']} spread={quote['spread']} "
            f"spread_points={quote['spread_points']} "
            f"digits={metadata.get('digits')} point={metadata.get('point')}"
        ),
    ]
    if metadata:
        lines.append(
            "Symbol metadata: "
            + ", ".join(
                f"{key}={value}"
                for key, value in metadata.items()
                if value is not None
            )
        )
    if account:
        lines.append(
            "Account: "
            + ", ".join(
                f"{key}={value}"
                for key, value in account.items()
                if value is not None
            )
        )
    lines.append(f"Positions: {len(payload['positions'])}")
    for label in ("M1", "M5", "M15", "H1"):
        lines.append(_summarize_series(label, getattr(snapshot, f"{label.lower()}_candles")))
    return "\n".join(lines)
