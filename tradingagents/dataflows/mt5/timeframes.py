from .errors import Mt5DependencyError, Mt5TimeframeError

SUPPORTED_TIMEFRAMES = ("M1", "M5", "M15", "M30", "H1", "H4", "D1")
TIMEFRAME_ATTRIBUTES = {name: f"TIMEFRAME_{name}" for name in SUPPORTED_TIMEFRAMES}

def normalize_timeframe(value: str) -> str:
    key = str(value).strip().upper()
    if key not in TIMEFRAME_ATTRIBUTES:
        raise Mt5TimeframeError(f"Unsupported MT5 timeframe {value!r}; supported: {', '.join(SUPPORTED_TIMEFRAMES)}")
    return key

def resolve_timeframe(value: str, api) -> int:
    key = normalize_timeframe(value)
    try:
        return getattr(api, TIMEFRAME_ATTRIBUTES[key])
    except AttributeError as exc:
        raise Mt5DependencyError(f"MT5 API does not expose {TIMEFRAME_ATTRIBUTES[key]}") from exc
