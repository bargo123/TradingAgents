"""Deterministic, pre-decision market feature extraction for Phase 8."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import Any

from .errors import FeatureExtractionIncompleteError

FEATURE_SCHEMA_VERSION = "experience-features.v1"
FEATURE_EXTRACTOR_VERSION = "phase8-feature-extractor.v1"
FEATURE_NAMES_V1 = (
    "spread_points",
    *(f"{tf}.{name}" for tf in ("M1", "M5", "M15", "H1") for name in
      ("return_over_bars", "range_pct", "close_position", "average_true_range", "direction_code")),
    "utc_hour_sin", "utc_hour_cos",
)
_DIRECTIONS = {"UP": 1.0, "FLAT": 0.0, "DOWN": -1.0}


@dataclass(frozen=True, slots=True)
class ExtractionDiagnostic:
    code: str
    path: str
    detail: str


@dataclass(frozen=True, slots=True)
class MarketStateVector:
    values: tuple[float, ...]
    mask: tuple[bool, ...]
    feature_names: tuple[str, ...]
    cohort: tuple[str, ...]
    fingerprint: str
    source_paths: tuple[str | None, ...] = ()
    missing_reasons: tuple[str | None, ...] = ()
    diagnostics: tuple[ExtractionDiagnostic, ...] = ()


def _fail(diags: list[ExtractionDiagnostic]) -> None:
    if diags:
        error = FeatureExtractionIncompleteError("; ".join(f"{d.code}:{d.path}" for d in diags))
        error.diagnostics = tuple(diags)
        raise error


def _timestamp(value: Any) -> datetime:
    try:
        if isinstance(value, str): value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError
        return value
    except (TypeError, ValueError):
        raise FeatureExtractionIncompleteError("TEMPORAL_INVALID:analysis_snapshot_timestamp")


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def extract_market_state(decision_row: Mapping[str, Any] | Any) -> MarketStateVector:
    """Extract only persisted snapshot and decision metadata; no outcomes are read."""
    row = decision_row if isinstance(decision_row, Mapping) else ({f.name: getattr(decision_row, f.name) for f in fields(decision_row)} if is_dataclass(decision_row) else vars(decision_row))
    diags: list[ExtractionDiagnostic] = []
    symbol = row.get("resolved_symbol") or row.get("symbol")
    profile = row.get("analysis_profile")
    timeframe = row.get("analysis_timeframe")
    if not isinstance(symbol, str) or not symbol.strip(): diags.append(ExtractionDiagnostic("IDENTITY_MISSING", "resolved_symbol", "required"))
    if not isinstance(profile, str) or not profile.strip(): diags.append(ExtractionDiagnostic("IDENTITY_MISSING", "analysis_profile", "required"))
    if not isinstance(timeframe, str) or not timeframe.strip(): diags.append(ExtractionDiagnostic("IDENTITY_MISSING", "analysis_timeframe", "required"))
    timestamp = row.get("analysis_snapshot_timestamp") or row.get("snapshot_timestamp")
    dt = _timestamp(timestamp) if timestamp is not None else None
    if dt is None: diags.append(ExtractionDiagnostic("TEMPORAL_INVALID", "analysis_snapshot_timestamp", "required"))
    raw = row.get("snapshot_json", {})
    if isinstance(raw, str):
        try: raw = json.loads(raw)
        except (TypeError, ValueError): raw = None
    if not isinstance(raw, Mapping): diags.append(ExtractionDiagnostic("SCHEMA_INVALID", "snapshot_json", "mapping required")); raw = {}
    quote = raw.get("quote")
    if not isinstance(quote, Mapping):
        diags.append(ExtractionDiagnostic("SCHEMA_INVALID", "snapshot_json.quote", "mapping required"))
        quote = {}
    metadata = raw.get("symbol_metadata", {})
    metadata = metadata if isinstance(metadata, Mapping) else {}
    point = raw.get("point", row.get("point", metadata.get("point"))); digits = raw.get("digits", row.get("digits", metadata.get("digits")))
    if _finite(point) is None or float(point) <= 0: diags.append(ExtractionDiagnostic("POINT_INVALID", "snapshot_json.point", "positive finite number required"))
    try: valid_digits = not isinstance(digits, bool) and isinstance(digits, (int, float)) and math.isfinite(float(digits)) and int(digits) == digits and 0 <= int(digits) <= 10
    except (TypeError, ValueError, OverflowError): valid_digits = False
    if not valid_digits:
        diags.append(ExtractionDiagnostic("DIGITS_INVALID", "snapshot_json.digits", "integer required"))
    _fail(diags)
    values: list[float] = []; mask: list[bool] = []; paths: list[str | None] = []; reasons: list[str | None] = []
    def add(name: str, value: Any, path: str, *, direction: bool = False) -> None:
        if direction:
            if value is None or str(value).upper() == "INSUFFICIENT_DATA": values.append(float("nan")); mask.append(False); paths.append(path); reasons.append("INSUFFICIENT_DATA"); return
            encoded = _DIRECTIONS.get(str(value).upper())
            if encoded is None:
                diags.append(ExtractionDiagnostic("DIRECTION_INVALID", path, "expected UP, FLAT, DOWN, or INSUFFICIENT_DATA"))
                values.append(float("nan")); mask.append(False); paths.append(path); reasons.append("DIRECTION_INVALID"); return
            value = encoded
        else: value = _finite(value)
        if value is None: values.append(float("nan")); mask.append(False); paths.append(path if path else None); reasons.append("MISSING" if value is None else "NON_FINITE")
        else: values.append(value); mask.append(True); paths.append(path); reasons.append(None)
    for quote_key in ("bid", "ask", "spread_points"):
        if _finite(quote.get(quote_key)) is None:
            diags.append(ExtractionDiagnostic("QUOTE_INVALID", f"snapshot_json.quote.{quote_key}", "finite value required"))
    spread = row.get("analysis_snapshot_spread_points", quote.get("spread_points"))
    add("spread_points", spread, "analysis_snapshot_spread_points" if row.get("analysis_snapshot_spread_points") is not None else "snapshot_json.quote.spread_points")
    features = raw.get("features", {})
    if not isinstance(features, Mapping):
        diags.append(ExtractionDiagnostic("SCHEMA_INVALID", "snapshot_json.features", "mapping required"))
        features = {}
    for tf in ("M1", "M5", "M15", "H1"):
        section = features.get(tf, features.get(tf.lower(), {})); section = section if isinstance(section, Mapping) else {}
        for key in ("return_over_bars", "range_pct", "close_position", "average_true_range"):
            add(f"{tf}.{key}", section.get(key), f"snapshot_json.features.{tf}.{key}")
        add(f"{tf}.direction_code", section.get("direction"), f"snapshot_json.features.{tf}.direction", direction=True)
    hour = dt.hour + dt.minute / 60 + dt.second / 3600
    import math as _math
    add("utc_hour_sin", _math.sin(2 * _math.pi * hour / 24), "analysis_snapshot_timestamp.utc_hour")
    add("utc_hour_cos", _math.cos(2 * _math.pi * hour / 24), "analysis_snapshot_timestamp.utc_hour")
    _fail(diags)
    cohort = (str(symbol), str(profile), str(timeframe), FEATURE_SCHEMA_VERSION, FEATURE_EXTRACTOR_VERSION)
    payload = {"version": FEATURE_EXTRACTOR_VERSION, "names": FEATURE_NAMES_V1,
               "values": [v if math.isfinite(v) else None for v in values], "mask": mask,
               "paths": paths, "missing": reasons, "cohort": cohort}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return MarketStateVector(tuple(values), tuple(mask), FEATURE_NAMES_V1, cohort, fingerprint, tuple(paths), tuple(reasons), tuple(diags))


__all__ = ["FEATURE_NAMES_V1", "FEATURE_SCHEMA_VERSION", "FEATURE_EXTRACTOR_VERSION", "ExtractionDiagnostic", "MarketStateVector", "extract_market_state"]
