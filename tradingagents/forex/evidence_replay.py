"""Read-only saved-snapshot A/B replay for Phase 9 evidence reasoning.

This module deliberately does not use :meth:`ForexShadowRunner.run`.  Replay
is an analysis comparison over an immutable snapshot and is therefore not a
normal shadow opportunity and cannot write the Phase 5/6 decision store.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from tradingagents.dataflows.mt5.clock import Mt5BrokerClock
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5SymbolInfo,
)

from .evidence_audit import _contains_forbidden
from .evidence_context import (
    EvidenceAuditStatus,
    EvidenceIntegrationStatus,
    EvidenceReferenceValidation,
    EvidenceUseStatus,
    validate_evidence_references,
)


class SnapshotReplayError(ValueError):
    """A persisted source row cannot safely be used as replay input."""


def _utc(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise SnapshotReplayError(f"{name} must be an ISO-8601 timestamp") from exc
    else:
        raise SnapshotReplayError(f"{name} must be a UTC timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise SnapshotReplayError(f"{name} must be timezone-aware UTC")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, name: str, *, integer: bool = False) -> float | int:
    if isinstance(value, bool):
        raise SnapshotReplayError(f"{name} must be numeric")
    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotReplayError(f"{name} must be numeric") from exc
    if isinstance(parsed, float) and not math.isfinite(parsed):
        raise SnapshotReplayError(f"{name} must be finite")
    return parsed


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SnapshotReplayError(f"{name} must be an object")
    return value


def _bar(value: Any, name: str) -> Mt5Bar:
    raw = _mapping(value, name)
    required = ("timestamp", "open", "high", "low", "close", "tick_volume")
    missing = [key for key in required if key not in raw]
    if missing:
        raise SnapshotReplayError(f"{name} missing fields: {', '.join(missing)}")
    return Mt5Bar(
        timestamp=_utc(raw["timestamp"], f"{name}.timestamp"),
        open=_number(raw["open"], f"{name}.open"),
        high=_number(raw["high"], f"{name}.high"),
        low=_number(raw["low"], f"{name}.low"),
        close=_number(raw["close"], f"{name}.close"),
        tick_volume=_number(raw["tick_volume"], f"{name}.tick_volume", integer=True),
        spread=None if raw.get("spread") is None else _number(raw["spread"], f"{name}.spread", integer=True),
        real_volume=None if raw.get("real_volume") is None else _number(raw["real_volume"], f"{name}.real_volume", integer=True),
    )


def _account(value: Any) -> Mt5AccountInfo | None:
    if value is None:
        return None
    raw = _mapping(value, "account")
    integer_fields = {"login", "leverage"}
    numeric_fields = {"balance", "equity", "profit", "margin", "free_margin"}
    kwargs: dict[str, Any] = {}
    for key in ("login", "server", "currency", "balance", "equity", "profit", "margin", "free_margin", "leverage"):
        item = raw.get(key)
        if item is not None and key in integer_fields:
            item = _number(item, f"account.{key}", integer=True)
        elif item is not None and key in numeric_fields:
            item = _number(item, f"account.{key}")
        kwargs[key] = item
    return Mt5AccountInfo(**kwargs)


def _position(value: Any, name: str) -> Mt5Position:
    raw = _mapping(value, name)
    if "ticket" not in raw or "symbol" not in raw:
        raise SnapshotReplayError(f"{name} must contain ticket and symbol")
    kwargs: dict[str, Any] = {
        "ticket": _number(raw["ticket"], f"{name}.ticket", integer=True),
        "symbol": str(raw["symbol"]),
    }
    for key in ("type",):
        kwargs[key] = None if raw.get(key) is None else _number(raw[key], f"{name}.{key}", integer=True)
    for key in ("volume", "price_open", "price_current", "profit"):
        kwargs[key] = None if raw.get(key) is None else _number(raw[key], f"{name}.{key}")
    kwargs["time"] = None if raw.get("time") is None else _utc(raw["time"], f"{name}.time")
    return Mt5Position(**kwargs)


def _broker_clock(value: Any) -> Mt5BrokerClock | None:
    if value is None:
        return None
    raw = dict(_mapping(value, "broker_clock"))
    if raw.get("calibrated_at_utc") is not None:
        raw["calibrated_at_utc"] = _utc(raw["calibrated_at_utc"], "broker_clock.calibrated_at_utc")
    if raw.get("offset_seconds") is not None:
        raw["offset_seconds"] = _number(raw["offset_seconds"], "broker_clock.offset_seconds")
    if raw.get("max_residual_seconds") is not None:
        raw["max_residual_seconds"] = _number(raw["max_residual_seconds"], "broker_clock.max_residual_seconds")
    if raw.get("sample_count") is not None:
        raw["sample_count"] = _number(raw["sample_count"], "broker_clock.sample_count", integer=True)
    allowed = {field.name for field in fields(Mt5BrokerClock)}
    return Mt5BrokerClock(**{key: value for key, value in raw.items() if key in allowed})


class SavedSnapshotCodec:
    """Decode the exact JSON snapshot persisted by Phase 5/6."""

    @staticmethod
    def from_source_row(row: Mapping[str, Any]) -> ForexMarketSnapshot:
        if not isinstance(row, Mapping) and hasattr(row, "keys"):
            row = {key: row[key] for key in row}
        if not isinstance(row, Mapping):
            raise SnapshotReplayError("source row must be a mapping")
        encoded = row.get("snapshot_json")
        if isinstance(encoded, (bytes, bytearray)):
            try:
                encoded = encoded.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SnapshotReplayError("snapshot_json is not UTF-8") from exc
        if isinstance(encoded, str):
            try:
                payload = json.loads(encoded)
            except json.JSONDecodeError as exc:
                raise SnapshotReplayError("snapshot_json is not valid JSON") from exc
        else:
            payload = encoded
        payload = _mapping(payload, "snapshot_json")

        required = ("timestamp", "symbol", "quote", "symbol_metadata", "account", "positions", "candles")
        missing = [key for key in required if key not in payload]
        if missing:
            raise SnapshotReplayError(f"snapshot_json missing fields: {', '.join(missing)}")
        timestamp = _utc(payload["timestamp"], "snapshot_json.timestamp")
        symbol = payload["symbol"]
        if not isinstance(symbol, str) or not symbol.strip():
            raise SnapshotReplayError("snapshot_json.symbol must be non-empty")
        symbol = symbol.strip()

        quote = _mapping(payload["quote"], "snapshot_json.quote")
        for key in ("bid", "ask", "spread", "spread_points"):
            if key not in quote:
                raise SnapshotReplayError(f"snapshot_json.quote missing {key}")
        bid = _number(quote["bid"], "snapshot_json.quote.bid")
        ask = _number(quote["ask"], "snapshot_json.quote.ask")
        spread = _number(quote["spread"], "snapshot_json.quote.spread")
        spread_points = _number(quote["spread_points"], "snapshot_json.quote.spread_points")
        if ask < bid or spread < 0 or spread_points < 0:
            raise SnapshotReplayError("snapshot_json.quote contains invalid spread values")

        metadata = _mapping(payload["symbol_metadata"], "snapshot_json.symbol_metadata")
        metadata_name = metadata.get("name")
        if metadata_name is not None and str(metadata_name).strip() != symbol:
            raise SnapshotReplayError("snapshot_json.symbol_metadata.name does not match symbol")
        symbol_info = None
        if metadata:
            if not metadata_name:
                raise SnapshotReplayError("snapshot_json.symbol_metadata.name is required")
            symbol_info = Mt5SymbolInfo(
                name=str(metadata_name),
                description=metadata.get("description"),
                digits=None if metadata.get("digits") is None else _number(metadata["digits"], "symbol_metadata.digits", integer=True),
                point=None if metadata.get("point") is None else _number(metadata["point"], "symbol_metadata.point"),
                visible=metadata.get("visible"),
                trade_mode=None if metadata.get("trade_mode") is None else _number(metadata["trade_mode"], "symbol_metadata.trade_mode", integer=True),
                currency_base=metadata.get("currency_base"),
                currency_profit=metadata.get("currency_profit"),
            )

        candles = _mapping(payload["candles"], "snapshot_json.candles")
        bars: dict[str, tuple[Mt5Bar, ...]] = {}
        for timeframe in ("M1", "M5", "M15", "H1"):
            values = candles.get(timeframe)
            if not isinstance(values, (list, tuple)):
                raise SnapshotReplayError(f"snapshot_json.candles.{timeframe} must be an array")
            bars[timeframe] = tuple(_bar(item, f"candles.{timeframe}[{index}]") for index, item in enumerate(values))

        positions = payload["positions"]
        if not isinstance(positions, (list, tuple)):
            raise SnapshotReplayError("snapshot_json.positions must be an array")
        snapshot = ForexMarketSnapshot(
            timestamp=timestamp,
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread=spread,
            spread_points=spread_points,
            m1_candles=bars["M1"],
            m5_candles=bars["M5"],
            m15_candles=bars["M15"],
            h1_candles=bars["H1"],
            account=_account(payload["account"]),
            positions=tuple(_position(item, f"positions[{index}]") for index, item in enumerate(positions)),
            symbol_info=symbol_info,
            broker_clock=_broker_clock(payload.get("broker_clock")),
        )
        if row.get("resolved_symbol") is not None and str(row["resolved_symbol"]).strip() != symbol:
            raise SnapshotReplayError("source resolved_symbol does not match snapshot symbol")
        if row.get("snapshot_timestamp") is not None and _utc(row["snapshot_timestamp"], "source snapshot_timestamp") != timestamp:
            raise SnapshotReplayError("source snapshot_timestamp does not match snapshot")
        if row.get("analysis_snapshot_timestamp") is not None and _utc(row["analysis_snapshot_timestamp"], "source analysis_snapshot_timestamp") != timestamp:
            raise SnapshotReplayError("source analysis_snapshot_timestamp does not match snapshot")
        for field, expected in (("analysis_snapshot_bid", bid), ("analysis_snapshot_ask", ask), ("analysis_snapshot_spread", spread), ("analysis_snapshot_spread_points", spread_points)):
            if row.get(field) is not None and _number(row[field], f"source {field}") != expected:
                raise SnapshotReplayError(f"source {field} does not match snapshot quote")
        return snapshot


@dataclass(frozen=True, slots=True)
class EvidenceReplayConfig:
    source_decision_id: str
    profile: str = "INTRADAY"
    analysts: tuple[str, ...] = ("market", "news")
    provider: str = ""
    models: Mapping[str, Any] = None
    model_settings: Mapping[str, Any] = None
    pinned_phase7_generation_id: str | None = None
    pinned_phase8_generation_id: str | None = None
    phase7_generation_id: str | None = None
    phase8_generation_id: str | None = None
    pinned_phase7_root: str | Path | None = None
    pinned_phase8_root: str | Path | None = None
    phase7_root: str | Path | None = None
    phase8_root: str | Path | None = None
    source_database_path: str | Path | None = None
    normalization_path: str | None = None
    audit_path: str | Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_decision_id, str) or not self.source_decision_id.strip():
            raise ValueError("source_decision_id must be non-empty")
        object.__setattr__(self, "analysts", tuple(self.analysts))
        object.__setattr__(self, "models", dict(self.models or {}))
        object.__setattr__(self, "model_settings", dict(self.model_settings or {}))

    @property
    def phase7_generation(self) -> str | None:
        return self.pinned_phase7_generation_id or self.phase7_generation_id

    @property
    def phase8_generation(self) -> str | None:
        return self.pinned_phase8_generation_id or self.phase8_generation_id


@dataclass(frozen=True, slots=True)
class EvidenceReplayReport:
    snapshot_fingerprint: str
    as_of: datetime
    provider: str
    models: Mapping[str, Any]
    model_settings: Mapping[str, Any]
    profile: str
    analysts: tuple[str, ...]
    baseline_action: str | None
    evidence_action: str | None
    did_action_change: bool | None
    evidence_context_hash: str | None
    bundle_status: str
    knowledge_count: int
    experience_count: int
    statistics_count: int
    knowledge_status: str | None
    experience_status: str | None
    statistics_status: str | None
    used_references: tuple[str, ...]
    rejected_references: tuple[Any, ...]
    citation_status: str | None
    baseline_latency_seconds: float | None
    evidence_latency_seconds: float | None
    retrieval_latency_seconds: float | None
    builder_latency_seconds: float | None
    baseline_telemetry: Mapping[str, Any]
    evidence_telemetry: Mapping[str, Any]
    pinned_phase7_generation_id: str | None
    pinned_phase8_generation_id: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    comparison_status: str = "VALID"
    source_integrity: Mapping[str, Any] = None
    integration_status: str = "DISABLED"
    normalization_status: str = "UNKNOWN"
    normalization_error: str | None = None
    context_integrity_status: str = "UNKNOWN"
    reference_validation_status: str = "NOT_RECORDED"
    evidence_use_status: str = "UNAVAILABLE"
    available_references: tuple[str, ...] = ()
    rendered_character_count: int = 0
    context_hash_valid: bool = False
    source_status: Mapping[str, Any] = None
    baseline_normalization_status: str = "UNKNOWN"
    baseline_normalization_error: str | None = None
    rendered_context: str = ""
    loopback_connection_attempts: int = 0
    external_network_attempts: int = 0
    # Safe deterministic retrieval metadata is retained so the Phase 9 audit
    # can prove exactly which query policy produced the injected context.
    knowledge_query: Any = None
    knowledge_query_fingerprint: str | None = None
    query_policy_version: str | None = None
    query_normalization_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)

    # Explicit aliases keep the report readable to callers that use the
    # terminology from the Phase 9 acceptance checklist.
    @property
    def knowledge_items_count(self) -> int:
        return self.knowledge_count

    @property
    def experience_items_count(self) -> int:
        return self.experience_count

    @property
    def statistics_items_count(self) -> int:
        return self.statistics_count

    @property
    def evidence_refs_used(self) -> tuple[str, ...]:
        return self.used_references

    @property
    def evidence_refs_rejected(self) -> tuple[Any, ...]:
        return self.rejected_references


_PRIVACY_FORBIDDEN = ("prompt", "completion", "reasoning", "credential", "apikey", "password", "token", "chainofthought")


def _privacy_forbidden(value: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", value.lower())
    return _contains_forbidden(value) or any(token in normalized for token in _PRIVACY_FORBIDDEN)


def _strict_privacy(value: Any, *, omit_rendered_context: bool = False) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        text = str(value)
        return "[REDACTED_SENSITIVE_TEXT]" if _privacy_forbidden(text) else text
    if is_dataclass(value):
        return {
            field.name: _strict_privacy(getattr(value, field.name), omit_rendered_context=omit_rendered_context)
            for field in fields(value)
            if not _privacy_forbidden(field.name)
            and (not omit_rendered_context or field.name != "rendered_context")
        }
    if isinstance(value, Mapping):
        return {
            key if isinstance(key, str) else "[REDACTED_UNSERIALIZABLE_KEY]": _strict_privacy(item, omit_rendered_context=omit_rendered_context)
            for key, item in value.items()
            if isinstance(key, str) and not _privacy_forbidden(key)
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_strict_privacy(item, omit_rendered_context=omit_rendered_context) for item in value]
    if isinstance(value, str):
        return "[REDACTED_SENSITIVE_TEXT]" if _privacy_forbidden(value) else value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return "[REDACTED_UNSERIALIZABLE]"


def _json_value(value: Any) -> Any:
    return _strict_privacy(value, omit_rendered_context=True)


def _fingerprint(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise SnapshotReplayError(f"source database does not exist: {source}")
    raw = source.read_bytes()
    result: dict[str, Any] = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "mtime_ns": source.stat().st_mtime_ns,
        "tables": {},
        "decision_row_count": None,
    }
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        for table in tables:
            columns = connection.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")').fetchall()
            result["tables"][table] = tuple((row[1], row[2], row[3], row[4], row[5]) for row in columns)
        if "shadow_decisions" in tables:
            result["decision_row_count"] = connection.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
    except sqlite3.Error as exc:
        raise SnapshotReplayError(f"source database is not readable SQLite: {source}") from exc
    finally:
        connection.close()
    return result


def _source_row(path: str | Path, decision_id: str) -> Mapping[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise SnapshotReplayError(f"source database does not exist: {source}")
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    row_dict: dict[str, Any] | None = None
    try:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT * FROM shadow_decisions WHERE decision_id = ?",
            (decision_id,),
        ).fetchone()
        if row is not None:
                row_dict = dict(zip(row.keys(), row, strict=True))
    except sqlite3.Error as exc:
        raise SnapshotReplayError("source database cannot be read for replay") from exc
    finally:
        connection.close()
    if row_dict is None:
        raise SnapshotReplayError(f"source decision not found: {decision_id}")
    return row_dict


def _contains_transient(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).lower().startswith("evidence_") or _contains_transient(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_transient(child) for child in value)
    return False


def _source_has_transient(path: str | Path) -> bool:
    source = Path(path)
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "shadow_decisions" not in tables:
            return False
        columns = {row[1] for row in connection.execute("PRAGMA table_info(shadow_decisions)")}
        for column in {"snapshot_json", "raw_portfolio_manager_result"} & columns:
            for (value,) in connection.execute(f'SELECT "{column}" FROM shadow_decisions'):
                if not isinstance(value, str):
                    continue
                try:
                    if _contains_transient(json.loads(value)):
                        return True
                except json.JSONDecodeError:
                    continue
        return False
    finally:
        connection.close()


def _generation(value: Any) -> tuple[str | None, str | None]:
    if isinstance(value, Mapping):
        return (
            value.get("phase7") or value.get("phase7_generation_id") or value.get("knowledge_generation_id"),
            value.get("phase8") or value.get("phase8_generation_id") or value.get("experience_generation_id"),
        )
    if isinstance(value, (tuple, list)) and len(value) >= 2:
        return (None if value[0] is None else str(value[0]), None if value[1] is None else str(value[1]))
    return (None, None)


def _factory(factory: Callable[..., Any], config: EvidenceReplayConfig) -> Any:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory()
    kwargs = {"config": config, "replay_config": config}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return factory(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in parameters}
    return factory(**accepted)


def _analysis_kwargs(snapshot: ForexMarketSnapshot, snapshot_bytes: bytes, config: EvidenceReplayConfig, enabled: bool) -> dict[str, Any]:
    fingerprint = hashlib.sha256(snapshot_bytes).hexdigest()
    return {
        "symbol": snapshot.symbol,
        "count": max((len(getattr(snapshot, f"{tf}_candles")) for tf in ("m1", "m5", "m15", "h1")), default=1),
        "analysis_date": snapshot.timestamp.date(),
        "analysts": config.analysts,
        "analysis_profile": config.profile,
        "forex_evidence_enabled": enabled,
        "evidence_enabled": enabled,
        "snapshot": snapshot,
        "snapshot_bytes": snapshot_bytes,
        "snapshot_fingerprint": fingerprint,
        "pinned_phase7_generation_id": config.phase7_generation,
        "pinned_phase8_generation_id": config.phase8_generation,
        "provider": config.provider,
        "models": config.models,
        "model_settings": config.model_settings,
        "normalization_path": config.normalization_path,
        "replay": True,
        "persist": False,
        "store": None,
    }


def _invoke_analysis(runner: Any, kwargs: Mapping[str, Any]) -> Any:
    analyze = getattr(runner, "analyze", None)
    if not callable(analyze):
        raise SnapshotReplayError("replay runner must expose analyze()")
    try:
        parameters = inspect.signature(analyze).parameters
    except (TypeError, ValueError):
        return analyze(**dict(kwargs))
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return analyze(**dict(kwargs))
    accepted = {name: value for name, value in kwargs.items() if name in parameters}
    return analyze(**accepted)


def _result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _action(result: Any) -> str | None:
    value = _result_value(result, "normalized_action")
    if value is None:
        decision = _result_value(result, "decision")
        value = _result_value(decision, "action")
    return None if value is None else str(value)


def _context_value(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _context_metrics(result: Any) -> dict[str, Any]:
    context = _result_value(result, "evidence_context")
    if context is None:
        normalization = str(_result_value(result, "normalization_status", "UNKNOWN"))
        return {"context": None, "hash": None, "hash_valid": False, "rendered": "", "rendered_characters": 0, "bundle": "EMPTY", "integration": "DISABLED", "knowledge": 0, "experience": 0, "statistics": 0, "statistics_status": None, "normalization": normalization, "normalization_error": _result_value(result, "normalization_error"), "context_integrity": str(_result_value(result, "context_integrity", {}).get("status", "UNKNOWN")) if isinstance(_result_value(result, "context_integrity", {}), Mapping) else "UNKNOWN", "available": (), "used": (), "rejected": (), "use_status": "DISABLED", "citation": "NOT_RECORDED", "source_status": {}, "loopback": 0, "external": 0, "knowledge_query": None, "knowledge_query_fingerprint": None, "query_policy_version": None, "query_normalization_fingerprint": None}
    diagnostics = _context_value(context, "diagnostics", {})
    source_status = diagnostics.get("source_status", {}) if isinstance(diagnostics, Mapping) else {}
    raw_result = _result_value(result, "raw_portfolio_manager_result", {})
    available = tuple(
        str(getattr(item, "display_id", ""))
        for items in (_context_value(context, "knowledge_items", ()), _context_value(context, "experience_items", ()), _context_value(context, "statistics_items", ()))
        for item in items
        if getattr(item, "display_id", None)
    )
    runtime_status = EvidenceIntegrationStatus(_context_value(context, "integration_status", EvidenceIntegrationStatus.FALLBACK))
    try:
        validation = validate_evidence_references(context, raw_result if isinstance(raw_result, Mapping) else {}, runtime_integration_status=runtime_status)
    except Exception:
        validation = EvidenceReferenceValidation(
            evidence_use_status=EvidenceUseStatus.UNAVAILABLE,
            evidence_audit_status=EvidenceAuditStatus.INVALID_REFERENCE,
        )
    rendered = str(_context_value(context, "rendered_context", ""))
    rendered_hash = _context_value(context, "rendered_context_hash")
    integrity = _result_value(result, "context_integrity", {})
    network = diagnostics.get("network", {}) if isinstance(diagnostics, Mapping) else {}
    return {
        "context": context,
        "hash": rendered_hash,
        "hash_valid": isinstance(rendered_hash, str) and rendered_hash == hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "rendered": rendered,
        "rendered_characters": int(_context_value(context, "rendered_character_count", len(rendered)) or 0),
        "bundle": str(_context_value(context, "bundle_status", "EMPTY")),
        "integration": str(runtime_status),
        "knowledge": int(_context_value(context, "selected_knowledge_count", 0) or 0),
        "experience": int(_context_value(context, "selected_experience_count", 0) or 0),
        "statistics": int(_context_value(context, "selected_statistics_count", 0) or 0),
        "knowledge_status": source_status.get("knowledge"),
        "experience_status": source_status.get("experience"),
        "statistics_status": _context_value(context, "statistics_status"),
        "normalization": str(_result_value(result, "normalization_status", "UNKNOWN")),
        "normalization_error": _result_value(result, "normalization_error"),
        "context_integrity": str(integrity.get("status", "UNKNOWN")) if isinstance(integrity, Mapping) else "UNKNOWN",
        "available": available,
        "used": tuple(validation.evidence_refs_used),
        "rejected": tuple(validation.evidence_refs_rejected),
        "use_status": str(validation.evidence_use_status),
        "citation": str(validation.evidence_audit_status),
        "source_status": source_status,
        "loopback": int(network.get("loopback_connection_attempts", 0) or 0) if isinstance(network, Mapping) else 0,
        "external": int(network.get("external_network_attempts", 0) or 0) if isinstance(network, Mapping) else 0,
        "knowledge_query": _context_value(context, "knowledge_query"),
        "knowledge_query_fingerprint": _context_value(context, "knowledge_query_fingerprint"),
        "query_policy_version": _context_value(context, "knowledge_query_policy_version"),
        "query_normalization_fingerprint": _context_value(context, "query_normalization_fingerprint"),
    }


def _telemetry(result: Any) -> Mapping[str, Any]:
    value = _result_value(result, "analysis_telemetry", {})
    if not isinstance(value, Mapping):
        return {}

    def redact(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                key: redact(child)
                for key, child in item.items()
                if not any(token in str(key).lower() for token in ("prompt", "completion", "reasoning", "credential", "api_key"))
            }
        if isinstance(item, (tuple, list)):
            return type(item)(redact(child) for child in item)
        return item

    return redact(value)


class SavedSnapshotReplay:
    """Run two injected, non-persisting analyses over one saved snapshot."""

    def __init__(
        self,
        runner_factory: Callable[..., Any] | None = None,
        generation_provider: Callable[[], Any] | Any | None = None,
        runner: Any | None = None,
    ) -> None:
        if runner is not None:
            if runner_factory is not None:
                raise ValueError("provide runner or runner_factory, not both")
            fixed_runner = runner

            def fixed_factory(**_: Any) -> Any:
                return fixed_runner

            runner_factory = fixed_factory
        elif runner_factory is not None and not callable(runner_factory) and callable(getattr(runner_factory, "analyze", None)):
            runner = runner_factory
            fixed_runner = runner

            def fixed_factory(**_: Any) -> Any:
                return fixed_runner

            runner_factory = fixed_factory
        self.runner_factory = runner_factory
        self.generation_provider = generation_provider

    def _generations(self, config: EvidenceReplayConfig) -> tuple[str | None, str | None]:
        if self.generation_provider is None:
            return (config.phase7_generation, config.phase8_generation)
        value = self.generation_provider() if callable(self.generation_provider) else self.generation_provider
        return _generation(value)

    def run(self, snapshot: ForexMarketSnapshot | None = None, *, snapshot_bytes: bytes | None = None, config: EvidenceReplayConfig) -> EvidenceReplayReport:
        if not isinstance(config, EvidenceReplayConfig):
            raise TypeError("config must be EvidenceReplayConfig")
        if config.audit_path is not None:
            raise SnapshotReplayError(
                "audit_path is not supported by replay; use a Phase 9 evidence_runtime audit sink"
            )
        if config.source_database_path is None:
            raise SnapshotReplayError("source database path is required for saved replay")
        source_row = None
        source_snapshot_bytes = None
        source_row = _source_row(config.source_database_path, config.source_decision_id)
        source_snapshot_value = source_row["snapshot_json"]
        if isinstance(source_snapshot_value, bytes):
            source_snapshot_bytes = source_snapshot_value
        elif isinstance(source_snapshot_value, str):
            source_snapshot_bytes = source_snapshot_value.encode("utf-8")
        else:
            raise SnapshotReplayError("source snapshot_json must be text or bytes")
        if snapshot_bytes is not None and snapshot_bytes != source_snapshot_bytes:
            raise SnapshotReplayError("snapshot bytes do not match source decision")
        snapshot_bytes = source_snapshot_bytes
        decoded_snapshot = SavedSnapshotCodec.from_source_row(dict(source_row))
        if snapshot is not None and snapshot != decoded_snapshot:
            raise SnapshotReplayError("snapshot does not match source decision")
        snapshot = decoded_snapshot
        if not isinstance(snapshot, ForexMarketSnapshot):
            raise SnapshotReplayError("snapshot must be loaded from the source decision")
        if not isinstance(snapshot_bytes, bytes):
            raise SnapshotReplayError("snapshot_bytes must be loaded from the source decision")
        source_before = _fingerprint(config.source_database_path)
        if _source_has_transient(config.source_database_path):
            raise SnapshotReplayError("source row contains transient Phase 9 evidence metadata")
        observed7, observed8 = self._generations(config)
        configured7, configured8 = config.phase7_generation, config.phase8_generation
        pinned7 = configured7 or observed7
        pinned8 = configured8 or observed8
        if pinned7 is None or pinned8 is None or (
            self.generation_provider is not None and (observed7 is None or observed8 is None)
        ):
            raise SnapshotReplayError(
                "both Phase 7 and Phase 8 generation IDs are required at replay start"
            )
        generation_changed = (
            (configured7 is not None and observed7 is not None and observed7 != configured7)
            or (configured8 is not None and observed8 is not None and observed8 != configured8)
        )
        effective_config = EvidenceReplayConfig(**{**{field.name: getattr(config, field.name) for field in fields(config)}, "pinned_phase7_generation_id": pinned7, "pinned_phase8_generation_id": pinned8})
        if self.runner_factory is None:
            raise SnapshotReplayError(
                "runner_factory is required for saved replay; inject a read-only ForexShadowRunner seam"
            )
        runner = _factory(self.runner_factory, effective_config)

        kwargs_a = _analysis_kwargs(snapshot, snapshot_bytes, effective_config, False)
        kwargs_b = _analysis_kwargs(snapshot, snapshot_bytes, effective_config, True)
        baseline = _invoke_analysis(runner, kwargs_a)
        before_b7, before_b8 = self._generations(effective_config)
        generation_changed = generation_changed or before_b7 is None or before_b8 is None or before_b7 != pinned7 or before_b8 != pinned8
        evidence = _invoke_analysis(runner, kwargs_b)
        after7, after8 = self._generations(effective_config)
        generation_changed = generation_changed or after7 is None or after8 is None or after7 != pinned7 or after8 != pinned8
        for result in (baseline, evidence):
            context = _result_value(result, "evidence_context")
            if context is not None:
                generation_changed = generation_changed or (pinned7 is not None and _context_value(context, "knowledge_generation_id") not in (pinned7,)) or (pinned8 is not None and _context_value(context, "experience_generation_id") not in (pinned8,))
            elif result is evidence:
                generation_changed = True
        source_after = _fingerprint(config.source_database_path)
        if source_before is not None and source_after != source_before:
            raise SnapshotReplayError("source database changed during replay")
        if _source_has_transient(config.source_database_path):
            raise SnapshotReplayError("source row gained transient Phase 9 evidence metadata")

        baseline_metrics = _context_metrics(baseline)
        evidence_metrics = _context_metrics(evidence)
        telemetry_a = _telemetry(baseline)
        telemetry_b = _telemetry(evidence)
        baseline_action, evidence_action = _action(baseline), _action(evidence)
        status = "INVALID_GENERATION_CHANGED" if generation_changed else "VALID"
        return EvidenceReplayReport(
            snapshot_fingerprint=hashlib.sha256(snapshot_bytes).hexdigest(),
            as_of=snapshot.timestamp,
            provider=config.provider,
            models=config.models,
            model_settings=config.model_settings,
            profile=config.profile,
            analysts=config.analysts,
            baseline_action=baseline_action,
            evidence_action=evidence_action,
            did_action_change=None if generation_changed else baseline_action != evidence_action,
            evidence_context_hash=evidence_metrics["hash"],
            bundle_status=evidence_metrics["bundle"],
            knowledge_count=evidence_metrics["knowledge"],
            experience_count=evidence_metrics["experience"],
            statistics_count=evidence_metrics["statistics"],
            knowledge_status=evidence_metrics.get("knowledge_status"),
            experience_status=evidence_metrics.get("experience_status"),
            statistics_status=evidence_metrics["statistics_status"],
            used_references=evidence_metrics["used"],
            rejected_references=evidence_metrics["rejected"],
            citation_status=evidence_metrics["citation"],
            baseline_latency_seconds=telemetry_a.get("analysis_latency_seconds"),
            evidence_latency_seconds=telemetry_b.get("analysis_latency_seconds"),
            retrieval_latency_seconds=telemetry_b.get("evidence_retrieval_latency_seconds"),
            builder_latency_seconds=telemetry_b.get("evidence_builder_latency_seconds"),
            baseline_telemetry=telemetry_a,
            evidence_telemetry=telemetry_b,
            pinned_phase7_generation_id=pinned7,
            pinned_phase8_generation_id=pinned8,
            warnings=(),
            errors=("generation changed during replay",) if generation_changed else (),
            comparison_status=status,
            source_integrity={"before": source_before, "after": source_after},
            integration_status=evidence_metrics["integration"],
            normalization_status=evidence_metrics["normalization"],
            normalization_error=evidence_metrics["normalization_error"],
            context_integrity_status=evidence_metrics["context_integrity"],
            reference_validation_status=evidence_metrics["citation"],
            evidence_use_status=evidence_metrics["use_status"],
            available_references=evidence_metrics["available"],
            rendered_character_count=evidence_metrics["rendered_characters"],
            context_hash_valid=evidence_metrics["hash_valid"],
            source_status=evidence_metrics["source_status"],
            baseline_normalization_status=baseline_metrics["normalization"],
            baseline_normalization_error=baseline_metrics["normalization_error"],
            rendered_context=evidence_metrics["rendered"],
            loopback_connection_attempts=evidence_metrics["loopback"],
            external_network_attempts=evidence_metrics["external"],
            knowledge_query=evidence_metrics["knowledge_query"],
            knowledge_query_fingerprint=evidence_metrics["knowledge_query_fingerprint"],
            query_policy_version=evidence_metrics["query_policy_version"],
            query_normalization_fingerprint=evidence_metrics["query_normalization_fingerprint"],
        )


__all__ = [
    "EvidenceReplayConfig",
    "EvidenceReplayReport",
    "SavedSnapshotCodec",
    "SavedSnapshotReplay",
    "SnapshotReplayError",
]
