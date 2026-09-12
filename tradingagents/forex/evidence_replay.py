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


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if is_dataclass(value):
        return {field.name: _json_value(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in value.items()
            if not any(
                token in str(key).lower()
                for token in ("prompt", "completion", "reasoning", "credential", "api_key")
            )
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_value(item) for item in value]
    return value


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
    try:
        with sqlite3.connect(uri, uri=True) as connection:
            tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            for table in tables:
                columns = connection.execute(f'PRAGMA table_info("{table.replace(chr(34), chr(34) * 2)}")').fetchall()
                result["tables"][table] = tuple((row[1], row[2], row[3], row[4], row[5]) for row in columns)
            if "shadow_decisions" in tables:
                result["decision_row_count"] = connection.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
    except sqlite3.Error as exc:
        raise SnapshotReplayError(f"source database is not readable SQLite: {source}") from exc
    return result


def _contains_transient(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(str(key).lower().startswith("evidence_") or _contains_transient(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_transient(child) for child in value)
    return False


def _source_has_transient(path: str | Path) -> bool:
    source = Path(path)
    uri = f"file:{source.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as connection:
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
        return {"context": None, "hash": None, "bundle": "EMPTY", "knowledge": 0, "experience": 0, "statistics": 0, "statistics_status": None, "used": (), "rejected": (), "citation": None}
    diagnostics = _context_value(context, "diagnostics", {})
    source_status = diagnostics.get("source_status", {}) if isinstance(diagnostics, Mapping) else {}
    used = _result_value(result, "evidence_refs_used", ())
    rejected = _result_value(result, "evidence_refs_rejected", ())
    raw_result = _result_value(result, "raw_portfolio_manager_result", {})
    if not used and isinstance(raw_result, Mapping):
        used = raw_result.get("evidence_refs_used", ())
    if not rejected and isinstance(raw_result, Mapping):
        rejected = raw_result.get("evidence_refs_rejected", ())
    if not isinstance(used, (tuple, list)):
        used = ()
    if not isinstance(rejected, (tuple, list)):
        rejected = ()
    return {
        "context": context,
        "hash": _context_value(context, "rendered_context_hash"),
        "bundle": str(_context_value(context, "bundle_status", "EMPTY")),
        "knowledge": int(_context_value(context, "selected_knowledge_count", 0) or 0),
        "experience": int(_context_value(context, "selected_experience_count", 0) or 0),
        "statistics": int(_context_value(context, "selected_statistics_count", 0) or 0),
        "knowledge_status": source_status.get("knowledge"),
        "experience_status": source_status.get("experience"),
        "statistics_status": _context_value(context, "statistics_status"),
        "used": tuple(str(value) for value in used),
        "rejected": tuple(rejected),
        "citation": _result_value(result, "citation_status")
        or (raw_result.get("evidence_audit_status") if isinstance(raw_result, Mapping) else None),
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
        value = self.generation_provider() if callable(self.generation_provider) else self.generation_provider
        phase7, phase8 = _generation(value)
        return (phase7 or config.phase7_generation, phase8 or config.phase8_generation)

    def run(self, snapshot: ForexMarketSnapshot, *, snapshot_bytes: bytes, config: EvidenceReplayConfig) -> EvidenceReplayReport:
        if not isinstance(snapshot, ForexMarketSnapshot):
            raise SnapshotReplayError("snapshot must be ForexMarketSnapshot")
        if not isinstance(snapshot_bytes, bytes):
            raise SnapshotReplayError("snapshot_bytes must be immutable bytes")
        if not isinstance(config, EvidenceReplayConfig):
            raise TypeError("config must be EvidenceReplayConfig")
        source_before = _fingerprint(config.source_database_path) if config.source_database_path is not None else None
        if config.source_database_path is not None and _source_has_transient(config.source_database_path):
            raise SnapshotReplayError("source row contains transient Phase 9 evidence metadata")
        pinned7, pinned8 = self._generations(config)
        effective_config = EvidenceReplayConfig(**{**{field.name: getattr(config, field.name) for field in fields(config)}, "pinned_phase7_generation_id": pinned7, "pinned_phase8_generation_id": pinned8})
        if self.runner_factory is None:
            raise SnapshotReplayError(
                "runner_factory is required for saved replay; inject a read-only ForexShadowRunner seam"
            )
        runner = _factory(self.runner_factory, effective_config)

        kwargs_a = _analysis_kwargs(snapshot, snapshot_bytes, effective_config, False)
        kwargs_b = _analysis_kwargs(snapshot, snapshot_bytes, effective_config, True)
        baseline = _invoke_analysis(runner, kwargs_a)
        evidence = _invoke_analysis(runner, kwargs_b)
        after7, after8 = self._generations(effective_config)
        generation_changed = (after7 is not None and pinned7 is not None and after7 != pinned7) or (after8 is not None and pinned8 is not None and after8 != pinned8)
        for result in (baseline, evidence):
            context = _result_value(result, "evidence_context")
            if context is not None:
                generation_changed = generation_changed or (pinned7 is not None and _context_value(context, "knowledge_generation_id") not in (None, pinned7)) or (pinned8 is not None and _context_value(context, "experience_generation_id") not in (None, pinned8))
        source_after = _fingerprint(config.source_database_path) if config.source_database_path is not None else None
        if source_before is not None and source_after != source_before:
            raise SnapshotReplayError("source database changed during replay")
        if config.source_database_path is not None and _source_has_transient(config.source_database_path):
            raise SnapshotReplayError("source row gained transient Phase 9 evidence metadata")

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
        )


__all__ = [
    "EvidenceReplayConfig",
    "EvidenceReplayReport",
    "SavedSnapshotCodec",
    "SavedSnapshotReplay",
    "SnapshotReplayError",
]
