from __future__ import annotations

import importlib.util
import json
import math
import re
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal


def _load_schemas_module():
    module_name = "tradingagents.agents.schemas"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    schemas_path = Path(__file__).resolve().parents[1] / "agents" / "schemas.py"
    spec = importlib.util.spec_from_file_location(module_name, schemas_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError("could not load tradingagents.agents.schemas")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module.PortfolioDecision, "model_rebuild"):
        module.PortfolioDecision.model_rebuild(
            _types_namespace={"PortfolioRating": module.PortfolioRating}
        )
    return module


_schemas = _load_schemas_module()
PortfolioDecision = _schemas.PortfolioDecision
PortfolioRating = _schemas.PortfolioRating

AllowedAction = Literal["BUY", "SELL", "HOLD"]
NormalizationStatus = Literal["NORMALIZED", "FAILED"]
DecisionContextStatus = Literal["COMPLETE", "INCOMPLETE"]
FutureEvaluationStatus = Literal["PENDING", "RESOLVED"]
DecisionReferenceStatus = Literal["AVAILABLE", "UNAVAILABLE", "INVALID_TEMPORAL"]

_RATING_TO_ACTION: dict[str, AllowedAction] = {
    PortfolioRating.BUY.value: "BUY",
    PortfolioRating.OVERWEIGHT.value: "BUY",
    PortfolioRating.HOLD.value: "HOLD",
    PortfolioRating.UNDERWEIGHT.value: "SELL",
    PortfolioRating.SELL.value: "SELL",
}

_FOREX_LONG_HORIZON_RE = re.compile(
    r"\b(?:month|months|year|years|quarter|quarters|long[- ]term|equity investment)\b",
    re.IGNORECASE,
)


def _validate_forex_payload(payload: Mapping[str, Any], profile_name: str) -> str | None:
    profile_value = payload.get("analysis_profile")
    if profile_value is not None and profile_value != profile_name:
        return (
            f"forex analysis_profile must be exactly {profile_name}, "
            f"got {profile_value!r}"
        )

    horizon = payload.get("time_horizon")
    if horizon is not None:
        if not isinstance(horizon, str):
            return "forex time_horizon must be a string when provided"
        if _FOREX_LONG_HORIZON_RE.search(horizon):
            return "forex time_horizon must be intraday minutes/hours, not months or years"

    validity = payload.get("valid_for_seconds")
    if validity is not None:
        if isinstance(validity, bool) or not isinstance(validity, int):
            return "forex valid_for_seconds must be an integer number of seconds"
        if not 60 <= validity <= 86400:
            return "forex valid_for_seconds must be between 60 and 86400 seconds"
    return None
def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value, key=repr)]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        utc_value = value.astimezone(timezone.utc)
        return utc_value.isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_safe(value.value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("non-finite float values are not allowed")
        return value
    raise TypeError(f"unsupported JSON value type: {type(value).__name__}")


def _parse_utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        if value.utcoffset() != timezone.utc.utcoffset(value):
            raise ValueError("timestamps must be UTC values")
        result = value.astimezone(timezone.utc)
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        if result.utcoffset() != timezone.utc.utcoffset(result):
            raise ValueError("timestamps must be UTC values")
        result = result.astimezone(timezone.utc)
    return result


def _canonical_rating(value: Any) -> str:
    if isinstance(value, PortfolioRating):
        return value.value
    if isinstance(value, str) and value in _RATING_TO_ACTION:
        return value
    raise ValueError("rating must be an exact PortfolioRating value")


def _normalize_from_payload(
    payload: Mapping[str, Any],
    *,
    forex_profile: str | None = None,
) -> ShadowNormalization:
    try:
        safe_payload = _json_safe(dict(payload))
    except TypeError as exc:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=str(exc),
            raw_result={"error": str(exc)},
        )

    if "rating" not in payload:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=safe_payload,
        )

    if forex_profile is not None:
        forex_error = _validate_forex_payload(payload, forex_profile)
        if forex_error is not None:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=forex_error,
                raw_result=safe_payload,
            )

    try:
        rating_text = _canonical_rating(payload["rating"])
    except ValueError:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                f"unknown PortfolioRating value: {payload['rating']!r}"
            ),
            raw_result=safe_payload,
        )
    except TypeError:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=safe_payload,
        )

    mapped_action = _RATING_TO_ACTION.get(rating_text)
    if mapped_action is None:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                f"unknown PortfolioRating value: {rating_text!r}"
            ),
            raw_result=safe_payload,
        )

    for alias in ("recommendation", "action"):
        if alias in payload and payload[alias] != payload["rating"]:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=(
                    "conflicting structured Portfolio Manager rating values"
                ),
                raw_result=safe_payload,
            )

    return ShadowNormalization(
        action=mapped_action,
        normalization_status="NORMALIZED",
        normalization_error=None,
        raw_result=safe_payload,
    )


@dataclass(frozen=True, slots=True)
class ShadowNormalization:
    action: AllowedAction | None
    normalization_status: NormalizationStatus
    normalization_error: str | None
    raw_result: dict[str, Any]


def normalize_portfolio_manager_result(
    raw: PortfolioDecision | Mapping[str, Any] | None,
    *,
    forex_profile: str | None = None,
) -> ShadowNormalization:
    if raw is None:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output is required and was None"
            ),
            raw_result={},
        )

    if isinstance(raw, str):
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must not be prose"
            ),
            raw_result={"value": raw},
        )

    if isinstance(raw, PortfolioDecision):
        payload = raw.model_dump(mode="json")
        return _normalize_from_payload(payload, forex_profile=forex_profile)

    if isinstance(raw, Mapping):
        return _normalize_from_payload(raw, forex_profile=forex_profile)

    return ShadowNormalization(
        action=None,
        normalization_status="FAILED",
        normalization_error=(
            "structured Portfolio Manager output must be a PortfolioDecision or mapping"
        ),
        raw_result={"value": str(type(raw).__name__)},
    )


@dataclass(frozen=True, slots=True)
class ShadowTradeDecision:
    decision_id: str
    created_at: datetime
    snapshot_timestamp: datetime
    analysis_date: date
    requested_symbol: str
    resolved_symbol: str
    action: AllowedAction | None
    raw_portfolio_manager_result: dict[str, Any]
    normalization_status: NormalizationStatus
    normalization_error: str | None
    confidence: float | None
    reference_bid: float
    reference_ask: float
    reference_mid: float
    spread: float
    spread_points: float
    analysis_timeframe: str
    trader_summary: str
    portfolio_manager_summary: str
    bull_summary: str | None = None
    bear_summary: str | None = None
    llm_provider: str | None = None
    quick_model: str | None = None
    deep_model: str | None = None
    snapshot_json: dict[str, Any] | None = None
    future_evaluation_status: FutureEvaluationStatus = "PENDING"
    outcome_raw: float | None = None
    outcome_alpha: float | None = None
    outcome_resolved_at: datetime | None = None
    reflection: str | None = None
    source_run_id: str | None = None
    analysis_profile: str = "INTRADAY"
    valid_for_seconds: int | None = None
    valid_until: datetime | None = None
    executed: bool = False
    decision_context_status: DecisionContextStatus = "INCOMPLETE"
    # Explicit temporal evidence added for Phase 5.  These fields are
    # defaulted at the end so existing callers that construct Phase 4
    # decisions positionally remain compatible.
    analysis_snapshot_timestamp: datetime | None = None
    analysis_snapshot_bid: float | None = None
    analysis_snapshot_ask: float | None = None
    analysis_snapshot_spread: float | None = None
    analysis_snapshot_spread_points: float | None = None
    decision_completed_timestamp: datetime | None = None
    analysis_latency_seconds: float | None = None
    decision_reference_timestamp: datetime | None = None
    decision_reference_bid: float | None = None
    decision_reference_ask: float | None = None
    decision_reference_spread: float | None = None
    decision_reference_spread_points: float | None = None
    decision_reference_status: DecisionReferenceStatus = "UNAVAILABLE"
    decision_reference_delay_seconds: float | None = None
    decision_reference_error: str | None = None

    def __post_init__(self) -> None:
        if self.executed is not False:
            raise ValueError("ShadowTradeDecision must always be created with executed=False")
        if self.normalization_status not in ("NORMALIZED", "FAILED"):
            raise ValueError("normalization_status must be NORMALIZED or FAILED")
        if self.decision_context_status not in ("COMPLETE", "INCOMPLETE"):
            raise ValueError("decision_context_status must be COMPLETE or INCOMPLETE")
        if self.future_evaluation_status not in ("PENDING", "RESOLVED"):
            raise ValueError("future_evaluation_status must be PENDING or RESOLVED")
        if not isinstance(self.analysis_profile, str) or not self.analysis_profile.strip():
            raise ValueError("analysis_profile must be a non-empty string")
        if self.valid_for_seconds is not None:
            if (
                isinstance(self.valid_for_seconds, bool)
                or not isinstance(self.valid_for_seconds, int)
                or not 60 <= self.valid_for_seconds <= 86400
            ):
                raise ValueError("valid_for_seconds must be an integer between 60 and 86400")
            if self.valid_until is None:
                raise ValueError("valid_until is required when valid_for_seconds is set")
        elif self.valid_until is not None:
            raise ValueError("valid_for_seconds is required when valid_until is set")
        if self.normalization_status == "NORMALIZED" and self.action is None:
            raise ValueError("normalized decisions must include an action")
        if self.normalization_status == "FAILED" and self.action is not None:
            raise ValueError("failed decisions must not include an action")
        if self.action is not None and self.action not in ("BUY", "SELL", "HOLD"):
            raise ValueError("action must be BUY, SELL, HOLD, or None")
        _parse_utc_datetime(self.snapshot_timestamp)
        _parse_utc_datetime(self.created_at)

        # Phase 4's snapshot/reference fields are the analysis quote.  Fill
        # the explicit analysis fields from those aliases for old callers and
        # old rows, while retaining the actual values for new decisions.
        analysis_timestamp = self.analysis_snapshot_timestamp or self.snapshot_timestamp
        analysis_timestamp = _parse_utc_datetime(analysis_timestamp)
        object.__setattr__(self, "analysis_snapshot_timestamp", analysis_timestamp)
        for explicit_name, legacy_name in (
            ("analysis_snapshot_bid", "reference_bid"),
            ("analysis_snapshot_ask", "reference_ask"),
            ("analysis_snapshot_spread", "spread"),
            ("analysis_snapshot_spread_points", "spread_points"),
        ):
            value = getattr(self, explicit_name)
            if value is None:
                value = getattr(self, legacy_name)
                object.__setattr__(self, explicit_name, value)

        if self.decision_completed_timestamp is not None:
            completed = _parse_utc_datetime(self.decision_completed_timestamp)
            object.__setattr__(self, "decision_completed_timestamp", completed)
            computed_latency = (
                completed - analysis_timestamp
            ).total_seconds()
            if computed_latency < 0:
                raise ValueError("decision_completed_timestamp cannot precede analysis snapshot")
            if self.analysis_latency_seconds is None:
                object.__setattr__(self, "analysis_latency_seconds", computed_latency)
            elif not math.isclose(
                float(self.analysis_latency_seconds), computed_latency, rel_tol=0.0, abs_tol=1e-6
            ):
                raise ValueError("analysis_latency_seconds does not match decision timestamps")
        elif self.analysis_latency_seconds is not None:
            raise ValueError("analysis_latency_seconds requires decision_completed_timestamp")

        if self.decision_reference_status not in (
            "AVAILABLE",
            "UNAVAILABLE",
            "INVALID_TEMPORAL",
        ):
            raise ValueError(
                "decision_reference_status must be AVAILABLE, UNAVAILABLE, or INVALID_TEMPORAL"
            )
        if self.decision_reference_timestamp is not None:
            reference_timestamp = _parse_utc_datetime(self.decision_reference_timestamp)
            object.__setattr__(self, "decision_reference_timestamp", reference_timestamp)
        if self.decision_reference_status == "UNAVAILABLE":
            if any(
                value is not None
                for value in (
                    self.decision_reference_timestamp,
                    self.decision_reference_bid,
                    self.decision_reference_ask,
                    self.decision_reference_spread,
                    self.decision_reference_spread_points,
                )
            ):
                raise ValueError("unavailable decision references must not include quote values")
            if self.decision_reference_delay_seconds is not None:
                raise ValueError("unavailable decision references must not include delay")
        else:
            if self.decision_reference_timestamp is None:
                raise ValueError("available or invalid decision references require a timestamp")
            if any(
                value is None
                for value in (
                    self.decision_reference_bid,
                    self.decision_reference_ask,
                    self.decision_reference_spread,
                    self.decision_reference_spread_points,
                )
            ):
                raise ValueError("decision references require a complete quote set")
            if self.decision_completed_timestamp is None:
                raise ValueError("decision references require decision_completed_timestamp")
            computed_delay = (
                self.decision_reference_timestamp - self.decision_completed_timestamp
            ).total_seconds()
            if self.decision_reference_delay_seconds is None:
                object.__setattr__(self, "decision_reference_delay_seconds", computed_delay)
            elif not math.isclose(
                float(self.decision_reference_delay_seconds),
                computed_delay,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("decision_reference_delay_seconds does not match reference timestamps")
            if self.decision_reference_status == "AVAILABLE" and computed_delay < 0:
                raise ValueError("AVAILABLE decision references cannot precede completion")
            if self.decision_reference_status == "INVALID_TEMPORAL" and computed_delay >= 0:
                raise ValueError("INVALID_TEMPORAL references must precede completion")
        if self.valid_until is not None:
            valid_until = _parse_utc_datetime(self.valid_until)
            expected = _parse_utc_datetime(self.snapshot_timestamp) + timedelta(
                seconds=self.valid_for_seconds
            )
            if valid_until != expected:
                raise ValueError("valid_until must equal snapshot_timestamp + valid_for_seconds")
        if self.outcome_resolved_at is not None:
            _parse_utc_datetime(self.outcome_resolved_at)
        if self.snapshot_json is None:
            object.__setattr__(self, "snapshot_json", {})
        else:
            _json_safe(self.snapshot_json)
        _json_safe(self.raw_portfolio_manager_result)
        for field_name in (
            "confidence",
            "reference_bid",
            "reference_ask",
            "reference_mid",
            "spread",
            "spread_points",
            "analysis_snapshot_bid",
            "analysis_snapshot_ask",
            "analysis_snapshot_spread",
            "analysis_snapshot_spread_points",
            "analysis_latency_seconds",
            "decision_reference_bid",
            "decision_reference_ask",
            "decision_reference_spread",
            "decision_reference_spread_points",
            "decision_reference_delay_seconds",
            "outcome_raw",
            "outcome_alpha",
        ):
            value = getattr(self, field_name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite when provided")
        for prefix in ("analysis_snapshot", "decision_reference"):
            bid = getattr(self, f"{prefix}_bid")
            ask = getattr(self, f"{prefix}_ask")
            if bid is not None and ask is not None and ask < bid:
                raise ValueError(f"{prefix} ask must be greater than or equal to bid")

    @property
    def raw_portfolio_manager_result_json(self) -> str:
        return json.dumps(
            _json_safe(self.raw_portfolio_manager_result),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )


class ShadowDecisionStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS shadow_decisions (
                    decision_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    requested_symbol TEXT NOT NULL,
                    resolved_symbol TEXT NOT NULL,
                    snapshot_timestamp TEXT NOT NULL,
                    action TEXT CHECK (action IS NULL OR action IN ('BUY','SELL','HOLD')),
                    normalization_status TEXT NOT NULL
                        CHECK (normalization_status IN ('NORMALIZED','FAILED')),
                    decision_context_status TEXT NOT NULL DEFAULT 'INCOMPLETE'
                        CHECK (decision_context_status IN ('COMPLETE','INCOMPLETE')),
                    normalization_error TEXT,
                    raw_portfolio_manager_result TEXT NOT NULL,
                    confidence REAL,
                    reference_bid REAL NOT NULL,
                    reference_ask REAL NOT NULL,
                    reference_mid REAL NOT NULL,
                    spread REAL NOT NULL,
                    spread_points REAL NOT NULL,
                    analysis_timeframe TEXT NOT NULL,
                    analysis_profile TEXT NOT NULL DEFAULT 'INTRADAY',
                    valid_for_seconds INTEGER,
                    valid_until TEXT,
                    trader_summary TEXT NOT NULL,
                    portfolio_manager_summary TEXT NOT NULL,
                    bull_summary TEXT,
                    bear_summary TEXT,
                    llm_provider TEXT,
                    quick_model TEXT,
                    deep_model TEXT,
                    snapshot_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0),
                    future_evaluation_status TEXT NOT NULL DEFAULT 'PENDING'
                        CHECK (future_evaluation_status IN ('PENDING','RESOLVED')),
                    outcome_raw REAL,
                    outcome_alpha REAL,
                    outcome_resolved_at TEXT,
                    reflection TEXT,
                    source_run_id TEXT,
                    CHECK (
                        (normalization_status = 'NORMALIZED' AND action IS NOT NULL
                            AND normalization_error IS NULL)
                        OR
                        (normalization_status = 'FAILED' AND action IS NULL)
                    )
                )
                """
            )
            existing_columns = {
                row[1]
                for row in conn.execute("PRAGMA table_info(shadow_decisions)").fetchall()
            }
            migrations = {
                "analysis_profile": "TEXT NOT NULL DEFAULT 'INTRADAY'",
                "valid_for_seconds": "INTEGER",
                "valid_until": "TEXT",
                "decision_context_status": (
                    "TEXT NOT NULL DEFAULT 'INCOMPLETE' "
                    "CHECK (decision_context_status IN ('COMPLETE','INCOMPLETE'))"
                ),
                "analysis_snapshot_timestamp": "TEXT",
                "analysis_snapshot_bid": "REAL",
                "analysis_snapshot_ask": "REAL",
                "analysis_snapshot_spread": "REAL",
                "analysis_snapshot_spread_points": "REAL",
                "decision_completed_timestamp": "TEXT",
                "analysis_latency_seconds": "REAL",
                "decision_reference_timestamp": "TEXT",
                "decision_reference_bid": "REAL",
                "decision_reference_ask": "REAL",
                "decision_reference_spread": "REAL",
                "decision_reference_spread_points": "REAL",
                "decision_reference_status": (
                    "TEXT NOT NULL DEFAULT 'UNAVAILABLE' "
                    "CHECK (decision_reference_status IN "
                    "('AVAILABLE','UNAVAILABLE','INVALID_TEMPORAL'))"
                ),
                "decision_reference_delay_seconds": "REAL",
                "decision_reference_error": "TEXT",
            }
            for column, declaration in migrations.items():
                if column not in existing_columns:
                    conn.execute(
                        f"ALTER TABLE shadow_decisions ADD COLUMN {column} {declaration}"
                    )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_decisions_resolved_symbol_date
                ON shadow_decisions (resolved_symbol, analysis_date)
                """
            )

    def record(self, decision: ShadowTradeDecision) -> None:
        self.initialize()
        columns = [
            "decision_id",
            "created_at",
            "analysis_date",
            "requested_symbol",
            "resolved_symbol",
            "snapshot_timestamp",
            "action",
            "normalization_status",
            "decision_context_status",
            "normalization_error",
            "raw_portfolio_manager_result",
            "confidence",
            "reference_bid",
            "reference_ask",
            "reference_mid",
            "spread",
            "spread_points",
            "analysis_timeframe",
            "analysis_profile",
            "valid_for_seconds",
            "valid_until",
            "trader_summary",
            "portfolio_manager_summary",
            "bull_summary",
            "bear_summary",
            "llm_provider",
            "quick_model",
            "deep_model",
            "snapshot_json",
            "executed",
            "future_evaluation_status",
            "outcome_raw",
            "outcome_alpha",
            "outcome_resolved_at",
            "reflection",
            "source_run_id",
            "analysis_snapshot_timestamp",
            "analysis_snapshot_bid",
            "analysis_snapshot_ask",
            "analysis_snapshot_spread",
            "analysis_snapshot_spread_points",
            "decision_completed_timestamp",
            "analysis_latency_seconds",
            "decision_reference_timestamp",
            "decision_reference_bid",
            "decision_reference_ask",
            "decision_reference_spread",
            "decision_reference_spread_points",
            "decision_reference_status",
            "decision_reference_delay_seconds",
            "decision_reference_error",
        ]
        placeholders = ", ".join(["?"] * len(columns))
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                f"""
                INSERT INTO shadow_decisions ({", ".join(columns)}) VALUES ({placeholders})
                ON CONFLICT(decision_id) DO NOTHING
                """,
                (
                    decision.decision_id,
                    _parse_utc_datetime(decision.created_at).isoformat().replace("+00:00", "Z"),
                    decision.analysis_date.isoformat(),
                    decision.requested_symbol,
                    decision.resolved_symbol,
                    _parse_utc_datetime(decision.snapshot_timestamp).isoformat().replace("+00:00", "Z"),
                    decision.action,
                    decision.normalization_status,
                    decision.decision_context_status,
                    decision.normalization_error,
                    decision.raw_portfolio_manager_result_json,
                    decision.confidence,
                    decision.reference_bid,
                    decision.reference_ask,
                    decision.reference_mid,
                    decision.spread,
                    decision.spread_points,
                    decision.analysis_timeframe,
                    decision.analysis_profile,
                    decision.valid_for_seconds,
                    None
                    if decision.valid_until is None
                    else _parse_utc_datetime(decision.valid_until)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    decision.trader_summary,
                    decision.portfolio_manager_summary,
                    decision.bull_summary,
                    decision.bear_summary,
                    decision.llm_provider,
                    decision.quick_model,
                    decision.deep_model,
                    json.dumps(_json_safe(decision.snapshot_json or {}), sort_keys=True),
                    0,
                    decision.future_evaluation_status,
                    decision.outcome_raw,
                    decision.outcome_alpha,
                    None
                    if decision.outcome_resolved_at is None
                    else _parse_utc_datetime(decision.outcome_resolved_at)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    decision.reflection,
                    decision.source_run_id,
                    _parse_utc_datetime(decision.analysis_snapshot_timestamp)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    decision.analysis_snapshot_bid,
                    decision.analysis_snapshot_ask,
                    decision.analysis_snapshot_spread,
                    decision.analysis_snapshot_spread_points,
                    None
                    if decision.decision_completed_timestamp is None
                    else _parse_utc_datetime(decision.decision_completed_timestamp)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    decision.analysis_latency_seconds,
                    None
                    if decision.decision_reference_timestamp is None
                    else _parse_utc_datetime(decision.decision_reference_timestamp)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    decision.decision_reference_bid,
                    decision.decision_reference_ask,
                    decision.decision_reference_spread,
                    decision.decision_reference_spread_points,
                    decision.decision_reference_status,
                    decision.decision_reference_delay_seconds,
                    decision.decision_reference_error,
                ),
            )

    def get(self, decision_id: str) -> ShadowTradeDecision:
        self.initialize()
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM shadow_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        return self._row_to_decision(row)

    def list_pending(self, resolved_symbol: str | None = None) -> list[ShadowTradeDecision]:
        self.initialize()
        query = "SELECT * FROM shadow_decisions WHERE future_evaluation_status = 'PENDING' AND executed = 0"
        params: list[Any] = []
        if resolved_symbol is not None:
            query += " AND resolved_symbol = ?"
            params.append(resolved_symbol)
        query += " ORDER BY created_at, decision_id"
        with sqlite3.connect(self.path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_decision(row) for row in rows]

    def update_outcome(
        self,
        decision_id: str,
        outcome_raw: float | None,
        outcome_alpha: float | None,
        outcome_resolved_at: str | datetime,
        reflection: str | None,
    ) -> None:
        self.initialize()
        resolved_at = _parse_utc_datetime(outcome_resolved_at).isoformat().replace("+00:00", "Z")
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                UPDATE shadow_decisions
                   SET outcome_raw = ?,
                       outcome_alpha = ?,
                       outcome_resolved_at = ?,
                       reflection = ?,
                       future_evaluation_status = 'RESOLVED'
                 WHERE decision_id = ?
                """,
                (outcome_raw, outcome_alpha, resolved_at, reflection, decision_id),
            )

    def _row_to_decision(self, row: sqlite3.Row) -> ShadowTradeDecision:
        row_keys = set(row.keys())
        return ShadowTradeDecision(
            decision_id=row["decision_id"],
            created_at=_parse_utc_datetime(row["created_at"]),
            analysis_date=date.fromisoformat(row["analysis_date"]),
            requested_symbol=row["requested_symbol"],
            resolved_symbol=row["resolved_symbol"],
            snapshot_timestamp=_parse_utc_datetime(row["snapshot_timestamp"]),
            action=row["action"],
            raw_portfolio_manager_result=json.loads(row["raw_portfolio_manager_result"]),
            normalization_status=row["normalization_status"],
            decision_context_status=(
                row["decision_context_status"]
                if "decision_context_status" in row_keys
                else "INCOMPLETE"
            ),
            normalization_error=row["normalization_error"],
            confidence=row["confidence"],
            reference_bid=row["reference_bid"],
            reference_ask=row["reference_ask"],
            reference_mid=row["reference_mid"],
            spread=row["spread"],
            spread_points=row["spread_points"],
            analysis_timeframe=row["analysis_timeframe"],
            analysis_profile=(
                row["analysis_profile"] if "analysis_profile" in row_keys else "INTRADAY"
            ),
            valid_for_seconds=(
                row["valid_for_seconds"] if "valid_for_seconds" in row_keys else None
            ),
            valid_until=(
                None
                if "valid_until" not in row_keys or row["valid_until"] is None
                else _parse_utc_datetime(row["valid_until"])
            ),
            trader_summary=row["trader_summary"],
            portfolio_manager_summary=row["portfolio_manager_summary"],
            bull_summary=row["bull_summary"],
            bear_summary=row["bear_summary"],
            llm_provider=row["llm_provider"],
            quick_model=row["quick_model"],
            deep_model=row["deep_model"],
            snapshot_json=json.loads(row["snapshot_json"]),
            future_evaluation_status=row["future_evaluation_status"],
            outcome_raw=row["outcome_raw"],
            outcome_alpha=row["outcome_alpha"],
            outcome_resolved_at=(
                None
                if row["outcome_resolved_at"] is None
                else _parse_utc_datetime(row["outcome_resolved_at"])
            ),
            reflection=row["reflection"],
            source_run_id=row["source_run_id"],
            executed=bool(row["executed"]),
            analysis_snapshot_timestamp=(
                None
                if "analysis_snapshot_timestamp" not in row_keys
                or row["analysis_snapshot_timestamp"] is None
                else _parse_utc_datetime(row["analysis_snapshot_timestamp"])
            ),
            analysis_snapshot_bid=(
                row["analysis_snapshot_bid"]
                if "analysis_snapshot_bid" in row_keys
                else None
            ),
            analysis_snapshot_ask=(
                row["analysis_snapshot_ask"]
                if "analysis_snapshot_ask" in row_keys
                else None
            ),
            analysis_snapshot_spread=(
                row["analysis_snapshot_spread"]
                if "analysis_snapshot_spread" in row_keys
                else None
            ),
            analysis_snapshot_spread_points=(
                row["analysis_snapshot_spread_points"]
                if "analysis_snapshot_spread_points" in row_keys
                else None
            ),
            decision_completed_timestamp=(
                None
                if "decision_completed_timestamp" not in row_keys
                or row["decision_completed_timestamp"] is None
                else _parse_utc_datetime(row["decision_completed_timestamp"])
            ),
            analysis_latency_seconds=(
                row["analysis_latency_seconds"]
                if "analysis_latency_seconds" in row_keys
                else None
            ),
            decision_reference_timestamp=(
                None
                if "decision_reference_timestamp" not in row_keys
                or row["decision_reference_timestamp"] is None
                else _parse_utc_datetime(row["decision_reference_timestamp"])
            ),
            decision_reference_bid=(
                row["decision_reference_bid"]
                if "decision_reference_bid" in row_keys
                else None
            ),
            decision_reference_ask=(
                row["decision_reference_ask"]
                if "decision_reference_ask" in row_keys
                else None
            ),
            decision_reference_spread=(
                row["decision_reference_spread"]
                if "decision_reference_spread" in row_keys
                else None
            ),
            decision_reference_spread_points=(
                row["decision_reference_spread_points"]
                if "decision_reference_spread_points" in row_keys
                else None
            ),
            decision_reference_status=(
                row["decision_reference_status"]
                if "decision_reference_status" in row_keys
                else "UNAVAILABLE"
            ),
            decision_reference_delay_seconds=(
                row["decision_reference_delay_seconds"]
                if "decision_reference_delay_seconds" in row_keys
                else None
            ),
            decision_reference_error=(
                row["decision_reference_error"]
                if "decision_reference_error" in row_keys
                else None
            ),
        )
