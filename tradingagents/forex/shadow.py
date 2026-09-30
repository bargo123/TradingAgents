from __future__ import annotations

import importlib.util
import json
import math
import re
import sqlite3
import sys
from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from tradingagents.path_utils import require_nonempty_path


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
ResearchManagerRecommendation = Literal[
    "BUY", "OVERWEIGHT", "HOLD", "UNDERWEIGHT", "SELL"
]
TraderAction = Literal["BUY", "SELL", "HOLD"]
PortfolioManagerRejectionReason = Literal[
    "RISK_REJECTED",
    "INSUFFICIENT_EDGE",
    "STALE",
    "TEMPORAL_INVALID",
    "SCHEMA_FAILURE",
    "OTHER_VALIDATED_REASON",
]

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


def _db_bool(value: Any, name: str) -> bool:
    """Decode persisted SQLite boolean flags without truthiness coercion."""

    if isinstance(value, bool):
        return value
    if type(value) is int and value in (0, 1):
        return bool(value)
    raise ValueError(f"{name} must be boolean 0/1")


def _validate_forex_payload(payload: Mapping[str, Any], profile_name: str) -> str | None:
    profile_value = payload.get("analysis_profile")
    if profile_value is not None and profile_value != profile_name:
        return (
            f"forex analysis_profile must be exactly {profile_name}, "
            f"received type {type(profile_value).__name__}"
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


def _invalid_structured_result() -> dict[str, str]:
    """Return bounded metadata for a rejected model payload."""

    return {"error": "STRUCTURED_OUTPUT_INVALID", "status": "FAILED"}


def _normalize_from_payload(
    payload: Mapping[str, Any],
    *,
    forex_profile: str | None = None,
) -> ShadowNormalization:
    try:
        safe_payload = _json_safe(dict(payload))
    except (TypeError, ValueError) as exc:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured output contains non-finite or unsupported values "
                f"(cause_type={type(exc).__name__})"
            ),
            raw_result=_invalid_structured_result(),
        )

    if "rating" not in payload:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=_invalid_structured_result(),
        )

    if forex_profile is not None:
        forex_error = _validate_forex_payload(payload, forex_profile)
        if forex_error is not None:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=forex_error,
                raw_result=_invalid_structured_result(),
            )

    try:
        rating_text = _canonical_rating(payload["rating"])
    except ValueError:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "unknown PortfolioRating value "
                f"(type={type(payload['rating']).__name__})"
            ),
            raw_result=_invalid_structured_result(),
        )
    except TypeError:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=_invalid_structured_result(),
        )

    mapped_action = _RATING_TO_ACTION.get(rating_text)
    if mapped_action is None:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error="unknown PortfolioRating value",
            raw_result=_invalid_structured_result(),
        )

    for alias in ("recommendation", "action"):
        if alias in payload and payload[alias] != payload["rating"]:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=(
                    "conflicting structured Portfolio Manager rating values"
                ),
                raw_result=_invalid_structured_result(),
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
            raw_result=_invalid_structured_result(),
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
    research_manager_recommendation: ResearchManagerRecommendation | None = None
    trader_action: TraderAction | None = None
    portfolio_manager_rejection_reason: PortfolioManagerRejectionReason | None = None

    def __post_init__(self) -> None:
        if self.executed is not False:
            raise ValueError("ShadowTradeDecision must always be created with executed=False")
        if not isinstance(self.decision_id, str) or not self.decision_id.strip():
            raise ValueError("decision_id must be a non-empty string")
        if isinstance(self.analysis_date, datetime) or not isinstance(self.analysis_date, date):
            raise ValueError("analysis_date must be a date")
        if self.source_run_id is not None and (
            not isinstance(self.source_run_id, str) or not self.source_run_id.strip()
        ):
            raise ValueError("source_run_id must be a non-empty string when provided")
        if self.normalization_status not in ("NORMALIZED", "FAILED"):
            raise ValueError("normalization_status must be NORMALIZED or FAILED")
        if self.normalization_status == "NORMALIZED":
            if self.normalization_error is not None:
                raise ValueError("normalization_error must be None for normalized decisions")
        elif not isinstance(self.normalization_error, str) or not self.normalization_error.strip():
            raise ValueError("normalization_error is required for failed decisions")
        if self.decision_context_status not in ("COMPLETE", "INCOMPLETE"):
            raise ValueError("decision_context_status must be COMPLETE or INCOMPLETE")
        if self.future_evaluation_status not in ("PENDING", "RESOLVED"):
            raise ValueError("future_evaluation_status must be PENDING or RESOLVED")
        for field_name in ("requested_symbol", "resolved_symbol", "analysis_timeframe"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.research_manager_recommendation is not None and self.research_manager_recommendation not in (
            "BUY",
            "OVERWEIGHT",
            "HOLD",
            "UNDERWEIGHT",
            "SELL",
        ):
            raise ValueError("research_manager_recommendation must use the canonical uppercase value")
        if self.trader_action is not None and self.trader_action not in ("BUY", "SELL", "HOLD"):
            raise ValueError("trader_action must use the canonical uppercase value")
        if self.portfolio_manager_rejection_reason is not None and self.portfolio_manager_rejection_reason not in (
            "RISK_REJECTED",
            "INSUFFICIENT_EDGE",
            "STALE",
            "TEMPORAL_INVALID",
            "SCHEMA_FAILURE",
            "OTHER_VALIDATED_REASON",
        ):
            raise ValueError("portfolio_manager_rejection_reason is not recognized")
        if not isinstance(self.analysis_profile, str) or not self.analysis_profile.strip():
            raise ValueError("analysis_profile must be a non-empty string")
        if self.analysis_profile != "INTRADAY":
            raise ValueError("analysis_profile must be exactly INTRADAY")
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
            else:
                try:
                    supplied_latency = float(self.analysis_latency_seconds)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("analysis_latency_seconds must be finite when provided") from exc
                if not math.isclose(
                    supplied_latency, computed_latency, rel_tol=0.0, abs_tol=1e-6
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
            else:
                try:
                    supplied_delay = float(self.decision_reference_delay_seconds)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError("decision_reference_delay_seconds must be finite when provided") from exc
                if not math.isclose(
                    supplied_delay,
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
        if not isinstance(self.snapshot_json, Mapping):
            raise ValueError("snapshot_json must be a mapping")
        if not isinstance(self.raw_portfolio_manager_result, Mapping):
            raise ValueError("raw_portfolio_manager_result must be a mapping")
        _json_safe(self.snapshot_json)
        _json_safe(self.raw_portfolio_manager_result)
        required_quote_fields = {
            "reference_bid",
            "reference_ask",
            "reference_mid",
            "spread",
            "spread_points",
        }
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
            if value is None:
                if field_name in required_quote_fields:
                    raise ValueError(f"{field_name} must be a finite numeric value")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field_name} must be a finite numeric value")
            try:
                finite_value = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{field_name} must be finite when provided") from exc
            if not math.isfinite(finite_value):
                raise ValueError(f"{field_name} must be finite when provided")
        for field_name in ("reference_bid", "reference_ask", "reference_mid"):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name in ("spread", "spread_points"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for prefix in ("analysis_snapshot", "decision_reference"):
            bid = getattr(self, f"{prefix}_bid")
            ask = getattr(self, f"{prefix}_ask")
            spread = getattr(self, f"{prefix}_spread")
            spread_points = getattr(self, f"{prefix}_spread_points")
            if bid is not None and bid <= 0:
                raise ValueError(f"{prefix} bid must be positive")
            if ask is not None and ask <= 0:
                raise ValueError(f"{prefix} ask must be positive")
            if bid is not None and ask is not None and ask < bid:
                raise ValueError(f"{prefix} ask must be greater than or equal to bid")
            if spread is not None and spread < 0:
                raise ValueError(f"{prefix} spread must be non-negative")
            if spread_points is not None and spread_points < 0:
                raise ValueError(f"{prefix} spread_points must be non-negative")
            if (
                bid is not None
                and ask is not None
                and spread is not None
                and not math.isclose(
                    ask - bid,
                    spread,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"{prefix} spread must equal ask minus bid")
        if not math.isclose(
            self.reference_ask - self.reference_bid,
            self.spread,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("reference spread must equal ask minus bid")
        if not math.isclose(
            (self.reference_bid + self.reference_ask) / 2,
            self.reference_mid,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("reference mid must equal bid/ask midpoint")

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
        self.path = require_nonempty_path(path, "path")

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as conn, conn:
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
                "research_manager_recommendation": "TEXT",
                "trader_action": "TEXT",
                "portfolio_manager_rejection_reason": "TEXT",
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
            "research_manager_recommendation",
            "trader_action",
            "portfolio_manager_rejection_reason",
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
        with closing(sqlite3.connect(self.path)) as conn, conn:
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
                    decision.research_manager_recommendation,
                    decision.trader_action,
                    decision.portfolio_manager_rejection_reason,
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
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM shadow_decisions WHERE decision_id = ?",
                (decision_id,),
            ).fetchone()
        if row is None:
            raise KeyError(decision_id)
        return self._row_to_decision(row)

    def find_by_source_run_id(self, source_run_id: str) -> tuple[ShadowTradeDecision, ...]:
        """Return decisions linked to a watcher source id without mutating data."""
        if not isinstance(source_run_id, str) or not source_run_id.strip():
            raise ValueError("source_run_id must be a non-empty string")
        self.initialize()
        with closing(sqlite3.connect(self.path)) as conn, conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM shadow_decisions WHERE source_run_id = ? ORDER BY created_at, decision_id",
                (source_run_id.strip(),),
            ).fetchall()
        return tuple(self._row_to_decision(row) for row in rows)

    def latest_eligible(self, resolved_symbol: str | None = None) -> ShadowTradeDecision | None:
        """Read the newest execution-eligible decision without writing the DB.

        Directional decisions are joined to their watcher run so freshness and
        reference validity cannot be lost when the HFT supervisor restarts.
        HOLD remains eligible as an explicit ``Direction.NONE`` update, which
        lets an invalid/non-directional analysis clear a previous plan.
        """

        if not self.path.is_file():
            return None
        normalized = str(self.path.expanduser().resolve()).replace("\\", "/")
        uri = f"file:{normalized}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=0) as conn:
            conn.row_factory = sqlite3.Row
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "shadow_decisions" not in tables:
                return None
            if "forex_watch_runs" in tables:
                query = (
                    "SELECT d.* FROM shadow_decisions AS d "
                    "LEFT JOIN forex_watch_runs AS r "
                    "ON r.decision_id = d.decision_id "
                    "AND r.source_run_id = d.source_run_id "
                    "WHERE d.executed = 0 AND d.normalization_status = 'NORMALIZED' "
                    "AND d.decision_context_status = 'COMPLETE' "
                    "AND (d.action = 'HOLD' OR ("
                    "d.action IN ('BUY','SELL') "
                    "AND d.decision_reference_status = 'AVAILABLE' "
                    "AND r.run_status IN ('SUCCEEDED','SUCCEEDED_SLOW') "
                    "AND r.decision_context_status = 'COMPLETE' "
                    "AND r.normalization_status = 'NORMALIZED' "
                    "AND r.decision_reference_status = 'AVAILABLE' "
                    "AND r.stale_by_completion = 0 "
                    "AND r.freshness_budget_seconds > 0 "
                    "AND d.analysis_latency_seconds IS NOT NULL "
                    "AND d.analysis_latency_seconds <= r.freshness_budget_seconds"
                    "))"
                )
            else:
                # A directional decision cannot be proven fresh without the
                # watcher evidence table. HOLD remains safe because it maps to
                # Direction.NONE.
                query = (
                    "SELECT d.* FROM shadow_decisions AS d "
                    "WHERE d.executed = 0 AND d.normalization_status = 'NORMALIZED' "
                    "AND d.decision_context_status = 'COMPLETE' "
                    "AND d.action = 'HOLD'"
                )
            params: tuple[Any, ...] = ()
            if resolved_symbol is not None:
                query += " AND UPPER(d.resolved_symbol) = ?"
                params = (str(resolved_symbol).strip().upper(),)
            query += " ORDER BY d.created_at DESC, d.decision_id DESC LIMIT 1"
            row = conn.execute(query, params).fetchone()
        return None if row is None else self._row_to_decision(row)

    def is_execution_eligible(self, decision: ShadowTradeDecision) -> bool:
        """Return whether a decision may activate a directional HFT plan.

        This is a read-only source-run check. It intentionally treats HOLD as
        eligible because HOLD maps to ``Direction.NONE`` and is used to clear
        an active plan; BUY/SELL require persisted watcher freshness and a
        post-completion broker-reference gate.
        """

        if not isinstance(decision, ShadowTradeDecision):
            return False
        if (
            decision.executed is not False
            or decision.decision_context_status != "COMPLETE"
            or decision.normalization_status != "NORMALIZED"
            or decision.action not in {"BUY", "SELL", "HOLD"}
        ):
            return False
        if decision.action == "HOLD":
            return True
        if decision.decision_reference_status != "AVAILABLE":
            return False
        if not self.path.is_file() or not decision.source_run_id:
            return False
        normalized = str(self.path.expanduser().resolve()).replace("\\", "/")
        uri = f"file:{normalized}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=0) as conn:
                conn.row_factory = sqlite3.Row
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                if "forex_watch_runs" not in tables:
                    return False
                row = conn.execute(
                    """
                    SELECT run_status, source_run_id,
                           decision_context_status, normalization_status,
                           decision_reference_status, stale_by_completion,
                           freshness_budget_seconds
                      FROM forex_watch_runs
                     WHERE decision_id = ? AND source_run_id = ?
                     ORDER BY COALESCE(completed_at, started_at) DESC
                     LIMIT 1
                    """,
                    (decision.decision_id, decision.source_run_id),
                ).fetchone()
        except (OSError, sqlite3.Error):
            return False
        if row is None:
            return False
        if row["run_status"] not in ("SUCCEEDED", "SUCCEEDED_SLOW"):
            return False
        if row["source_run_id"] != decision.source_run_id:
            return False
        if row["decision_context_status"] != "COMPLETE" or row["normalization_status"] != "NORMALIZED":
            return False
        if row["decision_reference_status"] != "AVAILABLE" or row["stale_by_completion"] != 0:
            return False
        budget = row["freshness_budget_seconds"]
        latency = decision.analysis_latency_seconds
        if budget is None or latency is None:
            return False
        try:
            budget_value = float(budget)
            latency_value = float(latency)
        except (TypeError, ValueError, OverflowError):
            return False
        return (
            math.isfinite(budget_value)
            and math.isfinite(latency_value)
            and budget_value > 0
            and latency_value <= budget_value
        )

    def list_pending(self, resolved_symbol: str | None = None) -> list[ShadowTradeDecision]:
        self.initialize()
        query = "SELECT * FROM shadow_decisions WHERE future_evaluation_status = 'PENDING' AND executed = 0"
        params: list[Any] = []
        if resolved_symbol is not None:
            query += " AND resolved_symbol = ?"
            params.append(resolved_symbol)
        query += " ORDER BY created_at, decision_id"
        with closing(sqlite3.connect(self.path)) as conn, conn:
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
        with closing(sqlite3.connect(self.path)) as conn, conn:
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
            executed=_db_bool(row["executed"], "executed"),
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
            research_manager_recommendation=(
                row["research_manager_recommendation"]
                if "research_manager_recommendation" in row_keys
                else None
            ),
            trader_action=(
                row["trader_action"]
                if "trader_action" in row_keys and row["trader_action"] in {"BUY", "SELL", "HOLD"}
                else None
            ),
            portfolio_manager_rejection_reason=(
                row["portfolio_manager_rejection_reason"]
                if "portfolio_manager_rejection_reason" in row_keys
                and row["portfolio_manager_rejection_reason"] in {
                    "RISK_REJECTED",
                    "INSUFFICIENT_EDGE",
                    "STALE",
                    "TEMPORAL_INVALID",
                    "SCHEMA_FAILURE",
                    "OTHER_VALIDATED_REASON",
                }
                else None
            ),
        )
