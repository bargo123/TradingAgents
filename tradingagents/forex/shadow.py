from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal


def _load_schemas_module():
    module_name = "tradingagents._forex_schemas"
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
    return module


_schemas = _load_schemas_module()
PortfolioDecision = _schemas.PortfolioDecision
PortfolioRating = _schemas.PortfolioRating

AllowedAction = Literal["BUY", "SELL", "HOLD"]
NormalizationStatus = Literal["NORMALIZED", "FAILED"]
FutureEvaluationStatus = Literal["PENDING", "RESOLVED"]

_RATING_TO_ACTION: dict[str, AllowedAction] = {
    PortfolioRating.BUY.value: "BUY",
    PortfolioRating.OVERWEIGHT.value: "BUY",
    PortfolioRating.HOLD.value: "HOLD",
    PortfolioRating.UNDERWEIGHT.value: "SELL",
    PortfolioRating.SELL.value: "SELL",
}
_RATING_ALIASES = ("rating", "recommendation", "action")


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
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        return _json_safe(value.value)
    return value


def _parse_utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        result = value.astimezone(timezone.utc)
    else:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware UTC values")
        result = result.astimezone(timezone.utc)
    if result.utcoffset() != timezone.utc.utcoffset(result):
        raise ValueError("timestamps must be UTC values")
    return result


def _canonical_rating(value: Any) -> str:
    if isinstance(value, PortfolioRating):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError("rating must be a PortfolioRating value")


def _normalized_payload(raw: PortfolioDecision | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, PortfolioDecision):
        return raw.model_dump(mode="json")
    if isinstance(raw, Mapping):
        return _json_safe(dict(raw))
    raise TypeError("raw result must be a PortfolioDecision, mapping, or None")


def _normalize_from_payload(payload: Mapping[str, Any]) -> ShadowNormalization:
    if "rating" not in payload:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=_json_safe(dict(payload)),
        )

    try:
        rating_text = _canonical_rating(payload["rating"]).strip()
    except TypeError:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                "structured Portfolio Manager output must include an exact rating"
            ),
            raw_result=_json_safe(dict(payload)),
        )

    mapped_action = _RATING_TO_ACTION.get(rating_text)
    if mapped_action is None:
        return ShadowNormalization(
            action=None,
            normalization_status="FAILED",
            normalization_error=(
                f"unknown PortfolioRating value: {rating_text!r}"
            ),
            raw_result=_json_safe(dict(payload)),
        )

    for alias in _RATING_ALIASES[1:]:
        if alias not in payload:
            continue
        try:
            alias_rating = _canonical_rating(payload[alias]).strip()
        except TypeError:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=(
                    "conflicting structured Portfolio Manager rating values"
                ),
                raw_result=_json_safe(dict(payload)),
            )
        if alias_rating != rating_text:
            return ShadowNormalization(
                action=None,
                normalization_status="FAILED",
                normalization_error=(
                    "conflicting structured Portfolio Manager rating values"
                ),
                raw_result=_json_safe(dict(payload)),
            )

    return ShadowNormalization(
        action=mapped_action,
        normalization_status="NORMALIZED",
        normalization_error=None,
        raw_result=_json_safe(dict(payload)),
    )


@dataclass(frozen=True, slots=True)
class ShadowNormalization:
    action: AllowedAction | None
    normalization_status: NormalizationStatus
    normalization_error: str | None
    raw_result: dict[str, Any]


def normalize_portfolio_manager_result(
    raw: PortfolioDecision | Mapping[str, Any] | None,
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
        return _normalize_from_payload(payload)

    if isinstance(raw, Mapping):
        return _normalize_from_payload(raw)

    return ShadowNormalization(
        action=None,
        normalization_status="FAILED",
        normalization_error=(
            "structured Portfolio Manager output must be a PortfolioDecision or mapping"
        ),
        raw_result={"value": _json_safe(raw)},
    )


@dataclass(frozen=True, slots=True)
class ShadowTradeDecision:
    decision_id: str
    created_at: datetime
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
    executed: bool = False

    def __post_init__(self) -> None:
        if self.executed is not False:
            raise ValueError("ShadowTradeDecision must always be created with executed=False")
        if self.normalization_status not in ("NORMALIZED", "FAILED"):
            raise ValueError("normalization_status must be NORMALIZED or FAILED")
        if self.normalization_status == "NORMALIZED" and self.action is None:
            raise ValueError("normalized decisions must include an action")
        if self.normalization_status == "FAILED" and self.action is not None:
            raise ValueError("failed decisions must not include an action")
        if self.action is not None and self.action not in ("BUY", "SELL", "HOLD"):
            raise ValueError("action must be BUY, SELL, HOLD, or None")
        _parse_utc_datetime(self.created_at)
        if self.outcome_resolved_at is not None:
            _parse_utc_datetime(self.outcome_resolved_at)
        if self.snapshot_json is None:
            object.__setattr__(self, "snapshot_json", {})

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
                    action TEXT CHECK (action IS NULL OR action IN ('BUY','SELL','HOLD')),
                    normalization_status TEXT NOT NULL
                        CHECK (normalization_status IN ('NORMALIZED','FAILED')),
                    normalization_error TEXT,
                    raw_portfolio_manager_result TEXT NOT NULL,
                    confidence REAL,
                    reference_bid REAL NOT NULL,
                    reference_ask REAL NOT NULL,
                    reference_mid REAL NOT NULL,
                    spread REAL NOT NULL,
                    spread_points REAL NOT NULL,
                    analysis_timeframe TEXT NOT NULL,
                    trader_summary TEXT NOT NULL,
                    portfolio_manager_summary TEXT NOT NULL,
                    bull_summary TEXT,
                    bear_summary TEXT,
                    llm_provider TEXT,
                    quick_model TEXT,
                    deep_model TEXT,
                    snapshot_json TEXT NOT NULL,
                    executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0),
                    future_evaluation_status TEXT NOT NULL DEFAULT 'PENDING',
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
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_shadow_decisions_resolved_symbol_date
                ON shadow_decisions (resolved_symbol, analysis_date)
                """
            )

    def record(self, decision: ShadowTradeDecision) -> None:
        self.initialize()
        placeholders = ", ".join(["?"] * 31)
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                f"""
                INSERT INTO shadow_decisions (
                    decision_id,
                    created_at,
                    analysis_date,
                    requested_symbol,
                    resolved_symbol,
                    action,
                    normalization_status,
                    normalization_error,
                    raw_portfolio_manager_result,
                    confidence,
                    reference_bid,
                    reference_ask,
                    reference_mid,
                    spread,
                    spread_points,
                    analysis_timeframe,
                    trader_summary,
                    portfolio_manager_summary,
                    bull_summary,
                    bear_summary,
                    llm_provider,
                    quick_model,
                    deep_model,
                    snapshot_json,
                    executed,
                    future_evaluation_status,
                    outcome_raw,
                    outcome_alpha,
                    outcome_resolved_at,
                    reflection,
                    source_run_id
                ) VALUES ({placeholders})
                ON CONFLICT(decision_id) DO NOTHING
                """,
                (
                    decision.decision_id,
                    _parse_utc_datetime(decision.created_at).isoformat().replace("+00:00", "Z"),
                    decision.analysis_date.isoformat(),
                    decision.requested_symbol,
                    decision.resolved_symbol,
                    decision.action,
                    decision.normalization_status,
                    decision.normalization_error,
                    decision.raw_portfolio_manager_result_json,
                    decision.confidence,
                    decision.reference_bid,
                    decision.reference_ask,
                    decision.reference_mid,
                    decision.spread,
                    decision.spread_points,
                    decision.analysis_timeframe,
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
        return ShadowTradeDecision(
            decision_id=row["decision_id"],
            created_at=_parse_utc_datetime(row["created_at"]),
            analysis_date=date.fromisoformat(row["analysis_date"]),
            requested_symbol=row["requested_symbol"],
            resolved_symbol=row["resolved_symbol"],
            action=row["action"],
            raw_portfolio_manager_result=json.loads(row["raw_portfolio_manager_result"]),
            normalization_status=row["normalization_status"],
            normalization_error=row["normalization_error"],
            confidence=row["confidence"],
            reference_bid=row["reference_bid"],
            reference_ask=row["reference_ask"],
            reference_mid=row["reference_mid"],
            spread=row["spread"],
            spread_points=row["spread_points"],
            analysis_timeframe=row["analysis_timeframe"],
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
        )
