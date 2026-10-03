"""Gateway-free observation of Phase 14B shadow challengers.

The observer consumes only causal features and validated runtime context. It
never receives a broker provider, an order gateway, retrieved text, or a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from tradingagents.forex.hft.execution import ShadowFillEngine, ShadowPosition, ShadowPositionLedger
from tradingagents.forex.hft.features import TickFeatures
from tradingagents.forex.hft.models import FastAction, PositionState, Tick
from tradingagents.forex.hft.regime import StrategicRegimeState
from tradingagents.forex.hft.risk import RiskConfig, RiskContext, RiskEngine
from tradingagents.forex.hft.strategies import SignalArbiter

from .book_strategies import BookStrategyRegistry, DeterministicBookStrategy
from .models import CandidateState, ExecutionMode, StrategyVersion
from .strategy_specs import StrategySpec, StrategySuitability


class ChallengerRegistryError(ValueError):
    """The Phase 14 catalog did not contain a trustworthy challenger registry."""


@dataclass(frozen=True, slots=True)
class ChallengerObservation:
    observation_id: str
    candidate_id: str
    spec_id: str
    strategy_id: str
    symbol: str
    timestamp: datetime
    event: str
    action: str
    reason_code: str
    entry_price: float | None
    exit_price: float | None
    spread_points: float
    slippage_points: float
    cost_points: float
    mfe_points: float
    mae_points: float
    holding_seconds: float
    gross_pnl: float | None
    commission_status: str = "UNKNOWN"
    executed: bool = False
    order_intent_emitted: bool = False

    def __post_init__(self) -> None:
        if self.event not in {"REJECTED", "ENTRY", "EXIT"}:
            raise ValueError("invalid challenger observation event")
        if self.action not in {FastAction.ENTER_LONG.value, FastAction.ENTER_SHORT.value, FastAction.EXIT.value}:
            raise ValueError("invalid challenger action")
        if self.executed is not False or self.order_intent_emitted is not False:
            raise ValueError("challenger observations cannot execute or emit order intents")
        if self.commission_status != "UNKNOWN":
            raise ValueError("challenger commission must remain UNKNOWN")
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("observation timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))
        for name in (
            "spread_points",
            "slippage_points",
            "cost_points",
            "mfe_points",
            "mae_points",
            "holding_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(slots=True)
class _CandidateRuntime:
    strategy: DeterministicBookStrategy
    ledger: ShadowPositionLedger
    fills: ShadowFillEngine
    position_size: float
    entry_spread_points: float | None = None


class _ObserverStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS challenger_observations (
                    observation_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    spec_id TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event TEXT NOT NULL CHECK(event IN ('REJECTED','ENTRY','EXIT')),
                    action TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    spread_points REAL NOT NULL,
                    slippage_points REAL NOT NULL,
                    cost_points REAL NOT NULL,
                    mfe_points REAL NOT NULL,
                    mae_points REAL NOT NULL,
                    holding_seconds REAL NOT NULL,
                    gross_pnl REAL,
                    commission_status TEXT NOT NULL CHECK(commission_status='UNKNOWN'),
                    executed INTEGER NOT NULL CHECK(executed=0),
                    order_intent_emitted INTEGER NOT NULL CHECK(order_intent_emitted=0)
                )
                """
            )

    def record(self, observation: ChallengerObservation) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute(
                """
                INSERT OR REPLACE INTO challenger_observations (
                    observation_id,candidate_id,spec_id,strategy_id,symbol,timestamp,
                    event,action,reason_code,entry_price,exit_price,spread_points,
                    slippage_points,cost_points,mfe_points,mae_points,holding_seconds,
                    gross_pnl,commission_status,executed,order_intent_emitted
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    observation.observation_id,
                    observation.candidate_id,
                    observation.spec_id,
                    observation.strategy_id,
                    observation.symbol,
                    observation.timestamp.isoformat(),
                    observation.event,
                    observation.action,
                    observation.reason_code,
                    observation.entry_price,
                    observation.exit_price,
                    observation.spread_points,
                    observation.slippage_points,
                    observation.cost_points,
                    observation.mfe_points,
                    observation.mae_points,
                    observation.holding_seconds,
                    observation.gross_pnl,
                    observation.commission_status,
                    0,
                    0,
                ),
            )


class ShadowChallengerObserver:
    """Evaluate promoted challengers beside—but never inside—the incumbent path."""

    def __init__(
        self,
        *,
        catalog_path: str | Path,
        observer_path: str | Path,
        slippage_points: float = 0.0,
        latency_ms: float = 0.0,
        position_size: float = 1.0,
        cost_safety_margin_points: float = 1.0,
        risk_config: RiskConfig | None = None,
    ) -> None:
        catalog = Path(catalog_path).expanduser().resolve()
        output = Path(observer_path).expanduser().resolve()
        if catalog == output:
            raise ValueError("observer output must be separate from the Phase 14 catalog")
        for name, value in (
            ("slippage_points", slippage_points),
            ("latency_ms", latency_ms),
            ("cost_safety_margin_points", cost_safety_margin_points),
        ):
            if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if isinstance(position_size, bool) or not math.isfinite(float(position_size)) or float(position_size) <= 0:
            raise ValueError("position_size must be finite and positive")
        if risk_config is not None and not isinstance(risk_config, RiskConfig):
            raise TypeError("risk_config must be RiskConfig")

        self._store = _ObserverStore(output)
        self._risk = RiskEngine(risk_config)
        self._arbiter = SignalArbiter(cost_safety_margin_points=float(cost_safety_margin_points))
        self._candidates = self._load_registry(catalog, slippage_points, latency_ms, position_size)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._candidates))

    def _load_registry(
        self,
        catalog: Path,
        slippage_points: float,
        latency_ms: float,
        position_size: float,
    ) -> dict[str, _CandidateRuntime]:
        if not catalog.is_file():
            raise ChallengerRegistryError("Phase 14 challenger catalog is missing")
        candidates: dict[str, _CandidateRuntime] = {}
        try:
            uri = f"{catalog.as_uri()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as db:
                db.row_factory = sqlite3.Row
                rows = db.execute(
                    """
                    SELECT c.candidate_id,c.payload_json,c.state
                    FROM candidates AS c
                    WHERE c.state=? AND EXISTS (
                        SELECT 1 FROM promotions AS p
                        WHERE p.experiment_id=c.experiment_id
                          AND p.candidate_id=c.candidate_id
                          AND p.decision=?
                    ) AND EXISTS (
                        SELECT 1 FROM deployments AS d
                        WHERE d.candidate_id=c.candidate_id
                          AND d.status=?
                    )
                    ORDER BY c.candidate_id
                    """,
                    (
                        CandidateState.SHADOW_CHALLENGER.value,
                        "SHADOW_CHALLENGER",
                        "SHADOW_CHALLENGER",
                    ),
                ).fetchall()
        except (sqlite3.Error, ValueError) as exc:
            raise ChallengerRegistryError("Phase 14 challenger catalog could not be read safely") from exc

        registry = BookStrategyRegistry()
        for row in rows:
            try:
                candidate_id = str(row["candidate_id"])
                payload = json.loads(row["payload_json"])
                if not isinstance(payload, dict) or payload.get("candidate_id") != candidate_id:
                    raise ValueError("candidate identity does not match its catalog row")
                if row["state"] != CandidateState.SHADOW_CHALLENGER.value:
                    raise ValueError("candidate is not registered as a shadow challenger")
                if payload.get("state") not in {CandidateState.EXTRACTED.value, CandidateState.SHADOW_CHALLENGER.value}:
                    raise ValueError("candidate payload has an invalid source state")
                if payload.get("execution_mode") != ExecutionMode.SHADOW.value or payload.get("real_money") is not False:
                    raise ValueError("candidate is not strictly shadow-only")
                spec_payload = payload.get("strategy_spec")
                spec = StrategySpec.from_dict(spec_payload)
                parent = payload.get("parent")
                if not isinstance(parent, dict) or set(parent) != {
                    "strategy_id",
                    "strategy_version",
                    "config_version",
                    "parameters",
                    "source_commit",
                    "config_hash",
                }:
                    raise ValueError("candidate parent provenance is invalid")
                version = StrategyVersion(
                    parent["strategy_id"],
                    parent["strategy_version"],
                    parent["config_version"],
                    parent["parameters"],
                    parent["source_commit"],
                )
                if parent["config_hash"] != version.config_hash:
                    raise ValueError("candidate parent config hash is invalid")
                expected_id = "book-spec-" + hashlib.sha256(
                    (version.config_hash + spec.content_hash).encode("utf-8")
                ).hexdigest()[:24]
                if candidate_id != expected_id or payload.get("strategy_id") != candidate_id:
                    raise ValueError("candidate ID is not bound to its StrategySpec")
                if spec.suitability is not StrategySuitability.HFT_SUITABLE or not spec.is_executable:
                    raise ValueError("candidate StrategySpec is not executable HFT material")
                strategy = registry.create(spec)
                candidates[candidate_id] = _CandidateRuntime(
                    strategy=strategy,
                    ledger=ShadowPositionLedger(),
                    fills=ShadowFillEngine(slippage_points=slippage_points, latency_ms=latency_ms),
                    position_size=float(position_size),
                )
            except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ChallengerRegistryError(
                    f"registered challenger {row['candidate_id']} failed strict StrategySpec validation"
                ) from exc
        return candidates

    def on_tick(
        self,
        features: TickFeatures,
        regime: StrategicRegimeState | None,
        risk_context: RiskContext,
    ) -> tuple[ChallengerObservation, ...]:
        if not isinstance(features, TickFeatures) or not isinstance(risk_context, RiskContext):
            return ()
        if not isinstance(regime, StrategicRegimeState):
            return ()
        if (
            regime.symbol != features.symbol
            or not regime.is_active(features.timestamp)
            or risk_context.tick_timestamp != features.timestamp
            or risk_context.observed_at < features.timestamp
        ):
            return ()
        try:
            tick = Tick(features.symbol, features.timestamp, features.bid, features.ask, features.point)
        except (TypeError, ValueError, OverflowError):
            return ()
        if (
            not math.isfinite(float(features.spread_points))
            or features.spread_points < 0
            or not math.isclose(tick.spread_points, features.spread_points, rel_tol=0.0, abs_tol=1e-6)
            or not math.isclose(tick.mid, features.mid, rel_tol=0.0, abs_tol=tick.point * 1e-6)
        ):
            return ()
        observations: list[ChallengerObservation] = []
        for candidate_id, runtime in self._candidates.items():
            position = runtime.ledger.position
            if position is not None and position.state in {PositionState.LONG, PositionState.SHORT}:
                observation = self._observe_position(candidate_id, runtime, features, tick, position)
                if observation is not None:
                    observations.append(observation)
                continue

            signal = runtime.strategy.evaluate(features)
            if signal is None:
                continue
            selected, gate_reason = self._arbiter.select(
                (signal,), regime, spread_points=features.spread_points
            )
            if selected is None:
                observations.append(
                    self._record(
                        candidate_id,
                        runtime,
                        features,
                        event="REJECTED",
                        action=signal.action,
                        reason_code=gate_reason,
                        entry_price=None,
                        exit_price=None,
                        cost_points=features.spread_points,
                        mfe_points=0.0,
                        mae_points=0.0,
                        holding_seconds=0.0,
                        gross_pnl=None,
                    )
                )
                continue
            isolated_context = replace(risk_context, open_exposure=0.0)
            risk = self._risk.evaluate(selected.action, tick, isolated_context)
            if not risk.accepted:
                observations.append(
                    self._record(
                        candidate_id,
                        runtime,
                        features,
                        event="REJECTED",
                        action=selected.action,
                        reason_code=risk.reason_code,
                        entry_price=None,
                        exit_price=None,
                        cost_points=features.spread_points,
                        mfe_points=0.0,
                        mae_points=0.0,
                        holding_seconds=0.0,
                        gross_pnl=None,
                    )
                )
                continue
            fill = runtime.fills.fill(selected.action, tick, size=runtime.position_size)
            target_delta = runtime.strategy.expected_move_points * features.point
            target = fill.price + target_delta if selected.action is FastAction.ENTER_LONG else fill.price - target_delta
            runtime.ledger.open(
                tick,
                selected.action,
                size=runtime.position_size,
                stop=None,
                target=target,
                strategy_id=runtime.strategy.strategy_id,
                entry_price=fill.price,
            )
            runtime.entry_spread_points = features.spread_points
            observations.append(
                self._record(
                    candidate_id,
                    runtime,
                    features,
                    event="ENTRY",
                    action=selected.action,
                    reason_code="SELECTED",
                    entry_price=fill.price,
                    exit_price=None,
                    cost_points=features.spread_points / 2.0 + runtime.fills.slippage_points,
                    mfe_points=0.0,
                    mae_points=0.0,
                    holding_seconds=0.0,
                    gross_pnl=0.0,
                )
            )
        return tuple(observations)

    def _observe_position(
        self,
        candidate_id: str,
        runtime: _CandidateRuntime,
        features: TickFeatures,
        tick: Tick,
        position: ShadowPosition,
    ) -> ChallengerObservation | None:
        observed = runtime.ledger.observe(tick)
        elapsed = (tick.timestamp - observed.entry_timestamp).total_seconds()
        target_reached = observed.mfe / features.point >= runtime.strategy.expected_move_points
        adverse_move = observed.mae < 0
        if not runtime.strategy.should_exit(
            features,
            elapsed_seconds=elapsed,
            target_reached=target_reached,
            adverse_move=adverse_move,
        ):
            return None
        reason = "MAX_HORIZON" if elapsed >= runtime.strategy.max_horizon_seconds else "BOOK_RULE_EXIT"
        fill = runtime.fills.fill(
            FastAction.EXIT,
            tick,
            size=observed.size,
            position_state=observed.state,
        )
        closed = runtime.ledger.close(tick, reason=reason, exit_price=fill.price)
        entry_spread = runtime.entry_spread_points or 0.0
        total_cost = entry_spread + features.spread_points + 2.0 * runtime.fills.slippage_points
        return self._record(
            candidate_id,
            runtime,
            features,
            event="EXIT",
            action=FastAction.EXIT,
            reason_code=reason,
            entry_price=closed.entry_price,
            exit_price=closed.exit_price,
            cost_points=total_cost,
            mfe_points=closed.mfe / features.point,
            mae_points=closed.mae / features.point,
            holding_seconds=closed.holding_seconds,
            gross_pnl=closed.gross_pnl,
        )

    def _record(
        self,
        candidate_id: str,
        runtime: _CandidateRuntime,
        features: TickFeatures,
        *,
        event: str,
        action: FastAction,
        reason_code: str,
        entry_price: float | None,
        exit_price: float | None,
        cost_points: float,
        mfe_points: float,
        mae_points: float,
        holding_seconds: float,
        gross_pnl: float | None,
    ) -> ChallengerObservation:
        observation_id = hashlib.sha256(
            f"{candidate_id}:{features.timestamp.isoformat()}:{event}".encode()
        ).hexdigest()
        observation = ChallengerObservation(
            observation_id=observation_id,
            candidate_id=candidate_id,
            spec_id=runtime.strategy.spec_id,
            strategy_id=runtime.strategy.strategy_id,
            symbol=features.symbol,
            timestamp=features.timestamp,
            event=event,
            action=action.value,
            reason_code=reason_code,
            entry_price=entry_price,
            exit_price=exit_price,
            spread_points=features.spread_points,
            slippage_points=runtime.fills.slippage_points,
            cost_points=cost_points,
            mfe_points=mfe_points,
            mae_points=mae_points,
            holding_seconds=holding_seconds,
            gross_pnl=gross_pnl,
        )
        self._store.record(observation)
        return observation


__all__ = [
    "ChallengerObservation",
    "ChallengerRegistryError",
    "ShadowChallengerObserver",
]
