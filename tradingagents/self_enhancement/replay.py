"""Causal Phase 14 replay over the existing deterministic HFT primitives."""

from __future__ import annotations

import json
import math
import sqlite3
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

from tradingagents.forex.hft.account import AccountSimulator, CompoundingMode
from tradingagents.forex.hft.engines import (
    MOMENTUM_EXIT_PROFILE,
    RANGE_EXIT_PROFILE,
    DynamicExitProfile,
    HftExecutionEngine,
)
from tradingagents.forex.hft.execution import ShadowFillEngine, ShadowPositionLedger
from tradingagents.forex.hft.features import TickFeatureEngine
from tradingagents.forex.hft.models import FastAction, PositionState, Tick
from tradingagents.forex.hft.regime import build_hft_bootstrap_neutral_regime
from tradingagents.forex.hft.risk import RiskContext, RiskEngine

from .models import CandidateSpec, ExitPolicyConfig

UTC = timezone.utc


class ReplayError(ValueError):
    """Raised when the replay input is not causally safe."""


@dataclass(frozen=True, slots=True)
class ChronologicalSplits:
    development: tuple[Tick, ...]
    validation: tuple[Tick, ...]
    unseen: tuple[Tick, ...]
    guard_gap_seconds: float


@dataclass(frozen=True, slots=True)
class ReplayMetrics:
    candidate_id: str
    strategy_id: str
    ticks_processed: int
    trades: int
    wins: int
    losses: int
    win_rate: float | None
    expectancy: float | None
    profit_factor: float | None
    total_return: float
    max_drawdown: float
    average_holding_seconds: float | None
    median_holding_seconds: float | None
    mfe_points: float
    mae_points: float
    mfe_capture_ratio: float | None
    profit_to_loss_flips: int
    spread_cost_points: float
    slippage_points: float
    broker_rejections: int
    exit_reasons: dict[str, int]
    long_trades: int
    short_trades: int
    future_leak_detected: bool = False
    execution_mode: str = "REPLAY"
    real_money: bool = False
    trades_per_hour: float | None = None
    sharpe_like: float | None = None
    commission_known: bool = False

    def to_dict(self) -> dict[str, object]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


def _validate_ticks(ticks: Iterable[Tick]) -> tuple[Tick, ...]:
    values = tuple(ticks)
    if not values:
        raise ReplayError("replay requires at least one tick")
    if any(not isinstance(item, Tick) for item in values):
        raise ReplayError("replay input must contain Tick objects")
    if any(right.timestamp <= left.timestamp for left, right in zip(values, values[1:], strict=False)):
        raise ReplayError("replay ticks must be strictly monotonic")
    if values[-1].timestamp > datetime.now(UTC):
        raise ReplayError("replay cannot consume future ticks")
    symbols = {item.symbol for item in values}
    if len(symbols) != 1:
        raise ReplayError("replay cannot change symbols")
    return values


def load_hft_ticks(path: str, *, symbol: str = "EURUSD") -> tuple[Tick, ...]:
    """Read normalized ticks through a SQLite read-only connection."""

    from pathlib import Path

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ReplayError(f"tick source does not exist: {source}")
    connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(
            "SELECT timestamp,symbol,bid,ask,features_json FROM hft_ticks WHERE symbol=? ORDER BY timestamp,rowid",
            (str(symbol).upper(),),
        ).fetchall()
    except sqlite3.Error as exc:
        raise ReplayError("tick source could not be read") from exc
    finally:
        connection.close()
    ticks: list[Tick] = []
    for index, row in enumerate(rows):
        try:
            from datetime import datetime

            timestamp = datetime.fromisoformat(str(row["timestamp"]).replace("Z", "+00:00"))
            features = json.loads(row["features_json"] or "{}")
            point = float(features.get("point", 0.00001)) if isinstance(features, dict) else 0.00001
            ticks.append(Tick(str(row["symbol"]), timestamp, float(row["bid"]), float(row["ask"]), point, index))
        except (TypeError, ValueError, OverflowError, json.JSONDecodeError) as exc:
            raise ReplayError(f"invalid tick row {index}") from exc
    return _validate_ticks(ticks)


def chronological_splits(
    ticks: Iterable[Tick],
    *,
    development_fraction: float = 0.5,
    validation_fraction: float = 0.25,
    guard_gap_seconds: float = 1.0,
) -> ChronologicalSplits:
    values = _validate_ticks(ticks)
    if not 0 < development_fraction < 1 or not 0 < validation_fraction < 1 or development_fraction + validation_fraction >= 1:
        raise ValueError("fractions must leave an unseen partition")
    if guard_gap_seconds < 0 or not math.isfinite(float(guard_gap_seconds)):
        raise ValueError("guard_gap_seconds must be finite and non-negative")
    n = len(values)
    first_boundary = max(1, int(n * development_fraction))
    second_boundary = max(first_boundary + 1, int(n * (development_fraction + validation_fraction)))
    gap = float(guard_gap_seconds)
    development = tuple(item for item in values[:first_boundary] if item.timestamp < values[first_boundary].timestamp - timedelta(seconds=gap))
    validation = tuple(
        item
        for item in values[first_boundary:second_boundary]
        if values[first_boundary].timestamp + timedelta(seconds=gap) <= item.timestamp < values[second_boundary].timestamp - timedelta(seconds=gap)
    )
    unseen = tuple(item for item in values[second_boundary:] if item.timestamp >= values[second_boundary].timestamp + timedelta(seconds=gap))
    if not development or not validation or not unseen:
        raise ReplayError("guarded chronological split is empty")
    return ChronologicalSplits(development, validation, unseen, gap)


def _profile(base: DynamicExitProfile, config: ExitPolicyConfig) -> DynamicExitProfile:
    floor = min(base.no_progress_floor_seconds, config.no_progress_cap_seconds)
    return replace(
        base,
        micro_reversal_mfe_fraction=config.micro_reversal_fraction,
        micro_reversal_giveback_fraction=config.micro_reversal_fraction,
        protection_arm_fraction=config.protection_arm_fraction,
        trailing_activation_fraction=max(base.trailing_activation_fraction, config.protection_arm_fraction),
        giveback_fraction=config.giveback_fraction,
        trailing_distance_fraction=config.trailing_distance_fraction,
        no_progress_floor_seconds=floor,
        no_progress_cap_seconds=max(floor, config.no_progress_cap_seconds),
        max_duration_seconds=max(config.max_duration_seconds, floor),
    )


class ReplayEvaluator:
    """Run one candidate through the same causal feature/risk/fill primitives."""

    def __init__(self, *, slippage_points: float = 0.0, latency_ms: float = 0.0) -> None:
        if slippage_points < 0 or latency_ms < 0 or not math.isfinite(float(slippage_points)) or not math.isfinite(float(latency_ms)):
            raise ValueError("replay cost assumptions must be finite and non-negative")
        self.slippage_points = float(slippage_points)
        self.latency_ms = float(latency_ms)

    def evaluate(self, ticks: Iterable[Tick], candidate: CandidateSpec, *, split: str | None = None) -> ReplayMetrics:
        if not isinstance(candidate, CandidateSpec):
            raise TypeError("candidate must be a CandidateSpec")
        values = _validate_ticks(ticks)
        config = candidate.exit_policy
        range_profile = _profile(RANGE_EXIT_PROFILE, config)
        momentum_profile = _profile(MOMENTUM_EXIT_PROFILE, config)
        engine = HftExecutionEngine(
            profit_target_fraction=config.profit_target_fraction,
            time_stop_seconds=max(1, int(config.max_duration_seconds)),
            hard_max_duration_seconds=config.max_duration_seconds,
            range_exit_profile=range_profile,
            momentum_exit_profile=momentum_profile,
        )
        features = TickFeatureEngine()
        risk = RiskEngine()
        fills = ShadowFillEngine(slippage_points=self.slippage_points, latency_ms=self.latency_ms)
        positions = ShadowPositionLedger()
        account = AccountSimulator(initial_balance=100.0, risk_fraction=0.005, mode=CompoundingMode.COMPOUNDING_RISK)
        regime = build_hft_bootstrap_neutral_regime(
            values[0].symbol,
            values[0].timestamp,
            ttl_seconds=max(1.0, (values[-1].timestamp - values[0].timestamp).total_seconds() + 5.0),
            git_commit=candidate.parent.source_commit,
        )
        regime = replace(
            regime,
            momentum_enabled=candidate.strategy_id == "momentum_continuation",
            range_enabled=candidate.strategy_id == "range_rejection",
        )
        position_state = PositionState.FLAT
        entry_price = entry_at = None
        expected_move_points: float | None = None
        pnls: list[float] = []
        durations: list[float] = []
        mfe: list[float] = []
        mae: list[float] = []
        long_trades = 0
        short_trades = 0
        exit_reasons: dict[str, int] = {}
        previous_positive = False
        profit_to_loss_flips = 0
        spread_cost = 0.0
        for tick in values:
            snapshot = features.update(tick)
            current = positions.position
            if current is not None and current.state in (PositionState.LONG, PositionState.SHORT):
                positions.observe(tick)
            decision = engine.on_tick(
                regime,
                snapshot,
                position_state=position_state,
                entry_price=entry_price,
                entry_at=entry_at,
                expected_move_points=expected_move_points,
                strategy_id=None if current is None else current.strategy_id,
            )
            if decision.action in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
                account_metrics = account.report()
                gate = risk.evaluate(
                    decision.action,
                    tick,
                    RiskContext(
                        float(account_metrics["equity"] or 100.0),
                        0.0,
                        0.0,
                        float(account_metrics["max_drawdown"] or 0.0),
                        0,
                        tick.timestamp,
                        tick.timestamp,
                        snapshot.session,
                    ),
                )
                if gate.accepted:
                    fill = fills.fill(decision.action, tick, size=1.0)
                    opened = positions.open(tick, decision.action, size=1.0, stop=None, target=None, strategy_id=decision.strategy_id or candidate.strategy_id, entry_price=fill.price)
                    position_state = opened.state
                    entry_price = opened.entry_price
                    entry_at = opened.entry_timestamp
                    expected_move_points = decision.expected_move_points
                    spread_cost += tick.spread_points
            elif decision.action is FastAction.EXIT and position_state in (PositionState.LONG, PositionState.SHORT):
                fill = fills.fill(decision.action, tick, size=1.0, position_state=position_state)
                closed = positions.close(tick, reason=decision.reason, exit_price=fill.price)
                account.record_trade(tick.timestamp, closed.net_pnl)
                pnls.append(closed.net_pnl)
                if closed.direction == "LONG":
                    long_trades += 1
                elif closed.direction == "SHORT":
                    short_trades += 1
                durations.append(closed.holding_seconds)
                mfe.append(closed.mfe / tick.point)
                mae.append(closed.mae / tick.point)
                exit_reasons[decision.reason] = exit_reasons.get(decision.reason, 0) + 1
                if previous_positive and closed.net_pnl < 0:
                    profit_to_loss_flips += 1
                previous_positive = closed.net_pnl > 0
                position_state = PositionState.FLAT
                entry_price = entry_at = None
                expected_move_points = None
        wins = [value for value in pnls if value > 0]
        losses = [value for value in pnls if value < 0]
        report = account.report()
        elapsed_seconds = max(0.0, (values[-1].timestamp - values[0].timestamp).total_seconds())
        standard_deviation = statistics.pstdev(pnls) if len(pnls) > 1 else 0.0
        sharpe_like = (
            (sum(pnls) / len(pnls)) / standard_deviation * math.sqrt(len(pnls))
            if len(pnls) > 1 and standard_deviation > 0
            else None
        )
        return ReplayMetrics(
            candidate_id=candidate.candidate_id,
            strategy_id=candidate.strategy_id,
            ticks_processed=len(values),
            trades=len(pnls),
            wins=len(wins),
            losses=len(losses),
            win_rate=len(wins) / len(pnls) if pnls else None,
            expectancy=sum(pnls) / len(pnls) if pnls else None,
            profit_factor=sum(wins) / abs(sum(losses)) if losses else None,
            total_return=float(report["compound_return"]),
            max_drawdown=float(report["max_drawdown"]),
            average_holding_seconds=sum(durations) / len(durations) if durations else None,
            median_holding_seconds=sorted(durations)[len(durations) // 2] if durations else None,
            mfe_points=sum(mfe) / len(mfe) if mfe else 0.0,
            mae_points=sum(mae) / len(mae) if mae else 0.0,
            mfe_capture_ratio=(sum(pnls) / sum(mfe) if mfe and sum(mfe) > 0 else None),
            profit_to_loss_flips=profit_to_loss_flips,
            spread_cost_points=spread_cost,
            slippage_points=self.slippage_points,
            broker_rejections=0,
            exit_reasons=exit_reasons,
            long_trades=long_trades,
            short_trades=short_trades,
            trades_per_hour=(len(pnls) * 3600.0 / elapsed_seconds) if elapsed_seconds > 0 else None,
            sharpe_like=sharpe_like,
            commission_known=False,
        )
__all__ = ["ChronologicalSplits", "ReplayError", "ReplayEvaluator", "ReplayMetrics", "chronological_splits", "load_hft_ticks"]
