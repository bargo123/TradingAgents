"""Deterministic risk gates for Phase 12 shadow actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import FastAction, RiskDecision, Tick, utc


@dataclass(frozen=True, slots=True)
class RiskConfig:
    max_risk_fraction: float = 0.01
    max_open_exposure: float = 1.0
    max_daily_loss: float = 0.02
    max_drawdown: float = 0.10
    max_consecutive_losses: int = 3
    max_spread_points: float = 20.0
    max_slippage_points: float = 5.0
    stale_after_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name in ("max_risk_fraction", "max_drawdown"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
            object.__setattr__(self, name, value)
        daily_loss = float(self.max_daily_loss)
        if daily_loss <= 0:
            raise ValueError("max_daily_loss must be positive")
        object.__setattr__(self, "max_daily_loss", daily_loss)
        for name in ("max_open_exposure", "max_spread_points", "max_slippage_points", "stale_after_seconds"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if isinstance(self.max_consecutive_losses, bool) or not isinstance(self.max_consecutive_losses, int) or self.max_consecutive_losses < 0:
            raise ValueError("max_consecutive_losses must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RiskContext:
    equity: float
    open_exposure: float
    daily_loss: float
    drawdown: float
    consecutive_losses: int
    tick_timestamp: datetime
    observed_at: datetime
    session: str
    slippage_points: float = 0.0

    def __post_init__(self) -> None:
        for name in ("equity", "open_exposure", "daily_loss", "drawdown", "slippage_points"):
            value = float(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if isinstance(self.consecutive_losses, bool) or not isinstance(self.consecutive_losses, int) or self.consecutive_losses < 0:
            raise ValueError("consecutive_losses must be a non-negative integer")
        object.__setattr__(self, "tick_timestamp", utc(self.tick_timestamp, "tick_timestamp"))
        object.__setattr__(self, "observed_at", utc(self.observed_at, "observed_at"))
        if not isinstance(self.session, str) or not self.session.strip():
            raise ValueError("session must be non-empty")


class RiskEngine:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def evaluate(self, action: FastAction, tick: Tick, context: RiskContext, *, plan_expired: bool = False) -> RiskDecision:
        if not isinstance(action, FastAction):
            action = FastAction(action)
        if plan_expired:
            return RiskDecision(False, "PLAN_EXPIRED", "strategic plan is expired")
        if action not in (FastAction.ENTER_LONG, FastAction.ENTER_SHORT):
            return RiskDecision(True, "NON_ENTRY", "non-entry action does not add exposure", 0.0)
        age = (context.observed_at - context.tick_timestamp).total_seconds()
        if age > self.config.stale_after_seconds:
            return RiskDecision(False, "STALE_TICK", "tick is older than the stale-data bound")
        if tick.spread_points > self.config.max_spread_points:
            return RiskDecision(False, "SPREAD_LIMIT", "spread exceeds risk ceiling")
        if context.slippage_points > self.config.max_slippage_points:
            return RiskDecision(False, "SLIPPAGE_LIMIT", "slippage assumption exceeds risk ceiling")
        if context.daily_loss >= self.config.max_daily_loss:
            return RiskDecision(False, "DAILY_LOSS_LIMIT", "daily loss limit reached")
        if context.drawdown >= self.config.max_drawdown:
            return RiskDecision(False, "DRAWDOWN_LIMIT", "drawdown limit reached")
        if context.consecutive_losses >= self.config.max_consecutive_losses:
            return RiskDecision(False, "CONSECUTIVE_LOSS_LIMIT", "consecutive-loss limit reached")
        if context.open_exposure >= self.config.max_open_exposure:
            return RiskDecision(False, "EXPOSURE_LIMIT", "open exposure limit reached")
        return RiskDecision(True, "ACCEPTED", "all deterministic risk gates passed", self.config.max_risk_fraction)


__all__ = ["RiskConfig", "RiskContext", "RiskEngine"]
