"""Bounded shadow account and descriptive compounding metrics."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date, datetime
from enum import Enum


class CompoundingMode(str, Enum):
    FIXED_RISK = "FIXED_RISK"
    COMPOUNDING_RISK = "COMPOUNDING_RISK"
    VOLATILITY_ADJUSTED = "VOLATILITY_ADJUSTED"


class AccountSimulator:
    def __init__(self, *, initial_balance: float, risk_fraction: float, mode: CompoundingMode = CompoundingMode.FIXED_RISK, max_position_size: float = 1_000_000.0) -> None:
        if initial_balance <= 0 or not 0 < risk_fraction <= 0.1 or max_position_size <= 0:
            raise ValueError("account bounds are invalid")
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.equity = float(initial_balance)
        self.peak_equity = float(initial_balance)
        self.max_drawdown = 0.0
        self.risk_fraction = float(risk_fraction)
        self.mode = CompoundingMode(mode)
        self.max_position_size = float(max_position_size)
        self._trades: list[float] = []
        self._daily: defaultdict[date, float] = defaultdict(float)

    def position_size(self, *, stop_distance_points: float, point_value: float, volatility_factor: float = 1.0) -> float:
        if stop_distance_points <= 0 or point_value <= 0 or volatility_factor <= 0:
            raise ValueError("sizing inputs must be positive")
        risk_base = self.initial_balance if self.mode is CompoundingMode.FIXED_RISK else self.equity
        if self.mode is CompoundingMode.VOLATILITY_ADJUSTED:
            risk_base /= volatility_factor
        size = risk_base * self.risk_fraction / (stop_distance_points * point_value)
        return min(self.max_position_size, max(0.0, size))

    def record_trade(self, timestamp: datetime, net_pnl: float) -> None:
        if timestamp.tzinfo is None or not math.isfinite(float(net_pnl)):
            raise ValueError("trade timestamp must be aware and pnl finite")
        pnl = float(net_pnl)
        self.balance += pnl
        self.equity = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)
        drawdown = max(0.0, (self.peak_equity - self.equity) / self.peak_equity)
        self.max_drawdown = max(self.max_drawdown, drawdown)
        self._trades.append(pnl)
        self._daily[timestamp.date()] += pnl

    def report(self) -> dict[str, float | int | None]:
        trades = self._trades
        wins = [value for value in trades if value > 0]
        losses = [value for value in trades if value < 0]
        daily_returns = [(value / self.initial_balance) for value in self._daily.values()]
        profit_factor = sum(wins) / abs(sum(losses)) if losses else None
        return {
            "balance": self.balance,
            "equity": self.equity,
            "compound_return": self.balance / self.initial_balance - 1.0,
            "max_drawdown": self.max_drawdown,
            "worst_day": min(daily_returns) if daily_returns else None,
            "best_day": max(daily_returns) if daily_returns else None,
            "benchmark_10pct_days": sum(value >= 0.10 for value in daily_returns),
            "trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else None,
            "profit_factor": profit_factor,
            "expectancy": sum(trades) / len(trades) if trades else None,
        }


__all__ = ["AccountSimulator", "CompoundingMode"]
