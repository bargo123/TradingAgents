"""Deterministic weakness observations over verified experience.

These findings are descriptive observations only.  They are never causal
conclusions and never mutate a strategy or configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import ExperienceTrade, FindingKind


@dataclass(frozen=True, slots=True)
class WeaknessFinding:
    category: str
    sample_size: int
    severity: str
    evidence_ids: tuple[str, ...]
    kind: FindingKind = FindingKind.OBSERVATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "sample_size": self.sample_size,
            "severity": self.severity,
            "evidence_ids": list(self.evidence_ids),
            "kind": self.kind.value,
        }


def detect_weaknesses(
    trades: tuple[ExperienceTrade, ...] | list[ExperienceTrade],
    *,
    minimum_samples: int = 20,
) -> tuple[WeaknessFinding, ...]:
    """Return only aggregate, minimum-sample observations."""

    if isinstance(minimum_samples, bool) or minimum_samples <= 0:
        raise ValueError("minimum_samples must be positive")
    values = tuple(trades)
    if any(not isinstance(item, ExperienceTrade) for item in values):
        raise TypeError("trades must contain ExperienceTrade values")
    if len(values) < minimum_samples:
        return ()
    ids = tuple(item.experience_id for item in values)
    findings: list[WeaknessFinding] = []
    pnl = [item.net_known_result for item in values]
    wins = [value for value in pnl if value > 0]
    losses = [value for value in pnl if value < 0]
    expectancy = sum(pnl) / len(pnl)
    profit_factor = sum(wins) / abs(sum(losses)) if losses else None
    if expectancy < 0:
        findings.append(WeaknessFinding("NEGATIVE_EXPECTANCY", len(values), "HIGH", ids))
    if profit_factor is not None and profit_factor < 1.0:
        findings.append(WeaknessFinding("POOR_PROFIT_FACTOR", len(values), "HIGH", ids))
    flips = sum(item.profit_to_loss_flip for item in values)
    if flips / len(values) >= 0.2:
        findings.append(WeaknessFinding("FREQUENT_PROFIT_TO_LOSS_FLIPS", len(values), "MEDIUM", ids))
    giveback = [item.mfe_capture_ratio for item in values if item.mfe_capture_ratio is not None]
    if giveback and sum(giveback) / len(giveback) < 0.25:
        findings.append(WeaknessFinding("LOW_MFE_CAPTURE", len(giveback), "MEDIUM", ids))
    for reason in ("STOP_LOSS", "TIME_STOP", "NO_PROGRESS"):
        count = sum(item.exit_reason.upper() == reason for item in values)
        if count / len(values) >= 0.5:
            findings.append(WeaknessFinding(f"EXCESSIVE_{reason}", count, "MEDIUM", tuple(item.experience_id for item in values if item.exit_reason.upper() == reason)))
    by_strategy: dict[str, list[ExperienceTrade]] = {}
    by_session: dict[str, list[ExperienceTrade]] = {}
    for item in values:
        by_strategy.setdefault(item.strategy_id, []).append(item)
        by_session.setdefault(item.session, []).append(item)
    for _label, groups, category in (
        ("strategy", by_strategy, "STRATEGY_SPECIFIC_DEGRADATION"),
        ("session", by_session, "SESSION_SPECIFIC_DEGRADATION"),
    ):
        for name, group in groups.items():
            if len(group) >= minimum_samples and sum(item.net_known_result for item in group) / len(group) < 0:
                findings.append(WeaknessFinding(f"{category}:{name}", len(group), "HIGH", tuple(item.experience_id for item in group)))
    return tuple(findings)


__all__ = ["WeaknessFinding", "detect_weaknesses"]
