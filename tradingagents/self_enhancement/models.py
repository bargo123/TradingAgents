"""Immutable Phase 14 contracts and bounded mutation specifications."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any

UTC = timezone.utc


class ExecutionMode(str, Enum):
    """Execution provenance allowed into the offline learning plane."""

    DEMO = "DEMO"
    REPLAY = "REPLAY"
    SHADOW = "SHADOW"
    TEST_ONLY = "TEST_ONLY"


class CandidateState(str, Enum):
    EXTRACTED = "EXTRACTED"
    EXPERIMENTAL = "EXPERIMENTAL"
    IMPLEMENTED = "IMPLEMENTED"
    REPLAYED = "REPLAYED"
    VALIDATED = "VALIDATED"
    UNSEEN_PASSED = "UNSEEN_PASSED"
    SHADOW_CHALLENGER = "SHADOW_CHALLENGER"
    DEMO_CHALLENGER = "DEMO_CHALLENGER"
    APPROVED = "APPROVED"
    INCUMBENT = "INCUMBENT"
    REJECTED = "REJECTED"
    ROLLED_BACK = "ROLLED_BACK"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ExperimentStatus(str, Enum):
    PLANNED = "PLANNED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    NO_EXPERIMENT = "NO_EXPERIMENT"
    ABORTED_RECOVERABLE = "ABORTED_RECOVERABLE"


class FindingKind(str, Enum):
    """Evidence maturity; observations are not causal conclusions."""

    OBSERVATION = "OBSERVATION"
    CAUSAL_HYPOTHESIS = "CAUSAL_HYPOTHESIS"
    CAUSAL_CONCLUSION = "CAUSAL_CONCLUSION"


class TriggerKind(str, Enum):
    NEW_VERIFIED_TRADES = "NEW_VERIFIED_TRADES"
    SESSION_COVERAGE = "SESSION_COVERAGE"
    PERFORMANCE_DETERIORATION = "PERFORMANCE_DETERIORATION"
    BOOK_CANDIDATE = "BOOK_CANDIDATE"
    TICK_DATASET = "TICK_DATASET"


def _text(value: Any, name: str, *, max_length: int = 256) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty bounded string")
    return value.strip()


def _utc(value: Any, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return number


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return MappingProxyType({str(key): _freeze_value(value[key]) for key in sorted(value, key=str)})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One explicitly approved automatic mutation parameter."""

    name: str
    value_type: str
    minimum: float
    maximum: float
    default: float
    safe_minimum: float | None = None
    safe_maximum: float | None = None

    def __post_init__(self) -> None:
        name = _text(self.name, "name")
        value_type = _text(self.value_type, "value_type")
        if value_type not in {"float", "int"}:
            raise ValueError("value_type must be float or int")
        minimum = _finite(self.minimum, "minimum")
        maximum = _finite(self.maximum, "maximum")
        default = _finite(self.default, "default")
        safe_min = minimum if self.safe_minimum is None else _finite(self.safe_minimum, "safe_minimum")
        safe_max = maximum if self.safe_maximum is None else _finite(self.safe_maximum, "safe_maximum")
        if not minimum <= safe_min <= default <= safe_max <= maximum:
            raise ValueError("parameter bounds are inconsistent")
        if value_type == "int" and any(float(v).is_integer() is False for v in (minimum, maximum, default, safe_min, safe_max)):
            raise ValueError("integer parameter bounds must be integral")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "safe_minimum", safe_min)
        object.__setattr__(self, "safe_maximum", safe_max)

    def validate(self, value: Any) -> float | int:
        if isinstance(value, bool):
            raise ValueError(f"{self.name} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{self.name} must be numeric") from exc
        if not math.isfinite(numeric) or not self.safe_minimum <= numeric <= self.safe_maximum:
            raise ValueError(f"{self.name} is outside its safe range")
        if self.value_type == "int":
            if not numeric.is_integer():
                raise ValueError(f"{self.name} must be an integer")
            return int(numeric)
        return numeric


_EXIT_PARAMETER_SPECS = {
    "profit_target_fraction": ParameterSpec("profit_target_fraction", "float", 0.5, 4.0, 1.5, 0.5, 4.0),
    "protection_arm_fraction": ParameterSpec("protection_arm_fraction", "float", 0.1, 1.0, 0.5, 0.1, 1.0),
    "giveback_fraction": ParameterSpec("giveback_fraction", "float", 0.05, 0.8, 0.25, 0.05, 0.8),
    "micro_reversal_fraction": ParameterSpec("micro_reversal_fraction", "float", 0.1, 1.0, 0.5, 0.1, 1.0),
    "no_progress_cap_seconds": ParameterSpec("no_progress_cap_seconds", "float", 5.0, 300.0, 30.0, 5.0, 300.0),
    "max_duration_seconds": ParameterSpec("max_duration_seconds", "float", 30.0, 900.0, 60.0, 30.0, 900.0),
    "trailing_distance_fraction": ParameterSpec("trailing_distance_fraction", "float", 0.05, 1.0, 0.5, 0.05, 1.0),
}


@dataclass(frozen=True, slots=True)
class ExitPolicyConfig:
    """Bounded exit-policy parameters; the only V1 automatic mutation surface."""

    profit_target_fraction: float = 1.5
    protection_arm_fraction: float = 0.5
    giveback_fraction: float = 0.25
    micro_reversal_fraction: float = 0.5
    no_progress_cap_seconds: float = 30.0
    max_duration_seconds: float = 60.0
    trailing_distance_fraction: float = 0.5

    def __post_init__(self) -> None:
        for name, spec in _EXIT_PARAMETER_SPECS.items():
            object.__setattr__(self, name, spec.validate(getattr(self, name)))
        if self.max_duration_seconds < self.no_progress_cap_seconds:
            raise ValueError("max_duration_seconds must cover no_progress_cap_seconds")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> ExitPolicyConfig:
        if not isinstance(values, Mapping):
            raise ValueError("exit policy must be a mapping")
        unknown = set(values) - set(_EXIT_PARAMETER_SPECS)
        if unknown:
            raise ValueError(f"unknown parameter: {sorted(unknown)[0]}")
        return cls(**dict(values))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in _EXIT_PARAMETER_SPECS}


@dataclass(frozen=True, slots=True)
class StrategyVersion:
    strategy_id: str
    strategy_version: str
    config_version: str
    parameters: Mapping[str, Any]
    source_commit: str
    config_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_id", _text(self.strategy_id, "strategy_id"))
        object.__setattr__(self, "strategy_version", _text(self.strategy_version, "strategy_version"))
        object.__setattr__(self, "config_version", _text(self.config_version, "config_version"))
        object.__setattr__(self, "source_commit", _text(self.source_commit, "source_commit"))
        params = _freeze_mapping(self.parameters, "parameters")
        object.__setattr__(self, "parameters", params)
        canonical = json.dumps(_json_value(params), sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "config_hash", hashlib.sha256(canonical.encode()).hexdigest())


@dataclass(frozen=True, slots=True)
class ExperienceTrade:
    """One verified, non-synthetic DEMO/replay observation."""

    experience_id: str
    source_database_id: str
    source_position_id: str
    strategy_id: str
    strategy_version: str
    config_version: str
    symbol: str
    direction: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_bid: float
    entry_ask: float
    entry_fill: float
    exit_bid: float
    exit_ask: float
    exit_fill: float
    spread_points: float
    slippage_points: float
    feature_snapshot: Mapping[str, Any]
    regime: str
    expected_move_points: float
    volume: float
    risk: float
    mfe_points: float
    mae_points: float
    exit_reason: str
    broker_execution_latency_ms: float
    gross_result: float
    net_known_result: float
    commission_known: bool
    profit_to_loss_flip: bool
    session: str
    volatility_state: str
    data_quality_state: str
    source_git_commit: str
    execution_mode: ExecutionMode | str
    real_money: bool = False
    synthetic: bool = False
    source_fingerprint: str | None = None
    mfe_capture_ratio: float | None = None
    signal_strength: float | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "experience_id", "source_database_id", "source_position_id", "strategy_id",
            "strategy_version", "config_version", "symbol", "direction", "regime",
            "exit_reason", "session", "volatility_state", "data_quality_state", "source_git_commit",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name))
        entry = _utc(self.entry_timestamp, "entry_timestamp")
        exit_at = _utc(self.exit_timestamp, "exit_timestamp")
        if exit_at <= entry:
            raise ValueError("exit_timestamp must be after entry_timestamp")
        object.__setattr__(self, "entry_timestamp", entry)
        object.__setattr__(self, "exit_timestamp", exit_at)
        for field_name in (
            "entry_bid", "entry_ask", "entry_fill", "exit_bid", "exit_ask", "exit_fill",
            "spread_points", "slippage_points", "expected_move_points", "volume", "risk",
            "mfe_points", "broker_execution_latency_ms", "gross_result", "net_known_result",
        ):
            object.__setattr__(self, field_name, _finite(getattr(self, field_name), field_name))
        object.__setattr__(self, "mae_points", _finite(self.mae_points, "mae_points"))
        if self.entry_bid > self.entry_ask or self.exit_bid > self.exit_ask:
            raise ValueError("bid cannot exceed ask")
        if not isinstance(self.commission_known, bool) or not isinstance(self.profit_to_loss_flip, bool):
            raise ValueError("boolean trade metadata is invalid")
        if self.real_money is not False:
            raise ValueError("real_money must be false")
        if self.synthetic is not False:
            raise ValueError("synthetic observations are excluded")
        mode = ExecutionMode(self.execution_mode)
        if mode not in {ExecutionMode.DEMO, ExecutionMode.REPLAY}:
            raise ValueError("execution_mode must be DEMO or REPLAY")
        object.__setattr__(self, "execution_mode", mode)
        object.__setattr__(self, "feature_snapshot", _freeze_mapping(self.feature_snapshot, "feature_snapshot"))
        if self.mfe_capture_ratio is not None:
            ratio = _finite(self.mfe_capture_ratio, "mfe_capture_ratio")
            object.__setattr__(self, "mfe_capture_ratio", ratio)
        if self.signal_strength is not None:
            object.__setattr__(self, "signal_strength", _finite(self.signal_strength, "signal_strength"))

    @property
    def holding_seconds(self) -> float:
        return (self.exit_timestamp - self.entry_timestamp).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return _json_value({field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()})


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    parent: StrategyVersion
    strategy_id: str
    exit_policy: ExitPolicyConfig
    hypothesis: str
    source_evidence: Sequence[Mapping[str, Any]] = ()
    state: CandidateState = CandidateState.EXPERIMENTAL
    execution_mode: ExecutionMode = ExecutionMode.SHADOW
    real_money: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, "candidate_id"))
        object.__setattr__(self, "strategy_id", _text(self.strategy_id, "strategy_id"))
        object.__setattr__(self, "hypothesis", _text(self.hypothesis, "hypothesis", max_length=2000))
        if not isinstance(self.parent, StrategyVersion) or not isinstance(self.exit_policy, ExitPolicyConfig):
            raise TypeError("candidate parent and exit_policy must be typed contracts")
        evidence = tuple(_freeze_value(item) for item in self.source_evidence)
        if any(not isinstance(item, Mapping) for item in evidence):
            raise ValueError("source_evidence must contain mappings")
        object.__setattr__(self, "source_evidence", evidence)
        object.__setattr__(self, "state", CandidateState(self.state))
        if self.execution_mode is not ExecutionMode.SHADOW or self.real_money is not False:
            raise ValueError("Phase 14 candidates are shadow-only")

    def with_state(self, state: CandidateState) -> CandidateSpec:
        return replace(self, state=CandidateState(state))


@dataclass(frozen=True, slots=True)
class LearningFinding:
    """Compact experience memory entry with an explicit evidence maturity."""

    finding_id: str
    kind: FindingKind | str
    statement: str
    evidence_ids: Sequence[str] = ()
    experiment_id: str | None = None
    verified: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "finding_id", _text(self.finding_id, "finding_id"))
        kind = FindingKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "statement", _text(self.statement, "statement", max_length=2000))
        evidence = tuple(_text(item, "evidence_id") for item in self.evidence_ids)
        object.__setattr__(self, "evidence_ids", evidence)
        if self.experiment_id is not None:
            object.__setattr__(self, "experiment_id", _text(self.experiment_id, "experiment_id"))
        if not isinstance(self.verified, bool):
            raise ValueError("verified must be boolean")
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance, "provenance"))
        if kind is FindingKind.CAUSAL_CONCLUSION and (not self.verified or self.experiment_id is None):
            raise ValueError("causal conclusions require a verified experiment")

    def to_dict(self) -> dict[str, Any]:
        return _json_value({field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()})


__all__ = [
        "CandidateSpec", "CandidateState", "ExecutionMode", "ExperimentStatus",
    "FindingKind", "LearningFinding", "TriggerKind",
    "ExitPolicyConfig", "ExperienceTrade", "ParameterSpec", "StrategyVersion",
]
