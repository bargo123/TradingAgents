from __future__ import annotations

import pytest

from tradingagents.dataflows.mt5.clock import BrokerClockConfig
from tradingagents.dataflows.mt5.models import _finite_number
from tradingagents.datasets.canonical import _snapshot
from tradingagents.distillation.splits import GroupedSplitter
from tradingagents.experience.features import _finite as experience_feature_finite
from tradingagents.experience.models import _finite_float as experience_finite_float
from tradingagents.forex.dashboard import _float as dashboard_float
from tradingagents.forex.decision_path_audit import _finite as decision_finite
from tradingagents.forex.evaluation import _finite as evaluation_finite
from tradingagents.forex.evidence_audit import _non_negative_float
from tradingagents.forex.evidence_context import EvidenceQueryPolicy
from tradingagents.forex.evidence_replay import _number as replay_number
from tradingagents.forex.evidence_runtime import ReadonlyEvidenceRuntimeConfiguration
from tradingagents.forex.hold_audit import _finite as hold_finite
from tradingagents.forex.ollama_runtime import DedicatedOllamaRuntime
from tradingagents.forex.revision_validation import _safe_nonnegative_float
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor
from tradingagents.knowledge.fusion import _finite_score
from tradingagents.knowledge.models import _finite_float as knowledge_finite_float

HUGE_INTEGER = 10**1000


@pytest.mark.parametrize(
    "factory",
    (
        lambda: ForexShadowRuntimeConfig(temperature=HUGE_INTEGER),
        lambda: EvidenceQueryPolicy(evidence_timeout_seconds=HUGE_INTEGER),
        lambda: ReadonlyEvidenceRuntimeConfiguration(evidence_timeout_seconds=HUGE_INTEGER),
        lambda: _non_negative_float(HUGE_INTEGER, "value"),
        lambda: BrokerClockConfig(max_tick_age_seconds=HUGE_INTEGER),
        lambda: DedicatedOllamaRuntime(
            ForexShadowRuntimeConfig(), probe_interval_seconds=HUGE_INTEGER
        ),
        lambda: experience_finite_float(HUGE_INTEGER, "value"),
        lambda: knowledge_finite_float(HUGE_INTEGER, "value"),
        lambda: _finite_number(HUGE_INTEGER, "value"),
        lambda: _finite_score(HUGE_INTEGER, "value"),
        lambda: replay_number(HUGE_INTEGER, "value"),
    ),
)
def test_numeric_configuration_rejects_oversized_integers(factory) -> None:
    with pytest.raises(ValueError):
        factory()


def test_feature_extractor_quarantines_oversized_numeric_values() -> None:
    assert experience_feature_finite(HUGE_INTEGER) is None


def test_dashboard_numeric_reader_quarantines_oversized_values() -> None:
    assert dashboard_float(HUGE_INTEGER) is None


def test_evaluation_numeric_reader_rejects_oversized_values() -> None:
    with pytest.raises(ValueError, match="numeric|finite"):
        evaluation_finite(HUGE_INTEGER, "value")


def test_supervisor_rejects_oversized_restart_backoff() -> None:
    supervisor = ForexSupervisor(
        runtime_factory=lambda _config: pytest.fail("runtime must not start"),
        store_factory=lambda _path: pytest.fail("store must not start"),
    )
    with pytest.raises(ValueError):
        supervisor.run(
            db_path="watch.db",
            watch_main=lambda: 0,
            restart_backoff_seconds=HUGE_INTEGER,
        )


def test_snapshot_projection_ignores_oversized_numeric_values() -> None:
    result = _snapshot(
        {"point": HUGE_INTEGER, "quote": {"bid": HUGE_INTEGER}},
        "EURUSD",
        "2026-01-01T00:00:00Z",
    )
    assert "point" not in result
    assert result["quote"] == {}


@pytest.mark.parametrize("reader", (decision_finite, hold_finite, _safe_nonnegative_float))
def test_audit_numeric_readers_quarantine_oversized_values(reader) -> None:
    assert reader(HUGE_INTEGER) is None


def test_grouped_splitter_rejects_oversized_ratio() -> None:
    with pytest.raises(ValueError):
        GroupedSplitter(ratios=(HUGE_INTEGER, 0.1, 0.1))
