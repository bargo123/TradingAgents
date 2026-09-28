from __future__ import annotations

import pytest

from tradingagents.dataflows.mt5.clock import BrokerClockConfig
from tradingagents.datasets.canonical import _snapshot
from tradingagents.forex.evidence_audit import _non_negative_float
from tradingagents.forex.evidence_context import EvidenceQueryPolicy
from tradingagents.forex.evidence_runtime import ReadonlyEvidenceRuntimeConfiguration
from tradingagents.forex.ollama_runtime import DedicatedOllamaRuntime
from tradingagents.forex.runtime_config import ForexShadowRuntimeConfig
from tradingagents.forex.supervisor import ForexSupervisor

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
    ),
)
def test_numeric_configuration_rejects_oversized_integers(factory) -> None:
    with pytest.raises(ValueError):
        factory()


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
