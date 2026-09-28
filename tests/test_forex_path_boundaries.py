from pathlib import Path

import pytest

from tradingagents.forex.evaluation import ShadowEvaluationStore
from tradingagents.forex.evidence_audit import EvidenceAuditStore
from tradingagents.forex.shadow import ShadowDecisionStore
from tradingagents.forex.watch_store import WatcherStore


@pytest.mark.parametrize(
    "store_factory",
    [WatcherStore, ShadowDecisionStore, ShadowEvaluationStore, EvidenceAuditStore],
)
@pytest.mark.parametrize("value", ["", "   ", Path("."), Path("   ")])
def test_sqlite_store_rejects_empty_path(store_factory, value):
    with pytest.raises((TypeError, ValueError), match="non-empty path"):
        store_factory(value)
