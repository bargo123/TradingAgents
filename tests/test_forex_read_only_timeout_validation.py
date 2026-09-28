from __future__ import annotations

from pathlib import Path

import pytest

from tradingagents.forex.dashboard import _read_only_connection as dashboard_connection
from tradingagents.forex.decision_path_audit import _readonly_connection as decision_path_connection
from tradingagents.forex.hold_audit import _readonly_connection as hold_audit_connection
from tradingagents.forex.research_signal_audit import (
    _readonly_connection as research_signal_connection,
)


@pytest.mark.parametrize(
    "reader",
    (decision_path_connection, hold_audit_connection, research_signal_connection),
)
@pytest.mark.parametrize("value", [True, "1", float("inf"), float("nan"), 10**1000])
def test_read_only_audit_connections_reject_malformed_busy_timeout(
    tmp_path: Path, reader, value
) -> None:
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        reader(tmp_path / "missing.sqlite3", value)


def test_dashboard_rejects_oversized_busy_timeout_without_overflow(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="busy_timeout_seconds"):
        dashboard_connection(tmp_path / "missing.sqlite3", 10**1000)
