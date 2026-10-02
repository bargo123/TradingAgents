import json

from tradingagents.self_enhancement.cli import main
from tradingagents.self_enhancement.dashboard import read_dashboard_status
from tradingagents.self_enhancement.store import SelfEnhancementStore


def test_status_for_missing_root_is_read_only(tmp_path):
    root = tmp_path / "missing"
    result = read_dashboard_status(root)
    assert result["status"] == "EMPTY"
    assert not root.exists()


def test_status_reports_scalar_catalog_state_without_initializing_writer(tmp_path):
    root = tmp_path / "phase14"
    store = SelfEnhancementStore(root / "phase14.sqlite3")
    store.initialize()
    result = read_dashboard_status(root)
    assert result["status"] == "OK"
    assert result["verified_experience"] == 0
    assert result["active_challengers"] == 0
    assert result["llm_calls"] == 0
    assert result["mt5_calls"] == 0


def test_cli_status_does_not_create_catalog(tmp_path, capsys):
    root = tmp_path / "status-only"
    assert main(["status", "--artifact-root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "EMPTY"
    assert not (root / "phase14.sqlite3").exists()
