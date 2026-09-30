from datetime import datetime, timedelta, timezone

from cli.forex_hft_shadow import main


def test_shadow_cli_refuses_active_watcher_before_provider_construction(tmp_path, monkeypatch, capsys):
    watcher_db = tmp_path / "watcher.sqlite3"
    import sqlite3

    with sqlite3.connect(watcher_db) as db:
        db.execute("CREATE TABLE forex_watcher_state (singleton_id INTEGER PRIMARY KEY, lifecycle_status TEXT, current_run_id TEXT, current_opportunity_key TEXT, owner_token TEXT, owner_pid INTEGER, owner_host TEXT, process_started_at TEXT, lease_acquired_at TEXT, heartbeat_at TEXT, lease_expires_at TEXT)")
        future = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
        db.execute("INSERT INTO forex_watcher_state VALUES (1,'ANALYZING','run-1','opp-1','owner',123,'host',?,?,?,?)", (future, future, future, future))

    called = {"provider": False}
    monkeypatch.setattr("cli.forex_hft_shadow._make_provider", lambda *args, **kwargs: called.update(provider=True))
    result = main(["--watcher-db-path", str(watcher_db), "--db-path", str(tmp_path / "hft.sqlite3"), "--max-ticks", "1", "--plan-json", str(tmp_path / "plan.json")])
    assert result == 1
    assert called["provider"] is False
    assert "WATCHER_ALREADY_RUNNING" in capsys.readouterr().err
