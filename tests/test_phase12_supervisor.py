from tradingagents.forex.supervisor import ForexSupervisor


def test_supervisor_hft_status_is_scalar_and_shadow_only(tmp_path):
    report = ForexSupervisor().hft_status(tmp_path / "hft.sqlite3")
    assert report["status"] == "NOT_INITIALIZED"
    assert report["executed"] is False

