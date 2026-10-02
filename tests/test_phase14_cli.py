from tradingagents.self_enhancement.cli import main


def test_status_cli_is_scalar_only_and_does_not_require_sources(tmp_path, capsys):
    assert main(["status", "--artifact-root", str(tmp_path / "phase14")]) == 0
    output = capsys.readouterr().out
    assert '"llm_calls": 0' in output
    assert '"mt5_calls": 0' in output
