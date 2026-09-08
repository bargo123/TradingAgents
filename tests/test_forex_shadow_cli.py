from __future__ import annotations

from pathlib import Path

import pytest

from cli.forex_shadow import build_parser, main


def test_forex_shadow_parser_defaults_and_positive_count():
    args = build_parser().parse_args([])

    assert args.symbol == "EURUSD"
    assert args.analysts == "market,news"
    assert args.count > 0

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--count", "0"])


def test_forex_shadow_parser_exposes_only_non_secret_runtime_overrides():
    args = build_parser().parse_args(
        [
            "--llm-provider",
            "ollama",
            "--quick-model",
            "qwen3",
            "--deep-model",
            "qwen3-thinking",
            "--backend-url",
            "http://127.0.0.1:11434/v1",
            "--temperature",
            "0.2",
            "--max-tokens",
            "512",
        ]
    )

    assert args.llm_provider == "ollama"
    assert args.quick_model == "qwen3"
    assert args.deep_model == "qwen3-thinking"
    assert args.backend_url.endswith("/v1")
    assert args.temperature == pytest.approx(0.2)
    assert args.max_tokens == 512


def test_forex_shadow_cli_prints_banner_and_record_message(capsys, monkeypatch, tmp_path):
    class FakeRunResult:
        decision = type(
            "Decision",
            (),
            {
                "decision_id": "decision-001",
                "action": "BUY",
                "normalization_status": "NORMALIZED",
                "executed": False,
            },
        )()
        provider_snapshot_calls = 1

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            return FakeRunResult()

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FakeRunner)

    exit_code = main(["--db-path", str(tmp_path / "shadow.db")])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "MT5 FOREX" in out
    assert "NO ORDER WILL BE SENT" in out
    assert "SHADOW DECISION RECORDED" in out
    assert "EXECUTED: FALSE" in out


def test_stock_cli_entry_point_string_remains_unchanged():
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'tradingagents = "cli.main:app"' in pyproject
    assert "forex-shadow" in pyproject


def test_cli_rejects_stock_analysts_before_constructing_runner(capsys, monkeypatch):
    class MustNotConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("provider runner must not be constructed")

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", MustNotConstruct)

    assert main(["--analysts", "market,fundamentals"]) == 2
    assert "unsupported" in capsys.readouterr().err.lower()


def test_cli_returns_nonzero_and_keeps_banner_when_runner_fails(capsys, monkeypatch):
    class FailingRunner:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("terminal unavailable")

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FailingRunner)

    assert main([]) == 1
    captured = capsys.readouterr()
    assert "MT5 FOREX" in captured.out
    assert "NO ORDER WILL BE SENT" in captured.out
    assert "terminal unavailable" in captured.err
