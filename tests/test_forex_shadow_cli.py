from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from cli.forex_shadow import build_parser, main
from tradingagents.forex.runner import ForexShadowRunner
from tradingagents.forex.watch_store import LeaseOwner, WatcherStore


def test_forex_shadow_parser_defaults_and_positive_count():
    args = build_parser().parse_args([])

    assert args.symbol == "EURUSD"
    assert args.analysts == "market,news"
    assert args.analysis_profile == "INTRADAY"
    assert args.count > 0
    assert args.evidence_enabled is None

    with pytest.raises(SystemExit):
        build_parser().parse_args(["--count", "0"])


def test_parser_exposes_evidence_switches_only_for_forex_shadow():
    assert build_parser().parse_args(["--evidence-enabled"]).evidence_enabled is True
    assert build_parser().parse_args(["--no-evidence"]).evidence_enabled is False
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--evidence-enabled", "--no-evidence"])


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


def test_runtime_config_uses_default_off():
    args = build_parser().parse_args([])
    assert "forex_evidence_enabled" not in __import__("cli.forex_shadow", fromlist=["_runtime_config"])._runtime_config(args)


def test_cli_passes_evidence_flag_without_constructing_stock_graph(monkeypatch, tmp_path):
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["config"] = kwargs.get("config")
            self.store = type("Store", (), {"path": tmp_path / "shadow.db"})()

        def run(self, **kwargs):
            decision = type("Decision", (), {
                "decision_id": "decision-001", "action": "HOLD",
                "normalization_status": "NORMALIZED", "executed": False,
            })()
            return type("Result", (), {"decision": decision, "metrics": {}, "elapsed_seconds": 0})()

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FakeRunner)
    assert main(["--evidence-enabled", "--db-path", str(tmp_path / "shadow.db")]) == 0
    assert captured["config"] == {"forex_evidence_enabled": True}


def test_runner_maps_forex_evidence_config_to_service(monkeypatch, tmp_path):
    captured = {}

    class FakeService:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("tradingagents.forex.runner.EvidenceIntegrationService", FakeService)
    runner = ForexShadowRunner(
        config={
            "forex_evidence_timeout_seconds": 3.5,
            "forex_evidence_knowledge_artifact_root": tmp_path / "knowledge",
            "forex_evidence_knowledge_embedding_model_path": tmp_path / "model",
            "forex_evidence_experience_artifact_root": tmp_path / "experience",
        }
    )

    runner._default_evidence_service_factory()

    assert captured["policy"].evidence_timeout_seconds == 3.5
    assert {
        key: str(value) for key, value in captured["artifact_roots"].items()
    } == {
        "knowledge": str(tmp_path / "knowledge"),
        "knowledge_embedding_model_path": str(tmp_path / "model"),
        "experience": str(tmp_path / "experience"),
    }


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


def test_forex_shadow_refuses_active_watcher_before_runner_construction(
    capsys, monkeypatch, tmp_path
):
    db_path = tmp_path / "shadow.db"
    store = WatcherStore(db_path)
    now = datetime.now(timezone.utc)
    store.acquire_lease(
        LeaseOwner(
            owner_token="watcher-owner",
            pid=123,
            host="host",
            process_started_at=now,
        ),
        now,
    )

    class MustNotConstruct:
        def __init__(self, **kwargs):
            raise AssertionError("ForexShadowRunner must not be constructed")

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", MustNotConstruct)
    monkeypatch.setattr("cli.forex_shadow.StatsCallbackHandler", MustNotConstruct)

    assert main(["--db-path", str(db_path)]) == 1
    captured = capsys.readouterr()
    assert "WATCHER_ALREADY_RUNNING" in captured.err
    assert "NO ORDER WILL BE SENT" in captured.out


def test_forex_shadow_runs_normally_without_active_watcher(
    capsys, monkeypatch, tmp_path
):
    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["constructed"] = True
            self.store = type("Store", (), {"path": tmp_path / "shadow.db"})()

        def run(self, **kwargs):
            captured["run"] = kwargs
            decision = type(
                "Decision",
                (),
                {
                    "decision_id": "decision-001",
                    "action": "HOLD",
                    "normalization_status": "NORMALIZED",
                    "decision_context_status": "COMPLETE",
                    "executed": False,
                    "requested_symbol": "EURUSD",
                    "resolved_symbol": "EURUSD",
                    "snapshot_timestamp": "2026-09-08T00:00:00Z",
                    "raw_portfolio_manager_result_json": '{"rating":"Hold"}',
                },
            )()
            return type("Result", (), {"decision": decision, "elapsed_seconds": 0, "metrics": {}})()

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FakeRunner)

    assert main(["--db-path", str(tmp_path / "shadow.db")]) == 0
    assert captured.get("constructed") is True


def test_cli_passes_stats_callback_and_prints_run_evidence(capsys, monkeypatch, tmp_path):
    captured = {}

    class FakeStats:
        def get_stats(self):
            return {"llm_calls": 9, "tool_calls": 3, "tokens_in": 10, "tokens_out": 20}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured["config"] = kwargs

        def run(self, **kwargs):
            captured["run"] = kwargs
            decision = type(
                "Decision",
                (),
                {
                    "decision_id": "decision-001",
                    "action": "HOLD",
                    "normalization_status": "NORMALIZED",
                    "decision_context_status": "COMPLETE",
                    "executed": False,
                    "requested_symbol": "EURUSD",
                    "resolved_symbol": "EURUSD",
                    "snapshot_timestamp": "2026-09-08T00:00:00Z",
                    "raw_portfolio_manager_result_json": '{"rating":"Hold"}',
                    "llm_provider": "openai",
                    "quick_model": "gpt-5.6-luna",
                    "deep_model": "gpt-5.6",
                    "reference_bid": 1.1,
                    "reference_ask": 1.2,
                    "spread": 0.1,
                    "spread_points": 10000,
                    "analysis_profile": "INTRADAY",
                    "valid_for_seconds": 3600,
                    "valid_until": "2026-09-08T01:00:00Z",
                },
            )()
            return type(
                "Result",
                (),
                {
                    "decision": decision,
                    "elapsed_seconds": 1.25,
                    "metrics": {
                        "llm_calls": 9,
                        "tool_calls": 3,
                        "tokens_in": 10,
                        "tokens_out": 20,
                        "reasoning_tokens": 4,
                        "macro_event_status": "MACRO/EVENT DATA UNAVAILABLE",
                        "bars_used": {"M1": 100, "M5": 100, "M15": 100, "H1": 100},
                        "agents": {
                            "Trader": {
                                "model": "gpt-5.6-luna",
                                "calls": 1,
                                "tokens_in": 2,
                                "tokens_out": 3,
                                "reasoning_tokens": 0,
                                "elapsed_seconds": 0.5,
                            }
                        },
                    },
                },
            )()

    monkeypatch.setattr("cli.forex_shadow.ForexShadowRunner", FakeRunner)
    monkeypatch.setattr("cli.forex_shadow.StatsCallbackHandler", FakeStats)

    assert main(["--db-path", str(tmp_path / "shadow.db")]) == 0
    assert len(captured["run"]["callbacks"]) == 1
    output = capsys.readouterr().out
    assert "LLM PROVIDER: openai" in output
    assert "LLM CALLS: 9" in output
    assert "NORMALIZED ACTION: HOLD" in output
    assert "DECISION CONTEXT STATUS: COMPLETE" in output
    assert "ANALYSIS PROFILE: INTRADAY" in output
    assert "VALID FOR SECONDS: 3600" in output
    assert "MACRO/EVENT STATUS: MACRO/EVENT DATA UNAVAILABLE" in output
    assert 'BARS USED: {"H1": 100, "M1": 100, "M15": 100, "M5": 100}' in output
    assert "AGENT METRICS: Trader" in output
    assert "EXECUTED: FALSE" in output
