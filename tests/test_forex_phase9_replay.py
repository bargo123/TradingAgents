from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tradingagents.agents.schemas import ForexPortfolioDecision
from tradingagents.dataflows.mt5.models import (
    ForexMarketSnapshot,
    Mt5AccountInfo,
    Mt5Bar,
    Mt5Position,
    Mt5SymbolInfo,
)
from tradingagents.forex.context import snapshot_to_dict
from tradingagents.forex.evidence_context import (
    CanonicalEvidenceItem,
    CanonicalKnowledgeQuery,
    EvidenceBundleStatus,
    EvidenceContext,
    EvidenceIntegrationStatus,
    EvidenceSourceKind,
)
from tradingagents.forex.evidence_replay import (
    EvidenceReplayConfig,
    SavedSnapshotCodec,
    SavedSnapshotReplay,
    SnapshotReplayError,
    _fingerprint,
    _source_has_transient,
    _source_row,
)
from tradingagents.forex.runner import ForexShadowRunner
from tradingagents.forex.telemetry import record_state_boundary


def _snapshot() -> ForexMarketSnapshot:
    timestamp = datetime(2026, 9, 12, 12, 0, tzinfo=timezone.utc)
    bar = Mt5Bar(timestamp, 1.1, 1.101, 1.099, 1.1005, 12, 2, 12)
    return ForexMarketSnapshot(
        timestamp=timestamp,
        symbol="EURUSDm",
        bid=1.1,
        ask=1.1002,
        spread=0.0002,
        spread_points=2.0,
        m1_candles=(bar,),
        m5_candles=(bar,),
        m15_candles=(bar,),
        h1_candles=(bar,),
        account=Mt5AccountInfo(login=7, server="demo", currency="USD", balance=1000.0),
        positions=(Mt5Position(ticket=1, symbol="EURUSDm", volume=0.1, time=timestamp),),
        symbol_info=Mt5SymbolInfo(name="EURUSDm", digits=5, point=0.00001, currency_base="EUR", currency_profit="USD"),
    )


def _row(snapshot: ForexMarketSnapshot | None = None) -> dict:
    snapshot = snapshot or _snapshot()
    return {
        "decision_id": "decision-1",
        "resolved_symbol": snapshot.symbol,
        "analysis_profile": "INTRADAY",
        "snapshot_json": json.dumps(snapshot_to_dict(snapshot), sort_keys=True),
    }


class _FakeRunner:
    def __init__(self, calls: list[dict], *, generations: tuple[str, str]):
        self.calls = calls
        self.generations = generations
        self.run_called = False

    def analyze(self, **kwargs):
        self.calls.append(kwargs)
        context = None
        if kwargs.get("forex_evidence_enabled"):
            context = EvidenceContext(
                integration_status=EvidenceIntegrationStatus.INJECTED,
                bundle_status=EvidenceBundleStatus.COMPLETE,
                as_of=kwargs["snapshot"].timestamp,
                knowledge_generation_id=self.generations[0],
                experience_generation_id=self.generations[1],
                rendered_context="bounded evidence",
                rendered_context_hash=hashlib.sha256(b"bounded evidence").hexdigest(),
            )
        return {
            "normalized_action": "BUY" if kwargs.get("forex_evidence_enabled") else "HOLD",
            "evidence_context": context,
            "analysis_telemetry": {"telemetry_status": "AVAILABLE"},
        }

    def run(self, **_kwargs):
        self.run_called = True
        raise AssertionError("replay must not call run")


def _config(tmp_path: Path) -> EvidenceReplayConfig:
    db = tmp_path / "source.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE shadow_decisions (decision_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL)")
        conn.execute("INSERT INTO shadow_decisions VALUES (?, ?)", ("decision-1", _row()["snapshot_json"]))
        conn.commit()
    finally:
        conn.close()
    return EvidenceReplayConfig(
        source_decision_id="decision-1",
        profile="INTRADAY",
        analysts=("market", "news"),
        provider="ollama",
        models={"quick": "qwen", "deep": "qwen"},
        model_settings={"temperature": 0},
        pinned_phase7_generation_id="p7",
        pinned_phase8_generation_id="p8",
        source_database_path=db,
    )


def _source_bytes() -> bytes:
    return _row()["snapshot_json"].encode("utf-8")


def test_saved_snapshot_codec_validates_row():
    snapshot = SavedSnapshotCodec.from_source_row(_row())
    assert snapshot.symbol == "EURUSDm"
    assert snapshot.timestamp.tzinfo == timezone.utc
    assert isinstance(snapshot.account, Mt5AccountInfo)
    assert isinstance(snapshot.symbol_info, Mt5SymbolInfo)
    assert isinstance(snapshot.m1_candles[0], Mt5Bar)
    assert isinstance(snapshot.positions[0], Mt5Position)

    with pytest.raises(SnapshotReplayError):
        SavedSnapshotCodec.from_source_row({"snapshot_json": "{}"})
    mismatched = _row()
    mismatched["analysis_snapshot_timestamp"] = "2026-09-12T12:01:00Z"
    with pytest.raises(SnapshotReplayError, match="analysis_snapshot_timestamp"):
        SavedSnapshotCodec.from_source_row(mismatched)


def test_replay_uses_identical_snapshot_fingerprint(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    snapshot = SavedSnapshotCodec.from_source_row(_row())
    result = replay.run(snapshot, snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert result.snapshot_fingerprint == hashlib.sha256(_source_bytes()).hexdigest()
    assert calls[0]["snapshot_bytes"] is calls[1]["snapshot_bytes"]
    assert calls[0]["snapshot_fingerprint"] == calls[1]["snapshot_fingerprint"] == result.snapshot_fingerprint


def test_replay_runs_baseline_then_evidence_sequentially(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert [call["forex_evidence_enabled"] for call in calls] == [False, True]
    assert report.baseline_action == "HOLD"
    assert report.evidence_action == "BUY"
    assert report.did_action_change is True
    assert runner.run_called is False


def test_replay_does_not_change_source_schema_or_rows(tmp_path: Path):
    config = _config(tmp_path)
    before = config.source_database_path.read_bytes()
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=config)
    assert config.source_database_path.read_bytes() == before
    with sqlite3.connect(config.source_database_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0] == 1


def test_replay_is_not_a_normal_opportunity(tmp_path: Path):
    runner = _FakeRunner([], generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert not hasattr(report, "decision")
    assert runner.run_called is False


def test_replay_pins_phase7_and_phase8_generations(tmp_path: Path):
    calls: list[dict] = []
    runner = _FakeRunner(calls, generations=("p7", "p8"))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: runner, generation_provider=lambda: ("p7", "p8"))
    replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert all(call["pinned_phase7_generation_id"] == "p7" for call in calls)
    assert all(call["pinned_phase8_generation_id"] == "p8" for call in calls)


def test_replay_propagates_pinned_generations_to_evidence_service(tmp_path: Path):
    seen_generation_providers: list[object] = []

    class _Graph:
        def __init__(self, **_kwargs):
            self.propagator = type(
                "P",
                (),
                {
                    "create_initial_state": lambda _self, *args, **kw: kw,
                    "get_graph_args": lambda _self, **_kw: {},
                },
            )()

        def invoke(self, _state, **_kwargs):
            return {
                "portfolio_manager_raw_result": {"rating": "Hold"},
                "final_trade_decision": {"rating": "Hold"},
            }

    class _Evidence:
        def __init__(self, generations: tuple[str | None, str | None]):
            self.generations = generations

        def retrieve(self, snapshot, **_kwargs):
            rendered = "saved evidence"
            return EvidenceContext(
                integration_status=EvidenceIntegrationStatus.INJECTED,
                bundle_status=EvidenceBundleStatus.COMPLETE,
                as_of=snapshot.timestamp,
                knowledge_generation_id=self.generations[0],
                experience_generation_id=self.generations[1],
                rendered_context=rendered,
                rendered_context_hash=hashlib.sha256(rendered.encode()).hexdigest(),
            )

    def evidence_factory(config, **_kwargs):
        provider = config.get("evidence_generation_provider", (None, None))
        seen_generation_providers.append(provider)
        return _Evidence(tuple(provider))

    runner = ForexShadowRunner(
        provider_factory=lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("saved replay must not construct MT5")
        ),
        graph_factory=lambda **kwargs: _Graph(**kwargs),
        evidence_service_factory=evidence_factory,
    )
    config = _config(tmp_path)
    replay = SavedSnapshotReplay(
        runner=runner,
        generation_provider=lambda: ("p7", "p8"),
    )

    report = replay.run(
        None,
        snapshot_bytes=_source_bytes(),
        config=config,
    )

    assert report.comparison_status == "VALID"
    assert seen_generation_providers == [("p7", "p8")]


def test_knowledge_generation_change_invalidates_replay(tmp_path: Path):
    generations = iter((("p7", "p8"), ("p7", "p8"), ("changed", "p8")))
    calls: list[dict] = []
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner(calls, generations=("p7", "p8")), generation_provider=lambda: next(generations))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_experience_generation_change_invalidates_replay(tmp_path: Path):
    generations = iter((("p7", "p8"), ("p7", "p8"), ("p7", "changed")))
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: next(generations))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_generation_change_between_legs_invalidates_even_if_restored(tmp_path: Path):
    samples = iter((("p7", "p8"), ("changed", "p8"), ("p7", "p8")))
    observed: list[tuple[str | None, str | None]] = []

    def generation_provider():
        value = next(samples)
        observed.append(value)
        return value

    replay = SavedSnapshotReplay(
        runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")),
        generation_provider=generation_provider,
    )
    report = replay.run(None, snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert len(observed) == 3
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_replay_report_excludes_prompts_and_reasoning(tmp_path: Path):
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    report = replay.run(SavedSnapshotCodec.from_source_row(_row()), snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    payload = report.to_dict()
    forbidden = {"prompt", "completion", "reasoning", "private_reasoning"}
    assert not forbidden.intersection(payload)
    assert all(not any(word in str(key).lower() for word in forbidden) for key in payload)


def test_replay_report_preserves_safe_knowledge_query_metadata(tmp_path: Path):
    class _Runner:
        def analyze(self, **kwargs):
            context = None
            if kwargs.get("forex_evidence_enabled"):
                context = EvidenceContext(
                    integration_status=EvidenceIntegrationStatus.INJECTED,
                    bundle_status=EvidenceBundleStatus.COMPLETE,
                    as_of=kwargs["snapshot"].timestamp,
                    knowledge_generation_id="p7",
                    experience_generation_id="p8",
                    knowledge_query=CanonicalKnowledgeQuery("forex liquidity", "query-fp", "v2"),
                    knowledge_query_fingerprint="query-fp",
                    knowledge_query_policy_version="v2",
                    rendered_context="bounded evidence",
                    rendered_context_hash=hashlib.sha256(b"bounded evidence").hexdigest(),
                )
            return {
                "normalized_action": "BUY" if kwargs.get("forex_evidence_enabled") else "HOLD",
                "evidence_context": context,
            }

    replay = SavedSnapshotReplay(runner=_Runner(), generation_provider=lambda: ("p7", "p8"))
    report = replay.run(None, snapshot_bytes=_source_bytes(), config=_config(tmp_path))
    assert report.knowledge_query == CanonicalKnowledgeQuery("forex liquidity", "query-fp", "v2")
    assert report.knowledge_query_fingerprint == "query-fp"
    assert report.query_policy_version == "v2"


def test_actual_runner_consumes_saved_snapshot_without_mt5_or_persistence(tmp_path: Path):
    class _NoMt5:
        def __call__(self, **_kwargs):
            raise AssertionError("saved replay must not construct MT5")

    class _Graph:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.propagator = type(
                "P",
                (),
                {
                    "create_initial_state": lambda _self, *args, **kw: kw,
                    "get_graph_args": lambda _self, **_kw: {},
                },
            )()

        def invoke(self, _state, **_kwargs):
            return {
                "portfolio_manager_raw_result": {"rating": "Hold"},
                "final_trade_decision": {"rating": "Hold"},
            }

    runner = ForexShadowRunner(
        provider_factory=_NoMt5(),
        graph_factory=lambda **kwargs: _Graph(**kwargs),
        store=type("Store", (), {"record": lambda *_args: (_ for _ in ()).throw(AssertionError("persisted"))})(),
    )
    snapshot = _snapshot()
    result = runner.analyze(
        symbol=snapshot.symbol,
        analysis_date=snapshot.timestamp.date(),
        snapshot=snapshot,
        snapshot_bytes=b"saved-row-bytes",
        forex_evidence_enabled=False,
        pinned_phase7_generation_id="p7",
        pinned_phase8_generation_id="p8",
        persist=False,
    )
    assert result.snapshot is snapshot
    assert result.normalized_action == "HOLD"


def test_actual_runner_replay_leg_passes_toggle_and_generation_pins(tmp_path: Path):
    class _NoMt5:
        def __call__(self, **_kwargs):
            raise AssertionError("saved replay must not construct MT5")

    graphs: list[object] = []

    class _Graph:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            graphs.append(self)
            self.propagator = type(
                "P",
                (),
                {
                    "create_initial_state": lambda _self, *args, **kw: kw,
                    "get_graph_args": lambda _self, **_kw: {},
                },
            )()

        def invoke(self, _state, **_kwargs):
            return {
                "portfolio_manager_raw_result": {"rating": "Hold"},
                "final_trade_decision": {"rating": "Hold"},
            }

    class _Evidence:
        def retrieve(self, snapshot, **_kwargs):
            rendered = "saved evidence"
            return EvidenceContext(
                integration_status=EvidenceIntegrationStatus.INJECTED,
                as_of=snapshot.timestamp,
                knowledge_generation_id="p7",
                experience_generation_id="p8",
                rendered_context=rendered,
                rendered_context_hash=hashlib.sha256(rendered.encode()).hexdigest(),
            )

    seen_configs: list[dict] = []
    runner = ForexShadowRunner(
        provider_factory=_NoMt5(),
        graph_factory=lambda **kwargs: _Graph(**kwargs),
        evidence_service_factory=lambda config, **_kwargs: (seen_configs.append(config) or _Evidence()),
    )
    snapshot = _snapshot()
    for enabled in (False, True):
        runner.analyze(
            symbol=snapshot.symbol,
            analysis_date=snapshot.timestamp.date(),
            snapshot=snapshot,
            snapshot_bytes=b"saved-row-bytes",
            evidence_enabled=enabled,
            pinned_phase7_generation_id="p7",
            pinned_phase8_generation_id="p8",
            provider="ollama",
            models={"quick": "qwen", "deep": "qwen"},
            model_settings={"temperature": 0},
            normalization_path="v1",
            persist=False,
        )
    assert len(graphs) == 2
    assert [graph.kwargs["config"]["forex_evidence_enabled"] for graph in graphs] == [False, True]
    assert graphs[1].kwargs["config"]["pinned_phase7_generation_id"] == "p7"
    assert graphs[1].kwargs["config"]["pinned_phase8_generation_id"] == "p8"
    assert graphs[1].kwargs["config"]["llm_provider"] == "ollama"
    assert graphs[1].kwargs["config"]["quick_think_llm"] == "qwen"
    assert graphs[1].kwargs["config"]["deep_think_llm"] == "qwen"
    assert graphs[1].kwargs["config"]["temperature"] == 0
    assert seen_configs and seen_configs[0]["pinned_phase7_generation_id"] == "p7"


def test_deterministic_structured_replay_populates_context_and_normalizes(tmp_path: Path):
    """A fake structured PM result proves replay wiring without MT5/Ollama."""

    class _NoMt5:
        def __call__(self, **_kwargs):
            raise AssertionError("saved replay must not construct MT5")

    class _Graph:
        def __init__(self, **_kwargs):
            self.propagator = type(
                "P",
                (),
                {
                    "create_initial_state": lambda _self, *args, **kw: kw,
                    "get_graph_args": lambda _self, **_kw: {},
                },
            )()

        def invoke(self, state, **_kwargs):
            decision = ForexPortfolioDecision(
                rating="Hold",
                executive_summary="bounded fake decision",
                investment_thesis="deterministic replay fixture",
                evidence_use_status="USED",
                evidence_refs_used=["K1"],
            )
            output = {
                **state,
                "market_report": "market report",
                "news_report": "news report",
                "investment_debate_state": {
                    "bull_history": "bull report",
                    "bear_history": "bear report",
                    "history": "research debate",
                },
                "investment_plan": "research manager plan",
                "trader_investment_plan": "trader plan",
                "risk_debate_state": {
                    "history": "risk debate",
                    "aggressive_history": "aggressive",
                    "conservative_history": "conservative",
                    "neutral_history": "neutral",
                },
                "portfolio_manager_raw_result": decision,
                "final_trade_decision": "HOLD",
            }
            for node in (
                "Market Analyst",
                "News Analyst",
                "Bull Researcher",
                "Bear Researcher",
                "Research Manager",
                "Trader",
                "Aggressive Analyst",
                "Conservative Analyst",
                "Neutral Analyst",
                "Portfolio Manager",
            ):
                record_state_boundary(node, "after", output)
            return output

    rendered = '{"items":[{"display_id":"K1"}]}'
    context = EvidenceContext(
        integration_status=EvidenceIntegrationStatus.INJECTED,
        bundle_status=EvidenceBundleStatus.COMPLETE,
        as_of=_snapshot().timestamp,
        knowledge_generation_id="p7",
        experience_generation_id="p8",
        knowledge_query=CanonicalKnowledgeQuery("forex liquidity", "query-fp", "v2"),
        knowledge_query_fingerprint="query-fp",
        knowledge_query_policy_version="v2",
        knowledge_items=(
            CanonicalEvidenceItem(
                display_id="K1",
                source_kind=EvidenceSourceKind.KNOWLEDGE,
                authoritative_id="chunk-1",
                text="liquidity",
                content_type="PROSE",
                score=1.0,
                provenance={"document_id": "doc-1", "chunk_id": "chunk-1"},
            ),
        ),
        selected_knowledge_count=1,
        rendered_context=rendered,
        rendered_context_hash=hashlib.sha256(rendered.encode()).hexdigest(),
        rendered_character_count=len(rendered),
    )

    class _Evidence:
        def retrieve(self, _snapshot, **_kwargs):
            return context

    runner = ForexShadowRunner(
        provider_factory=_NoMt5(),
        graph_factory=lambda **kwargs: _Graph(**kwargs),
        evidence_service_factory=lambda **_kwargs: _Evidence(),
    )
    replay = SavedSnapshotReplay(runner=runner, generation_provider=lambda: ("p7", "p8"))
    report = replay.run(None, snapshot_bytes=_source_bytes(), config=_config(tmp_path))

    assert report.comparison_status == "VALID"
    assert report.evidence_action == "HOLD"
    assert report.normalization_status == "NORMALIZED"
    assert report.context_integrity_status == "COMPLETE"
    assert report.knowledge_count == 1
    assert report.evidence_use_status == "USED"
    assert report.used_references == ("K1",)


def test_normal_runner_keeps_scalar_llm_provider_when_mt5_provider_is_live(tmp_path: Path):
    snapshot = _snapshot()
    graph_configs: list[dict] = []

    class _Provider:
        market_snapshot_calls = 0

        def initialize(self):
            return True

        def shutdown(self):
            return None

        def ensure_symbol(self, _symbol):
            return snapshot.symbol

        def get_market_snapshot(self, _symbol, count=100):
            self.market_snapshot_calls += 1
            return snapshot

        def get_spread(self, symbol):
            return type("Spread", (), {"symbol": symbol, "bid": snapshot.bid, "ask": snapshot.ask, "price": snapshot.spread, "points": snapshot.spread_points, "timestamp": snapshot.timestamp})()

    class _Graph:
        def __init__(self, **kwargs):
            graph_configs.append(kwargs["config"])
            self.propagator = type(
                "P",
                (),
                {
                    "create_initial_state": lambda _self, *args, **kw: kw,
                    "get_graph_args": lambda _self, **_kw: {},
                },
            )()

        def invoke(self, _state, **_kwargs):
            return {"portfolio_manager_raw_result": {"rating": "Hold"}, "final_trade_decision": {"rating": "Hold"}}

    runner = ForexShadowRunner(
        provider_factory=lambda **_kwargs: _Provider(),
        graph_factory=lambda **kwargs: _Graph(**kwargs),
        config={"llm_provider": "openai", "data_cache_dir": str(tmp_path / "cache")},
    )
    runner.analyze(symbol="EURUSD", analysis_date=snapshot.timestamp.date())
    assert graph_configs and graph_configs[0]["llm_provider"] == "openai"


def test_replay_binds_to_source_decision_and_snapshot_bytes(tmp_path: Path):
    config = _config(tmp_path)
    source_bytes = _row()["snapshot_json"].encode("utf-8")
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    with pytest.raises(SnapshotReplayError, match="source"):
        replay.run(replace(_snapshot(), bid=2.0), snapshot_bytes=source_bytes, config=config)

    replay.run(None, snapshot_bytes=source_bytes, config=config)


def test_replay_requires_read_only_source_database_binding(tmp_path: Path):
    config = replace(_config(tmp_path), source_database_path=None)
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    with pytest.raises(SnapshotReplayError, match="source database"):
        replay.run(_snapshot(), snapshot_bytes=_source_bytes(), config=config)


def test_source_helpers_release_windows_sqlite_handles(tmp_path: Path):
    helpers = (
        lambda path: _source_row(path, "decision-1"),
        lambda path: _fingerprint(path)["decision_row_count"],
        _source_has_transient,
    )
    for helper in helpers:
        config = _config(tmp_path)
        result = helper(config.source_database_path)
        assert result is not None
        config.source_database_path.unlink()
        assert not config.source_database_path.exists()


def test_replay_rejects_requested_audit_path_instead_of_ignoring_it(tmp_path: Path):
    config = _config(tmp_path)
    config = replace(config, audit_path=tmp_path / "audit.sqlite3")
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: ("p7", "p8"))
    with pytest.raises(SnapshotReplayError, match="audit"):
        replay.run(None, snapshot_bytes=_row()["snapshot_json"].encode(), config=config)


def test_missing_generation_or_evidence_context_invalidates_replay(tmp_path: Path):
    config = _config(tmp_path)
    replay = SavedSnapshotReplay(runner_factory=lambda **_: _FakeRunner([], generations=("p7", "p8")), generation_provider=lambda: (None, "p8"))
    with pytest.raises(SnapshotReplayError, match="generation"):
        replay.run(None, snapshot_bytes=_row()["snapshot_json"].encode(), config=config)

    class _NoContext:
        def analyze(self, **_kwargs):
            return {"normalized_action": "HOLD", "evidence_context": None}

    replay = SavedSnapshotReplay(runner=_NoContext(), generation_provider=lambda: ("p7", "p8"))
    report = replay.run(None, snapshot_bytes=_row()["snapshot_json"].encode(), config=config)
    assert report.comparison_status == "INVALID_GENERATION_CHANGED"
    assert report.did_action_change is None


def test_report_sanitizes_forbidden_fields_inside_dataclasses(tmp_path: Path):
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Telemetry:
        reasoning: str
        nested: dict[str, str]

    class _Runner:
        def analyze(self, **kwargs):
            return {
                "normalized_action": "HOLD",
                "evidence_context": None,
                "analysis_telemetry": _Telemetry("secret", {"completion": "secret"}),
            }

    replay = SavedSnapshotReplay(runner=_Runner(), generation_provider=lambda: ("p7", "p8"))
    report = replay.run(None, snapshot_bytes=_row()["snapshot_json"].encode(), config=_config(tmp_path))
    report = replace(report, models={"nested": _Telemetry("secret", {"completion": "secret"})})
    payload = report.to_dict()
    assert "reasoning" not in json.dumps(payload).lower()
    assert "completion" not in json.dumps(payload).lower()
