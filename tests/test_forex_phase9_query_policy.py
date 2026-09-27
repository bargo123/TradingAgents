from datetime import datetime, timezone

import pytest

from tradingagents.dataflows.mt5.models import ForexMarketSnapshot, Mt5Bar, Mt5SymbolInfo
from tradingagents.experience.features import FEATURE_NAMES_V1
from tradingagents.experience.models import EvidenceRequest
from tradingagents.experience.orchestrator import EvidenceOrchestrator
from tradingagents.forex.evidence_context import EvidenceQueryPolicy, EvidenceSnapshotAdapter


def _snapshot() -> ForexMarketSnapshot:
    bars = tuple(
        Mt5Bar(datetime(2026, 1, 1, 12, i, tzinfo=timezone.utc), 1.1 + i * .0001,
               1.101 + i * .0001, 1.099 + i * .0001, 1.1005 + i * .0001, 10)
        for i in range(2)
    )
    return ForexMarketSnapshot(
        timestamp=datetime(2026, 1, 1, 12, tzinfo=timezone.utc), symbol=" eurusd ",
        bid=1.1001, ask=1.1002, spread=.0001, spread_points=10, m1_candles=bars,
        m5_candles=bars, m15_candles=bars, h1_candles=bars, account=None, positions=(),
        symbol_info=Mt5SymbolInfo("EURUSD", digits=5, point=.00001),
    )


def test_query_uses_only_present_snapshot_fields():
    query = EvidenceQueryPolicy().build_knowledge_query(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15"
    )
    assert "EURUSD" in query.text and "INTRADAY" in query.text and "M15" in query.text
    assert "UP" in query.text
    assert "spread" in query.text.lower()
    assert "=" not in query.text


def test_query_omits_l2_and_final_action():
    query = EvidenceQueryPolicy().build_knowledge_query(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15"
    )
    lowered = query.text.lower()
    assert all(term not in lowered for term in ("l2", "order book", "final action", "future", "fundamental", "social"))


def test_query_fingerprint_is_versioned():
    query = EvidenceQueryPolicy().build_knowledge_query(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15"
    )
    assert query.policy_version == "v2"
    assert len(query.fingerprint) == 64


def test_snapshot_adapter_uses_phase8_feature_vector_shape():
    state = EvidenceSnapshotAdapter().to_market_state(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15"
    )
    assert tuple(state["feature_names"]) == FEATURE_NAMES_V1
    assert len(state["values"]) == len(FEATURE_NAMES_V1)
    assert tuple(state["cohort"])[0:3] == ("EURUSD", "INTRADAY", "M15")


def test_default_statistics_status_is_not_requested():
    request, _ = EvidenceQueryPolicy().build_request(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15",
        as_of=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
    )
    assert isinstance(request, EvidenceRequest)
    assert request.evaluation_basis is None and request.horizon_seconds is None


def test_explicit_basis_and_horizon_are_forwarded():
    policy = EvidenceQueryPolicy(statistics_horizon_seconds=900)
    request, _ = policy.build_request(
        _snapshot(), resolved_symbol="EURUSD", analysis_profile="INTRADAY", analysis_timeframe="M15",
        as_of=datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
    )
    assert request.evaluation_basis == "ANALYSIS_SNAPSHOT" and request.horizon_seconds == 900


def test_tier_c_is_never_an_experience_item():
    assert all("TIER_C" not in tier.value for tier in EvidenceQueryPolicy().trust_tiers)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("knowledge_top_k", True),
        ("experience_top_k", 1.0),
        ("max_knowledge_items", "4"),
        ("max_experience_items", None),
        ("max_statistics_items", False),
        ("max_rendered_characters", 6000.0),
        ("evidence_timeout_seconds", True),
        ("evidence_timeout_seconds", "10"),
        ("statistics_horizon_seconds", True),
        ("statistics_horizon_seconds", 900.0),
        ("statistics_horizon_seconds", "900"),
    ],
)
def test_query_policy_rejects_non_contract_numeric_types(field, value):
    with pytest.raises(ValueError):
        EvidenceQueryPolicy(**{field: value})


def test_query_policy_requires_string_evaluation_basis_when_statistics_are_enabled():
    with pytest.raises(ValueError):
        EvidenceQueryPolicy(statistics_horizon_seconds=900, evaluation_basis=1)


@pytest.mark.parametrize("basis", ["", "UNKNOWN", "analysis_snapshot", 1])
def test_query_policy_rejects_unsupported_evaluation_basis(basis):
    with pytest.raises(ValueError):
        EvidenceQueryPolicy(evaluation_basis=basis)


@pytest.mark.parametrize("field", ["query_policy_version", "budget_policy_version"])
@pytest.mark.parametrize("value", [False, 0, [], {}])
def test_query_policy_rejects_non_text_version_fields(field, value):
    with pytest.raises(ValueError, match=field):
        EvidenceQueryPolicy(**{field: value})


def test_canonical_query_is_fts_safe_and_returns_provenance_fixture():
    """The deterministic query must be usable by the lexical Phase 7 seam."""

    snapshot = _snapshot()
    policy = EvidenceQueryPolicy()
    request, query = policy.build_request(
        snapshot,
        resolved_symbol="EURUSD",
        analysis_profile="INTRADAY",
        analysis_timeframe="M15",
        as_of=snapshot.timestamp,
    )

    class _KnowledgeFixture:
        calls = 0

        def search(self, knowledge_request):
            self.calls += 1
            # SQLite FTS5 treats the old key=value form as an expression and
            # rejects it.  The fixture models that real lexical boundary.
            if "=" in knowledge_request.text:
                return ()
            return ("provenance-bearing-knowledge-hit",)

    fixture = _KnowledgeFixture()
    bundle = EvidenceOrchestrator(
        fixture,
        type("Experience", (), {"search": lambda _self, _request: ()})(),
        type("Statistics", (), {"calculate": lambda _self, _request: None})(),
    ).query(request)

    assert "=" not in query.text
    assert bundle.status == "COMPLETE"
    assert len(bundle.knowledge) == 1
    assert fixture.calls == 1
    lowered = query.text.lower()
    assert all(term not in lowered for term in ("buy", "sell", "hold", "outcome", "future"))
    assert request.evaluation_basis is None and request.horizon_seconds is None
