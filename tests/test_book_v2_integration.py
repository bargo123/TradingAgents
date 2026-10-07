import builtins

import pytest

from tests.test_book_natural_language import source_lookup
from tests.test_book_natural_language_sources import source_cases
from tests.test_book_v2_extraction import _Transport, extractor, sentence
from tradingagents.self_enhancement.book_v2_artifacts import (
    evaluate_v2_artifacts,
    load_v2_discovery_artifacts,
    publish_v2,
)
from tradingagents.self_enhancement.book_v2_registry import map_v2
from tradingagents.self_enhancement.phase14c_mapping import capture_current_strategy_contracts


def test_model_to_proof_artifact_reload_keeps_real_rules_nonexecutable(tmp_path, monkeypatch):
    from tradingagents.dataflows.mt5.provider import MT5Provider
    from tradingagents.forex.hft.demo_gateway import VerifiedDemoExecutionGateway
    from tradingagents.forex.supervisor import ForexSupervisor
    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("offline qualification cannot construct a broker or supervisor")
    for cls in (MT5Provider, VerifiedDemoExecutionGateway, ForexSupervisor):
        monkeypatch.setattr(cls, "__init__", forbidden_constructor)
    original = builtins.__import__
    def block_broker(name, *args, **kwargs):
        if name == "MetaTrader5" or "forex_supervisor" in name:
            raise AssertionError("offline path must not construct trading runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", block_broker)

    def propose(user, number):
        quote = user["evidence"][0]["text"]
        stage = "EXPECTED_MOVE" if "target on each trade" in quote else "STOP_BEHAVIOR"
        return {"rules": [{"family": "OTHER_SUPPORTED", "stage": stage, "evidence_indices": [0]}]}
    worker = extractor(_Transport(propose))
    results = worker.extract_actionable_v2(tuple(sentence(c) for c in source_cases()[:2]), source_lookup=source_lookup)
    root = tmp_path / "v2"
    identity = {"generation_id": source_cases()[0]["evidence"]["generation_id"], "generation_fingerprint": "a" * 64, "population_hash": "sha256:" + "b" * 64}
    report = publish_v2(root, identity=identity, results=results, source_lookup=source_lookup, telemetry=[c.to_dict() for c in worker.calls])
    assert report["llm_calls"] == 2
    assert report["supported_rule_count"] == 2
    assert report["eligible_candidate_count"] == 0
    candidates = load_v2_discovery_artifacts(root, source_lookup=source_lookup)
    assert len(candidates) == 2
    for candidate in candidates:
        assert candidate.status.value == "SUPPORTED_NONEXECUTABLE"
        mappings = map_v2(candidate, snapshot=capture_current_strategy_contracts(), source_lookup=source_lookup)
        assert all(c.status.value == "NO_DIRECT_MATCH" for m in mappings for c in m.components)
    assert evaluate_v2_artifacts(root, source_lookup=source_lookup)["status"] == "NOT_RUN"


def test_artifact_telemetry_cannot_persist_raw_model_output(tmp_path):
    identity = {"generation_id": "generation", "generation_fingerprint": "a" * 64, "population_hash": "sha256:" + "b" * 64}
    with pytest.raises(ValueError, match="telemetry"):
        publish_v2(tmp_path / "unsafe", identity=identity, results=(), source_lookup=source_lookup, telemetry=[{"raw_output": "untrusted reasoning"}])
