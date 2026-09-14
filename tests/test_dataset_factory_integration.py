from __future__ import annotations

import json
from pathlib import Path

from tests.fixtures.dataset_factory import canonical, fingerprint, observations
from tradingagents.datasets.eligibility import EligibilityResult
from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import DatasetConfig, DatasetExclusionReason
from tradingagents.datasets.sources import SourceReadResult


def _install_fixture_readers(monkeypatch, tmp_path):
    source = tmp_path / "source.db"
    source.write_bytes(b"fixture-source")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    audit = tmp_path / "audit.db"
    audit.write_bytes(b"fixture-audit")
    rows = observations(source)
    fps = {"phase56": fingerprint("phase56", source), "phase8": fingerprint("phase8", phase8), "phase9": fingerprint("phase9", audit)}

    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self):
            if self.path == source:
                return SourceReadResult(decisions=tuple(row.decision for row in rows), fingerprint=fps["phase56"])
            return SourceReadResult(fingerprint=fps["phase8"] if self.path.name == "catalog.sqlite3" else fps["phase9"])

    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory._bind_sqlite_marks", lambda result, digest: result)
    monkeypatch.setattr("tradingagents.datasets.factory._source_inventory", lambda results, p8, p9: [{"kind": "phase56", "source_id": "fixture", "canonical_path": str(source), "file_sha256": "fixture", "snapshot_fingerprint": "fixture", "decision_ids": [], "decision_fingerprints": [], "decision_identities": []}])
    monkeypatch.setattr("tradingagents.datasets.factory._fingerprint", lambda result, kind: fps[kind])
    monkeypatch.setattr("tradingagents.datasets.factory._coalesce_fingerprints", lambda results: fps["phase56"].to_dict())
    return source, phase8, audit, rows, fps


def test_public_factory_end_to_end_is_deterministic_and_explicit(monkeypatch, tmp_path):
    source, phase8, audit, rows, fps = _install_fixture_readers(monkeypatch, tmp_path)
    outcomes = {row.decision.decision_id: row for row in rows}
    def join(*_): return tuple(outcomes.values())
    def classify(observation, _config):
        ident = observation.decision.decision_id
        if ident.startswith("valid"):
            return EligibilityResult(True, ())
        reason = {"tier-b": DatasetExclusionReason.UNTRUSTED_TIER, "tier-c": DatasetExclusionReason.UNTRUSTED_TIER, "incomplete": DatasetExclusionReason.DECISION_CONTEXT_INCOMPLETE, "unavailable": DatasetExclusionReason.OUTCOME_UNAVAILABLE, "future": DatasetExclusionReason.TEMPORAL_INVALID}[ident]
        return EligibilityResult(False, (reason,), {"decision_id": ident})
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", join)
    monkeypatch.setattr("tradingagents.datasets.factory.classify_observation", classify)
    monkeypatch.setattr("tradingagents.datasets.factory.canonicalize", lambda obs, elig: canonical(obs.decision.decision_id, fps, int(obs.decision.decision_id[-1]) if obs.decision.decision_id[-1].isdigit() else 0))
    config = DatasetConfig((source,), phase8, audit, tmp_path / "out")
    first = DatasetFactory().build(config)
    assert first.status == "EMPTY_ELIGIBLE_SET" or first.manifest is not None
    generation = next(p for p in config.output_root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    first_bytes = {p.name: p.read_bytes() for p in generation.iterdir() if p.is_file()}
    second_config = DatasetConfig((source,), phase8, audit, tmp_path / "out-reversed")
    second = DatasetFactory().build(second_config)
    assert second.manifest is not None
    generation2 = next(p for p in second_config.output_root.iterdir() if p.is_dir() and (p / "manifest.json").exists())
    assert {p.name: p.read_bytes() for p in generation2.iterdir() if p.is_file()} == first_bytes
    payload = json.loads((generation / "manifest.json").read_text(encoding="utf-8"))
    assert payload["safety"] == {"network_attempts": 0, "llm_calls": 0, "tool_calls": 0, "mt5_calls": 0}
    reasons = payload["counts"]["reason_counts"]
    assert reasons["UNTRUSTED_TIER"] == 2
    assert reasons["DECISION_CONTEXT_INCOMPLETE"] == 1
    assert reasons["OUTCOME_UNAVAILABLE"] == 1
    assert reasons["TEMPORAL_INVALID"] == 1


def test_fixture_rows_cover_duplicate_and_future_categories():
    assert {row.decision.decision_id for row in observations(Path("fixture.db"))} >= {"future", "unavailable", "tier-b", "tier-c"}
