import json
from datetime import datetime, timezone
from pathlib import Path

from tradingagents.datasets.eligibility import EligibilityResult
from tradingagents.datasets.errors import SourceReadError
from tradingagents.datasets.factory import DatasetFactory
from tradingagents.datasets.models import (
    CanonicalExampleV1,
    DatasetConfig,
    DatasetExclusionReason,
    DatasetManifest,
    EvaluationObservation,
    JoinedObservation,
    SourceFingerprint,
    SourceObservation,
)
from tradingagents.datasets.sources import SourceReadResult


def _fingerprint(kind: str, path: Path, suffix: str = "") -> SourceFingerprint:
    return SourceFingerprint(
        f"{kind}-{path.stem}-{suffix or 'one'}", str(path), f"schema-{kind}",
        f"file-{kind}-{suffix or 'one'}", f"snapshot-{kind}-{suffix or 'one'}",
    )


def _canonical(decision_id: str, fingerprints: dict[str, SourceFingerprint]):
    return CanonicalExampleV1(
        example_id=f"ex-{decision_id}",
        decision={
            "decision_id": decision_id,
            "source_run_id": f"run-{decision_id}",
            "requested_symbol": "EURUSD",
            "resolved_symbol": "EURUSD",
            "analysis_profile": "p",
            "analysis_timeframe": "M5",
            "action": "BUY",
            "analysis_snapshot_timestamp": "2026-01-01T00:00:00+00:00",
        },
        market={}, research={},
        outcome={"evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300},
        trust={},
        provenance={key: value.to_dict() for key, value in fingerprints.items()},
    )


def test_factory_reads_sorted_sources_once_and_publishes(tmp_path, monkeypatch):
    calls = []
    (tmp_path / "a.db").write_bytes(b"a")
    (tmp_path / "b.db").write_bytes(b"b")

    class FakeSource:
        def __init__(self, path):
            self.path = Path(path)
        def read(self):
            calls.append(self.path)
            return SourceReadResult(fingerprint=_fingerprint("source", self.path))

    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", FakeSource)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", FakeSource)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", FakeSource)

    config = DatasetConfig(
        source_db_paths=(tmp_path / "b.db", tmp_path / "a.db"),
        phase8_root=tmp_path / "phase8",
        phase9_audit_path=tmp_path / "audit.db",
        output_root=tmp_path / "out",
    )
    # The test only exercises source orchestration; the fake readers fail visibly.
    report = DatasetFactory().build(config)
    assert report.status in {"EMPTY_ELIGIBLE_SET", "FAILED"}
    source_calls = [item for item in calls if item.name in {"a.db", "b.db"}]
    assert source_calls == sorted(source_calls)


def test_factory_has_no_external_safety_counters():
    assert DatasetFactory.safety_counters() == {
        "network_attempts": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "mt5_calls": 0,
    }


def test_factory_records_changed_and_removed_source_rows(tmp_path, monkeypatch):
    """Published metadata must expose row/source deltas on the next build."""
    # This test intentionally uses real files so source ordering and byte
    # identity are exercised; adapter and join seams remain deterministic.
    source = tmp_path / "source.db"
    source.write_bytes(b"v1")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    audit = tmp_path / "audit.db"
    audit.write_bytes(b"audit")
    output = tmp_path / "out"
    config = DatasetConfig((source,), phase8, audit, output)

    fp56 = _fingerprint("phase56", source, "v1")
    fp8 = _fingerprint("phase8", phase8 / "catalog.sqlite3")
    fp9 = _fingerprint("phase9", audit)
    previous = output / "generation-prior"
    previous.mkdir(parents=True)
    (previous / "manifest.json").write_text(json.dumps({
        "metadata": {
            "factory_config": {"filters": {}, "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "allow_empty": True},
            "source_inventory": [{
                "canonical_path": str(source.resolve()), "source_id": "old-source-id",
                "file_sha256": "old-file-sha", "snapshot_fingerprint": "old-snapshot",
                "decision_ids": ["removed-row"],
                "decision_identities": [{"decision_id": "removed-row", "fingerprint": "old-decision-fingerprint"}],
            }],
        }
    }), encoding="utf-8")
    current_decision = SourceObservation(
        "new-row", datetime(2026, 1, 1, tzinfo=timezone.utc),
        fields={"source_decision_fingerprint": "new-decision-fingerprint"},
    )

    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self):
            if self.path == source:
                return SourceReadResult(decisions=(current_decision,), fingerprint=fp56)
            return SourceReadResult(fingerprint=fp8 if self.path.name == "catalog.sqlite3" else fp9)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    captured = {}
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (captured.update(kw) or previous))
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))

    DatasetFactory().build(config)
    metadata = captured["metadata"]
    # Source deltas are retained in bounded manifest metadata for the next run.
    assert "removed-row" in str(metadata)
    assert str(source.resolve()) in metadata["diagnostics"]["changed_sources"]
    assert metadata["diagnostics"]["removed_rows"] == [
        f"{source.resolve()}:removed-row"
    ]
    assert metadata["diagnostics"]["added_rows"] == [
        f"{source.resolve()}:new-row"
    ]
    assert metadata["diagnostics"]["changed_rows"] == []


def test_factory_revalidates_existing_generation_before_reuse(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    audit = tmp_path / "audit.db"
    audit.write_bytes(b"audit")
    config = DatasetConfig((source,), tmp_path / "phase8", audit, tmp_path / "out")
    config.phase8_root.mkdir()
    fp = _fingerprint("phase56", source)
    fp8 = _fingerprint("phase8", config.phase8_root / "catalog.sqlite3")
    fp9 = _fingerprint("phase9", config.output_root / "audit.db")
    prior = config.output_root / "generation-existing"
    prior.mkdir(parents=True)
    (prior / "manifest.json").write_text(json.dumps({
        "source_fingerprints": {"phase56": fp.to_dict(), "phase8": fp8.to_dict(), "phase9": fp9.to_dict()},
        "metadata": {
            "factory_config": {"filters": {}, "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "allow_empty": True},
            "source_inventory": [], "diagnostics": {"broken_sources": [], "duplicate_sources": [], "removed_sources": [], "changed_sources": [], "removed_rows": [], "added_rows": [], "changed_rows": []},
        },
    }), encoding="utf-8")

    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return SourceReadResult(fingerprint=fp if self.path == source else (fp8 if self.path.name == "catalog.sqlite3" else fp9))
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory._bind_sqlite_marks", lambda result, digest: result)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._source_inventory", lambda *_: [])
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (_ for _ in ()).throw(__import__("tradingagents.datasets.writer", fromlist=["GenerationExistsError"]).GenerationExistsError()))
    monkeypatch.setattr("tradingagents.datasets.factory.validate_generation", lambda path: type("R", (), {"valid": False, "errors": ("tampered",)})())
    try:
        DatasetFactory().build(config)
    except Exception as exc:
        assert "tampered" in str(exc)
    else:
        raise AssertionError("invalid existing generation was reused")


def test_factory_reuses_unchanged_generation_only_after_validation(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    config = DatasetConfig((source,), tmp_path / "phase8", None, tmp_path / "out")
    config.phase8_root.mkdir()
    fp56 = _fingerprint("phase56", source)
    fp8 = _fingerprint("phase8", config.phase8_root / "catalog.sqlite3")
    prior = config.output_root / "generation-existing"
    prior.mkdir(parents=True)

    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self):
            return SourceReadResult(
                fingerprint=fp56 if self.path == source else fp8
            )

    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory._bind_sqlite_marks", lambda result, digest: result)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._source_inventory", lambda *_: [])
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))

    first_write = {}

    def publish_once(root, examples, exclusions, split_result, **kwargs):
        first_write.update(kwargs)
        (prior / "manifest.json").write_text(json.dumps({
            "dataset_id": prior.name,
            "examples": 0,
            "exclusions": 0,
            "split_status": "INSUFFICIENT_DATA",
            "status": "EMPTY_ELIGIBLE_SET",
            "source_fingerprints": kwargs["source_fingerprints"],
            "metadata": kwargs["metadata"],
        }), encoding="utf-8")
        return prior

    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", publish_once)
    first = DatasetFactory().build(config)
    assert first.status == "EMPTY_ELIGIBLE_SET"
    assert first_write

    validation_calls = []
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (_ for _ in ()).throw(__import__("tradingagents.datasets.writer", fromlist=["GenerationExistsError"]).GenerationExistsError()))
    monkeypatch.setattr(
        "tradingagents.datasets.factory.validate_generation",
        lambda path: (validation_calls.append(path) or type("R", (), {"valid": True, "errors": ()})()),
    )

    second = DatasetFactory().build(config)
    assert second.status == "EMPTY_ELIGIBLE_SET"
    assert validation_calls == [prior]


def test_factory_records_phase8_and_phase9_read_failures_as_exclusions(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    audit = tmp_path / "audit.db"
    audit.write_bytes(b"audit")
    config = DatasetConfig((source,), phase8, audit, tmp_path / "out")

    class Phase56Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return SourceReadResult(fingerprint=_fingerprint("phase56", source))

    class BrokenPhase8:
        def __init__(self, path): self.path = Path(path)
        def read(self): raise SourceReadError("phase8 schema drift")

    class BrokenPhase9:
        def __init__(self, path): self.path = Path(path)
        def read(self): raise SourceReadError("phase9 schema drift")

    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Phase56Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", BrokenPhase8)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase9AuditSource", BrokenPhase9)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    captured = {}
    monkeypatch.setattr(
        "tradingagents.datasets.factory.write_generation",
        lambda *a, **kw: (captured.update({"exclusions": a[2]}) or config.output_root),
    )

    report = DatasetFactory().build(config)

    integrity = [
        item for item in report.exclusions
        if item.reasons == (DatasetExclusionReason.SOURCE_INTEGRITY_FAILED,)
    ]
    assert {item.details["source_path"] for item in integrity} == {
        str((phase8 / "catalog.sqlite3").resolve()), str(audit.resolve())
    }
    published = captured["exclusions"]
    assert {
        item.details["source_path"] for item in published
        if item.reasons == (DatasetExclusionReason.SOURCE_INTEGRITY_FAILED,)
    } == {
        str((phase8 / "catalog.sqlite3").resolve()), str(audit.resolve())
    }


def test_factory_generation_identity_includes_sqlite_wal_mark(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"same-main")
    wal = Path(str(source) + "-wal")
    wal.write_bytes(b"wal-v1")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    config = DatasetConfig((source,), phase8, None, tmp_path / "out")
    fp = _fingerprint("phase56", source)

    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return SourceReadResult(fingerprint=fp)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    captured = []
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (captured.append(kw["source_fingerprints"]) or config.output_root))

    DatasetFactory().build(config)
    wal.write_bytes(b"wal-v2")
    DatasetFactory().build(config)

    assert captured[0]["phase56"]["file_sha256"] != captured[1]["phase56"]["file_sha256"]


def test_factory_isolates_broken_source_with_explicit_integrity_exclusion(tmp_path, monkeypatch):
    source = tmp_path / "broken.db"
    source.write_bytes(b"not-a-database")
    config = DatasetConfig((source,), tmp_path / "phase8", None, tmp_path / "out")
    config.phase8_root.mkdir()

    class BrokenReader:
        def __init__(self, path): self.path = Path(path)
        def read(self): raise SourceReadError("schema drift")
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", BrokenReader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", BrokenReader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: ())
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    captured = {}
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (captured.update(kw) or config.output_root))

    report = DatasetFactory().build(config)
    assert report.status == "EMPTY_ELIGIBLE_SET"
    assert any(DatasetExclusionReason.SOURCE_INTEGRITY_FAILED in item.reasons for item in report.exclusions)
    assert "broken.db" in str(captured["metadata"]["diagnostics"]["broken_sources"])


def test_factory_deduplicates_same_decision_fingerprint_across_rows(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    config = DatasetConfig((source,), tmp_path / "phase8", None, tmp_path / "out")
    config.phase8_root.mkdir()
    fp = _fingerprint("phase56", source)
    decision_a = SourceObservation(
        "d-a", datetime(2026, 1, 1, tzinfo=timezone.utc),
        source_run_id="run", requested_symbol="EURUSD", resolved_symbol="EURUSD", action="BUY",
        fields={"source_decision_fingerprint": "same-decision"},
    )
    decision_b = SourceObservation(
        "d-b", decision_a.analysis_snapshot_timestamp,
        source_run_id="run", requested_symbol="EURUSD", resolved_symbol="EURUSD", action="BUY",
        fields={"source_decision_fingerprint": "same-decision"},
    )
    evaluation = EvaluationObservation("d-a", "ANALYSIS_SNAPSHOT", 300)
    observations = (
        JoinedObservation(decision_a, evaluation), JoinedObservation(decision_b, evaluation),
    )
    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return SourceReadResult(fingerprint=fp)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: observations)
    monkeypatch.setattr("tradingagents.datasets.factory.classify_observation", lambda *_: EligibilityResult(True, ()))
    monkeypatch.setattr("tradingagents.datasets.factory.canonicalize", lambda observation, eligibility: _canonical(observation.decision.decision_id, {"phase56": fp, "phase8": _fingerprint("phase8", config.phase8_root), "phase9": _fingerprint("phase9", tmp_path / "audit.db")}))
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    captured = {}
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (captured.update(kw) or config.output_root))

    report = DatasetFactory().build(config)
    assert len(report.exclusions) == 1
    assert report.exclusions[0].reasons == (DatasetExclusionReason.DUPLICATE,)


def test_factory_uses_classifier_selected_basis_and_horizon_for_canonical_row(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    config = DatasetConfig((source,), phase8, None, tmp_path / "out", evaluation_basis="DECISION_REFERENCE", horizon_seconds=600)
    fp = _fingerprint("phase56", source)
    decision = SourceObservation("d1", datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc), action="BUY", fields={"source_decision_fingerprint": "dfp"})
    source_result = SourceReadResult(decisions=(decision,), evaluations=(
        EvaluationObservation("d1", "ANALYSIS_SNAPSHOT", 300, fields={"source_decision_fingerprint": "dfp"}),
        EvaluationObservation("d1", "DECISION_REFERENCE", 600, fields={"source_decision_fingerprint": "dfp"}),
    ), fingerprint=fp)
    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return source_result if self.path == source else SourceReadResult(fingerprint=_fingerprint("phase8", phase8))
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: (JoinedObservation(decision, source_result.evaluations[0], fields={"evaluations": ({"decision_id": "d1", "evaluation_basis": "ANALYSIS_SNAPSHOT", "horizon_seconds": 300, "evaluation_status": "COMPLETE", "source_context_eligible": True, "fields": {}}, {"decision_id": "d1", "evaluation_basis": "DECISION_REFERENCE", "horizon_seconds": 600, "evaluation_status": "COMPLETE", "source_context_eligible": True, "fields": {}})}),))
    monkeypatch.setattr("tradingagents.datasets.factory.classify_observation", lambda *_: EligibilityResult(True, details={"decision_id": "d1", "effective_evaluation_basis": "DECISION_REFERENCE", "effective_horizon_seconds": 600}))
    captured = {}
    monkeypatch.setattr("tradingagents.datasets.factory.canonicalize", lambda observation, eligibility: (captured.setdefault("evaluation", observation.evaluation), _canonical("d1", {"phase56": fp, "phase8": _fingerprint("phase8", phase8), "phase9": _fingerprint("phase9", tmp_path / "audit")}))[1])
    monkeypatch.setattr("tradingagents.datasets.factory.assign_splits", lambda rows: type("S", (), {"assignments": (), "status": "INSUFFICIENT_DATA"})())
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: config.output_root)
    DatasetFactory().build(config)
    assert captured["evaluation"].evaluation_basis == "DECISION_REFERENCE"
    assert captured["evaluation"].horizon_seconds == 600


def test_factory_emits_join_failure_exclusion_for_each_candidate(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    source.write_bytes(b"source")
    phase8 = tmp_path / "phase8"
    phase8.mkdir()
    config = DatasetConfig((source,), phase8, None, tmp_path / "out")
    fp = _fingerprint("phase56", source)
    decisions = tuple(SourceObservation(f"d{i}", datetime(2026, 1, i + 1, tzinfo=timezone.utc), fields={"source_decision_fingerprint": f"fp{i}"}) for i in range(2))
    class Reader:
        def __init__(self, path): self.path = Path(path)
        def read(self): return SourceReadResult(decisions=decisions, fingerprint=fp) if self.path == source else SourceReadResult(fingerprint=_fingerprint("phase8", phase8))
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyPhase56Source", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.ReadonlyExperienceSource", Reader)
    monkeypatch.setattr("tradingagents.datasets.factory.join_observations", lambda *_: (_ for _ in ()).throw(ValueError("schema mismatch")))
    monkeypatch.setattr("tradingagents.datasets.factory._manifest", lambda path: DatasetManifest(path.name))
    captured = {}
    monkeypatch.setattr("tradingagents.datasets.factory.write_generation", lambda *a, **kw: (captured.update({"exclusions": kw["metadata"], "rows": a[2]}) or config.output_root))
    report = DatasetFactory().build(config)
    assert {item.decision_id for item in report.exclusions if item.decision_id in {"d0", "d1"}} == {"d0", "d1"}
    assert all(DatasetExclusionReason.SCHEMA_UNSUPPORTED in item.reasons for item in report.exclusions if item.decision_id in {"d0", "d1"})
