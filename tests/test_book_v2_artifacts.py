import hashlib
import importlib
import json

import pytest

from tests.test_book_natural_language import resolve, source_lookup
from tests.test_book_natural_language_sources import source_cases


def artifacts():
    return importlib.import_module("tradingagents.self_enhancement.book_v2_artifacts")


def publish(root):
    identity = {"generation_id": source_cases()[0]["evidence"]["generation_id"], "generation_fingerprint": "a" * 64, "population_hash": "sha256:" + "b" * 64}
    return artifacts().publish_v2(root, identity=identity, results=(resolve(source_cases()[0]),), source_lookup=source_lookup)


def test_v2_artifacts_reload_proofs_and_report_actual_gates(tmp_path):
    root = tmp_path / "v2"
    report = publish(root)
    assert report["supported_rule_count"] == 1
    assert report["nonexecutable_candidate_count"] == 1
    assert report["eligible_candidate_count"] == 0
    assert report["evaluation"]["status"] == "NOT_RUN"
    values = artifacts().load_v2_discovery_artifacts(root, source_lookup=source_lookup)
    assert values[0].status.value == "SUPPORTED_NONEXECUTABLE"
    with pytest.raises(FileExistsError):
        publish(root)


def rewrite_record(root, change):
    path = root / "normalized-rules.json"
    data = json.loads(path.read_text())
    change(data)
    payload = json.dumps(data, sort_keys=True).encode()
    path.write_bytes(payload)
    report_path = root / "phase14c-report.json"
    report = json.loads(report_path.read_text())
    report["rules_sha256"] = hashlib.sha256(payload).hexdigest()
    report_path.write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize("field,value", [("value", "11"), ("schema_version", "unknown"), ("stage", "ENTRY"), ("semantic_fingerprint", "c" * 64)])
def test_recomputed_artifact_hash_cannot_authorize_tampered_semantics(tmp_path, field, value):
    root = tmp_path / "v2"
    publish(root)
    rewrite_record(root, lambda data: data["results"][0]["rule"].update({field: value}))
    with pytest.raises(ValueError):
        artifacts().load_v2_discovery_artifacts(root, source_lookup=source_lookup)


def test_identity_generation_mismatch_rejects_even_with_rehashed_identity(tmp_path):
    root = tmp_path / "v2"
    publish(root)
    path = root / "book-v2-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["generation_id"] = "other-generation"
    manifest["identity_sha256"] = artifacts().identity_digest(manifest["identity"])
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        artifacts().load_v2_discovery_artifacts(root, source_lookup=source_lookup)


def test_v2_evaluation_cannot_use_claimed_eligible_artifact(tmp_path):
    root = tmp_path / "v2"
    publish(root)
    result = artifacts().evaluate_v2_artifacts(root, source_lookup=source_lookup)
    assert result["status"] == "NOT_RUN"
    assert result["order_send_calls"] == 0
    assert result["reason_code"] == "NO_ELIGIBLE_CANDIDATES"


def test_real_catalog_lookup_rejects_rehashed_wrong_manifest_fingerprint(tmp_path):
    from tests.test_book_natural_language_sources import CATALOG
    if not CATALOG.exists():
        pytest.skip("pinned Phase 7 catalog absent")
    lookup = artifacts().pinned_source_lookup(CATALOG.parent,
        generation_id=source_cases()[0]["evidence"]["generation_id"],
        population_hash="sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17",
        generation_fingerprint="428929b8060d96dd59be8e10ca338ded9534c12265a657aff5fad5cc97ffe3d1",
        catalog_sha256=hashlib.sha256(CATALOG.read_bytes()).hexdigest())
    identity = {"generation_id": source_cases()[0]["evidence"]["generation_id"], "generation_fingerprint": "0" * 64,
        "population_hash": "sha256:fd3ee0c846f2969747aca70576f58e55ffe1c07e8190570d2f9f7e2cb20b9e17"}
    with pytest.raises(ValueError, match="pin|fingerprint"):
        artifacts().publish_v2(tmp_path / "wrong-pin", identity=identity,
            results=(resolve(source_cases()[0]),), source_lookup=lookup)


def pinned_fixture(tmp_path):
    from tests.test_phase14c_inventory import _seed_catalog
    from tradingagents.knowledge.catalog import KnowledgeCatalog
    from tradingagents.self_enhancement.phase14c_inventory import _canonical_generation_fingerprint
    path = _seed_catalog(tmp_path / "knowledge", document_count=1)
    generation = KnowledgeCatalog(path).active_generation()
    identity = {"generation_id": generation.generation_id,
        "generation_fingerprint": _canonical_generation_fingerprint(generation),
        "population_hash": generation.population_hash}
    lookup = artifacts().pinned_source_lookup(path.parent, **identity, catalog_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return path, identity, lookup


@pytest.mark.parametrize("mutation", [
    "UPDATE knowledge_index_generations SET lexical_index_version='changed' WHERE is_active=1",
    "UPDATE knowledge_resources SET display_path='changed'",
])
def test_reload_rejects_catalog_drift_even_when_selected_chunks_unchanged(tmp_path, mutation):
    import sqlite3
    path, identity, lookup = pinned_fixture(tmp_path)
    root = tmp_path / "v2"
    artifacts().publish_v2(root, identity=identity, results=(), source_lookup=lookup)
    with sqlite3.connect(path) as con:
        before = con.execute("SELECT provenance_json FROM knowledge_chunks").fetchall()
        con.execute(mutation)
        assert con.execute("SELECT provenance_json FROM knowledge_chunks").fetchall() == before
    with pytest.raises((ValueError, RuntimeError), match="pin|fingerprint|generation"):
        artifacts().load_v2_discovery_artifacts(root, source_lookup=lookup)


def test_reload_rejects_rehashed_wrong_fingerprint_against_pinned_lookup(tmp_path):
    _, identity, lookup = pinned_fixture(tmp_path)
    root = tmp_path / "v2"
    artifacts().publish_v2(root, identity=identity, results=(), source_lookup=lookup)
    path = root / "book-v2-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["identity"]["generation_fingerprint"] = "0" * 64
    digest = artifacts().identity_digest(manifest["identity"])
    manifest["identity_sha256"] = digest
    path.write_text(json.dumps(manifest))
    report_path = root / "phase14c-report.json"
    report = json.loads(report_path.read_text())
    report["identity_sha256"] = digest
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="pin"):
        artifacts().load_v2_discovery_artifacts(root, source_lookup=lookup)


def test_actual_v2_runner_evaluation_checks_bound_source_location_and_pin(tmp_path):
    from tests.test_phase14c_integration import _paths
    from tradingagents.self_enhancement.phase14c_runner import (
        _fingerprint_sources,
        run_phase14c_evaluation,
    )
    _, identity, lookup = pinned_fixture(tmp_path)
    source_paths, _ = _paths(tmp_path)
    identity["source_fingerprints"] = _fingerprint_sources(source_paths)
    root = tmp_path / "v2"
    artifacts().publish_v2(root, identity=identity, results=(), source_lookup=lookup)
    location = root / "v2-source-location.json"
    location.write_text(json.dumps({"knowledge_root": lookup.pin["knowledge_root"]}))
    assert run_phase14c_evaluation(artifact_root=root, source_paths=source_paths, source_commit="fixture")["status"] == "NOT_RUN"
    location.write_text(json.dumps({"knowledge_root": str(tmp_path)}))
    with pytest.raises(ValueError, match="source location"):
        run_phase14c_evaluation(artifact_root=root, source_paths=source_paths, source_commit="fixture")
