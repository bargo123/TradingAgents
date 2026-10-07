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
