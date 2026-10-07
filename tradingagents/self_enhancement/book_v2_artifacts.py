"""Versioned offline artifacts and pinned read-only evidence reload."""
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from .book_natural_language import verify_rule
from .book_normalization_models import (
    FEATURE_CONTRACT_VERSION,
    GRAMMAR_VERSION,
    SCHEMA_VERSION,
    NormalizationResult,
    NormalizationStatus,
)
from .book_v2_pipeline import assemble_v2


def _bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def identity_digest(identity):
    return hashlib.sha256(_bytes(identity)).hexdigest()


def pinned_source_lookup(knowledge_root, *, generation_id, population_hash):
    catalog = Path(knowledge_root).resolve(strict=True) / "catalog.sqlite3"
    def lookup(span):
        if span.generation_id != generation_id:
            raise ValueError("unpinned generation")
        with sqlite3.connect(catalog.as_uri() + "?mode=ro", uri=True) as con:
            con.execute("PRAGMA query_only=ON")
            row = con.execute(
                "SELECT document_id,source_hash,projection_generation,projection_population_hash,provenance_json "
                "FROM knowledge_chunks WHERE chunk_id=? AND active=1", (span.chunk_id,),
            ).fetchone()
        if row is None or row[:4] != (span.document_id, span.source_hash, generation_id, population_hash):
            raise ValueError("pinned evidence identity mismatch")
        payload = json.loads(row[4])
        return payload["text"]
    return lookup


def _unique(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError("duplicate JSON key")
        values[key] = value
    return values


def _read(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 4_000_000:
        raise ValueError("invalid or oversized artifact")
    return json.loads(path.read_bytes(), object_pairs_hook=_unique)


def publish_v2(root, *, identity, results, source_lookup, telemetry=(), coverage=None, existing_run=False):
    from .phase14c_runner import _safe_telemetry_record
    safe_telemetry = [_safe_telemetry_record(item) for item in telemetry]
    root = Path(root).resolve()
    if existing_run:
        run = _read(root / "run-manifest.json")
        if run.get("identity") != identity or run.get("status") != "IN_PROGRESS" or identity.get("assembly_schema_version") != SCHEMA_VERSION:
            raise ValueError("not an owned V2 run directory")
    else:
        root.mkdir(parents=False, exist_ok=False)
    # Exclusive publication: no prior artifact is ever replaced.
    for name in ("book-v2-manifest.json", "normalized-rules.json", "phase14c-report.json"):
        if (root / name).exists():
            raise FileExistsError("V2 artifact already exists")
    checked = []
    for result in results:
        if not isinstance(result, NormalizationResult):
            raise ValueError("typed normalization results required")
        if result.rule is not None:
            actual = verify_rule(result.rule, source_lookup=source_lookup)
            if actual != result:
                raise ValueError("unverified result or claimed status")
            if any(s.generation_id != identity["generation_id"] for s in result.rule.evidence.spans):
                raise ValueError("evidence generation differs from run identity")
        checked.append(result)
    candidates = assemble_v2(checked, source_lookup=source_lookup)
    payload = {"schema_version": SCHEMA_VERSION, "results": [r.to_dict() for r in checked]}
    encoded = _bytes(payload)
    if len(encoded) > 4_000_000:
        raise ValueError("normalization artifact exceeds bounded reader size")
    reasons = Counter(code for r in checked for code in r.reason_codes)
    report = {
        "schema_version": "phase14c-normalization-report.v2",
        "status": "INCOMPLETE", "outcome": "NO_ELIGIBLE_CANDIDATES",
        "artifact_integrity": "VALID", "generation_id": identity["generation_id"],
        "generation_fingerprint": identity["generation_fingerprint"],
        "population_hash": identity["population_hash"],
        "normalization_schema": SCHEMA_VERSION,
        "supported_rule_count": sum(r.rule is not None and r.status is not NormalizationStatus.REJECTED for r in checked),
        "rejected_rule_count": sum(r.status is NormalizationStatus.REJECTED for r in checked),
        "nonexecutable_candidate_count": sum(c.status is NormalizationStatus.SUPPORTED_NONEXECUTABLE for c in candidates),
        "eligible_candidate_count": sum(c.status is NormalizationStatus.EXECUTABLE_ELIGIBLE for c in candidates),
        "reason_code_counts": dict(sorted(reasons.items())),
        "candidate_reason_code_counts": dict(sorted(Counter(code for c in candidates for code in c.reason_codes).items())),
        "missing_stages": {c.envelope.strategy_id: [s.value for s in c.missing_stages] for c in candidates},
        "rules_sha256": hashlib.sha256(encoded).hexdigest(),
        "identity_sha256": identity_digest(identity),
        "coverage": coverage or {}, "telemetry": safe_telemetry,
        "llm_calls": sum(not item["cache_hit"] for item in safe_telemetry),
        "cache_hits": sum(item["cache_hit"] for item in safe_telemetry),
        "call_outcomes": dict(sorted(Counter(item["outcome"] for item in safe_telemetry).items())),
        "evaluation": {"status": "NOT_RUN", "reason_code": "NO_ELIGIBLE_CANDIDATES"},
        "real_money": False,
    }
    manifest = {"schema_version": SCHEMA_VERSION, "grammar_version": GRAMMAR_VERSION,
                "feature_contract_version": FEATURE_CONTRACT_VERSION,
                "identity": identity, "identity_sha256": identity_digest(identity)}
    for name, data in (("book-v2-manifest.json", _bytes(manifest)), ("normalized-rules.json", encoded), ("phase14c-report.json", _bytes(report))):
        with (root / name).open("xb") as stream:
            stream.write(data)
    return report


def load_v2_discovery_artifacts(artifact_root, *, source_lookup):
    root = Path(artifact_root).resolve(strict=True)
    manifest = _read(root / "book-v2-manifest.json")
    report = _read(root / "phase14c-report.json")
    if (manifest.get("schema_version"), manifest.get("grammar_version"), manifest.get("feature_contract_version")) != (SCHEMA_VERSION, GRAMMAR_VERSION, FEATURE_CONTRACT_VERSION):
        raise ValueError("unsupported V2 artifact version")
    identity = manifest["identity"]
    digest = identity_digest(identity)
    if manifest["identity_sha256"] != digest or report["identity_sha256"] != digest:
        raise ValueError("V2 run identity mismatch")
    path = root / "normalized-rules.json"
    payload = _read(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["rules_sha256"] or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("V2 artifact integrity mismatch")
    if set(payload) != {"schema_version", "results"} or type(payload["results"]) is not list:
        raise ValueError("invalid normalized result schema")
    results = tuple(NormalizationResult.from_dict(item) for item in payload["results"])
    for r in results:
        if r.rule is not None:
            if any(s.generation_id != identity["generation_id"] for s in r.rule.evidence.spans):
                raise ValueError("V2 source generation mismatch")
            if verify_rule(r.rule, source_lookup=source_lookup) != r:
                raise ValueError("V2 rule proof mismatch")
    return assemble_v2(results, source_lookup=source_lookup)


def evaluate_v2_artifacts(root, *, source_lookup):
    from .book_v2_registry import create_research_strategy
    candidates = load_v2_discovery_artifacts(root, source_lookup=source_lookup)
    eligible = [c for c in candidates if c.status is NormalizationStatus.EXECUTABLE_ELIGIBLE]
    for c in eligible:
        create_research_strategy(c, source_lookup=source_lookup)
    if eligible:
        # No current reviewed V2 primitive can reach this path. Never substitute
        # an uncosted or unordered evaluator when a future grammar is extended.
        raise ValueError("V2 replay adapter needs a reviewed primitive contract")
    return {"status": "NOT_RUN", "reason_code": "NO_ELIGIBLE_CANDIDATES",
            "candidate_count": len(candidates), "order_send_calls": 0,
            "gate_reasons": sorted({r for c in candidates for r in c.reason_codes}), "real_money": False}
