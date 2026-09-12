import hashlib
import sqlite3
from datetime import datetime, timezone

import pytest

from tradingagents.forex.evidence_audit import (
    AuditWriteError,
    EvidenceAuditStore,
    EvidenceUsageAudit,
)


def audit(**overrides):
    context = "K1: evidence"
    values = {
        "decision_id": "D1", "source_run_id": "R1", "integration_status": "INJECTED", "bundle_status": "COMPLETE",
        "as_of": datetime(2026, 1, 1, tzinfo=timezone.utc), "rendered_context": context,
        "rendered_context_hash": hashlib.sha256(context.encode()).hexdigest(), "available_knowledge_ids": ("K1",),
        "evidence_refs_used": ("K1",), "diagnostics": {"code": "ok"}, "source_errors": {}, "retrieval_count": 1,
    }
    values.update(overrides)
    return EvidenceUsageAudit(**values)


def test_audit_schema_contains_required_metadata(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    with sqlite3.connect(tmp_path / "audit.sqlite3") as db:
        columns = {row[1] for row in db.execute("PRAGMA table_info(evidence_usage_audit)")}
    assert {"decision_id", "source_run_id", "rendered_context", "rendered_context_hash", "as_of",
            "integration_status", "bundle_status", "evidence_refs_used", "diagnostics"} <= columns


def test_audit_retains_exact_context_payload_and_hash(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    row = store.get("D1")
    assert row["rendered_context"] == "K1: evidence"
    assert row["rendered_context_hash"] == hashlib.sha256(b"K1: evidence").hexdigest()


def test_audit_rejects_prompt_completion_reasoning_and_credentials(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    with pytest.raises(ValueError):
        store.append(audit(diagnostics={"nested": {"prompt": "secret"}}))
    with pytest.raises(ValueError):
        store.append(audit(source_errors={"credential": "x"}))


def test_audit_is_append_only(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    other = "different"
    store.append(audit(rendered_context=other, rendered_context_hash=hashlib.sha256(other.encode()).hexdigest()))
    assert len(store.list_for_source_run("R1")) == 2


def test_audit_duplicate_append_is_idempotent(tmp_path):
    store = EvidenceAuditStore(tmp_path / "audit.sqlite3")
    store.initialize()
    store.append(audit())
    store.append(audit())
    assert len(store.list_for_source_run("R1")) == 1


def test_audit_write_failure_is_typed(tmp_path):
    path = tmp_path / "audit.sqlite3"
    path.write_text("not sqlite")
    with pytest.raises(AuditWriteError):
        EvidenceAuditStore(path).initialize()
