"""Stable source-row and source-snapshot identities."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

_SCHEMA = "phase5-6-source"
_VERSION = "phase8.source-fingerprint.v1"


def _fingerprint(kind: str, value: Mapping[str, Any]) -> str:
    payload = {
        "source_schema": _SCHEMA,
        "source_schema_version": _VERSION,
        "fingerprint_kind": kind,
        "value": value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def source_decision_fingerprint(row: Mapping[str, Any]) -> str:
    return _fingerprint("decision", row)


def source_evaluation_fingerprint(row: Mapping[str, Any]) -> str:
    return _fingerprint("evaluation", row)


def source_snapshot_fingerprint(metadata: Mapping[str, Any]) -> str:
    return _fingerprint("snapshot", metadata)


__all__ = ["source_decision_fingerprint", "source_evaluation_fingerprint", "source_snapshot_fingerprint"]
