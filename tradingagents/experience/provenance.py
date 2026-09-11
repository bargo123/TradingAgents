"""Small, deterministic provenance helpers for Phase 8 evidence."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .errors import ProvenanceViolationError


def build_evaluation_provenance(
    evaluation_fingerprint: str,
    *,
    training_eligible: bool | None = None,
    training_eligibility_reason: str | None = None,
    **fields: Any,
) -> dict[str, Any]:
    if not evaluation_fingerprint:
        raise ProvenanceViolationError("evaluation fingerprint is required")
    result = {"evaluation_fingerprint": str(evaluation_fingerprint)}
    if training_eligible is not None:
        if not isinstance(training_eligible, bool):
            raise ProvenanceViolationError("training_eligible must be boolean")
        result["training_eligible"] = training_eligible
    if training_eligibility_reason is not None:
        result["training_eligibility_reason"] = str(training_eligibility_reason)[:256]
    result.update(fields)
    return result


def validate_evaluation_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceViolationError("provenance must be a mapping")
    result = dict(value)
    if "training_eligible" in result and not isinstance(result["training_eligible"], bool):
        raise ProvenanceViolationError("training_eligible must be boolean")
    # The catalog supplies the row fingerprint when callers pass only the
    # optional provenance fields.
    return result


def provenance_fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["build_evaluation_provenance", "validate_evaluation_provenance", "provenance_fingerprint"]
