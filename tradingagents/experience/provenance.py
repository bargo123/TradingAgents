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
    if not isinstance(evaluation_fingerprint, str) or not evaluation_fingerprint.strip():
        raise ProvenanceViolationError("evaluation fingerprint must be non-empty text")
    result = {"evaluation_fingerprint": evaluation_fingerprint}
    if training_eligible is not None:
        if not isinstance(training_eligible, bool):
            raise ProvenanceViolationError("training_eligible must be boolean")
        result["training_eligible"] = training_eligible
    if training_eligibility_reason is not None:
        if (
            not isinstance(training_eligibility_reason, str)
            or not training_eligibility_reason.strip()
            or len(training_eligibility_reason) > 256
        ):
            raise ProvenanceViolationError(
                "training eligibility reason must be non-empty text"
            )
        result["training_eligibility_reason"] = training_eligibility_reason
    result.update(fields)
    return result


def validate_evaluation_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProvenanceViolationError("provenance must be a mapping")
    result = dict(value)
    if "evaluation_fingerprint" in result and (
        not isinstance(result["evaluation_fingerprint"], str)
        or not result["evaluation_fingerprint"].strip()
    ):
        raise ProvenanceViolationError("evaluation fingerprint must be non-empty text")
    if "training_eligible" in result and not isinstance(result["training_eligible"], bool):
        raise ProvenanceViolationError("training_eligible must be boolean")
    if "training_eligibility_reason" in result and (
        not isinstance(result["training_eligibility_reason"], str)
        or not result["training_eligibility_reason"].strip()
        or len(result["training_eligibility_reason"]) > 256
    ):
        raise ProvenanceViolationError(
            "training eligibility reason must be non-empty text"
        )
    # The catalog supplies the row fingerprint when callers pass only the
    # optional provenance fields.
    return result


def provenance_fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "build_evaluation_provenance",
    "validate_evaluation_provenance",
    "provenance_fingerprint",
]
