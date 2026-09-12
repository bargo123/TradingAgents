from __future__ import annotations

import pytest

from tradingagents.experience.catalog import ExperienceCatalog
from tradingagents.experience.errors import ProvenanceViolationError
from tradingagents.experience.provenance import (
    build_evaluation_provenance,
    provenance_fingerprint,
    validate_evaluation_provenance,
)


def test_training_eligibility_is_preserved_as_provenance_only() -> None:
    provenance = build_evaluation_provenance(
        "eval-fp", training_eligible=False, training_eligibility_reason="missing_context"
    )
    assert provenance["evaluation_fingerprint"] == "eval-fp"
    assert provenance["training_eligible"] is False
    assert provenance["training_eligibility_reason"] == "missing_context"
    assert provenance_fingerprint(provenance)


def test_invalid_evaluation_provenance_is_rejected() -> None:
    with pytest.raises(ProvenanceViolationError):
        validate_evaluation_provenance({"training_eligible": "yes"})


def test_catalog_snapshot_retains_provenance_and_status(tmp_path) -> None:
    catalog = ExperienceCatalog(tmp_path / "experience")
    catalog.append_evaluation_snapshot(
        "exp1",
        {"evaluation_status": "COMPLETE", "net_points": 2.5},
        "eval-fp",
        provenance={"training_eligible": True, "training_eligibility_reason": "eligible"},
    )
    snapshot = catalog.evaluation_snapshots("exp1")[0]
    assert snapshot["evaluation_status"] == "COMPLETE"
    assert snapshot["provenance"]["training_eligible"] is True
    assert snapshot["fingerprint"] == "eval-fp"


def test_catalog_rejects_mismatched_provenance_fingerprint(tmp_path) -> None:
    catalog = ExperienceCatalog(tmp_path / "experience")
    with pytest.raises(ProvenanceViolationError):
        catalog.append_evaluation_snapshot(
            "exp1",
            {"evaluation_status": "COMPLETE"},
            "eval-fp",
            provenance={"evaluation_fingerprint": "different-fp"},
        )
