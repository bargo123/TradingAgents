from tradingagents.experience.identity import (
    source_decision_fingerprint, source_evaluation_fingerprint, source_snapshot_fingerprint,
)


def test_source_fingerprint_is_key_order_independent() -> None:
    assert source_decision_fingerprint({"b": 2, "a": 1}) == source_decision_fingerprint({"a": 1, "b": 2})


def test_fingerprints_are_distinct_by_explicit_source_contract() -> None:
    row = {"decision_id": "d1", "evaluation_basis": "A"}
    assert source_decision_fingerprint(row) != source_evaluation_fingerprint(row)
    assert source_snapshot_fingerprint(row) != source_decision_fingerprint(row)
