from tradingagents.distillation.dedup import DedupIndex, canonical_example_id
from tradingagents.distillation.splits import GroupedSplitter, validate_no_group_leakage

def examples():
    return [{"document_id": f"d{i}", "chapter":"c", "section":f"s{i}", "text":f"lesson {i}"} for i in range(10)]

def test_dedup_rejects_exact_and_near_duplicate():
    idx=DedupIndex(near_threshold=.6)
    assert idx.check("Order flow imbalance measures pressure") is None
    assert idx.check("Order flow imbalance measures pressure") == "DUPLICATE"
    assert idx.check("Order flow imbalance measures pressure today") == "NEAR_DUPLICATE"

def test_canonical_id_is_stable():
    assert canonical_example_id({"b":2,"a":1}) == canonical_example_id({"a":1,"b":2})

def test_grouped_split_is_stable_and_has_no_leakage():
    rows=examples(); splitter=GroupedSplitter(); assignments=splitter.assign(rows)
    assert assignments == splitter.assign(list(reversed(rows)))
    assert validate_no_group_leakage(rows, assignments)
    assert set(assignments.values()) == {"train","validation","test"}

def test_grouped_split_reports_insufficient_data():
    try: GroupedSplitter().assign(examples()[:2])
    except ValueError as exc: assert str(exc) == "INSUFFICIENT_DATA"
    else: raise AssertionError("expected insufficient-data failure")
