from dataclasses import replace
import json

import pytest

from tests.test_phase14b_book_atomic_extraction import _Transport
from tests.test_book_natural_language_sources import source_cases
from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_atomic_extraction import AtomicStrategyExtractor, EvidenceSentence


def sentence(case=None):
    case = case or source_cases()[0]
    e = case["evidence"]
    text = " " * e["start_offset"] + e["quote"]
    hit = KnowledgeHit(chunk_id=e["chunk_id"], document_id=e["document_id"], source_hash=e["source_hash"], text=text, source_filename="fixture", title="fixture", extra={"projection_generation": e["generation_id"], "projection_population_hash": "sha256:" + "b" * 64})
    return EvidenceSentence("fixture-" + case["case_id"], hit, 0, 0, e["start_offset"], e["end_offset"], e["quote"], 0, 0)


def extractor(transport, cache=None, version="digest"):
    return AtomicStrategyExtractor("http://127.0.0.1:11434", "qwen3.5:2b", timeout_seconds=300, max_output_tokens=512, context_tokens=8192, model_version=version, transport=transport, cache_path=cache)


def selection(user, n):
    return {"rules": [{"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [0]}]}


def test_v2_local_proposal_normalizes_source_and_reuses_selection_cache(tmp_path):
    transport = _Transport(selection)
    worker = extractor(transport, tmp_path / "cache.db")
    first = worker.extract_actionable_v2((sentence(),))
    assert first[0].rule.value == 10
    assert first[0].status.value == "SUPPORTED_NONEXECUTABLE"
    assert worker.extract_actionable_v2((sentence(),)) == first
    assert len(transport.calls) == 1
    assert json.loads(transport.calls[0]["body"]["messages"][-1]["content"])["stage"] == "NATURAL_RULE_EXTRACTION"
    assert transport.calls[0]["body"]["options"] == {"temperature": 0, "num_predict": 256, "num_ctx": 8192}
    changed = extractor(transport, tmp_path / "cache.db", version="other-digest")
    changed.extract_actionable_v2((sentence(),))
    assert len(transport.calls) == 2


@pytest.mark.parametrize("rule", [
    {"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [0], "value": 10},
    {"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [9]},
    {"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [0, 0]},
    {"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [True]},
])
def test_untrusted_fields_and_indexes_are_observable_rejections(rule):
    results = extractor(_Transport(lambda u, n: {"rules": [rule]})).extract_actionable_v2((sentence(),))
    assert results and results[0].status.value == "REJECTED"
    assert results[0].rule is None


def test_changed_quote_same_id_does_not_reuse_cached_meaning(tmp_path):
    transport = _Transport(selection)
    worker = extractor(transport, tmp_path / "cache.db")
    worker.extract_actionable_v2((sentence(),))
    original = sentence()
    altered = replace(original, hit=replace(original.hit, text=original.hit.text.replace("10 pip", "11 pip")), text=original.text.replace("10 pip", "11 pip"))
    assert worker.extract_actionable_v2((altered,))[0].rule.value == 11
    assert len(transport.calls) == 2


def test_v2_does_not_send_to_nonloopback_provider():
    worker = extractor(_Transport(selection))
    worker.endpoint = "http://example.com"
    with pytest.raises(ValueError):
        worker.extract_actionable_v2((sentence(),))


def test_cross_chunk_selection_rejects():
    worker = extractor(_Transport(lambda u, n: {"rules": [{"family": "OTHER_SUPPORTED", "stage": "EXPECTED_MOVE", "evidence_indices": [0, 1]}]}))
    # Each chunk is sent separately; index 1 is unavailable in both requests.
    result = worker.extract_actionable_v2((sentence(), sentence(source_cases()[1])))
    assert all(r.status.value == "REJECTED" for r in result)
