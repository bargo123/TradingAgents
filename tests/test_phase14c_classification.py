from __future__ import annotations

import json

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_atomic_extraction import (
    AtomicStrategyExtractor,
    EvidenceGroup,
    PresenceStatus,
    prepare_evidence_sentences,
)

GENERATION_ID = "gen-phase14c-fixture"
POPULATION_HASH = "sha256:" + "c" * 64


class _Response:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


class _Transport:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def post(self, url, *, json, timeout):
        prompt = jsonlib.loads(json["messages"][-1]["content"])
        self.calls.append({"url": url, "body": json, "timeout": timeout, "stage": prompt["stage"]})
        result = self.respond(prompt, len(self.calls))
        if result == "MALFORMED":
            content = "not a structured response"
        elif isinstance(result, str):
            content = result
        else:
            content = jsonlib.dumps(result, sort_keys=True, separators=(",", ":"))
        return _Response(
            {
                "message": {"role": "assistant", "content": content},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 32,
                "eval_count": 8,
            }
        )


jsonlib = json


def _hit(text: str, chunk_id: str = "chunk-1") -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_hash=("a" if chunk_id.endswith("1") else "b") * 64,
        text=text,
        source_filename="fixture.pdf",
        source_relative_path="fixture.pdf",
        title="Fixture evidence",
        page=1,
        parser_version="parser-v1",
        chunker_version="chunker-v1",
        index_version="index-v1",
        extra={
            "projection_generation": GENERATION_ID,
            "projection_population_hash": POPULATION_HASH,
        },
    )


def _group(family_id: str, group_id: str, text: str) -> EvidenceGroup:
    sentence = prepare_evidence_sentences((_hit(text),))[0]
    return EvidenceGroup(family_id, group_id, (sentence,))


def _extract_response(prompt, _call_number):
    evidence = prompt["evidence"]
    indexes = [item["index"] for item in evidence]
    if prompt["stage"] == "CONCEPT_GROUPING":
        return {"candidates": [{"family": "MOMENTUM_CONTINUATION", "evidence_indices": indexes[:1]}]}
    if prompt["stage"] == "ATOMIC_RULE_EXTRACTION":
        return {"rules": [{"stage": "ENTRY", "evidence_index": indexes[0]}]}
    raise AssertionError("actionable-only extraction must not classify presence")


def test_evidence_group_accepts_the_closed_strategy_family_code():
    group = _group(
        "MOMENTUM_CONTINUATION",
        "group-momentum",
        "LONG when momentum > 1.2 points after confirmation.",
    )

    assert group.family_id == "MOMENTUM_CONTINUATION"


def test_presence_classification_returns_one_ordered_safe_result_per_group():
    def respond(prompt, _call_number):
        marker = prompt["evidence"][0]["text"]
        if marker.startswith("NONE"):
            return {"decision": "NONE"}
        if marker.startswith("AMBIGUOUS"):
            return {"decision": "AMBIGUOUS"}
        if marker.startswith("BROKEN"):
            return "MALFORMED"
        return {"decision": "ACTIONABLE"}

    groups = (
        _group("momentum", "group-none", "NONE source sentence."),
        _group("breakout", "group-ambiguous", "AMBIGUOUS source sentence."),
        _group("range_rejection", "group-broken", "BROKEN source sentence."),
    )
    transport = _Transport(respond)
    report = AtomicStrategyExtractor(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        timeout_seconds=30,
        max_output_tokens=512,
        context_tokens=16384,
        transport=transport,
    ).classify_presence(groups)

    assert [result.group for result in report.results] == list(groups)
    assert [result.status for result in report.results] == [
        PresenceStatus.NONE,
        PresenceStatus.AMBIGUOUS,
        None,
    ]
    assert [result.failure_code for result in report.results] == [None, None, "SCHEMA_INVALID"]
    assert all(result.generated_text is None for result in report.results)
    assert [call["stage"] for call in transport.calls] == ["PRESENCE"] * 3
    assert report.llm_calls == 3
    assert report.schema_failure_count == 1
    assert "not a structured response" not in repr(report)


def test_actionable_group_can_be_extracted_without_repeating_presence():
    sentence = _group(
        "momentum",
        "actionable-group",
        "LONG when momentum > 1.2 points after confirmation.",
    ).sentences[0]
    transport = _Transport(_extract_response)
    extractor = AtomicStrategyExtractor(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        timeout_seconds=30,
        max_output_tokens=512,
        context_tokens=16384,
        transport=transport,
    )

    report = extractor.extract_actionable((sentence,))

    assert [call["stage"] for call in transport.calls] == [
        "CONCEPT_GROUPING",
        "ATOMIC_RULE_EXTRACTION",
    ]
    assert all(call.stage != "PRESENCE" for call in report.call_telemetry)
    assert report.concept_count == 1
    assert report.atomic_rule_count == 1
    assert report.concepts[0].rules[0].evidence.evidence_id == sentence.evidence_id


def test_legacy_extract_skips_none_ambiguous_and_failed_presence_groups():
    def respond(prompt, _call_number):
        text = prompt["evidence"][0]["text"]
        if text.startswith("NONE"):
            return {"decision": "NONE"}
        if text.startswith("AMBIGUOUS"):
            return {"decision": "AMBIGUOUS"}
        return "MALFORMED"

    hits = (
        _hit("NONE source sentence.", "chunk-1"),
        _hit("AMBIGUOUS source sentence.", "chunk-2"),
        _hit("BROKEN source sentence.", "chunk-3"),
    )
    transport = _Transport(respond)
    report = AtomicStrategyExtractor(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        timeout_seconds=30,
        max_output_tokens=512,
        context_tokens=16384,
        transport=transport,
    ).extract(hits)

    assert report.concepts == ()
    assert [call["stage"] for call in transport.calls] == ["PRESENCE"] * 3


def test_presence_cache_reuses_same_ordered_evidence_across_family_metadata(tmp_path):
    calls = []

    def respond(prompt, _call_number):
        calls.append(prompt["stage"])
        return {"decision": "ACTIONABLE"}

    cache_path = tmp_path / "atomic.sqlite3"
    sentence = _group("momentum", "first-group", "LONG when momentum > 1.2 points.").sentences
    extractor = AtomicStrategyExtractor(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        timeout_seconds=30,
        max_output_tokens=512,
        context_tokens=16384,
        cache_path=cache_path,
        transport=_Transport(respond),
    )

    first = extractor.classify_presence((EvidenceGroup("momentum", "first-group", sentence),))
    second = extractor.classify_presence((EvidenceGroup("breakout", "different-group", sentence),))

    assert first.results[0].status is PresenceStatus.ACTIONABLE
    assert second.results[0].status is PresenceStatus.ACTIONABLE
    assert calls == ["PRESENCE"]
    assert second.cache_hits == 1
    assert second.llm_calls == 0
