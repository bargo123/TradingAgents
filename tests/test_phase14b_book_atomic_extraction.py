from __future__ import annotations

import json as jsonlib
import sqlite3

from tradingagents.knowledge.models import KnowledgeHit
from tradingagents.self_enhancement.book_drafter import (
    AtomicRuleBatch,
    ConceptGrouping,
    OllamaStrategyDrafter,
    PresenceDecision,
    prepare_evidence_sentences,
)

GENERATION = "gen_atomic_fixture"
FINGERPRINT = "sha256:" + "b" * 64


def _hit(text: str, *, chunk_id: str = "chunk-1", section: str = "Rules") -> KnowledgeHit:
    return KnowledgeHit(
        chunk_id=chunk_id,
        document_id="doc-book-1",
        source_hash=(chunk_id[-1].lower() * 64)[:64],
        text=text,
        source_filename="short-horizon.pdf",
        title="Short Horizon Rules",
        page=7,
        section=section,
        extra={
            "projection_generation": GENERATION,
            "projection_population_hash": FINGERPRINT,
        },
    )


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
        request = {"url": url, "body": json, "timeout": timeout}
        self.calls.append(request)
        user = jsonlib.loads(json["messages"][-1]["content"])
        result = self.respond(user, len(self.calls))
        if isinstance(result, dict) and "message" in result:
            return _Response(result)
        return _Response(
            {
                "model": "qwen3.5:2b",
                "message": {"role": "assistant", "content": json_dumps(result)},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 41,
                "eval_count": 23,
            }
        )


def json_dumps(value):
    return jsonlib.dumps(value, sort_keys=True, separators=(",", ":"))


def _staged_response(user, _call_number):
    if user["stage"] == "PRESENCE":
        return {"decision": "ACTIONABLE"}
    indexes = [record["index"] for record in user["evidence"]]
    if user["stage"] == "CONCEPT_GROUPING":
        return {
            "candidates": [
                {"family": "MOMENTUM_CONTINUATION", "evidence_indices": indexes}
            ]
        }
    if user["stage"] == "ATOMIC_RULE_EXTRACTION":
        return {"rules": [{"stage": "ENTRY", "evidence_index": indexes[0]}]}
    raise AssertionError(f"unexpected stage: {user['stage']}")


def test_evidence_sentence_ids_and_offsets_are_stable_and_exact():
    hit = _hit("Opening thought. LONG when momentum > 1.2 points after confirmation. Closing thought.")

    first = prepare_evidence_sentences((hit,))
    second = prepare_evidence_sentences((hit,))

    assert [item.evidence_id for item in first] == [item.evidence_id for item in second]
    rule = next(item for item in first if item.text.startswith("LONG when momentum"))
    assert hit.text[rule.start_offset : rule.end_offset] == rule.text
    assert rule.text == "LONG when momentum > 1.2 points after confirmation."


def test_stage_schemas_are_small_closed_and_do_not_accept_strategy_prose():
    assert PresenceDecision.model_validate_json('{"decision":"ACTIONABLE"}').decision == "ACTIONABLE"
    assert ConceptGrouping.model_validate_json(
        '{"candidates":[{"family":"RANGE_REJECTION","evidence_indices":[0]}]}'
    )
    assert AtomicRuleBatch.model_validate_json(
        '{"rules":[{"stage":"ENTRY","evidence_index":0}]}'
    )
    atomic_properties = AtomicRuleBatch.model_json_schema()["$defs"]["AtomicRuleReference"]["properties"]
    assert set(atomic_properties) == {"stage", "evidence_index"}
    assert AtomicRuleBatch.model_json_schema()["additionalProperties"] is False


def test_drafter_resolves_model_evidence_ids_to_exact_source_and_uses_small_stage_caps():
    transport = _Transport(_staged_response)
    source = "LONG when momentum > 1.2 points after confirmation."
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=transport
    ).draft((_hit(source),), max_specs=2)

    assert result.ok is True
    assert len(transport.calls) == 3
    assert [jsonlib.loads(call["body"]["messages"][-1]["content"])["stage"] for call in transport.calls] == [
        "PRESENCE",
        "CONCEPT_GROUPING",
        "ATOMIC_RULE_EXTRACTION",
    ]
    assert [call["body"]["think"] for call in transport.calls] == [False, False, False]
    assert [call["body"]["options"]["temperature"] for call in transport.calls] == [0, 0, 0]
    assert all(call["body"]["options"]["num_predict"] <= 300 for call in transport.calls)
    claim = result.specs[0].rule_claims[0]
    assert claim.quote == source
    assert claim.start_offset == 0
    assert claim.end_offset == len(source)
    assert result.atomic_rule_count == 1
    assert result.llm_calls == 3


def test_stage_json_schemas_constrain_references_to_the_current_evidence_batch():
    transport = _Transport(_staged_response)
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=transport
    ).draft(
        (
            _hit(
                "LONG when momentum > 1.2 points after confirmation. "
                "Exit when momentum < 0.4 points after confirmation."
            ),
        )
    )

    assert result.llm_calls >= 3
    grouping_request = next(
        call for call in transport.calls
        if jsonlib.loads(call["body"]["messages"][-1]["content"])["stage"] == "CONCEPT_GROUPING"
    )
    grouping_indexes = [
        record["index"]
        for record in jsonlib.loads(grouping_request["body"]["messages"][-1]["content"])["evidence"]
    ]
    grouping_schema = grouping_request["body"]["format"]
    grouping_reference = grouping_schema["$defs"]["ConceptReference"]
    assert grouping_reference["properties"]["evidence_indices"]["items"]["enum"] == grouping_indexes
    assert grouping_reference["properties"]["evidence_indices"]["maxItems"] <= len(grouping_indexes)

    atomic_request = next(
        call for call in transport.calls
        if jsonlib.loads(call["body"]["messages"][-1]["content"])["stage"] == "ATOMIC_RULE_EXTRACTION"
    )
    atomic_indexes = [
        record["index"]
        for record in jsonlib.loads(atomic_request["body"]["messages"][-1]["content"])["evidence"]
    ]
    atomic_reference = atomic_request["body"]["format"]["$defs"]["AtomicRuleReference"]
    assert atomic_reference["properties"]["evidence_index"]["enum"] == atomic_indexes


def test_invalid_evidence_index_is_rejected_before_atomic_extraction():
    def respond(user, _call_number):
        if user["stage"] == "PRESENCE":
            return {"decision": "ACTIONABLE"}
        return {
            "candidates": [
                {"family": "MOMENTUM_CONTINUATION", "evidence_indices": [999]}
            ]
        }

    transport = _Transport(respond)
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=transport
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))

    assert result.specs == ()
    assert result.provenance_failure_count == 1
    assert len(transport.calls) == 2


def test_unsupported_numeric_rule_is_not_repaired_or_emitted():
    def respond(user, _call_number):
        if user["stage"] == "PRESENCE":
            return {"decision": "ACTIONABLE"}
        if user["stage"] == "CONCEPT_GROUPING":
            return {
                "candidates": [
                    {"family": "MOMENTUM_CONTINUATION", "evidence_indices": [user["evidence"][0]["index"]]}
                ]
            }
        return {"rules": [{"stage": "ENTRY", "evidence_index": user["evidence"][0]["index"]}]}

    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=_Transport(respond)
    ).draft((_hit("LONG when momentum > 1.2 invented_units after confirmation."),))

    assert result.specs == ()
    assert result.unsupported_rule_count == 1


def test_truncated_evidence_group_is_split_once_and_does_not_destroy_other_units():
    actionable = "LONG when momentum > 1.2 points after confirmation."

    def respond(user, call_number):
        evidence = user["evidence"]
        if user["stage"] == "PRESENCE":
            texts = [item["text"] for item in evidence]
            if call_number == 1:
                return {
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "length",
                    "prompt_eval_count": 70,
                    "eval_count": 64,
                }
            return {"decision": "ACTIONABLE" if actionable in texts else "NONE"}
        if user["stage"] == "CONCEPT_GROUPING":
            return {"candidates": [{"family": "MOMENTUM_CONTINUATION", "evidence_indices": [evidence[0]["index"]]}]}
        return {"rules": [{"stage": "ENTRY", "evidence_index": evidence[0]["index"]}]}

    hits = (
        _hit("BROKEN evidence. Ordinary context. More context.", chunk_id="chunk-1"),
        _hit(actionable, chunk_id="chunk-2"),
    )
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=_Transport(respond)
    ).draft(hits)

    assert len(result.specs) == 1
    assert result.specs[0].rule_claims[0].quote == actionable
    assert result.truncation_count == 1
    assert result.retry_count >= 1
    assert max(call.retry_depth for call in result.call_telemetry) == 1


def test_successful_atomic_stages_resume_from_cache_without_new_provider_calls(tmp_path):
    cache = tmp_path / "atomic.sqlite3"
    first_transport = _Transport(_staged_response)
    first = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=first_transport, cache_path=cache
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))

    second_transport = _Transport(lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))
    second = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=second_transport, cache_path=cache
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))

    assert first.specs == second.specs
    assert len(first_transport.calls) == 3
    assert second_transport.calls == []
    assert second.cache_hits == 3
    with sqlite3.connect(cache) as connection:
        row_count = connection.execute("SELECT COUNT(*) FROM atomic_results").fetchone()[0]
    assert row_count == 3


def test_model_version_change_invalidates_atomic_cache(tmp_path):
    cache = tmp_path / "atomic.sqlite3"
    first_transport = _Transport(_staged_response)
    first = OllamaStrategyDrafter(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        model_version="sha256:model-revision-1",
        transport=first_transport,
        cache_path=cache,
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))
    second_transport = _Transport(_staged_response)
    second = OllamaStrategyDrafter(
        "http://127.0.0.1:11434",
        "qwen3.5:2b",
        model_version="sha256:model-revision-2",
        transport=second_transport,
        cache_path=cache,
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))

    assert first.model_version == "sha256:model-revision-1"
    assert second.model_version == "sha256:model-revision-2"
    assert len(first_transport.calls) == len(second_transport.calls) == 3


def test_malformed_completion_is_not_persisted_or_semantically_repaired(tmp_path):
    marker = "completion-must-never-be-persisted"
    transport = _Transport(
        lambda *_: {
            "message": {"content": marker},
            "done": True,
            "done_reason": "stop",
        }
    )
    cache = tmp_path / "atomic.sqlite3"
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=transport, cache_path=cache
    ).draft((_hit("LONG when momentum > 1.2 points after confirmation."),))

    assert result.specs == ()
    assert result.schema_failure_count > 0
    assert marker not in jsonlib.dumps(result.to_dict())
    assert marker not in cache.read_bytes().decode("utf-8", errors="ignore")


def test_model_reference_contract_uses_bounded_compact_evidence_indices():
    def indexed_response(user, _call_number):
        if user["stage"] == "PRESENCE":
            return {"decision": "ACTIONABLE"}
        indexes = [record["index"] for record in user["evidence"]]
        if user["stage"] == "CONCEPT_GROUPING":
            return {
                "candidates": [
                    {"family": "MOMENTUM_CONTINUATION", "evidence_indices": indexes}
                ]
            }
        return {"rules": [{"stage": "ENTRY", "evidence_index": indexes[0]}]}

    transport = _Transport(indexed_response)
    source = "LONG when momentum > 1.2 points after confirmation."
    result = OllamaStrategyDrafter(
        "http://127.0.0.1:11434", "qwen3.5:2b", transport=transport
    ).draft((_hit(source),))

    assert result.ok is True
    assert result.specs[0].rule_claims[0].quote == source
    for call in transport.calls:
        user = jsonlib.loads(call["body"]["messages"][-1]["content"])
        if user["stage"] == "PRESENCE":
            assert all(set(record) == {"id", "text"} for record in user["evidence"])
        else:
            assert all(set(record) == {"index", "text"} for record in user["evidence"])
            assert [record["index"] for record in user["evidence"]] == list(
                range(len(user["evidence"]))
            )
