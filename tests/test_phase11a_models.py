from datetime import UTC, datetime
from pathlib import Path

import pytest

from tradingagents.distillation.models import (
    CONTRACT_VERSION,
    Difficulty,
    GroundingClaim,
    KnowledgeExample,
    LessonType,
    SourceBlock,
    SourcePacket,
    SourcePlan,
    SourceRef,
    canonical_hash,
    canonical_json,
)


def ref():
    return SourceRef("d", "book.pdf", "h", "c", "g", page=2, content_type="EQUATION")


def test_contracts_are_versioned_and_canonical():
    r = ref()
    assert CONTRACT_VERSION.endswith(".v1")
    assert canonical_hash(r) == canonical_hash(r)
    with pytest.raises(ValueError):
        SourceRef("", "book.pdf", "h", "c", "g")
    with pytest.raises(ValueError):
        SourceBlock(r, "x", metadata={"prompt": "bad"})


def test_packet_and_example_contracts_are_json_safe():
    block = SourceBlock(ref(), "OFI measures signed order flow")
    packet = SourcePacket("p", (block,))
    assert packet.refs == (ref(),)
    ex = KnowledgeExample(
        "e",
        LessonType.DEFINITION,
        "OFI",
        Difficulty.FOUNDATIONAL,
        "sys",
        "user",
        "assistant",
        (ref(),),
        (GroundingClaim("OFI", (ref(),)),),
    )
    assert ex.to_dict()["lesson_type"] == "DEFINITION"
    assert ex.system_instruction == ex.system == "sys"
    assert ex.user_instruction == ex.user == "user"
    assert ex.assistant_target == ex.assistant == "assistant"


def test_canonical_serialization_handles_paths_datetimes_enums_and_sets():
    payload = {
        "path": Path("catalog.sqlite3"),
        "when": datetime(2026, 1, 1, tzinfo=UTC),
        "enum": LessonType.DEFINITION,
        "values": {"b", "a"},
    }
    assert canonical_json(payload) == canonical_json(
        {
            "values": {"a", "b"},
            "enum": "DEFINITION",
            "when": "2026-01-01T00:00:00+00:00",
            "path": "catalog.sqlite3",
        }
    )
    assert canonical_hash(payload) == canonical_hash(
        {
            "values": {"a", "b"},
            "enum": "DEFINITION",
            "when": "2026-01-01T00:00:00+00:00",
            "path": "catalog.sqlite3",
        }
    )


def test_nested_sensitive_and_action_fields_are_rejected():
    with pytest.raises(ValueError, match="unsafe field"):
        SourceBlock(ref(), "equation", metadata={"nested": [{"reasoning_trace": "x"}]})
    with pytest.raises(ValueError, match="unsafe field"):
        SourceBlock(ref(), "equation", metadata={"future_outcome": "x"})


def test_knowledge_example_persists_policy_quality_split_and_fingerprints():
    ex = KnowledgeExample(
        "e",
        LessonType.DEFINITION,
        "OFI",
        Difficulty.FOUNDATIONAL,
        "sys",
        "user",
        "assistant",
        (ref(),),
        (GroundingClaim("OFI", (ref(),)),),
        policy_versions={"grounding": "g1"},
        quality_status="ACCEPTED",
        source_fingerprints={"phase7": "fp"},
        split="train",
    )
    data = ex.to_dict()
    assert data["policy_versions"] == {"grounding": "g1"}
    assert data["quality_status"] == "ACCEPTED"
    assert data["source_fingerprints"] == {"phase7": "fp"}
    assert data["split"] == "train"


def test_knowledge_example_persists_grounding_and_quality_reasons():
    ex = KnowledgeExample(
        "e",
        LessonType.DEFINITION,
        "OFI",
        Difficulty.FOUNDATIONAL,
        "sys",
        "user",
        "assistant",
        (ref(),),
        (GroundingClaim("OFI", (ref(),)),),
        grounding_status="GROUNDED",
        quality_status="ACCEPTED",
        quality_reasons=("PARAPHRASE",),
    )
    data = ex.to_dict()
    assert data["grounding_status"] == "GROUNDED"
    assert data["quality_status"] == "ACCEPTED"
    assert data["quality_reasons"] == ["PARAPHRASE"]


def test_contract_rejects_unknown_provenance_and_source_type():
    with pytest.raises(ValueError, match="content_type"):
        SourceRef("d", "book.pdf", "h", "c", "g", content_type="AUDIO")
    with pytest.raises(ValueError, match="contract_version"):
        KnowledgeExample(
            "e",
            LessonType.DEFINITION,
            "OFI",
            Difficulty.FOUNDATIONAL,
            "sys",
            "user",
            "assistant",
            (ref(),),
            (GroundingClaim("OFI", (ref(),)),),
            contract_version="phase11a-knowledge-example.v2",
        )


def test_knowledge_example_normalizes_status_and_requires_typed_refs():
    ex = KnowledgeExample(
        "e",
        LessonType.DEFINITION,
        "OFI",
        Difficulty.FOUNDATIONAL,
        "sys",
        "user",
        "assistant",
        (ref(),),
        (GroundingClaim("OFI", (ref(),)),),
        status="ACCEPTED",
        source_type="BOOK_KNOWLEDGE",
    )
    assert ex.status.value == "ACCEPTED"
    with pytest.raises(ValueError, match="source_type"):
        KnowledgeExample(
            "e",
            LessonType.DEFINITION,
            "OFI",
            Difficulty.FOUNDATIONAL,
            "sys",
            "user",
            "assistant",
            (ref(),),
            (GroundingClaim("OFI", (ref(),)),),
            source_type="TRADING_EXPERIENCE",
        )


def test_grounding_claim_rejects_untyped_refs():
    with pytest.raises(ValueError, match="SourceRef"):
        GroundingClaim("OFI", ("chunk-id",))


def test_source_plan_rejects_private_diagnostics():
    with pytest.raises(ValueError, match="unsafe private text"):
        SourcePlan(
            "g",
            (),
            "fingerprint",
            diagnostics=({"diagnostic": "private reasoning trace"},),
        )


def test_knowledge_example_requires_nonempty_grounding_provenance():
    with pytest.raises(ValueError, match="source_refs"):
        KnowledgeExample(
            "e",
            LessonType.DEFINITION,
            "OFI",
            Difficulty.FOUNDATIONAL,
            "sys",
            "user",
            "assistant",
            (),
            (GroundingClaim("OFI", (ref(),)),),
        )
    with pytest.raises(ValueError, match="claims"):
        KnowledgeExample(
            "e",
            LessonType.DEFINITION,
            "OFI",
            Difficulty.FOUNDATIONAL,
            "sys",
            "user",
            "assistant",
            (ref(),),
            (),
        )
