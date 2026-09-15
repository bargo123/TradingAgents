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
