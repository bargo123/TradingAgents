import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

from tradingagents.experience.models import EvidenceBundle
from tradingagents.forex.evidence_context import (
    CanonicalKnowledgeQuery,
    EvidenceQueryPolicy,
    Phase9EvidenceContextBuilder,
)

AS_OF = datetime(2026, 9, 12, 12, tzinfo=timezone.utc)


def item(kind, ident, text, score=0.0, **extra):
    return SimpleNamespace(
        chunk_id=extra.pop("chunk_id", ident),
        document_id=extra.pop("document_id", f"doc-{ident}"),
        experience_id=extra.pop("experience_id", ident),
        content_type=extra.pop("content_type", "PROSE"),
        text=text,
        score=score,
        similarity_score=score,
        provenance=extra.pop("provenance", {"database_id": f"db-{ident}", "kind": kind}),
        source_database_id=extra.pop("source_database_id", f"db-{ident}"),
        source_decision_id=extra.pop("source_decision_id", f"decision-{ident}"),
        metadata=extra,
    )


def bundle(*, knowledge=(), experience=(), statistics=(), provenance=None, status="COMPLETE"):
    return EvidenceBundle(
        status=status,
        knowledge=tuple(knowledge),
        experience=tuple(experience),
        statistics=statistics,
        source_status={"knowledge": "COMPLETE", "experience": "COMPLETE", "statistics": "COMPLETE"},
        provenance=provenance or {"query_normalization_fingerprint": "norm-1"},
    )


def build_context(bundle_value, policy=None, as_of=AS_OF, **kwargs):
    return Phase9EvidenceContextBuilder().build(
        bundle_value,
        policy=policy or EvidenceQueryPolicy(),
        as_of=as_of,
        knowledge_query=CanonicalKnowledgeQuery("q", "qfp", "qv1"),
        phase7_generation_id="p7",
        phase8_generation_id="p8",
        **kwargs,
    )


def test_builder_preserves_phase8_experience_order():
    context = build_context(bundle(experience=[item("EXPERIENCE", "first", "one"), item("EXPERIENCE", "second", "two")]))
    assert [x.authoritative_id for x in context.experience_items] == ["first", "second"]


def test_builder_preserves_phase7_knowledge_order():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "first", "one"), item("KNOWLEDGE", "second", "two")]))
    assert [x.authoritative_id for x in context.knowledge_items] == ["first", "second"]


def test_builder_does_not_resort_by_raw_score():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "low", "low", 0.1), item("KNOWLEDGE", "high", "high", 0.9)]))
    assert [x.authoritative_id for x in context.knowledge_items] == ["low", "high"]


def test_builder_caps_sources_before_total_budget():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", str(i), str(i)) for i in range(6)], experience=[item("EXPERIENCE", str(i), str(i)) for i in range(6)]))
    assert len(context.knowledge_items) <= 4 and len(context.experience_items) <= 4
    assert [x.display_id for x in context.knowledge_items] == ["K1", "K2", "K3", "K4"]


def test_builder_assigns_stable_k_e_s_ids():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "k", "k")], experience=[item("EXPERIENCE", "e", "e")], statistics=[item("STATISTICS", "s", "s")]))
    assert [x.display_id for x in context.knowledge_items] == ["K1"]
    assert [x.display_id for x in context.experience_items] == ["E1"]
    assert [x.display_id for x in context.statistics_items] == ["S1"]


def test_builder_drops_whole_items_when_character_budget_is_exceeded():
    policy = EvidenceQueryPolicy(max_rendered_characters=300)
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "a", "x" * 200), item("KNOWLEDGE", "b", "y" * 200)]), policy=policy)
    assert context.rendered_character_count <= 300
    assert context.dropped_knowledge_count >= 1
    assert all("x" * 200 not in context.rendered_context or "y" * 200 not in context.rendered_context for _ in [0])


def test_context_hash_covers_exact_rendered_payload():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "k", "payload")]))
    assert context.rendered_context_hash == hashlib.sha256(context.rendered_context.encode()).hexdigest()
    assert json.loads(context.rendered_context)


def test_context_hash_changes_for_cutoff_or_policy():
    b = bundle(knowledge=[item("KNOWLEDGE", "k", "payload")])
    assert build_context(b).rendered_context_hash != build_context(b, as_of=AS_OF.replace(hour=13)).rendered_context_hash
    assert build_context(b).rendered_context_hash != build_context(b, policy=EvidenceQueryPolicy(budget_policy_version="v2")).rendered_context_hash


def test_provenance_ids_are_complete():
    context = build_context(bundle(knowledge=[item("KNOWLEDGE", "k", "text", document_id="doc-k", chunk_id="chunk-k", source_database_id="db-k")], experience=[item("EXPERIENCE", "e", "text", source_database_id="db-e", source_decision_id="dec-e")]))
    assert context.knowledge_items[0].provenance["document_id"] == "doc-k"
    assert context.knowledge_items[0].provenance["chunk_id"] == "chunk-k"
    assert context.experience_items[0].provenance["experience_id"] == "e"
    assert context.experience_items[0].provenance["source_database_id"] == "db-e"
    assert context.experience_items[0].provenance["source_decision_id"] == "dec-e"


def test_authoritative_ids_fall_back_to_structured_provenance():
    knowledge = SimpleNamespace(
        text="knowledge", content_type="PROSE", score=1.0,
        provenance={"document_id": "doc-from-prov", "chunk_id": "chunk-from-prov"},
    )
    experience = SimpleNamespace(
        text="experience", content_type="PROSE", score=1.0,
        provenance={"experience_id": "exp-from-prov", "source_database_id": "db-from-prov"},
    )
    context = build_context(bundle(knowledge=[knowledge], experience=[experience]))
    assert context.knowledge_items[0].authoritative_id == "chunk-from-prov"
    assert context.experience_items[0].authoritative_id == "exp-from-prov"


def test_builder_is_publicly_exported():
    import tradingagents.forex.evidence_context as module
    assert "Phase9EvidenceContextBuilder" in module.__all__


def test_builder_preserves_statistics_as_separate_items():
    context = build_context(bundle(statistics=[item("STATISTICS", "s1", "one"), item("STATISTICS", "s2", "two")]))
    assert [x.source_kind.value for x in context.statistics_items] == ["STATISTICS", "STATISTICS"]


def test_builder_never_calls_an_llm(monkeypatch):
    import builtins
    original_import = builtins.__import__
    monkeypatch.setattr("builtins.__import__", lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM import")) if "openai" in a[0].lower() else original_import(*a, **k))
    assert json.loads(build_context(bundle()).rendered_context)["items"] == []
