from tradingagents.self_enhancement.book_factory import BookStrategyFactory
from tradingagents.self_enhancement.candidates import CandidateGenerator
from tradingagents.self_enhancement.models import CandidateState, ExitPolicyConfig, StrategyVersion


def _parent():
    return StrategyVersion(
        "range_rejection",
        "incumbent-v1",
        "cfg-v1",
        ExitPolicyConfig().to_dict(),
        "abc123",
    )


def test_candidate_generator_is_deterministic_and_bounded():
    first = CandidateGenerator().generate(_parent())
    second = CandidateGenerator().generate(_parent())
    assert [item.candidate_id for item in first] == [item.candidate_id for item in second]
    assert len(first) == 6
    assert all(item.execution_mode.value == "SHADOW" for item in first)
    assert all(item.state is CandidateState.EXPERIMENTAL for item in first)
    assert all(item.real_money is False for item in first)


def test_book_factory_preserves_exact_phase7_provenance_and_never_approves():
    evidence = {
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "source_filename": "paper.pdf",
        "source_hash": "hash-1",
        "page": 4,
        "section": "3.1",
        "content_type": "PROSE",
        "text": "Protect favorable excursion after failed continuation.",
    }
    candidate = BookStrategyFactory().from_hits((evidence,), parent=_parent())[0]
    assert candidate.state is CandidateState.EXTRACTED
    assert candidate.source_evidence == (evidence,)
    assert candidate.execution_mode.value == "SHADOW"


def test_book_factory_rejects_anonymous_evidence():
    try:
        BookStrategyFactory().from_hits(({"text": "unproven"},), parent=_parent())
    except ValueError as exc:
        assert "provenance" in str(exc)
    else:
        raise AssertionError("anonymous book evidence must fail closed")
