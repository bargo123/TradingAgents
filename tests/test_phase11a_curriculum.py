import json

from tests.fixtures.phase11a_knowledge import fixture_source, grounded_candidate
from tradingagents.distillation.curriculum import (
    KnowledgeDatasetBinding,
    KnowledgeGenerationInvalidError,
)
from tradingagents.distillation.factory import _coerce_candidate
from tradingagents.distillation.writer import write_generation


def test_curriculum_exposes_only_train_and_validation(tmp_path):
    source = fixture_source()
    packet = next(iter(source.blocks()))
    from tradingagents.distillation.models import SourcePacket

    row = _coerce_candidate(
        grounded_candidate(SourcePacket("p", (packet,))), SourcePacket("p", (packet,))
    )
    generation = write_generation(tmp_path, [row], [], {row.example_id: "train"}, metadata={})
    binding = KnowledgeDatasetBinding.open(generation)
    assert len(binding.train_rows) == 1
    assert binding.validation_rows == ()
    assert binding.test_rows == ()
    assert binding.training_source.source_type == "BOOK_KNOWLEDGE"


def test_curriculum_rejects_tampered_manifest(tmp_path):
    source = fixture_source()
    packet = next(iter(source.blocks()))
    from tradingagents.distillation.models import SourcePacket

    packet = SourcePacket("p", (packet,))
    row = _coerce_candidate(grounded_candidate(packet), packet)
    generation = write_generation(tmp_path, [row], [], {row.example_id: "train"}, metadata={})
    manifest = generation / "manifest.json"
    value = json.loads(manifest.read_text(encoding="utf-8"))
    value["schema_version"] = "bad"
    manifest.write_text(json.dumps(value), encoding="utf-8")
    try:
        KnowledgeDatasetBinding.open(generation)
    except KnowledgeGenerationInvalidError:
        pass
    else:
        raise AssertionError("expected invalid generation")


def test_curriculum_rejects_insufficient_split_status(tmp_path):
    destination = write_generation(
        tmp_path, [], [], {}, metadata={"split_status": "INSUFFICIENT_DATA"}
    )
    try:
        KnowledgeDatasetBinding.open(destination)
    except KnowledgeGenerationInvalidError:
        return
    raise AssertionError("expected insufficient split rejection")
