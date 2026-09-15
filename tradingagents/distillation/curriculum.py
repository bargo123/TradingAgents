"""Explicit knowledge-only curriculum handoff to the Phase 11 trainer."""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

@dataclass(frozen=True)
class CurriculumPlan:
    stages: tuple[str, ...] = ("KNOWLEDGE", "EXPERIENCE")

@dataclass(frozen=True)
class KnowledgeTrainingSource:
    path: Path
    source_type: str = "BOOK_KNOWLEDGE"
    def rows(self) -> Iterator[dict]:
        for name in ("train.jsonl", "validation.jsonl"):
            file = self.path / name
            if file.is_file():
                for line in file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("source_type", self.source_type) != self.source_type:
                            raise ValueError("knowledge source type mismatch")
                        yield row

    def test_rows(self) -> Iterator[dict]:
        """Tests are intentionally unavailable through the training binding."""
        return iter(())

@dataclass(frozen=True)
class KnowledgeDatasetBinding:
    path: Path
    curriculum: CurriculumPlan = CurriculumPlan()
    @classmethod
    def open(cls, path: str | Path) -> "KnowledgeDatasetBinding":
        root = Path(path)
        if not (root / "manifest.json").is_file():
            raise FileNotFoundError("knowledge generation manifest is missing")
        return cls(root)
    @property
    def training_source(self) -> KnowledgeTrainingSource:
        return KnowledgeTrainingSource(self.path)
    def rows(self) -> Iterator[dict]:
        return self.training_source.rows()

__all__ = ["KnowledgeDatasetBinding", "KnowledgeTrainingSource", "CurriculumPlan"]
