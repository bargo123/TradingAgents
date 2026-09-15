"""Stable source-grouped train/validation/test assignment."""

from __future__ import annotations

import hashlib
from typing import Any


class GroupedSplitter:
    def __init__(self, *, ratios=(0.70, 0.15, 0.15), min_groups=3):
        self.ratios = ratios
        self.min_groups = min_groups

    def assign(self, examples: list[Any]) -> dict[str, str]:
        groups = {self._group(e) for e in examples}
        if len(groups) < self.min_groups:
            raise ValueError("INSUFFICIENT_DATA")
        ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        n = len(ordered)
        cut1 = max(1, round(n * self.ratios[0]))
        cut2 = max(cut1 + 1, round(n * (self.ratios[0] + self.ratios[1])))
        group_splits = {
            g: ("train" if i < cut1 else "validation" if i < cut2 else "test")
            for i, g in enumerate(ordered)
        }
        # Writer consumes example_id -> split; grouping remains durable because
        # every example derives its assignment from the same group key.
        return {self._example_id(e): group_splits[self._group(e)] for e in examples}

    def _group(self, e):
        if isinstance(e, dict):
            return (
                str(e.get("document_id", ""))
                + "|"
                + str(e.get("chapter", ""))
                + "|"
                + str(e.get("section", e.get("section_path", "")))
            )
        return (
            str(getattr(e, "document_id", ""))
            + "|"
            + str(getattr(e, "chapter", ""))
            + "|"
            + str(getattr(e, "section", getattr(e, "section_path", "")))
        )

    def _example_id(self, e):
        if isinstance(e, dict):
            return str(
                e.get("example_id")
                or e.get("id")
                or hashlib.sha256(repr(sorted(e.items())).encode()).hexdigest()
            )
        return str(getattr(e, "example_id", ""))


def validate_no_group_leakage(examples: list[Any], assignments: dict[str, str]) -> bool:
    seen = {}
    for e in examples:
        g = GroupedSplitter()._group(e)
        eid = GroupedSplitter()._example_id(e)
        split = assignments.get(eid, assignments.get(g))
        if g in seen and seen[g] != split:
            return False
        seen[g] = split
    return True
