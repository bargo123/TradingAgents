"""Stable source-grouped train/validation/test assignment."""

from __future__ import annotations

import hashlib
from typing import Any


class GroupedSplitter:
    def __init__(self, *, ratios=(0.70, 0.15, 0.15), min_groups=3):
        if len(ratios) != 3 or any(float(x) <= 0 for x in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three positive values summing to one")
        if min_groups < 3:
            raise ValueError("min_groups must be at least three")
        self.ratios = ratios
        self.min_groups = min_groups

    def assign(self, examples: list[Any]) -> dict[str, str]:
        groups = {self._group(e) for e in examples}
        if len(groups) < self.min_groups:
            raise ValueError("INSUFFICIENT_DATA")
        ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        n = len(ordered)
        cut1 = min(n - 2, max(1, round(n * self.ratios[0])))
        cut2 = min(n - 1, max(cut1 + 1, round(n * (self.ratios[0] + self.ratios[1]))))
        group_splits = {
            g: ("train" if i < cut1 else "validation" if i < cut2 else "test")
            for i, g in enumerate(ordered)
        }
        # Writer consumes example_id -> split; grouping remains durable because
        # every example derives its assignment from the same group key.
        return {self._example_id(e): group_splits[self._group(e)] for e in examples}

    def _group(self, e):
        if isinstance(e, dict):
            if e.get("source_refs"):
                refs = e["source_refs"]
                locations = []
                for ref in refs:
                    get = ref.get if isinstance(ref, dict) else lambda key, default=None: getattr(ref, key, default)
                    locations.append(
                        f"{get('document_id', '')}|{get('chapter', '')}|{get('section', '')}"
                    )
                return "||".join(sorted(set(locations)))
            return (
                str(e.get("document_id", ""))
                + "|"
                + str(e.get("chapter", ""))
                + "|"
                + str(e.get("section", e.get("section_path", "")))
            )
        refs = getattr(e, "source_refs", ())
        if refs:
            # A multi-source lesson is one indivisible provenance group.  Use
            # all source locations, order-independent, so no source can leak
            # across partitions by virtue of appearing after the first ref.
            locations = []
            for ref in refs:
                section = getattr(ref, "section", None) or " / ".join(
                    getattr(ref, "section_path", ()) or ()
                )
                locations.append(
                    f"{getattr(ref, 'document_id', '')}|{getattr(ref, 'chapter', '')}|{section}"
                )
            return "||".join(sorted(set(locations)))
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
