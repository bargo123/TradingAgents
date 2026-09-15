"""Stable source-grouped train/validation/test assignment."""

from __future__ import annotations

import hashlib
from typing import Any


class GroupedSplitter:
    def __init__(self, *, ratios=(0.70, 0.15, 0.15), min_groups=3):
        if len(ratios) != 3 or any(float(x) <= 0 for x in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
            raise ValueError("ratios must be three positive values summing to one")
        if isinstance(min_groups, bool) or not isinstance(min_groups, int) or min_groups < 1:
            raise ValueError("min_groups must be a positive integer")
        self.ratios = ratios
        self.min_groups = min_groups

    def assign(self, examples: list[Any]) -> dict[str, str]:
        component_groups = self._component_groups(examples)
        groups = set(component_groups)
        if len(groups) < self.min_groups:
            raise ValueError("INSUFFICIENT_DATA")
        ordered = sorted(groups, key=lambda g: hashlib.sha256(g.encode()).hexdigest())
        n = len(ordered)
        counts = self._split_counts(n)
        labels = (
            ("train",) * counts[0]
            + ("validation",) * counts[1]
            + ("test",) * counts[2]
        )
        group_splits = {g: labels[i] for i, g in enumerate(ordered)}
        # Writer consumes example_id -> split; grouping remains durable because
        # every example derives its assignment from the same group key.
        return {
            self._example_id(e): group_splits[component_groups[index]]
            for index, e in enumerate(examples)
        }

    def _split_counts(self, n: int) -> tuple[int, int, int]:
        if n <= 0:
            return (0, 0, 0)
        if n < 3:
            return (n, 0, 0)
        remaining = n - 3
        raw = [remaining * float(ratio) for ratio in self.ratios]
        counts = [1 + int(value) for value in raw]
        left = remaining - sum(int(value) for value in raw)
        fractions = sorted(
            ((raw[index] - int(raw[index]), index) for index in range(3)),
            key=lambda item: (-item[0], item[1]),
        )
        for _ in range(left):
            counts[fractions[_][1]] += 1
        return tuple(counts)  # type: ignore[return-value]

    def _locations(self, e: Any) -> set[str]:
        refs = e.get("source_refs", ()) if isinstance(e, dict) else getattr(e, "source_refs", ())
        locations: set[str] = set()
        for ref in refs or ():
            if isinstance(ref, dict):
                document = ref.get("document_id", "")
                chapter = ref.get("chapter", "")
                section = ref.get("section") or " / ".join(ref.get("section_path", ()) or ())
            else:
                document = getattr(ref, "document_id", "")
                chapter = getattr(ref, "chapter", "")
                section = getattr(ref, "section", None) or " / ".join(
                    getattr(ref, "section_path", ()) or ()
                )
            locations.add(f"{document}|{chapter}|{section}")
        return locations or {self._group(e)}

    def _component_groups(self, examples: list[Any]) -> list[str]:
        """Return connected provenance components for each example.

        Multi-source lessons and single-source lessons that share a source
        boundary must remain in one split; otherwise a synthesis lesson could
        leak its component into another partition.
        """

        parent: dict[str, str] = {}

        def find(value: str) -> str:
            parent.setdefault(value, value)
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(left: str, right: str) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[max(root_left, root_right)] = min(root_left, root_right)

        locations = [self._locations(example) for example in examples]
        for values in locations:
            values = sorted(values)
            for value in values:
                find(value)
            for value in values[1:]:
                union(values[0], value)
        members_by_root: dict[str, set[str]] = {}
        for member in parent:
            members_by_root.setdefault(find(member), set()).add(member)
        key_by_root = {
            root: "||".join(sorted(members))
            for root, members in members_by_root.items()
        }
        for values in locations:
            first = next(iter(values))
            find(first)
        return [key_by_root[find(next(iter(values)))] for values in locations]

    def _group(self, e):
        if isinstance(e, dict):
            if e.get("source_refs"):
                refs = e["source_refs"]
                locations = []
                for ref in refs:
                    if isinstance(ref, dict):
                        get = ref.get
                    else:
                        def get(key, default=None, _ref=ref):
                            return getattr(_ref, key, default)
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
    splitter = GroupedSplitter()
    groups = splitter._component_groups(examples)
    seen = {}
    for index, e in enumerate(examples):
        g = groups[index]
        eid = splitter._example_id(e)
        split = assignments.get(eid, assignments.get(g))
        if g in seen and seen[g] != split:
            return False
        seen[g] = split
    return True
