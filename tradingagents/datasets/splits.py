"""Deterministic chronological, grouped dataset splitting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .models import CanonicalExampleV1, SplitAssignment

SPLIT_STATUS_COMPLETE = "COMPLETE"
SPLIT_STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitResult:
    """Assignments plus the explicit population status used by the manifest."""

    assignments: tuple[SplitAssignment, ...]
    status: str

    def __iter__(self):
        return iter(self.assignments)

    def __len__(self):
        return len(self.assignments)


def _value(example: Any, name: str, default: Any = None) -> Any:
    if isinstance(example, Mapping):
        return example.get(name, default)
    return getattr(example, name, default)


def _mapping_value(example: Any, section: str, name: str, default: Any = None) -> Any:
    value = _value(example, section, {})
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _timestamp(example: Any) -> datetime | str:
    value = _mapping_value(example, "decision", "analysis_snapshot_timestamp")
    if value is None:
        raise ValueError("example is missing analysis_snapshot_timestamp")
    if not isinstance(value, (datetime, str)):
        raise ValueError("analysis_snapshot_timestamp must be datetime or ISO string")
    return value


def _timestamp_key(value: datetime | str) -> str:
    # ISO UTC strings sort chronologically after canonicalization. Datetimes are
    # normalized to their ISO representation without mutating the source row.
    return value.isoformat() if isinstance(value, datetime) else value


def _group_id(example: Any) -> str:
    run = _mapping_value(example, "decision", "source_run_id", "")
    decision = _mapping_value(example, "decision", "decision_id", "")
    return f"run:{run}" if run else f"decision:{decision}"


def _sort_key(example: Any) -> tuple[str, str, str, int, str]:
    decision = str(_mapping_value(example, "decision", "decision_id", ""))
    basis = str(_mapping_value(example, "outcome", "evaluation_basis", ""))
    horizon = _mapping_value(example, "outcome", "horizon_seconds", 0)
    example_id = str(_value(example, "example_id", ""))
    return (_timestamp_key(_timestamp(example)), decision, basis, int(horizon), example_id)


def _allocation(group_count: int) -> tuple[int, int, int]:
    """Largest-remainder allocation with deterministic split-order tie breaks."""
    targets = (group_count * 70, group_count * 15, group_count * 15)
    counts = [target // 100 for target in targets]
    remaining = group_count - sum(counts)
    remainders = [target % 100 for target in targets]
    for index in sorted(range(3), key=lambda i: (-remainders[i], i))[:remaining]:
        counts[index] += 1
    return tuple(counts)  # type: ignore[return-value]


def assign_splits(examples: Iterable[CanonicalExampleV1]) -> SplitResult:
    """Assign rows by chronological source group, independent of input order.

    Fewer than three groups cannot populate all three requested partitions and
    therefore returns an explicit insufficient status with no fabricated rows.
    """
    rows = sorted(examples, key=_sort_key)
    if len({_group_id(row) for row in rows}) < 3:
        return SplitResult((), SPLIT_STATUS_INSUFFICIENT_DATA)

    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(_group_id(row), []).append(row)
    groups = sorted(
        grouped.items(),
        key=lambda item: (_sort_key(item[1][0]), item[0]),
    )
    counts = _allocation(len(groups))
    group_split: dict[str, str] = {}
    offset = 0
    for split, count in zip(_SPLITS, counts, strict=True):
        for group_id, _ in groups[offset : offset + count]:
            group_split[group_id] = split
        offset += count
    assignments = tuple(
        SplitAssignment(_value(row, "example_id"), group_split[_group_id(row)], _group_id(row))
        for row in rows
    )
    result = SplitResult(assignments, SPLIT_STATUS_COMPLETE)
    validate_split_assignments(result, rows)
    return result


def validate_split_assignments(
    assignments: SplitResult | Iterable[SplitAssignment],
    examples: Iterable[CanonicalExampleV1] | None = None,
    *,
    status: str | None = None,
) -> None:
    """Validate uniqueness, chronology, and group isolation of assignments."""
    if isinstance(assignments, SplitResult):
        result_status = assignments.status
        rows = tuple(assignments.assignments)
    else:
        result_status = status or SPLIT_STATUS_COMPLETE
        rows = tuple(assignments)
    if result_status not in {SPLIT_STATUS_COMPLETE, SPLIT_STATUS_INSUFFICIENT_DATA}:
        raise ValueError("malformed split status")
    if result_status == SPLIT_STATUS_INSUFFICIENT_DATA:
        if rows:
            raise ValueError("insufficient split status must have no assignments")
        return
    if not rows:
        raise ValueError("complete split status requires assignments")
    if len({a.example_id for a in rows}) != len(rows):
        raise ValueError("duplicate example assignment")
    groups: dict[str, set[str]] = {}
    for assignment in rows:
        if assignment.split not in _SPLITS:
            raise ValueError("malformed split assignment")
        groups.setdefault(assignment.group_id, set()).add(assignment.split)
    if any(len(splits) > 1 for splits in groups.values()):
        raise ValueError("cross-split group")
    if examples is None:
        return
    example_rows = tuple(examples)
    by_id = {_value(row, "example_id"): row for row in example_rows}
    if len(by_id) != len(example_rows):
        raise ValueError("duplicate example key")
    if set(by_id) != {a.example_id for a in rows}:
        raise ValueError("assignment/example mismatch")
    canonical_keys = {
        (
            _mapping_value(row, "decision", "decision_id", ""),
            _mapping_value(row, "outcome", "evaluation_basis", ""),
            _mapping_value(row, "outcome", "horizon_seconds", 0),
        )
        for row in example_rows
    }
    if len(canonical_keys) != len(example_rows):
        raise ValueError("duplicate example key")
    input_order = [by_id[a.example_id] for a in rows]
    if any(
        _sort_key(left) > _sort_key(right)
        for left, right in zip(input_order, input_order[1:], strict=False)
    ):
        raise ValueError("non-chronological split rows")
    ordered = sorted(rows, key=lambda a: _sort_key(by_id[a.example_id]))
    rank = {split: i for i, split in enumerate(_SPLITS)}
    if any(rank[a.split] > rank[b.split] for a, b in zip(ordered, ordered[1:], strict=False)):
        raise ValueError("non-chronological split rows")
    for assignment in ordered:
        if assignment.group_id != _group_id(by_id[assignment.example_id]):
            raise ValueError("assignment group mismatch")


__all__ = [
    "SPLIT_STATUS_COMPLETE",
    "SPLIT_STATUS_INSUFFICIENT_DATA",
    "SplitResult",
    "assign_splits",
    "validate_split_assignments",
]
