from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .errors import SequenceTooLongError


def reduce_context(
    messages: tuple[dict[str, Any], ...],
    max_length: int,
    length_of: Callable[[tuple[dict[str, Any], ...]], int] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    """Reduce optional context without truncating the target or current state.

    ``length_of`` is supplied by the actual tokenizer.  The reducer therefore
    never relies on character counts or an assumed token/character ratio.
    Structural bundles are represented by deterministic parent/part IDs and
    are packed in source order; lower-priority rejected/available evidence is
    removed first.
    """
    if len(messages) < 2 or not isinstance(messages[1].get("content"), str):
        return messages, None
    try:
        context = json.loads(messages[1]["content"])
    except (ValueError, TypeError):
        return messages, None
    if not isinstance(context, dict):
        return messages, None
    reduced = dict(context)
    changes: dict[str, Any] = {"removed": [], "split_parts": []}

    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise SequenceTooLongError("maximum sequence length must be positive")

    def candidate(value: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        result = list(messages)
        result[1] = dict(result[1])
        result[1]["content"] = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return tuple(result)

    def fits(value: dict[str, Any]) -> bool:
        return length_of is None or length_of(candidate(value)) <= max_length

    if length_of is None or length_of(messages) <= max_length:
        return messages, None

    def reduction_record(value: dict[str, Any]) -> dict[str, Any]:
        # Preserve the reduced context keys for callers that need to audit
        # state preservation, with a namespaced metadata entry for removals.
        record = dict(value)
        record["_phase11_reduction"] = {
            "removed": list(changes["removed"]),
            "split_parts": list(changes["split_parts"]),
        }
        return record

    # Rejected/available evidence has the lowest priority.  Keep a bounded
    # audit of what was reduced; the audit itself never contains text.
    for key in ("rejected_evidence", "evidence_refs_rejected", "available_evidence", "dropped_counts"):
        if reduced.get(key):
            reduced[key] = [] if isinstance(reduced[key], list) else {}
            changes["removed"].append(key)
            if fits(reduced):
                return candidate(reduced), reduction_record(reduced)

    # Split tables/equation bundles into deterministic parts.  Header/caption
    # metadata is repeated on each part, and parts are packed in order until
    # the actual tokenizer says the sequence fits.
    for key in ("tables", "equations", "definitions"):
        items = reduced.get(key)
        if not isinstance(items, list):
            continue
        expanded: list[Any] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                expanded.append(item)
                continue
            rows_key = "rows" if isinstance(item.get("rows"), list) else ("items" if isinstance(item.get("items"), list) else None)
            rows = item.get(rows_key) if rows_key else None
            if not isinstance(rows, list) or len(rows) <= 1:
                expanded.append(item)
                continue
            parent = str(item.get("id", f"{key}-{index + 1}"))
            for part_no, start in enumerate(range(0, len(rows)), 1):
                part = dict(item)
                part[rows_key] = rows[start : start + 1]
                part["parent_id"] = parent
                part["part_id"] = f"{parent}/part-{part_no}"
                expanded.append(part)
                changes["split_parts"].append(part["part_id"])
        reduced[key] = expanded
        if fits(reduced):
            return candidate(reduced), reduction_record(reduced)
        # Keep structural bundles while removing trailing parts.  Never remove
        # a non-bundle item or the first part of a bundle.
        while len(expanded) > 1:
            trial = dict(reduced)
            trial[key] = expanded[:-1]
            if fits(trial):
                reduced = trial
                return candidate(reduced), reduction_record(reduced)
            expanded = expanded[:-1]
        reduced[key] = expanded

    if reduced != context and (length_of is None or fits(reduced)):
        return candidate(reduced), reduction_record(reduced)
    return messages, None


def ensure_capacity(length: int, max_length: int, *, target_length: int = 1) -> None:
    if length > max_length or target_length > max_length:
        raise SequenceTooLongError(f"sequence length {length} exceeds maximum {max_length}")


__all__ = ["SequenceTooLongError", "reduce_context", "ensure_capacity"]
