from __future__ import annotations

import json
from typing import Any

from .errors import SequenceTooLongError


def reduce_context(messages: tuple[dict[str, Any], ...], max_length: int, length_of) -> tuple[tuple[dict[str, Any], ...], dict[str, Any] | None]:
    """Reduce only optional evidence bundles, retaining state and target."""
    if not messages or not isinstance(messages[1].get("content"), str):
        return messages, None
    try:
        context = json.loads(messages[1]["content"])
    except (ValueError, TypeError):
        return messages, None
    if not isinstance(context, dict):
        return messages, None
    reduced = dict(context)
    changed = False
    # Rejected evidence is lowest priority and can be removed first.
    for key in ("rejected_evidence", "evidence_refs_rejected", "available_evidence"):
        if key in reduced and reduced[key]:
            reduced[key] = []
            changed = True
    # Tables/equations are split into deterministic parts, never dropped.
    for key in ("tables", "equations", "definitions"):
        items = reduced.get(key)
        if not isinstance(items, list):
            continue
        out = []
        for item in items:
            if not isinstance(item, dict):
                out.append(item); continue
            rows_key = "rows" if "rows" in item else ("items" if "items" in item else None)
            rows = item.get(rows_key) if rows_key else None
            if isinstance(rows, list) and len(rows) > 1:
                # Stable, bounded chunks; header/caption/equation definitions repeat.
                # Start with one row per part.  This is intentionally conservative:
                # tokenizers can have very different expansion ratios.
                chunk = 1
                for start in range(0, len(rows), chunk):
                    part = dict(item); part[rows_key] = rows[start:start + chunk]
                    if len(rows) > chunk:
                        part["parent_id"] = str(item.get("id", key))
                        part["part_id"] = f"{part['parent_id']}/part-{start // chunk + 1}"
                    out.append(part)
                    # Keep the first structural part in this example; the
                    # remaining deterministic parts are represented by IDs
                    # and can be materialized by a later packing pass.
                    if start == 0 and len(rows) > chunk:
                        break
                changed = changed or len(out) != len(items)
            else:
                out.append(item)
        reduced[key] = out
    if not changed:
        return messages, None
    candidate = list(messages)
    candidate[1] = dict(candidate[1]); candidate[1]["content"] = json.dumps(reduced, sort_keys=True, separators=(",", ":"))
    return tuple(candidate), reduced


def ensure_capacity(length: int, max_length: int, *, target_length: int = 1) -> None:
    if length > max_length or target_length > max_length:
        raise SequenceTooLongError(f"sequence length {length} exceeds maximum {max_length}")


__all__ = ["SequenceTooLongError", "reduce_context", "ensure_capacity"]
