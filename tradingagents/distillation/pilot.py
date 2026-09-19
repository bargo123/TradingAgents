"""Bounded representative packet selection for explicit Phase 11A pilots."""

from __future__ import annotations

from dataclasses import replace

from .models import SourcePlan, canonical_hash


def select_representative_packets(plan: SourcePlan, limit: int) -> SourcePlan:
    """Return a deterministic 1..10 packet subset without touching Phase 7."""

    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 10:
        raise ValueError("pilot packet count must be between 1 and 10")
    packets = tuple(plan.packets)
    if len(packets) <= limit:
        selected = packets
    elif limit == 1:
        selected = (packets[0],)
    else:
        indexes = sorted({round(index * (len(packets) - 1) / (limit - 1)) for index in range(limit)})
        selected = tuple(packets[index] for index in indexes)
    fingerprint = canonical_hash(
        {
            "parent_request_fingerprint": plan.request_fingerprint,
            "selected_packet_ids": [packet.packet_id for packet in selected],
            "limit": limit,
        }
    )
    diagnostics = tuple(plan.diagnostics) + (
        {"code": "PILOT_SUBSET", "selected_packets": len(selected), "requested_limit": limit},
    )
    return replace(plan, packets=selected, request_fingerprint=fingerprint, diagnostics=diagnostics)


__all__ = ["select_representative_packets"]
