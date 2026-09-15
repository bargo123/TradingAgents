"""Point-in-time, deterministic projection of Phase 10 rows for SFT."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ContractError
from .models import SFT_FORMAT_VERSION, SFTExample, canonical_json

_ACTION = {"BUY", "SELL", "HOLD"}
_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")
_SENSITIVE = re.compile(r"(?:secret|password|credential|api[_ -]?key|access[_ -]?token|private[_ -]?key|prompt|completion|reasoning|chain[_ -]?of[_ -]?thought|\bcot\b|scratch)", re.I)

_DECISION = frozenset({
    "action", "resolved_symbol", "requested_symbol", "analysis_profile", "analysis_timeframe",
    "decision_context_status", "normalization_status", "analysis_snapshot_timestamp",
    "snapshot_timestamp", "decision_completed_timestamp", "valid_for_seconds", "valid_until",
})
_MARKET = frozenset({"symbol", "point", "digits", "snapshot_timestamp", "analysis_snapshot_timestamp", "quote", "features", "bars", "indicators", "current_state"})
_RESEARCH = frozenset({"evidence_refs", "evidence_refs_used", "used_evidence", "available_evidence", "evidence_refs_rejected"})
_PROVENANCE = frozenset({"source_id", "canonical_path", "schema_fingerprint", "file_sha256", "snapshot_fingerprint", "contract_version", "phase56", "phase8", "phase9"})


def _value(row: Any, name: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(name, default)
    return getattr(row, name, default)


def _safe_source(value: Any, *, key: str = "") -> None:
    if key and _SENSITIVE.search(key):
        raise ContractError(f"forbidden sensitive field: {key}")
    if isinstance(value, Mapping):
        for name, child in value.items():
            _safe_source(child, key=str(name))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _safe_source(child)
    elif isinstance(value, str) and _SENSITIVE.search(value):
        raise ContractError("forbidden sensitive value")


def _project(value: Mapping[str, Any], allowed: frozenset[str], *, reject_unknown: bool = False) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError("source section must be a mapping")
    for key, child in value.items():
        if _SENSITIVE.search(str(key)):
            raise ContractError(f"forbidden sensitive field: {key}")
        if reject_unknown and key not in allowed:
            raise ContractError(f"unknown source field: {key}")
        if key in allowed:
            _safe_source(child, key=str(key))
    return {str(key): value[key] for key in sorted(value) if key in allowed}


def _refs(*sections: Mapping[str, Any]) -> list[str]:
    found: set[str] = set()
    for section in sections:
        for key in ("evidence_refs", "evidence_refs_used", "used_evidence", "available_evidence"):
            raw = section.get(key, ())
            if isinstance(raw, str):
                raw = [raw]
            if raw is None:
                continue
            if not isinstance(raw, (list, tuple, set, frozenset)):
                raise ContractError("evidence references must be a sequence")
            for ref in raw:
                if not isinstance(ref, str) or not _REF.fullmatch(ref):
                    raise ContractError("invalid evidence reference")
                found.add(ref)
    return sorted(found)


def parse_target(target: Mapping[str, Any] | str) -> dict[str, Any]:
    """Parse the closed assistant target schema without accepting extensions."""
    if isinstance(target, str):
        try:
            target = json.loads(target)
        except json.JSONDecodeError as exc:
            raise ContractError("target is not valid JSON") from exc
    if not isinstance(target, Mapping) or set(target) != {"action", "evidence_refs"}:
        raise ContractError("target keys must be exactly action and evidence_refs")
    action = target["action"]
    if not isinstance(action, str) or action not in _ACTION:
        raise ContractError("target action must be BUY, SELL, or HOLD")
    evidence = target["evidence_refs"]
    if not isinstance(evidence, (list, tuple)) or any(not isinstance(x, str) or not _REF.fullmatch(x) for x in evidence):
        raise ContractError("invalid evidence references")
    normalized = list(evidence)
    if normalized != sorted(set(normalized)):
        raise ContractError("evidence references must be unique and sorted")
    return {"action": action, "evidence_refs": normalized}


@dataclass(frozen=True, slots=True)
class SFTFormatter:
    policy_version: str = SFT_FORMAT_VERSION

    def __post_init__(self) -> None:
        if self.policy_version != SFT_FORMAT_VERSION:
            raise ContractError("unsupported formatter policy version")

    def format(self, row: Any, split: str | None = None) -> SFTExample:
        example_id = _value(row, "example_id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ContractError("row example_id is required")
        split = split or _value(row, "split", "train")
        if split not in {"train", "validation"}:
            raise ContractError("invalid SFT split")
        decision = _value(row, "decision", {})
        market = _value(row, "market", {})
        research = _value(row, "research", {})
        provenance = _value(row, "provenance", {})
        # Trust and outcome are deliberately inspected only for forbidden keys,
        # then dropped. Their values can never reach either prompt or target.
        _safe_source(_value(row, "trust", {}))
        _safe_source(_value(row, "outcome", {}))
        d = _project(decision, _DECISION)
        m = _project(market, _MARKET)
        r = _project(research, _RESEARCH)
        p = _project(provenance, _PROVENANCE)
        action = d.get("action")
        if not isinstance(action, str) or action not in _ACTION:
            raise ContractError("source action must be BUY, SELL, or HOLD")
        target = parse_target({"action": action, "evidence_refs": _refs(r, p)})
        user = {"decision": d, "market": m, "research": r, "provenance": p}
        messages = (
            {"role": "system", "content": "Use only the supplied point-in-time state and evidence references."},
            {"role": "user", "content": canonical_json(user)},
            {"role": "assistant", "content": canonical_json(target)},
        )
        return SFTExample(example_id=example_id, split=split, messages=tuple(messages), target=target)

    format_example = format


__all__ = ["SFTFormatter", "parse_target"]
